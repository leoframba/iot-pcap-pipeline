"""Phase 1C.3a streaming Parquet storage for frozen V1 feature rows.

Pipeline:
  PCAP → iter_packets → iter_windows → extract_features → small buffer → Parquet

Does not change windowing or any of the 27 feature definitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from iot_pcap_pipeline.features.extractor import FeatureVector, extract_features
from iot_pcap_pipeline.features.schema import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    V1_FEATURE_NAMES,
    V1_FEATURE_SPECS,
    write_feature_schema,
)
from iot_pcap_pipeline.features.validate import validate_window_and_features
from iot_pcap_pipeline.paths import (
    DEFAULT_FEATURES_DIR,
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import frozen_window_policy
from iot_pcap_pipeline.windowing.stream import WindowStreamStats, iter_windows

DEFAULT_BUFFER_ROWS = 2_048
PARQUET_COMPRESSION = "zstd"

DEFAULT_PARQUET_SMOKE_DIR = DEFAULT_FEATURES_DIR / "v1" / "smoke" / "parquet"
DEFAULT_FEATURE_CHECKPOINT_DIR = DEFAULT_FEATURES_DIR / "v1" / ".work" / "train"

# Row identity + window coords + frozen V1 features (no attack/device/path columns).
PARQUET_IDENTITY_COLUMNS: tuple[str, ...] = (
    "pcap_id",
    "binary_label",
    "segment_index",
    "window_index",
    "packet_index_start",
    "packet_index_end",
)
PARQUET_COLUMN_NAMES: tuple[str, ...] = PARQUET_IDENTITY_COLUMNS + V1_FEATURE_NAMES

DEFAULT_PARQUET_SMOKE_PCAP_PATHS: tuple[str, ...] = (
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Benign_train.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Idle/Idle.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Recon-VulScan_train.pcap",
)


def pcap_id_from_path(
    pcap_path: Path | str,
    *,
    project_root: Path | None = None,
) -> str:
    """Path-stable shard id: ``<stem>-<sha256(repo-relative)[:16]>``."""
    path = Path(pcap_path)
    rel = to_repo_relative(path, project_root=project_root)
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]
    return f"{path.stem}-{digest}"


def feature_schema_sha256(schema_path: Path | str | None = None) -> str:
    """SHA-256 of the on-disk V1 feature schema JSON (utf-8 bytes)."""
    path = Path(schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not path.is_file():
        write_feature_schema(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_parquet_arrow_schema() -> pa.Schema:
    """Explicit Arrow schema for one V1 feature shard row."""
    fields: list[pa.Field] = [
        pa.field("pcap_id", pa.string(), nullable=False),
        pa.field("binary_label", pa.string(), nullable=False),
        pa.field("segment_index", pa.int64(), nullable=False),
        pa.field("window_index", pa.int64(), nullable=False),
        pa.field("packet_index_start", pa.int64(), nullable=False),
        pa.field("packet_index_end", pa.int64(), nullable=False),
    ]
    for spec in V1_FEATURE_SPECS:
        arrow_type = pa.float64() if spec.dtype == "float64" else pa.int64()
        fields.append(pa.field(spec.name, arrow_type, nullable=False))
    return pa.schema(fields)


def _row_from_window(
    *,
    pcap_id: str,
    binary_label: str,
    window: Any,
    features: FeatureVector,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pcap_id": pcap_id,
        "binary_label": binary_label,
        "segment_index": int(window.segment_index),
        "window_index": int(window.window_index),
        "packet_index_start": int(window.packet_index_start),
        "packet_index_end": int(window.packet_index_end),
    }
    for name in V1_FEATURE_NAMES:
        value = getattr(features, name)
        if name in ("unique_ip_count", "unique_port_count"):
            row[name] = int(value)
        else:
            row[name] = float(value)
    return row


class StreamingFeatureParquetWriter:
    """Bounded-memory Parquet writer: buffer → batch → clear → continue."""

    def __init__(
        self,
        tmp_path: Path | str,
        *,
        buffer_rows: int = DEFAULT_BUFFER_ROWS,
        compression: str = PARQUET_COMPRESSION,
        schema: pa.Schema | None = None,
    ) -> None:
        if buffer_rows < 1:
            raise ValueError("buffer_rows must be >= 1")
        self.tmp_path = Path(tmp_path)
        self.buffer_rows = buffer_rows
        self.compression = compression
        self.schema = schema or feature_parquet_arrow_schema()
        self._buffer: list[dict[str, Any]] = []
        self._rows_written = 0
        self._closed = False
        self._aborted = False
        self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path.unlink(missing_ok=True)
        self._writer = pq.ParquetWriter(
            where=str(self.tmp_path),
            schema=self.schema,
            compression=self.compression,
        )

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def append(self, row: dict[str, Any]) -> None:
        if self._closed or self._aborted:
            raise RuntimeError("writer is closed")
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        assert self._writer is not None
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> int:
        """Flush remaining rows and close the underlying writer."""
        if self._aborted:
            raise RuntimeError("writer was aborted")
        if self._closed:
            return self._rows_written
        try:
            self._flush()
            assert self._writer is not None
            self._writer.close()
            self._writer = None
            self._closed = True
            return self._rows_written
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Close without promoting; remove the temporary shard if present."""
        if self._aborted:
            return
        self._aborted = True
        self._buffer.clear()
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.close()
            except (OSError, pa.ArrowInvalid, pa.ArrowIOError):
                pass
        self.tmp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class BuildResult:
    pcap_id: str
    pcap_path: str
    output_path: Path
    checkpoint_path: Path
    row_count: int
    packets_processed: int
    windows_written: int
    input_file_size: int
    output_file_size: int
    elapsed_seconds: float
    resumed: bool
    feature_strategy_version: str
    feature_build_strategy_version: str
    feature_schema_sha256: str


def checkpoint_path_for(
    pcap_id: str,
    *,
    checkpoint_dir: Path | None = None,
) -> Path:
    root = Path(checkpoint_dir or DEFAULT_FEATURE_CHECKPOINT_DIR)
    return root / f"{pcap_id}.json"


def write_build_checkpoint(payload: dict[str, Any], path: Path) -> Path:
    """Atomically write a completed per-PCAP build checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_build_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def parquet_schema_matches(path: Path, expected: pa.Schema | None = None) -> bool:
    expected = expected or feature_parquet_arrow_schema()
    try:
        actual = pq.read_schema(path)
    except (OSError, pa.ArrowInvalid, pa.ArrowIOError):
        return False
    return actual.equals(expected, check_metadata=False)


def parquet_row_count(path: Path) -> int | None:
    try:
        meta = pq.ParquetFile(path).metadata
    except (OSError, pa.ArrowInvalid, pa.ArrowIOError):
        return None
    if meta is None:
        return None
    return int(meta.num_rows)


def checkpoint_is_reusable(
    *,
    checkpoint_path: Path,
    output_path: Path,
    pcap_path: Path,
    input_file_size: int,
    schema_sha256: str,
    binary_label: str,
    project_root: Path | None = None,
    expected_schema: pa.Schema | None = None,
) -> bool:
    """Return True only when checkpoint + shard pass all resume checks."""
    root = (project_root or PROJECT_ROOT).resolve()
    payload = load_build_checkpoint(checkpoint_path)
    if payload is None:
        return False

    pcap = Path(pcap_path)
    out = Path(output_path)
    expected_rel = to_repo_relative(pcap, project_root=root)
    expected_out_rel = to_repo_relative(out, project_root=root)
    expected_id = pcap_id_from_path(pcap, project_root=root)

    if payload.get("pcap_id") != expected_id:
        return False
    if payload.get("pcap_path") != expected_rel:
        return False
    if payload.get("binary_label") != binary_label:
        return False
    if payload.get("output_path") != expected_out_rel:
        return False
    if int(payload.get("input_file_size", -1)) != int(input_file_size):
        return False
    if payload.get("feature_strategy_version") != FEATURE_STRATEGY_VERSION:
        return False
    if payload.get("feature_build_strategy_version") != FEATURE_BUILD_STRATEGY_VERSION:
        return False
    if payload.get("feature_schema_sha256") != schema_sha256:
        return False

    if not out.is_file():
        return False
    if int(payload.get("output_file_size", -1)) != int(out.stat().st_size):
        return False
    if parquet_row_count(out) != int(payload.get("output_row_count", -1)):
        return False
    return parquet_schema_matches(out, expected_schema)


def read_feature_rows_from_parquet(path: Path | str) -> list[dict[str, Any]]:
    """Read all shard rows as plain dicts (test / verification helper)."""
    table = pq.read_table(path, schema=feature_parquet_arrow_schema())
    return table.to_pylist()


def build_pcap_parquet(
    pcap_path: Path | str,
    metadata: dict[str, Any] | None,
    output_path: Path | str,
    *,
    checkpoint_path: Path | str | None = None,
    project_root: Path | None = None,
    schema_path: Path | str | None = None,
    resume: bool = True,
    buffer_rows: int = DEFAULT_BUFFER_ROWS,
    validate: bool = True,
    max_windows: int | None = None,
) -> BuildResult:
    """Stream one PCAP into an atomic Parquet shard with a resume checkpoint.

    Reuses ``iter_packets`` → ``iter_windows`` → ``extract_features`` →
    ``validate_window_and_features``. No second extractor.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    pcap = Path(pcap_path)
    if not pcap.is_absolute():
        pcap = root / pcap
    pcap = pcap.resolve()
    if not pcap.is_file():
        raise FileNotFoundError(f"PCAP not found: {pcap}")

    out = Path(output_path)
    if not out.is_absolute():
        out = root / out
    pcap_id = pcap_id_from_path(pcap, project_root=root)
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else checkpoint_path_for(pcap_id)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    schema_file = Path(schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not schema_file.is_absolute():
        schema_file = root / schema_file
    write_feature_schema(schema_file)
    schema_hash = feature_schema_sha256(schema_file)
    input_size = pcap.stat().st_size
    rel = to_repo_relative(pcap, project_root=root)
    out_rel = to_repo_relative(out, project_root=root)
    binary_label = ""
    if metadata:
        binary_label = str(metadata.get("binary_label") or "")

    if resume and checkpoint_is_reusable(
        checkpoint_path=ckpt,
        output_path=out,
        pcap_path=pcap,
        input_file_size=input_size,
        schema_sha256=schema_hash,
        binary_label=binary_label,
        project_root=root,
    ):
        payload = load_build_checkpoint(ckpt) or {}
        return BuildResult(
            pcap_id=pcap_id,
            pcap_path=rel,
            output_path=out,
            checkpoint_path=ckpt,
            row_count=int(payload["output_row_count"]),
            packets_processed=int(payload.get("packets_processed", 0)),
            windows_written=int(payload["output_row_count"]),
            input_file_size=input_size,
            output_file_size=int(payload.get("output_file_size", out.stat().st_size)),
            elapsed_seconds=0.0,
            resumed=True,
            feature_strategy_version=FEATURE_STRATEGY_VERSION,
            feature_build_strategy_version=FEATURE_BUILD_STRATEGY_VERSION,
            feature_schema_sha256=schema_hash,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out.with_suffix(out.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    out.unlink(missing_ok=True)

    policy = frozen_window_policy()
    stats = WindowStreamStats()
    writer = StreamingFeatureParquetWriter(tmp_path, buffer_rows=buffer_rows)
    started = time.perf_counter()
    try:
        for window in iter_windows(
            iter_packets(pcap),
            policy,
            stats=stats,
            max_windows=max_windows,
        ):
            features = extract_features(window)
            if validate:
                validate_window_and_features(window, features)
            writer.append(
                _row_from_window(
                    pcap_id=pcap_id,
                    binary_label=binary_label,
                    window=window,
                    features=features,
                )
            )
        row_count = writer.close()
        if stats.full_window_count != row_count:
            raise RuntimeError(
                f"Parquet rows ({row_count}) != emitted windows "
                f"({stats.full_window_count}) for {pcap}"
            )
        os.replace(tmp_path, out)
    except Exception:
        writer.abort()
        tmp_path.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    output_size = out.stat().st_size
    payload = {
        "pcap_id": pcap_id,
        "pcap_path": rel,
        "binary_label": binary_label,
        "input_file_size": input_size,
        "feature_strategy_version": FEATURE_STRATEGY_VERSION,
        "feature_build_strategy_version": FEATURE_BUILD_STRATEGY_VERSION,
        "feature_schema_sha256": schema_hash,
        "output_path": out_rel,
        "output_row_count": row_count,
        "output_file_size": output_size,
        "packets_processed": stats.packets_seen,
        "windows_written": row_count,
    }
    write_build_checkpoint(payload, ckpt)

    return BuildResult(
        pcap_id=pcap_id,
        pcap_path=rel,
        output_path=out,
        checkpoint_path=ckpt,
        row_count=row_count,
        packets_processed=stats.packets_seen,
        windows_written=row_count,
        input_file_size=input_size,
        output_file_size=output_size,
        elapsed_seconds=elapsed,
        resumed=False,
        feature_strategy_version=FEATURE_STRATEGY_VERSION,
        feature_build_strategy_version=FEATURE_BUILD_STRATEGY_VERSION,
        feature_schema_sha256=schema_hash,
    )


def run_parquet_smoke(
    *,
    pcap_paths: list[Path] | None = None,
    inventory_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    project_root: Path | None = None,
    resume: bool = True,
    buffer_rows: int = DEFAULT_BUFFER_ROWS,
) -> dict[str, Any]:
    """TRAIN-only 1C.3a smoke: a few PCAPs → Parquet shards + checkpoints."""
    from iot_pcap_pipeline.features.build import load_inventory_index
    from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

    root = (project_root or PROJECT_ROOT).resolve()
    inv = Path(inventory_path or (root / "data" / "manifests" / "pcap_inventory.csv"))
    if not inv.is_absolute():
        inv = root / inv
    index = load_inventory_index(inv)

    if pcap_paths is None:
        paths: list[Path] = []
        for rel in DEFAULT_PARQUET_SMOKE_PCAP_PATHS:
            meta = index.get(rel, {})
            split = meta.get("split")
            if split and split != "train":
                raise FeatureExtractionError(
                    f"DEFAULT_PARQUET_SMOKE_PCAP_PATHS contains non-TRAIN PCAP: "
                    f"{rel} (split={split!r})"
                )
            if index and not meta:
                raise FeatureExtractionError(
                    f"DEFAULT_PARQUET_SMOKE_PCAP_PATHS path missing from inventory: {rel}"
                )
            paths.append(root / rel)
    else:
        paths = [p if p.is_absolute() else (root / p) for p in pcap_paths]

    out_dir = Path(output_dir or DEFAULT_PARQUET_SMOKE_DIR)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ckpt_dir = Path(checkpoint_dir or DEFAULT_FEATURE_CHECKPOINT_DIR)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir

    results: list[BuildResult] = []
    for path in paths:
        rel = to_repo_relative(path, project_root=root)
        meta = dict(index.get(rel, {}))
        pcap_id = pcap_id_from_path(path, project_root=root)
        result = build_pcap_parquet(
            path,
            meta,
            out_dir / f"{pcap_id}.parquet",
            checkpoint_path=ckpt_dir / f"{pcap_id}.json",
            project_root=root,
            resume=resume,
            buffer_rows=buffer_rows,
        )
        results.append(result)

    return {
        "feature_strategy_version": FEATURE_STRATEGY_VERSION,
        "feature_build_strategy_version": FEATURE_BUILD_STRATEGY_VERSION,
        "output_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),
        "results": results,
    }


def format_parquet_smoke_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 1C.3a — streaming Parquet smoke",
        f"feature_strategy_version: {payload['feature_strategy_version']}",
        f"feature_build_strategy_version: {payload['feature_build_strategy_version']}",
        f"output_dir: {payload['output_dir']}",
        f"checkpoint_dir: {payload['checkpoint_dir']}",
        "",
    ]
    for result in payload["results"]:
        assert isinstance(result, BuildResult)
        if result.resumed:
            pps_s = "n/a (resumed)"
            wps_s = "n/a (resumed)"
            elapsed_s = "0.000 (resumed)"
        elif result.elapsed_seconds > 0:
            pps_s = f"{result.packets_processed / result.elapsed_seconds:,.1f}"
            wps_s = f"{result.windows_written / result.elapsed_seconds:,.1f}"
            elapsed_s = f"{result.elapsed_seconds:.3f}"
        else:
            pps_s = "n/a"
            wps_s = "n/a"
            elapsed_s = f"{result.elapsed_seconds:.3f}"
        lines.extend(
            [
                f"=== {result.pcap_id} ===",
                f"  resumed:              {result.resumed}",
                f"  packets_processed:    {result.packets_processed:,}",
                f"  windows_written:      {result.windows_written:,}",
                f"  parquet_rows:         {result.row_count:,}",
                f"  parquet_size_bytes:   {result.output_file_size:,}",
                f"  elapsed_seconds:      {elapsed_s}",
                f"  packets_per_sec:      {pps_s}",
                f"  windows_per_sec:      {wps_s}",
                f"  output:               {result.output_path}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
