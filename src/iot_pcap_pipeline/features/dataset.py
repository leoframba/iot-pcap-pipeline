"""Phase 1C.3b corpus orchestration for frozen V1 Parquet shards.

Inventory → TRAIN selection → size check → largest-first → process pool →
``build_pcap_parquet`` → deterministic ``build_manifest.csv``.

No TRAIN-wide statistics, TEST extraction, or model logic.
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

from iot_pcap_pipeline.features.parquet import (
    DEFAULT_BUFFER_ROWS,
    DEFAULT_FEATURE_CHECKPOINT_DIR,
    BuildResult,
    build_pcap_parquet,
    pcap_id_from_path,
)
from iot_pcap_pipeline.features.schema import write_feature_schema
from iot_pcap_pipeline.paths import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_DIR,
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.characterize import load_train_inventory_rows
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

EXPECTED_TRAIN_PCAP_COUNT = 85
DEFAULT_FEATURE_DATASET_WORKERS = 4
DEFAULT_TRAIN_PARQUET_DIR = DEFAULT_FEATURES_DIR / "v1" / "train"
DEFAULT_BUILD_MANIFEST_PATH = DEFAULT_FEATURES_DIR / "v1" / "build_manifest.csv"
DEFAULT_SMOKE_DATASET_DIR = DEFAULT_FEATURES_DIR / "v1" / "smoke" / "dataset"
DEFAULT_SMOKE_BUILD_MANIFEST_PATH = (
    DEFAULT_FEATURES_DIR / "v1" / "smoke" / "build_manifest_smoke.csv"
)
DEFAULT_SMOKE_CHECKPOINT_DIR = DEFAULT_FEATURES_DIR / "v1" / ".work" / "smoke"

# Modest TRAIN set for orchestration smoke (not the giant flood files).
ORCHESTRATION_SMOKE_PCAP_PATHS: tuple[str, ...] = (
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Recon-Ping_Sweep_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Recon-VulScan_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Recon-OS_Scan_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/ARP_Spoofing_train.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Idle/Idle.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Power/Blink_Mini_Camera_Power.pcap",
)

SplitName = Literal["train"]
JobStatus = Literal["ok", "size_mismatch", "error"]

BUILD_MANIFEST_COLUMNS: tuple[str, ...] = (
    "pcap_id",
    "pcap_path",
    "binary_label",
    "status",
    "error_message",
    "manifest_file_size",
    "disk_file_size",
    "input_file_size",
    "output_path",
    "output_row_count",
    "output_file_size",
    "packets_processed",
    "windows_written",
    "resumed",
    "elapsed_seconds",
    "feature_strategy_version",
    "feature_build_strategy_version",
    "feature_schema_sha256",
)


@dataclass(frozen=True)
class PcapJobResult:
    """Picklable per-PCAP outcome for the parent aggregator."""

    pcap_path: str
    pcap_id: str
    binary_label: str
    status: JobStatus
    error_message: str = ""
    manifest_file_size: int | None = None
    disk_file_size: int | None = None
    input_file_size: int | None = None
    output_path: str = ""
    output_row_count: int | None = None
    output_file_size: int | None = None
    packets_processed: int | None = None
    windows_written: int | None = None
    resumed: bool = False
    elapsed_seconds: float = 0.0
    feature_strategy_version: str = FEATURE_STRATEGY_VERSION
    feature_build_strategy_version: str = FEATURE_BUILD_STRATEGY_VERSION
    feature_schema_sha256: str = ""

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "pcap_id": self.pcap_id,
            "pcap_path": self.pcap_path,
            "binary_label": self.binary_label,
            "status": self.status,
            "error_message": self.error_message,
            "manifest_file_size": (
                "" if self.manifest_file_size is None else self.manifest_file_size
            ),
            "disk_file_size": (
                "" if self.disk_file_size is None else self.disk_file_size
            ),
            "input_file_size": (
                "" if self.input_file_size is None else self.input_file_size
            ),
            "output_path": self.output_path,
            "output_row_count": (
                "" if self.output_row_count is None else self.output_row_count
            ),
            "output_file_size": (
                "" if self.output_file_size is None else self.output_file_size
            ),
            "packets_processed": (
                "" if self.packets_processed is None else self.packets_processed
            ),
            "windows_written": (
                "" if self.windows_written is None else self.windows_written
            ),
            "resumed": str(self.resumed).lower(),
            "elapsed_seconds": f"{self.elapsed_seconds:.6f}",
            "feature_strategy_version": self.feature_strategy_version,
            "feature_build_strategy_version": self.feature_build_strategy_version,
            "feature_schema_sha256": self.feature_schema_sha256,
        }


@dataclass
class FeatureDatasetBuildResult:
    split: str
    rows: list[dict[str, Any]]
    manifest_path: Path
    output_dir: Path
    checkpoint_dir: Path
    workers: int
    resume: bool
    ok_count: int = 0
    failed_count: int = 0
    resumed_count: int = 0
    elapsed_seconds: float = 0.0
    results_by_path: dict[str, PcapJobResult] = field(default_factory=dict)


def _result_from_build(
    build: BuildResult,
    *,
    binary_label: str,
    manifest_file_size: int,
    disk_file_size: int,
    project_root: Path,
) -> PcapJobResult:
    return PcapJobResult(
        pcap_path=build.pcap_path,
        pcap_id=build.pcap_id,
        binary_label=binary_label,
        status="ok",
        manifest_file_size=manifest_file_size,
        disk_file_size=disk_file_size,
        input_file_size=build.input_file_size,
        output_path=to_repo_relative(build.output_path, project_root=project_root),
        output_row_count=build.row_count,
        output_file_size=build.output_file_size,
        packets_processed=build.packets_processed,
        windows_written=build.windows_written,
        resumed=build.resumed,
        elapsed_seconds=build.elapsed_seconds,
        feature_strategy_version=build.feature_strategy_version,
        feature_build_strategy_version=build.feature_build_strategy_version,
        feature_schema_sha256=build.feature_schema_sha256,
    )


def build_one_pcap_job(payload: dict[str, Any]) -> PcapJobResult:
    """Picklable worker: size-check one PCAP then call ``build_pcap_parquet``."""
    root = Path(payload["project_root"])
    rel = str(payload["pcap_path"])
    binary_label = str(payload.get("binary_label") or "")
    manifest_size = int(payload["manifest_file_size"])
    output_dir = Path(payload["output_dir"])
    checkpoint_dir = Path(payload["checkpoint_dir"])
    schema_path = Path(payload["schema_path"]) if payload.get("schema_path") else None
    resume = bool(payload.get("resume", True))
    buffer_rows = int(payload.get("buffer_rows", DEFAULT_BUFFER_ROWS))

    abs_path = Path(rel) if Path(rel).is_absolute() else (root / rel)
    pcap_id = pcap_id_from_path(abs_path, project_root=root)
    out_path = output_dir / f"{pcap_id}.parquet"
    ckpt_path = checkpoint_dir / f"{pcap_id}.json"

    if not abs_path.is_file():
        return PcapJobResult(
            pcap_path=rel,
            pcap_id=pcap_id,
            binary_label=binary_label,
            status="error",
            error_message=f"PCAP not found: {abs_path}",
            manifest_file_size=manifest_size,
            disk_file_size=None,
        )

    disk_size = abs_path.stat().st_size
    if disk_size != manifest_size:
        return PcapJobResult(
            pcap_path=rel,
            pcap_id=pcap_id,
            binary_label=binary_label,
            status="size_mismatch",
            error_message=(
                f"disk file size {disk_size} != manifest file_size {manifest_size}"
            ),
            manifest_file_size=manifest_size,
            disk_file_size=disk_size,
        )

    metadata = {
        "binary_label": binary_label,
        "pcap_path": rel,
        "split": payload.get("split", "train"),
        "attack_family": payload.get("attack_family", ""),
        "attack_type": payload.get("attack_type", ""),
        "profiling_type": payload.get("profiling_type", ""),
        "profiling_variant": payload.get("profiling_variant", ""),
        "device": payload.get("device", ""),
        "capture_session": payload.get("capture_session", ""),
    }
    try:
        build = build_pcap_parquet(
            abs_path,
            metadata,
            out_path,
            checkpoint_path=ckpt_path,
            project_root=root,
            schema_path=schema_path,
            resume=resume,
            buffer_rows=buffer_rows,
        )
    except (FeatureExtractionError, OSError, ValueError, RuntimeError) as exc:
        return PcapJobResult(
            pcap_path=rel,
            pcap_id=pcap_id,
            binary_label=binary_label,
            status="error",
            error_message=str(exc),
            manifest_file_size=manifest_size,
            disk_file_size=disk_size,
        )

    return _result_from_build(
        build,
        binary_label=binary_label,
        manifest_file_size=manifest_size,
        disk_file_size=disk_size,
        project_root=root,
    )


def select_train_rows(
    inventory_path: Path,
    *,
    require_expected_count: bool = True,
    pcap_paths: list[str] | None = None,
) -> list[dict[str, str]]:
    """Load TRAIN inventory rows; optionally restrict to an explicit path list."""
    train = load_train_inventory_rows(inventory_path)
    if pcap_paths is not None:
        wanted = set(pcap_paths)
        by_path = {r["pcap_path"]: r for r in train}
        missing = sorted(wanted - set(by_path))
        if missing:
            raise FeatureExtractionError(
                "Requested TRAIN PCAPs missing from inventory: "
                + ", ".join(missing)
            )
        selected = [by_path[p] for p in pcap_paths]
    else:
        selected = list(train)
        if require_expected_count and len(selected) != EXPECTED_TRAIN_PCAP_COUNT:
            raise FeatureExtractionError(
                f"Expected {EXPECTED_TRAIN_PCAP_COUNT} TRAIN PCAPs, "
                f"found {len(selected)} in {inventory_path}"
            )
    return selected


def schedule_largest_first(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a new list sorted by manifest file_size descending."""
    return sorted(
        rows,
        key=lambda row: int(row.get("file_size") or 0),
        reverse=True,
    )


def write_build_manifest(path: Path, results: list[PcapJobResult]) -> Path:
    """Write deterministic CSV sorted by ``pcap_path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda r: r.pcap_path)
    rows = [r.to_manifest_row() for r in ordered]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BUILD_MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in BUILD_MANIFEST_COLUMNS})
    tmp.replace(path)
    return path


def _job_payload(
    row: dict[str, str],
    *,
    project_root: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    schema_path: Path,
    resume: bool,
    buffer_rows: int,
    split: str,
) -> dict[str, Any]:
    return {
        "project_root": str(project_root),
        "pcap_path": row["pcap_path"],
        "binary_label": row.get("binary_label") or "",
        "manifest_file_size": int(row["file_size"]),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "schema_path": str(schema_path),
        "resume": resume,
        "buffer_rows": buffer_rows,
        "split": split,
        "attack_family": row.get("attack_family") or "",
        "attack_type": row.get("attack_type") or "",
        "profiling_type": row.get("profiling_type") or "",
        "profiling_variant": row.get("profiling_variant") or "",
        "device": row.get("device") or "",
        "capture_session": row.get("capture_session") or "",
    }


def build_feature_dataset(
    *,
    split: SplitName = "train",
    inventory_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
    schema_path: Path | str | None = None,
    project_root: Path | None = None,
    workers: int = DEFAULT_FEATURE_DATASET_WORKERS,
    resume: bool = True,
    buffer_rows: int = DEFAULT_BUFFER_ROWS,
    pcap_paths: list[str] | None = None,
    smoke: bool = False,
    progress_file: TextIO | None = None,
) -> FeatureDatasetBuildResult:
    """Build per-PCAP Parquet shards for one split (TRAIN only in 1C.3b step 1)."""
    if split != "train":
        raise FeatureExtractionError(
            "Phase 1C.3b step 1 only supports --split train "
            f"(got {split!r})"
        )

    root = (project_root or PROJECT_ROOT).resolve()
    inv = Path(inventory_path or (DEFAULT_MANIFEST_DIR / "pcap_inventory.csv"))
    if not inv.is_absolute():
        inv = root / inv

    default_out = DEFAULT_SMOKE_DATASET_DIR if smoke else DEFAULT_TRAIN_PARQUET_DIR
    default_ckpt = (
        DEFAULT_SMOKE_CHECKPOINT_DIR if smoke else DEFAULT_FEATURE_CHECKPOINT_DIR
    )
    default_manifest = (
        DEFAULT_SMOKE_BUILD_MANIFEST_PATH if smoke else DEFAULT_BUILD_MANIFEST_PATH
    )

    out_dir = Path(output_dir or default_out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ckpt_dir = Path(checkpoint_dir or default_ckpt)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    man_path = Path(manifest_path or default_manifest)
    if not man_path.is_absolute():
        man_path = root / man_path
    schema_file = Path(schema_path) if schema_path else None
    if schema_file is not None and not schema_file.is_absolute():
        schema_file = root / schema_file
    schema_file = write_feature_schema(schema_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    paths_filter = list(ORCHESTRATION_SMOKE_PCAP_PATHS) if smoke else pcap_paths
    require_count = paths_filter is None
    selected = select_train_rows(
        inv,
        require_expected_count=require_count,
        pcap_paths=paths_filter,
    )
    scheduled = schedule_largest_first(selected)

    worker_count = max(1, int(workers))
    if progress_file is not None:
        progress_file.write(
            f"Feature dataset build: split={split} pcaps={len(scheduled)} "
            f"workers={worker_count} resume={resume}\n"
        )
        progress_file.write(f"output_dir: {out_dir}\n")
        progress_file.write(f"checkpoint_dir: {ckpt_dir}\n")
        progress_file.flush()

    payloads = [
        _job_payload(
            row,
            project_root=root,
            output_dir=out_dir,
            checkpoint_dir=ckpt_dir,
            schema_path=schema_file,
            resume=resume,
            buffer_rows=buffer_rows,
            split=split,
        )
        for row in scheduled
    ]

    results_by_path: dict[str, PcapJobResult] = {}
    started = time.perf_counter()

    def _ingest(result: PcapJobResult) -> None:
        results_by_path[result.pcap_path] = result
        if progress_file is not None:
            progress_file.write(
                f"[{len(results_by_path)}/{len(scheduled)}] "
                f"{Path(result.pcap_path).name}: status={result.status} "
                f"rows={result.output_row_count} resumed={result.resumed}\n"
            )
            progress_file.flush()

    if worker_count == 1:
        for payload in payloads:
            _ingest(build_one_pcap_job(payload))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(build_one_pcap_job, payload): payload
                for payload in payloads
            }
            for future in as_completed(future_map):
                payload = future_map[future]
                rel = str(payload["pcap_path"])
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 — isolate worker crashes
                    abs_path = root / rel
                    pcap_id = pcap_id_from_path(abs_path, project_root=root)
                    result = PcapJobResult(
                        pcap_path=rel,
                        pcap_id=pcap_id,
                        binary_label=str(payload.get("binary_label") or ""),
                        status="error",
                        error_message=f"worker crash: {exc}",
                        manifest_file_size=int(payload["manifest_file_size"]),
                    )
                _ingest(result)

    elapsed = time.perf_counter() - started
    ordered_results = [results_by_path[r["pcap_path"]] for r in selected]
    # selected is path-sorted from inventory order when filtered by pcap_paths list
    # order; for full train, load_train_inventory_rows sorts by path. Prefer
    # deterministic path sort for the manifest regardless of schedule order.
    ordered_results = sorted(results_by_path.values(), key=lambda r: r.pcap_path)
    write_build_manifest(man_path, ordered_results)

    ok = sum(1 for r in ordered_results if r.status == "ok")
    failed = len(ordered_results) - ok
    resumed = sum(1 for r in ordered_results if r.resumed and r.status == "ok")

    return FeatureDatasetBuildResult(
        split=split,
        rows=[r.to_manifest_row() for r in ordered_results],
        manifest_path=man_path,
        output_dir=out_dir,
        checkpoint_dir=ckpt_dir,
        workers=worker_count,
        resume=resume,
        ok_count=ok,
        failed_count=failed,
        resumed_count=resumed,
        elapsed_seconds=elapsed,
        results_by_path=results_by_path,
    )


def format_feature_dataset_summary(result: FeatureDatasetBuildResult) -> str:
    lines = [
        "Phase 1C.3b — feature dataset build",
        f"split: {result.split}",
        f"feature_strategy_version: {FEATURE_STRATEGY_VERSION}",
        f"feature_build_strategy_version: {FEATURE_BUILD_STRATEGY_VERSION}",
        f"workers: {result.workers}",
        f"resume: {result.resume}",
        f"pcaps: {len(result.rows)} (ok={result.ok_count}, failed={result.failed_count})",
        f"checkpoint_hits: {result.resumed_count}",
        f"elapsed_seconds: {result.elapsed_seconds:.3f}",
        f"output_dir: {result.output_dir}",
        f"checkpoint_dir: {result.checkpoint_dir}",
        f"manifest: {result.manifest_path}",
    ]
    return "\n".join(lines) + "\n"


def logical_manifest_fingerprint(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Comparable view of manifest rows across worker counts / output dirs.

    Excludes timing, resume flags, and Parquet byte size (compression may vary).
    """
    skip = {"elapsed_seconds", "resumed", "output_file_size"}
    out: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda r: str(r.get("pcap_path", ""))):
        item = {
            key: str(row.get(key, ""))
            for key in BUILD_MANIFEST_COLUMNS
            if key not in skip
        }
        # Compare shard basename only — parent output_dir may differ across runs.
        if item.get("output_path"):
            item["output_path"] = Path(item["output_path"]).name
        out.append(item)
    return out
