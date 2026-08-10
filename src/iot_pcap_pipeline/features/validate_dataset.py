"""Read-only validation of a completed TRAIN feature Parquet build.

Does not decode PCAPs or rewrite shards. Produces feature summaries and, when
all checks pass, ``train_build_complete.json``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from iot_pcap_pipeline.features.dataset import (
    DEFAULT_BUILD_MANIFEST_PATH,
    EXPECTED_TRAIN_PCAP_COUNT,
)
from iot_pcap_pipeline.features.parquet import (
    feature_parquet_arrow_schema,
    feature_schema_sha256,
    parquet_row_count,
    parquet_schema_matches,
)
from iot_pcap_pipeline.features.schema import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    V1_FEATURE_NAMES,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_FEATURES_DIR,
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    WINDOWING_STRATEGY_VERSION,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.characterize import DEFAULT_CHARACTERIZATION_CSV
from iot_pcap_pipeline.windowing.policy import (
    BACKWARD_RESET_SECONDS,
    INACTIVITY_TIMEOUT_SECONDS,
    WINDOW_SIZE,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

SplitName = Literal["train"]

DEFAULT_INTEGRITY_CSV = DEFAULT_AUDIT_DIR / "pcap_integrity.csv"
DEFAULT_TRAIN_FEATURE_SUMMARY_CSV = (
    DEFAULT_FEATURES_DIR / "v1" / "train_feature_summary.csv"
)
DEFAULT_TRAIN_CONSTANT_FEATURES_CSV = (
    DEFAULT_FEATURES_DIR / "v1" / "train_constant_features.csv"
)
DEFAULT_TRAIN_BUILD_COMPLETE_JSON = (
    DEFAULT_FEATURES_DIR / "v1" / "train_build_complete.json"
)

FEATURE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "count",
    "nonfinite_count",
    "min",
    "max",
    "mean",
    "std",
    "is_constant",
)


@dataclass
class _OnlineStats:
    count: int = 0
    nonfinite_count: int = 0
    min_v: float | None = None
    max_v: float | None = None
    mean: float = 0.0
    m2: float = 0.0

    def update_many(self, values: Any) -> None:
        for raw in values:
            if raw is None:
                self.nonfinite_count += 1
                continue
            value = float(raw)
            if not math.isfinite(value):
                self.nonfinite_count += 1
                continue
            self.count += 1
            if self.min_v is None or value < self.min_v:
                self.min_v = value
            if self.max_v is None or value > self.max_v:
                self.max_v = value
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    def population_std(self) -> float:
        if self.count == 0:
            return float("nan")
        if self.count == 1:
            return 0.0
        return math.sqrt(self.m2 / self.count)

    def to_row(self, name: str) -> dict[str, Any]:
        if self.count == 0:
            return {
                "feature_name": name,
                "count": 0,
                "nonfinite_count": self.nonfinite_count,
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "is_constant": "",
            }
        is_constant = self.min_v == self.max_v
        return {
            "feature_name": name,
            "count": self.count,
            "nonfinite_count": self.nonfinite_count,
            "min": self.min_v,
            "max": self.max_v,
            "mean": self.mean,
            "std": self.population_std(),
            "is_constant": str(bool(is_constant)).lower(),
        }


@dataclass
class ValidationIssue:
    code: str
    pcap_path: str
    message: str


@dataclass
class FeatureDatasetValidationResult:
    split: str
    passed: bool
    pcap_count: int
    total_feature_rows: int
    issues: list[ValidationIssue] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    constant_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_path: Path | None = None
    constant_path: Path | None = None
    complete_path: Path | None = None
    feature_strategy_version: str = FEATURE_STRATEGY_VERSION
    feature_build_strategy_version: str = FEATURE_BUILD_STRATEGY_VERSION
    feature_schema_sha256: str = ""
    windowing_strategy_version: str = WINDOWING_STRATEGY_VERSION


def _load_csv_index(path: Path, key: str = "pcap_path") -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FeatureExtractionError(f"Required CSV missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {r[key]: r for r in rows if r.get(key)}


def _load_frozen_window_index(path: Path) -> dict[str, dict[str, str]]:
    """Index characterization rows for the frozen 25 / 5s / 1s policy only."""
    if not path.is_file():
        raise FeatureExtractionError(f"Required CSV missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            wsize = int(float(row.get("window_size") or -1))
            inactive = float(row.get("inactivity_timeout_seconds") or -1)
            backward = float(row.get("backward_reset_seconds") or -1)
        except ValueError:
            continue
        if (
            wsize == WINDOW_SIZE
            and inactive == INACTIVITY_TIMEOUT_SECONDS
            and backward == BACKWARD_RESET_SECONDS
        ):
            pcap_path = row.get("pcap_path") or ""
            if pcap_path:
                out[pcap_path] = row
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _accumulate_feature_stats(
    parquet_path: Path,
    stats: dict[str, _OnlineStats],
    *,
    batch_size: int = 65_536,
) -> None:
    """Stream only the 27 V1 feature columns into online aggregators."""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(V1_FEATURE_NAMES)):
        table = pa.Table.from_batches([batch])
        for name in V1_FEATURE_NAMES:
            stats[name].update_many(table.column(name).to_pylist())


def validate_feature_dataset(
    *,
    split: SplitName = "train",
    manifest_path: Path | str | None = None,
    integrity_path: Path | str | None = None,
    characterization_path: Path | str | None = None,
    schema_path: Path | str | None = None,
    summary_output: Path | str | None = None,
    constant_output: Path | str | None = None,
    complete_output: Path | str | None = None,
    project_root: Path | None = None,
    progress_file: TextIO | None = None,
) -> FeatureDatasetValidationResult:
    """Validate an existing TRAIN Parquet build without re-extracting."""
    if split != "train":
        raise FeatureExtractionError(
            "validate-feature-dataset currently supports --split train only "
            f"(got {split!r})"
        )

    root = (project_root or PROJECT_ROOT).resolve()
    man_path = Path(manifest_path or DEFAULT_BUILD_MANIFEST_PATH)
    if not man_path.is_absolute():
        man_path = root / man_path
    integ_path = Path(integrity_path or DEFAULT_INTEGRITY_CSV)
    if not integ_path.is_absolute():
        integ_path = root / integ_path
    char_path = Path(characterization_path or DEFAULT_CHARACTERIZATION_CSV)
    if not char_path.is_absolute():
        char_path = root / char_path
    schema_file = Path(schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not schema_file.is_absolute():
        schema_file = root / schema_file
    summary_out = Path(summary_output or DEFAULT_TRAIN_FEATURE_SUMMARY_CSV)
    if not summary_out.is_absolute():
        summary_out = root / summary_out
    constant_out = Path(constant_output or DEFAULT_TRAIN_CONSTANT_FEATURES_CSV)
    if not constant_out.is_absolute():
        constant_out = root / constant_out
    complete_out = Path(complete_output or DEFAULT_TRAIN_BUILD_COMPLETE_JSON)
    if not complete_out.is_absolute():
        complete_out = root / complete_out

    if not man_path.is_file():
        raise FeatureExtractionError(f"build_manifest.csv missing: {man_path}")

    with man_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    issues: list[ValidationIssue] = []
    if len(manifest_rows) != EXPECTED_TRAIN_PCAP_COUNT:
        issues.append(
            ValidationIssue(
                code="manifest_count",
                pcap_path="",
                message=(
                    f"expected {EXPECTED_TRAIN_PCAP_COUNT} manifest rows, "
                    f"found {len(manifest_rows)}"
                ),
            )
        )

    try:
        schema_hash = feature_schema_sha256(schema_file)
    except (OSError, FileNotFoundError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                code="schema_file",
                pcap_path="",
                message=str(exc),
            )
        )
        schema_hash = ""

    expected_schema = feature_parquet_arrow_schema()
    integrity = _load_csv_index(integ_path)
    frozen_windows = _load_frozen_window_index(char_path)

    stats = {name: _OnlineStats() for name in V1_FEATURE_NAMES}
    total_rows = 0
    ok_shards = 0

    if progress_file is not None:
        progress_file.write(
            f"Validating {len(manifest_rows)} TRAIN shards "
            f"(read-only; no PCAP decode)\n"
        )
        progress_file.flush()

    for i, row in enumerate(manifest_rows, start=1):
        pcap_path = row.get("pcap_path") or ""
        status = (row.get("status") or "").strip()
        if status != "ok":
            issues.append(
                ValidationIssue(
                    code="manifest_status",
                    pcap_path=pcap_path,
                    message=f"manifest status={status!r}, expected 'ok'",
                )
            )
            continue

        if row.get("feature_strategy_version") != FEATURE_STRATEGY_VERSION:
            issues.append(
                ValidationIssue(
                    code="feature_strategy_version",
                    pcap_path=pcap_path,
                    message=(
                        f"got {row.get('feature_strategy_version')!r}, "
                        f"expected {FEATURE_STRATEGY_VERSION!r}"
                    ),
                )
            )
        if row.get("feature_build_strategy_version") != FEATURE_BUILD_STRATEGY_VERSION:
            issues.append(
                ValidationIssue(
                    code="feature_build_strategy_version",
                    pcap_path=pcap_path,
                    message=(
                        f"got {row.get('feature_build_strategy_version')!r}, "
                        f"expected {FEATURE_BUILD_STRATEGY_VERSION!r}"
                    ),
                )
            )
        if schema_hash and row.get("feature_schema_sha256") != schema_hash:
            issues.append(
                ValidationIssue(
                    code="feature_schema_sha256",
                    pcap_path=pcap_path,
                    message=(
                        f"manifest hash {row.get('feature_schema_sha256')!r} "
                        f"!= on-disk schema hash {schema_hash!r}"
                    ),
                )
            )

        out_rel = row.get("output_path") or ""
        shard = Path(out_rel)
        if not shard.is_absolute():
            shard = root / shard
        if not shard.is_file():
            issues.append(
                ValidationIssue(
                    code="shard_missing",
                    pcap_path=pcap_path,
                    message=f"Parquet shard missing: {shard}",
                )
            )
            continue

        try:
            expected_rows = int(row["output_row_count"])
            packets_processed = int(row["packets_processed"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    code="manifest_parse",
                    pcap_path=pcap_path,
                    message=f"invalid row counts: {exc}",
                )
            )
            continue

        if not parquet_schema_matches(shard, expected_schema):
            issues.append(
                ValidationIssue(
                    code="parquet_schema",
                    pcap_path=pcap_path,
                    message=f"Parquet schema mismatch: {shard}",
                )
            )
            continue

        actual_rows = parquet_row_count(shard)
        if actual_rows is None:
            issues.append(
                ValidationIssue(
                    code="parquet_unreadable",
                    pcap_path=pcap_path,
                    message=f"Parquet failed to open: {shard}",
                )
            )
            continue
        if actual_rows != expected_rows:
            issues.append(
                ValidationIssue(
                    code="parquet_row_count",
                    pcap_path=pcap_path,
                    message=(
                        f"Parquet rows {actual_rows} != "
                        f"manifest output_row_count {expected_rows}"
                    ),
                )
            )
            continue

        integ = integrity.get(pcap_path)
        if integ is None:
            issues.append(
                ValidationIssue(
                    code="integrity_missing",
                    pcap_path=pcap_path,
                    message=f"pcap_path missing from {integ_path}",
                )
            )
        else:
            try:
                audit_packets = int(integ["packet_count"])
            except (KeyError, TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        code="integrity_parse",
                        pcap_path=pcap_path,
                        message="invalid packet_count in pcap_integrity.csv",
                    )
                )
            else:
                if packets_processed != audit_packets:
                    issues.append(
                        ValidationIssue(
                            code="packet_count_mismatch",
                            pcap_path=pcap_path,
                            message=(
                                f"packets_processed {packets_processed} != "
                                f"integrity packet_count {audit_packets}"
                            ),
                        )
                    )

        win = frozen_windows.get(pcap_path)
        if win is None:
            issues.append(
                ValidationIssue(
                    code="characterization_missing",
                    pcap_path=pcap_path,
                    message=(
                        f"no frozen {WINDOW_SIZE}/{INACTIVITY_TIMEOUT_SECONDS}/"
                        f"{BACKWARD_RESET_SECONDS} row in {char_path}"
                    ),
                )
            )
        else:
            try:
                full_windows = int(float(win["full_window_count"]))
            except (KeyError, TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        code="characterization_parse",
                        pcap_path=pcap_path,
                        message="invalid full_window_count",
                    )
                )
            else:
                if expected_rows != full_windows:
                    issues.append(
                        ValidationIssue(
                            code="window_count_mismatch",
                            pcap_path=pcap_path,
                            message=(
                                f"output_row_count {expected_rows} != "
                                f"characterization full_window_count {full_windows}"
                            ),
                        )
                    )

        try:
            _accumulate_feature_stats(shard, stats)
        except (OSError, pa.ArrowInvalid, pa.ArrowIOError) as exc:
            issues.append(
                ValidationIssue(
                    code="feature_stream",
                    pcap_path=pcap_path,
                    message=f"failed streaming features: {exc}",
                )
            )
            continue

        total_rows += actual_rows
        ok_shards += 1
        if progress_file is not None and (
            i == 1 or i == len(manifest_rows) or i % 10 == 0
        ):
            progress_file.write(
                f"[{i}/{len(manifest_rows)}] {Path(pcap_path).name}: "
                f"rows={actual_rows}\n"
            )
            progress_file.flush()

    summary_rows = [stats[name].to_row(name) for name in V1_FEATURE_NAMES]
    for row in summary_rows:
        if row["count"] and int(row["nonfinite_count"]) > 0:
            issues.append(
                ValidationIssue(
                    code="nonfinite_features",
                    pcap_path="",
                    message=(
                        f"{row['feature_name']} has "
                        f"{row['nonfinite_count']} non-finite values"
                    ),
                )
            )

    constant_rows = [
        row for row in summary_rows if str(row.get("is_constant")).lower() == "true"
    ]

    _write_csv(summary_out, summary_rows, list(FEATURE_SUMMARY_COLUMNS))
    _write_csv(constant_out, constant_rows, list(FEATURE_SUMMARY_COLUMNS))

    passed = len(issues) == 0 and len(manifest_rows) == EXPECTED_TRAIN_PCAP_COUNT
    complete_path: Path | None = None
    if passed:
        payload = {
            "validation_status": "passed",
            "split": split,
            "pcap_count": len(manifest_rows),
            "total_feature_rows": total_rows,
            "feature_strategy_version": FEATURE_STRATEGY_VERSION,
            "feature_build_strategy_version": FEATURE_BUILD_STRATEGY_VERSION,
            "feature_schema_sha256": schema_hash,
            "windowing_strategy_version": WINDOWING_STRATEGY_VERSION,
            "windowing": {
                "window_size": WINDOW_SIZE,
                "inactivity_timeout_seconds": INACTIVITY_TIMEOUT_SECONDS,
                "backward_reset_seconds": BACKWARD_RESET_SECONDS,
            },
            "constant_feature_count": len(constant_rows),
            "constant_features": [r["feature_name"] for r in constant_rows],
            "note": (
                "Constant features are reported only; exclusion from model "
                "input requires an explicit pre-training contract decision. "
                "TEST must not be consulted."
            ),
            "artifacts": {
                "build_manifest": to_repo_relative(man_path, project_root=root),
                "feature_summary": to_repo_relative(summary_out, project_root=root),
                "constant_features": to_repo_relative(
                    constant_out, project_root=root
                ),
            },
        }
        complete_out.parent.mkdir(parents=True, exist_ok=True)
        complete_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        complete_path = complete_out
    elif complete_out.is_file():
        # Stale pass marker must not survive a failed re-validation.
        complete_out.unlink(missing_ok=True)

    return FeatureDatasetValidationResult(
        split=split,
        passed=passed,
        pcap_count=len(manifest_rows),
        total_feature_rows=total_rows,
        issues=issues,
        summary_rows=summary_rows,
        constant_rows=constant_rows,
        summary_path=summary_out,
        constant_path=constant_out,
        complete_path=complete_path,
        feature_strategy_version=FEATURE_STRATEGY_VERSION,
        feature_build_strategy_version=FEATURE_BUILD_STRATEGY_VERSION,
        feature_schema_sha256=schema_hash,
        windowing_strategy_version=WINDOWING_STRATEGY_VERSION,
    )


def format_validation_summary(
    result: FeatureDatasetValidationResult,
    *,
    project_root: Path | None = None,
) -> str:
    root = (project_root or PROJECT_ROOT).resolve()

    def _rel(path: Path | None) -> str:
        if path is None:
            return ""
        return to_repo_relative(path, project_root=root)

    lines = [
        "Phase 1C.3b — validate feature dataset (read-only)",
        f"split: {result.split}",
        f"validation_status: {'passed' if result.passed else 'failed'}",
        f"pcap_count: {result.pcap_count}",
        f"total_feature_rows: {result.total_feature_rows:,}",
        f"issue_count: {len(result.issues)}",
        f"constant_features: {len(result.constant_rows)}",
        f"feature_strategy_version: {result.feature_strategy_version}",
        f"feature_build_strategy_version: {result.feature_build_strategy_version}",
        f"feature_schema_sha256: {result.feature_schema_sha256}",
    ]
    if result.summary_path is not None:
        lines.append(f"feature_summary: {_rel(result.summary_path)}")
    if result.constant_path is not None:
        lines.append(f"constant_features_csv: {_rel(result.constant_path)}")
    if result.complete_path is not None:
        lines.append(f"build_complete: {_rel(result.complete_path)}")
    if result.constant_rows:
        names = ", ".join(r["feature_name"] for r in result.constant_rows)
        lines.append(f"constant feature names: {names}")
        lines.append(
            "Note: constants are reported only; do not drop from V1 without "
            "an explicit pre-training contract (TEST must not decide)."
        )
    if result.issues:
        lines.append("issues:")
        for issue in result.issues[:50]:
            loc = issue.pcap_path or "(global)"
            lines.append(f"  - [{issue.code}] {loc}: {issue.message}")
        if len(result.issues) > 50:
            lines.append(f"  ... and {len(result.issues) - 50} more")
    return "\n".join(lines) + "\n"
