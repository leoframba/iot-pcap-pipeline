"""TRAIN-only per-group feature characterization over existing Parquet shards.

Read-only: does not decode PCAPs or rewrite shards. Used to inspect whether
features encode source/capture shortcuts before any feature drops.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from iot_pcap_pipeline.features.characterize import summary_category
from iot_pcap_pipeline.features.dataset import DEFAULT_BUILD_MANIFEST_PATH
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.paths import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_DIR,
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

SplitName = Literal["train"]

DEFAULT_PERCENTILE_SAMPLE_CAP = 200_000
DEFAULT_PERCENTILE_SAMPLE_SEED = 42

DEFAULT_GROUP_SUMMARY_CSV = (
    DEFAULT_FEATURES_DIR / "v1" / "train_feature_group_summary.csv"
)
DEFAULT_PCAP_DIAGNOSTICS_CSV = (
    DEFAULT_FEATURES_DIR / "v1" / "train_feature_pcap_diagnostics.csv"
)
DEFAULT_GROUP_CHARACTERIZATION_JSON = (
    DEFAULT_FEATURES_DIR / "v1" / "train_feature_group_characterization.json"
)

GROUP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "group",
    "group_kind",
    "feature_name",
    "pcap_count",
    "row_count",
    "nonfinite_count",
    "zero_count",
    "nonzero_count",
    "min",
    "max",
    "mean",
    "std",
    "p01",
    "p50",
    "p95",
    "p99",
    "p99_9",
    "percentile_method",
    "percentile_sample_size",
    "is_constant",
)

PCAP_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "pcap_path",
    "pcap_id",
    "group",
    "binary_label",
    "attack_family",
    "attack_type",
    "profiling_type",
    "row_count",
    "tcp_urg_ratio_nonzero_windows",
    "tcp_urg_ratio_max",
    "frame_length_max_max",
    "frame_length_max_eq_8754_windows",
    "ipv6_ratio_nonzero_windows",
    "llc_ratio_nonzero_windows",
    "igmp_ratio_nonzero_windows",
    "icmpv6_ratio_nonzero_windows",
    "window_span_seconds_mean",
    "iat_mean_seconds_mean",
)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile; ``p`` in 0..100."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    weight = rank - lo
    return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight


@dataclass
class _Reservoir:
    cap: int
    rng: random.Random
    values: list[float] = field(default_factory=list)
    seen: int = 0

    def offer_array(self, values: np.ndarray) -> None:
        """Bounded sample for percentiles.

        Exact while ``seen <= cap``. Afterwards, randomly replace at most a
        small number of slots per batch (avoids O(cap) list copies every batch).
        """
        if values.size == 0:
            return
        # Fill phase.
        if len(self.values) < self.cap:
            need = self.cap - len(self.values)
            take = min(need, int(values.size))
            self.values.extend(values[:take].astype(float, copy=False).tolist())
            self.seen += take
            values = values[take:]
            if values.size == 0:
                return
        # Replacement phase: O(k) with small k, not O(cap) merges.
        n = int(values.size)
        self.seen += n
        # Expected replacements ~ cap * n / seen; clamp for throughput.
        expected = max(1, int(self.cap * n / max(self.seen, 1)))
        k = min(n, expected, 512)
        if k <= 0:
            return
        chosen = (
            values
            if k == n
            else values[np.asarray(self.rng.sample(range(n), k), dtype=np.int64)]
        )
        for value in chosen.tolist():
            slot = self.rng.randrange(self.cap)
            self.values[slot] = float(value)


@dataclass
class _FeatureAgg:
    count: int = 0
    nonfinite_count: int = 0
    zero_count: int = 0
    nonzero_count: int = 0
    min_v: float | None = None
    max_v: float | None = None
    mean: float = 0.0
    m2: float = 0.0
    reservoir: _Reservoir | None = None

    def update_array(self, values: np.ndarray) -> None:
        """Batch-update from a 1-D float64 array."""
        if values.size == 0:
            return
        finite = np.isfinite(values)
        self.nonfinite_count += int((~finite).sum())
        xs = values[finite]
        if xs.size == 0:
            return
        n0 = self.count
        n1 = int(xs.size)
        batch_mean = float(xs.mean())
        batch_m2 = float(((xs - batch_mean) ** 2).sum())
        if n0 == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = n0 + n1
            self.mean = self.mean + delta * n1 / total
            self.m2 = self.m2 + batch_m2 + delta * delta * n0 * n1 / total
        self.count = n0 + n1
        zeros = int((xs == 0.0).sum())
        self.zero_count += zeros
        self.nonzero_count += n1 - zeros
        batch_min = float(xs.min())
        batch_max = float(xs.max())
        if self.min_v is None or batch_min < self.min_v:
            self.min_v = batch_min
        if self.max_v is None or batch_max > self.max_v:
            self.max_v = batch_max
        if self.reservoir is not None:
            self.reservoir.offer_array(xs)

    def population_std(self) -> float:
        if self.count == 0:
            return float("nan")
        if self.count == 1:
            return 0.0
        return math.sqrt(self.m2 / self.count)

    def to_row(
        self,
        *,
        group: str,
        group_kind: str,
        feature_name: str,
        pcap_count: int,
    ) -> dict[str, Any]:
        if self.count == 0:
            return {
                "group": group,
                "group_kind": group_kind,
                "feature_name": feature_name,
                "pcap_count": pcap_count,
                "row_count": 0,
                "nonfinite_count": self.nonfinite_count,
                "zero_count": 0,
                "nonzero_count": 0,
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "p01": "",
                "p50": "",
                "p95": "",
                "p99": "",
                "p99_9": "",
                "percentile_method": "",
                "percentile_sample_size": 0,
                "is_constant": "",
            }
        assert self.reservoir is not None
        sample = sorted(self.reservoir.values)
        method = (
            "exact"
            if self.reservoir.seen <= self.reservoir.cap
            else "bounded_sample"
        )
        return {
            "group": group,
            "group_kind": group_kind,
            "feature_name": feature_name,
            "pcap_count": pcap_count,
            "row_count": self.count,
            "nonfinite_count": self.nonfinite_count,
            "zero_count": self.zero_count,
            "nonzero_count": self.nonzero_count,
            "min": self.min_v,
            "max": self.max_v,
            "mean": self.mean,
            "std": self.population_std(),
            "p01": _percentile(sample, 1),
            "p50": _percentile(sample, 50),
            "p95": _percentile(sample, 95),
            "p99": _percentile(sample, 99),
            "p99_9": _percentile(sample, 99.9),
            "percentile_method": method,
            "percentile_sample_size": len(sample),
            "is_constant": str(self.min_v == self.max_v).lower(),
        }


@dataclass
class _GroupState:
    group: str
    group_kind: str
    pcap_paths: set[str] = field(default_factory=set)
    features: dict[str, _FeatureAgg] = field(default_factory=dict)


@dataclass
class _PcapDiag:
    pcap_path: str
    pcap_id: str
    group: str
    binary_label: str
    attack_family: str
    attack_type: str
    profiling_type: str
    row_count: int = 0
    tcp_urg_ratio_nonzero_windows: int = 0
    tcp_urg_ratio_max: float = 0.0
    frame_length_max_max: float = 0.0
    frame_length_max_eq_8754_windows: int = 0
    ipv6_ratio_nonzero_windows: int = 0
    llc_ratio_nonzero_windows: int = 0
    igmp_ratio_nonzero_windows: int = 0
    icmpv6_ratio_nonzero_windows: int = 0
    window_span_sum: float = 0.0
    iat_mean_sum: float = 0.0

    def to_row(self) -> dict[str, Any]:
        n = max(self.row_count, 1)
        return {
            "pcap_path": self.pcap_path,
            "pcap_id": self.pcap_id,
            "group": self.group,
            "binary_label": self.binary_label,
            "attack_family": self.attack_family,
            "attack_type": self.attack_type,
            "profiling_type": self.profiling_type,
            "row_count": self.row_count,
            "tcp_urg_ratio_nonzero_windows": self.tcp_urg_ratio_nonzero_windows,
            "tcp_urg_ratio_max": self.tcp_urg_ratio_max,
            "frame_length_max_max": self.frame_length_max_max,
            "frame_length_max_eq_8754_windows": self.frame_length_max_eq_8754_windows,
            "ipv6_ratio_nonzero_windows": self.ipv6_ratio_nonzero_windows,
            "llc_ratio_nonzero_windows": self.llc_ratio_nonzero_windows,
            "igmp_ratio_nonzero_windows": self.igmp_ratio_nonzero_windows,
            "icmpv6_ratio_nonzero_windows": self.icmpv6_ratio_nonzero_windows,
            "window_span_seconds_mean": self.window_span_sum / n,
            "iat_mean_seconds_mean": self.iat_mean_sum / n,
        }


@dataclass
class GroupCharacterizationResult:
    group_rows: list[dict[str, Any]]
    pcap_rows: list[dict[str, Any]]
    group_summary_path: Path
    pcap_diagnostics_path: Path
    summary_json_path: Path
    pcap_count: int
    total_feature_rows: int
    group_count: int


def _group_kind(group: str) -> str:
    if group == "all_train":
        return "global"
    if group == "publisher_benign":
        return "publisher_benign"
    if group.startswith("profiling_"):
        return "profiling"
    if group.startswith("attack_family_"):
        return "attack_family"
    if group.startswith("attack_type_"):
        return "attack_type"
    return "other"


def _rng_for_group(group: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _ensure_group(
    groups: dict[str, _GroupState],
    group: str,
    *,
    sample_cap: int,
    sample_seed: int,
) -> _GroupState:
    state = groups.get(group)
    if state is not None:
        return state
    rng = _rng_for_group(group, sample_seed)
    features = {
        name: _FeatureAgg(reservoir=_Reservoir(cap=sample_cap, rng=rng))
        for name in V1_FEATURE_NAMES
    }
    # Independent streams: re-seed per feature for stability under parallel offers.
    for i, name in enumerate(V1_FEATURE_NAMES):
        feat_rng = _rng_for_group(f"{group}:{name}", sample_seed + i)
        features[name].reservoir = _Reservoir(cap=sample_cap, rng=feat_rng)
    state = _GroupState(group=group, group_kind=_group_kind(group), features=features)
    groups[group] = state
    return state


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def characterize_train_feature_groups(
    *,
    manifest_path: Path | str | None = None,
    inventory_path: Path | str | None = None,
    group_summary_output: Path | str | None = None,
    pcap_diagnostics_output: Path | str | None = None,
    summary_json_output: Path | str | None = None,
    project_root: Path | None = None,
    percentile_sample_cap: int = DEFAULT_PERCENTILE_SAMPLE_CAP,
    percentile_sample_seed: int = DEFAULT_PERCENTILE_SAMPLE_SEED,
    progress_file: TextIO | None = None,
) -> GroupCharacterizationResult:
    """Stream TRAIN Parquet shards into per-group feature statistics."""
    root = (project_root or PROJECT_ROOT).resolve()
    man_path = Path(manifest_path or DEFAULT_BUILD_MANIFEST_PATH)
    if not man_path.is_absolute():
        man_path = root / man_path
    inv_path = Path(inventory_path or (DEFAULT_MANIFEST_DIR / "pcap_inventory.csv"))
    if not inv_path.is_absolute():
        inv_path = root / inv_path
    group_out = Path(group_summary_output or DEFAULT_GROUP_SUMMARY_CSV)
    if not group_out.is_absolute():
        group_out = root / group_out
    diag_out = Path(pcap_diagnostics_output or DEFAULT_PCAP_DIAGNOSTICS_CSV)
    if not diag_out.is_absolute():
        diag_out = root / diag_out
    json_out = Path(summary_json_output or DEFAULT_GROUP_CHARACTERIZATION_JSON)
    if not json_out.is_absolute():
        json_out = root / json_out

    if not man_path.is_file():
        raise FeatureExtractionError(f"build_manifest.csv missing: {man_path}")
    if not inv_path.is_file():
        raise FeatureExtractionError(f"inventory missing: {inv_path}")

    with man_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with inv_path.open(newline="", encoding="utf-8") as handle:
        inventory = {r["pcap_path"]: r for r in csv.DictReader(handle)}

    groups: dict[str, _GroupState] = {}
    pcap_diags: list[_PcapDiag] = []
    total_rows = 0

    if progress_file is not None:
        progress_file.write(
            f"Characterizing {len(manifest_rows)} TRAIN shards "
            f"(read-only Parquet; sample_cap={percentile_sample_cap:,})\n"
        )
        progress_file.flush()

    for i, row in enumerate(manifest_rows, start=1):
        if (row.get("status") or "") != "ok":
            raise FeatureExtractionError(
                f"manifest row not ok for {row.get('pcap_path')}: {row.get('status')}"
            )
        pcap_path = row["pcap_path"]
        meta = inventory.get(pcap_path)
        if meta is None:
            raise FeatureExtractionError(f"pcap missing from inventory: {pcap_path}")

        primary = summary_category(meta) or "unknown"
        family = (meta.get("attack_family") or "").strip()
        attack_type = (meta.get("attack_type") or "").strip()
        labels = ["all_train", primary]
        if family:
            labels.append(f"attack_family_{family}")
        if attack_type:
            labels.append(f"attack_type_{attack_type}")
        # Deduplicate while preserving order.
        seen: set[str] = set()
        group_names: list[str] = []
        for name in labels:
            if name not in seen:
                seen.add(name)
                group_names.append(name)

        out_rel = row["output_path"]
        shard = Path(out_rel)
        if not shard.is_absolute():
            shard = root / shard
        if not shard.is_file():
            raise FeatureExtractionError(f"Parquet shard missing: {shard}")

        diag = _PcapDiag(
            pcap_path=pcap_path,
            pcap_id=row.get("pcap_id") or "",
            group=primary,
            binary_label=meta.get("binary_label") or "",
            attack_family=family,
            attack_type=attack_type,
            profiling_type=(meta.get("profiling_type") or "").strip(),
        )

        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(
            batch_size=65_536, columns=list(V1_FEATURE_NAMES)
        ):
            table = pa.Table.from_batches([batch])
            n = table.num_rows
            diag.row_count += n
            total_rows += n

            arrays = {
                name: np.asarray(table.column(name), dtype=np.float64)
                for name in V1_FEATURE_NAMES
            }
            for group_name in group_names:
                state = _ensure_group(
                    groups,
                    group_name,
                    sample_cap=percentile_sample_cap,
                    sample_seed=percentile_sample_seed,
                )
                state.pcap_paths.add(pcap_path)
                for name in V1_FEATURE_NAMES:
                    state.features[name].update_array(arrays[name])

            urg = arrays["tcp_urg_ratio"]
            diag.tcp_urg_ratio_nonzero_windows += int((urg > 0.0).sum())
            if urg.size:
                diag.tcp_urg_ratio_max = max(diag.tcp_urg_ratio_max, float(urg.max()))
            flmax = arrays["frame_length_max"]
            if flmax.size:
                diag.frame_length_max_max = max(
                    diag.frame_length_max_max, float(flmax.max())
                )
                diag.frame_length_max_eq_8754_windows += int((flmax == 8754.0).sum())
            diag.ipv6_ratio_nonzero_windows += int(
                (arrays["ipv6_ratio"] > 0.0).sum()
            )
            diag.llc_ratio_nonzero_windows += int((arrays["llc_ratio"] > 0.0).sum())
            diag.igmp_ratio_nonzero_windows += int(
                (arrays["igmp_ratio"] > 0.0).sum()
            )
            diag.icmpv6_ratio_nonzero_windows += int(
                (arrays["icmpv6_ratio"] > 0.0).sum()
            )
            diag.window_span_sum += float(arrays["window_span_seconds"].sum())
            diag.iat_mean_sum += float(arrays["iat_mean_seconds"].sum())

        pcap_diags.append(diag)
        if progress_file is not None and (
            i == 1 or i == len(manifest_rows) or i % 10 == 0
        ):
            progress_file.write(
                f"[{i}/{len(manifest_rows)}] {Path(pcap_path).name}: "
                f"rows={diag.row_count} group={primary}\n"
            )
            progress_file.flush()

    group_rows: list[dict[str, Any]] = []
    for group_name in sorted(groups):
        state = groups[group_name]
        for feature_name in V1_FEATURE_NAMES:
            group_rows.append(
                state.features[feature_name].to_row(
                    group=group_name,
                    group_kind=state.group_kind,
                    feature_name=feature_name,
                    pcap_count=len(state.pcap_paths),
                )
            )

    pcap_rows = [d.to_row() for d in sorted(pcap_diags, key=lambda d: d.pcap_path)]
    _write_csv(group_out, group_rows, list(GROUP_SUMMARY_COLUMNS))
    _write_csv(diag_out, pcap_rows, list(PCAP_DIAGNOSTIC_COLUMNS))

    def _pcaps_with(pred) -> list[dict[str, Any]]:
        out = []
        for d in pcap_diags:
            if pred(d):
                out.append(
                    {
                        "pcap_path": d.pcap_path,
                        "group": d.group,
                        "attack_family": d.attack_family,
                        "attack_type": d.attack_type,
                        "profiling_type": d.profiling_type,
                        "row_count": d.row_count,
                    }
                )
        return sorted(out, key=lambda r: r["pcap_path"])

    urg_pcaps = _pcaps_with(lambda d: d.tcp_urg_ratio_nonzero_windows > 0)
    jumbo_pcaps = _pcaps_with(lambda d: d.frame_length_max_eq_8754_windows > 0)
    ipv6_pcaps = _pcaps_with(lambda d: d.ipv6_ratio_nonzero_windows > 0)
    llc_pcaps = _pcaps_with(lambda d: d.llc_ratio_nonzero_windows > 0)
    igmp_pcaps = _pcaps_with(lambda d: d.igmp_ratio_nonzero_windows > 0)
    icmpv6_pcaps = _pcaps_with(lambda d: d.icmpv6_ratio_nonzero_windows > 0)

    # Temporal separation snapshot: group means for span/IAT.
    temporal = {}
    for group_name in ("publisher_benign", "profiling_idle", "profiling_active", "profiling_power", "profiling_interaction"):
        if group_name not in groups:
            continue
        temporal[group_name] = {
            "window_span_seconds_mean": groups[group_name]
            .features["window_span_seconds"]
            .mean,
            "iat_mean_seconds_mean": groups[group_name].features["iat_mean_seconds"].mean,
            "row_count": groups[group_name].features["window_span_seconds"].count,
        }
    for family in sorted(
        g for g in groups if g.startswith("attack_family_")
    ):
        temporal[family] = {
            "window_span_seconds_mean": groups[family]
            .features["window_span_seconds"]
            .mean,
            "iat_mean_seconds_mean": groups[family].features["iat_mean_seconds"].mean,
            "row_count": groups[family].features["window_span_seconds"].count,
        }

    payload = {
        "split": "train",
        "feature_strategy_version": FEATURE_STRATEGY_VERSION,
        "feature_build_strategy_version": FEATURE_BUILD_STRATEGY_VERSION,
        "pcap_count": len(manifest_rows),
        "total_feature_rows": total_rows,
        "group_count": len(groups),
        "percentile_sample_cap": percentile_sample_cap,
        "percentile_sample_seed": percentile_sample_seed,
        "artifacts": {
            "group_summary": to_repo_relative(group_out, project_root=root),
            "pcap_diagnostics": to_repo_relative(diag_out, project_root=root),
        },
        "diagnostics": {
            "tcp_urg_ratio_nonzero_pcaps": urg_pcaps,
            "frame_length_max_eq_8754_pcaps": jumbo_pcaps,
            "ipv6_ratio_nonzero_pcaps": ipv6_pcaps,
            "llc_ratio_nonzero_pcaps": llc_pcaps,
            "igmp_ratio_nonzero_pcaps": igmp_pcaps,
            "icmpv6_ratio_nonzero_pcaps": icmpv6_pcaps,
            "temporal_group_means": temporal,
        },
        "note": (
            "TRAIN-only characterization for shortcut diagnostics. "
            "Do not drop features from this report alone."
        ),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return GroupCharacterizationResult(
        group_rows=group_rows,
        pcap_rows=pcap_rows,
        group_summary_path=group_out,
        pcap_diagnostics_path=diag_out,
        summary_json_path=json_out,
        pcap_count=len(manifest_rows),
        total_feature_rows=total_rows,
        group_count=len(groups),
    )


def format_group_characterization_summary(
    result: GroupCharacterizationResult,
    *,
    project_root: Path | None = None,
) -> str:
    root = (project_root or PROJECT_ROOT).resolve()
    lines = [
        "Phase 1C.3b — TRAIN per-group feature characterization (read-only)",
        f"pcap_count: {result.pcap_count}",
        f"total_feature_rows: {result.total_feature_rows:,}",
        f"group_count: {result.group_count}",
        f"group_summary: {to_repo_relative(result.group_summary_path, project_root=root)}",
        f"pcap_diagnostics: {to_repo_relative(result.pcap_diagnostics_path, project_root=root)}",
        f"summary_json: {to_repo_relative(result.summary_json_path, project_root=root)}",
    ]
    return "\n".join(lines) + "\n"
