"""V2A A4/A5: FIT-only ARP semantic feature probe (no model training, no TEST).

Processes ARP Spoofing FIT + all BENIGN FIT PCAPs under the frozen V1
25-packet window policy. Compares candidate ARP features against ``arp_ratio``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.features.arp_v2 import (
    ARP_V2_FEATURE_NAMES,
    ARP_V2_STRATEGY_VERSION,
    extract_arp_semantic_features,
)
from iot_pcap_pipeline.modeling.baselines.data import reject_test_path
from iot_pcap_pipeline.modeling.view import DEFAULT_SPLIT_MANIFEST_PATH
from iot_pcap_pipeline.paths import PROJECT_ROOT, to_repo_relative
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE, frozen_window_policy
from iot_pcap_pipeline.windowing.stream import iter_windows
from iot_pcap_pipeline.windowing.window import PacketWindow

DEFAULT_ARP_PROBE_DIR = (
    PROJECT_ROOT / "data" / "experiments" / "v2_arp" / "phase_v2a1"
)

PROBE_GROUP_SPOOFING = "spoofing"
PROBE_GROUP_PUBLISHER_BENIGN = "publisher benign"
PROBE_GROUP_PROFILING_BENIGN = "profiling benign"

# V1 parity: arp_ratio = n_arp / WINDOW_SIZE
ARP_RATIO_NAME = "arp_ratio"

STAT_FEATURE_NAMES: tuple[str, ...] = ARP_V2_FEATURE_NAMES + (ARP_RATIO_NAME,)


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100)."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    weight = rank - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {
            "window_count": 0,
            "nonzero_count": 0,
            "nonzero_rate": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    nonzero = sum(1 for v in values if v != 0.0)
    return {
        "window_count": n,
        "nonzero_count": nonzero,
        "nonzero_rate": nonzero / n,
        "mean": sum(values) / n,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def probe_group_for_row(row: dict[str, str]) -> str | None:
    """Map a modeling-split FIT row to an A4 probe group, or None to skip."""
    if row.get("modeling_split") != "fit":
        return None
    label = (row.get("binary_label") or "").strip()
    family = (row.get("attack_family") or "").strip()
    benign_cat = (row.get("benign_category") or "").strip()
    group_kind = (row.get("group_kind") or "").strip()

    if family == "Spoofing" or group_kind == "spoofing":
        return PROBE_GROUP_SPOOFING
    if label != "BENIGN":
        return None
    if benign_cat == "publisher_benign" or group_kind == "publisher_benign":
        return PROBE_GROUP_PUBLISHER_BENIGN
    if benign_cat.startswith("profiling_") or group_kind.startswith("profiling_"):
        return PROBE_GROUP_PROFILING_BENIGN
    return None


def load_arp_probe_targets(
    split_manifest_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """FIT ARP Spoofing + all FIT BENIGN rows; rejects TEST paths."""
    root = project_root or PROJECT_ROOT
    path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    for row in rows:
        group = probe_group_for_row(row)
        if group is None:
            continue
        pcap_rel = (row.get("pcap_path") or "").strip()
        if not pcap_rel:
            raise ValueError(f"missing pcap_path in split manifest row: {row}")
        reject_test_path(pcap_rel)
        abs_path = root / pcap_rel
        if not abs_path.is_file():
            raise FileNotFoundError(f"FIT probe PCAP missing: {abs_path}")
        out = dict(row)
        out["probe_group"] = group
        selected.append(out)

    selected.sort(key=lambda r: (r["probe_group"], r["pcap_id"]))
    return selected


def _arp_ratio(window: PacketWindow) -> float:
    n_arp = sum(1 for p in window.packets if p.is_arp)
    return n_arp / float(WINDOW_SIZE)


def iter_window_feature_rows(
    pcap_path: Path,
) -> Iterator[dict[str, float]]:
    """Yield one feature dict per full window (ARP V2 + V1 arp_ratio)."""
    reject_test_path(pcap_path)
    policy = frozen_window_policy()
    for window in iter_windows(iter_packets(pcap_path), policy):
        arp = extract_arp_semantic_features(window)
        row = {name: float(arp.to_feature_dict()[name]) for name in ARP_V2_FEATURE_NAMES}
        row[ARP_RATIO_NAME] = _arp_ratio(window)
        yield row


@dataclass
class _ValueBucket:
    values: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in STAT_FEATURE_NAMES}
    )

    def add(self, row: dict[str, float]) -> None:
        for name in STAT_FEATURE_NAMES:
            self.values[name].append(float(row[name]))

    def feature_stats(self, name: str) -> dict[str, float | int]:
        return _stats(self.values[name])


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_rows(
    *,
    scope_key: str,
    scope_value: str,
    bucket: _ValueBucket,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in STAT_FEATURE_NAMES:
        stats = bucket.feature_stats(name)
        rows.append(
            {
                scope_key: scope_value,
                "feature": name,
                **stats,
            }
        )
    return rows


def _vs_arp_ratio_rows(
    group_buckets: dict[str, _ValueBucket],
) -> list[dict[str, Any]]:
    """A5: per-feature group comparison against arp_ratio."""
    rows: list[dict[str, Any]] = []
    for group, bucket in sorted(group_buckets.items()):
        arp_stats = bucket.feature_stats(ARP_RATIO_NAME)
        for name in ARP_V2_FEATURE_NAMES:
            feat_stats = bucket.feature_stats(name)
            rows.append(
                {
                    "group": group,
                    "feature": name,
                    "feature_mean": feat_stats["mean"],
                    "feature_nonzero_rate": feat_stats["nonzero_rate"],
                    "feature_p95": feat_stats["p95"],
                    "feature_max": feat_stats["max"],
                    "arp_ratio_mean": arp_stats["mean"],
                    "arp_ratio_nonzero_rate": arp_stats["nonzero_rate"],
                    "arp_ratio_p95": arp_stats["p95"],
                    "arp_ratio_max": arp_stats["max"],
                }
            )
    return rows


def _conditional_conflict_evidence(
    group_buckets: dict[str, _ValueBucket],
) -> dict[str, Any]:
    """Among windows with arp_ratio > 0, do conflict features fire on spoof only?"""
    evidence: dict[str, Any] = {}
    conflict_features = (
        "arp_sender_ip_conflict_count",
        "arp_sender_ip_conflict_ratio",
        "arp_max_macs_per_sender_ip",
        "arp_mapping_change_count",
    )
    for feat in conflict_features:
        per_group: dict[str, Any] = {}
        for group, bucket in sorted(group_buckets.items()):
            arp_vals = bucket.values[ARP_RATIO_NAME]
            feat_vals = bucket.values[feat]
            assert len(arp_vals) == len(feat_vals)
            with_arp = [
                (a, f) for a, f in zip(arp_vals, feat_vals, strict=True) if a > 0.0
            ]
            n = len(with_arp)
            if n == 0:
                per_group[group] = {
                    "windows_with_arp": 0,
                    "feature_nonzero_given_arp": 0,
                    "feature_nonzero_rate_given_arp": 0.0,
                    "feature_mean_given_arp": 0.0,
                    "arp_ratio_mean_given_arp": 0.0,
                }
                continue
            nonzero = sum(1 for _a, f in with_arp if f != 0.0)
            per_group[group] = {
                "windows_with_arp": n,
                "feature_nonzero_given_arp": nonzero,
                "feature_nonzero_rate_given_arp": nonzero / n,
                "feature_mean_given_arp": sum(f for _a, f in with_arp) / n,
                "arp_ratio_mean_given_arp": sum(a for a, _f in with_arp) / n,
            }
        evidence[feat] = per_group
    return evidence


def run_arp_fit_probe(
    *,
    split_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    progress_file: TextIO | None = None,
    max_windows_per_pcap: int | None = None,
) -> dict[str, Any]:
    """Run A4/A5 FIT-only ARP probe and write summary artifacts."""
    root = project_root or PROJECT_ROOT
    out_dir = Path(output_dir or DEFAULT_ARP_PROBE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_arp_probe_targets(split_manifest_path, project_root=root)
    if not targets:
        raise ValueError("no FIT ARP spoofing / benign PCAPs selected for probe")

    group_buckets: dict[str, _ValueBucket] = defaultdict(_ValueBucket)
    pcap_buckets: dict[str, _ValueBucket] = {}
    pcap_meta: dict[str, dict[str, str]] = {}

    for idx, row in enumerate(targets, start=1):
        pcap_id = row["pcap_id"]
        group = row["probe_group"]
        rel = row["pcap_path"]
        reject_test_path(rel)
        abs_path = root / rel
        if progress_file is not None:
            print(
                f"[{idx}/{len(targets)}] {group}: {rel}",
                file=progress_file,
                flush=True,
            )

        bucket = _ValueBucket()
        n_windows = 0
        for feat_row in iter_window_feature_rows(abs_path):
            bucket.add(feat_row)
            group_buckets[group].add(feat_row)
            n_windows += 1
            if max_windows_per_pcap is not None and n_windows >= max_windows_per_pcap:
                break

        pcap_buckets[pcap_id] = bucket
        pcap_meta[pcap_id] = {
            "group": group,
            "pcap_id": pcap_id,
            "pcap_path": rel,
            "modeling_group_key": row.get("modeling_group_key") or "",
            "binary_label": row.get("binary_label") or "",
            "benign_category": row.get("benign_category") or "",
            "attack_family": row.get("attack_family") or "",
        }
        if progress_file is not None:
            print(f"  windows={n_windows}", file=progress_file, flush=True)

    by_pcap_rows: list[dict[str, Any]] = []
    for pcap_id, bucket in sorted(pcap_buckets.items(), key=lambda kv: kv[0]):
        meta = pcap_meta[pcap_id]
        for feat_row in _summary_rows(
            scope_key="pcap_id", scope_value=pcap_id, bucket=bucket
        ):
            by_pcap_rows.append(
                {
                    "group": meta["group"],
                    "pcap_path": meta["pcap_path"],
                    "pcap_id": meta["pcap_id"],
                    "feature": feat_row["feature"],
                    "window_count": feat_row["window_count"],
                    "nonzero_count": feat_row["nonzero_count"],
                    "nonzero_rate": feat_row["nonzero_rate"],
                    "mean": feat_row["mean"],
                    "p50": feat_row["p50"],
                    "p95": feat_row["p95"],
                    "p99": feat_row["p99"],
                    "max": feat_row["max"],
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for group, bucket in sorted(group_buckets.items()):
        for feat_row in _summary_rows(
            scope_key="group", scope_value=group, bucket=bucket
        ):
            summary_rows.append(
                {
                    "group": group,
                    "feature": feat_row["feature"],
                    "window_count": feat_row["window_count"],
                    "nonzero_count": feat_row["nonzero_count"],
                    "nonzero_rate": feat_row["nonzero_rate"],
                    "mean": feat_row["mean"],
                    "p50": feat_row["p50"],
                    "p95": feat_row["p95"],
                    "p99": feat_row["p99"],
                    "max": feat_row["max"],
                }
            )

    nonzero_rows = [
        {
            "group": r["group"],
            "feature": r["feature"],
            "window_count": r["window_count"],
            "nonzero_count": r["nonzero_count"],
            "nonzero_rate": r["nonzero_rate"],
        }
        for r in summary_rows
    ]
    # Also emit PCAP-level nonzero rates (requested: group, PCAP, ...).
    nonzero_by_pcap_rows = [
        {
            "group": r["group"],
            "pcap_path": r["pcap_path"],
            "pcap_id": r["pcap_id"],
            "feature": r["feature"],
            "window_count": r["window_count"],
            "nonzero_count": r["nonzero_count"],
            "nonzero_rate": r["nonzero_rate"],
        }
        for r in by_pcap_rows
    ]

    vs_rows = _vs_arp_ratio_rows(group_buckets)
    conditional = _conditional_conflict_evidence(group_buckets)

    summary_path = out_dir / "arp_feature_summary.csv"
    by_pcap_path = out_dir / "arp_feature_by_pcap.csv"
    nonzero_path = out_dir / "arp_feature_nonzero_rates.csv"
    vs_path = out_dir / "arp_vs_arp_ratio.csv"
    complete_path = out_dir / "arp_probe_complete.json"

    _write_csv(
        summary_path,
        [
            "group",
            "feature",
            "window_count",
            "nonzero_count",
            "nonzero_rate",
            "mean",
            "p50",
            "p95",
            "p99",
            "max",
        ],
        summary_rows,
    )
    _write_csv(
        by_pcap_path,
        [
            "group",
            "pcap_path",
            "pcap_id",
            "feature",
            "window_count",
            "nonzero_count",
            "nonzero_rate",
            "mean",
            "p50",
            "p95",
            "p99",
            "max",
        ],
        by_pcap_rows,
    )
    # Nonzero-rate artifact: group-level + per-PCAP detail stacked with scope column.
    nonzero_combined = [
        {
            "scope": "group",
            "group": r["group"],
            "pcap_path": "",
            "pcap_id": "",
            "feature": r["feature"],
            "window_count": r["window_count"],
            "nonzero_count": r["nonzero_count"],
            "nonzero_rate": r["nonzero_rate"],
        }
        for r in nonzero_rows
    ] + [
        {
            "scope": "pcap",
            "group": r["group"],
            "pcap_path": r["pcap_path"],
            "pcap_id": r["pcap_id"],
            "feature": r["feature"],
            "window_count": r["window_count"],
            "nonzero_count": r["nonzero_count"],
            "nonzero_rate": r["nonzero_rate"],
        }
        for r in nonzero_by_pcap_rows
    ]
    _write_csv(
        nonzero_path,
        [
            "scope",
            "group",
            "pcap_path",
            "pcap_id",
            "feature",
            "window_count",
            "nonzero_count",
            "nonzero_rate",
        ],
        nonzero_combined,
    )
    _write_csv(
        vs_path,
        [
            "group",
            "feature",
            "feature_mean",
            "feature_nonzero_rate",
            "feature_p95",
            "feature_max",
            "arp_ratio_mean",
            "arp_ratio_nonzero_rate",
            "arp_ratio_p95",
            "arp_ratio_max",
        ],
        vs_rows,
    )

    spoof = group_buckets.get(PROBE_GROUP_SPOOFING)
    pub = group_buckets.get(PROBE_GROUP_PUBLISHER_BENIGN)
    prof = group_buckets.get(PROBE_GROUP_PROFILING_BENIGN)

    def _nz(bucket: _ValueBucket | None, name: str) -> float:
        if bucket is None:
            return 0.0
        return float(bucket.feature_stats(name)["nonzero_rate"])

    semantic_signal_notes: list[str] = []
    for feat in (
        "arp_sender_ip_conflict_count",
        "arp_sender_ip_conflict_ratio",
        "arp_mapping_change_count",
        "arp_max_macs_per_sender_ip",
    ):
        s_nz = _nz(spoof, feat)
        b_nz = max(_nz(pub, feat), _nz(prof, feat))
        s_arp = _nz(spoof, ARP_RATIO_NAME)
        b_arp = max(_nz(pub, ARP_RATIO_NAME), _nz(prof, ARP_RATIO_NAME))
        if s_nz > 0 and b_nz == 0.0:
            semantic_signal_notes.append(
                f"{feat}: nonzero on spoofing ({s_nz:.4f}) and zero on both benign "
                f"groups — semantic signal beyond presence of ARP "
                f"(spoof arp_ratio nonzero={s_arp:.4f}, benign max={b_arp:.4f})"
            )
        elif s_nz > b_nz * 5 and s_nz > 0:
            semantic_signal_notes.append(
                f"{feat}: spoofing nonzero_rate {s_nz:.4f} >> benign {b_nz:.4f}"
            )
        elif s_nz == 0.0:
            semantic_signal_notes.append(
                f"{feat}: zero on spoofing FIT windows — stateless within-window "
                "conflict may be insufficient for this capture"
            )

    complete = {
        "status": "complete",
        "strategy_version": ARP_V2_STRATEGY_VERSION,
        "phase": "v2a1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "windowing": {
            "window_size": WINDOW_SIZE,
            "policy": "frozen_window_policy",
        },
        "data_access": {
            "development_data": "FIT only",
            "v1_final_test_access": False,
            "pcaps": [
                {
                    "probe_group": t["probe_group"],
                    "pcap_id": t["pcap_id"],
                    "pcap_path": t["pcap_path"],
                }
                for t in targets
            ],
            "pcap_count": len(targets),
            "group_counts": {
                g: sum(1 for t in targets if t["probe_group"] == g)
                for g in (
                    PROBE_GROUP_SPOOFING,
                    PROBE_GROUP_PUBLISHER_BENIGN,
                    PROBE_GROUP_PROFILING_BENIGN,
                )
            },
        },
        "artifacts": {
            "arp_feature_summary": to_repo_relative(summary_path, project_root=root),
            "arp_feature_by_pcap": to_repo_relative(by_pcap_path, project_root=root),
            "arp_feature_nonzero_rates": to_repo_relative(
                nonzero_path, project_root=root
            ),
            "arp_vs_arp_ratio": to_repo_relative(vs_path, project_root=root),
            "arp_probe_complete": to_repo_relative(complete_path, project_root=root),
        },
        "window_counts_by_group": {
            g: int(bucket.feature_stats(ARP_RATIO_NAME)["window_count"])
            for g, bucket in sorted(group_buckets.items())
        },
        "a5_conditional_conflict_given_arp": conditional,
        "a5_semantic_signal_notes": semantic_signal_notes,
        "max_windows_per_pcap": max_windows_per_pcap,
    }
    complete_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    return complete


__all__ = [
    "ARP_RATIO_NAME",
    "DEFAULT_ARP_PROBE_DIR",
    "PROBE_GROUP_PROFILING_BENIGN",
    "PROBE_GROUP_PUBLISHER_BENIGN",
    "PROBE_GROUP_SPOOFING",
    "load_arp_probe_targets",
    "probe_group_for_row",
    "run_arp_fit_probe",
]
