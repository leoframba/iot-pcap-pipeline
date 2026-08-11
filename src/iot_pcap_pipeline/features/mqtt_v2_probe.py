"""V2M M4: FIT-only MQTT structural characterization probe (no training, no TEST)."""

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

from iot_pcap_pipeline.features.mqtt_v2 import (
    MQTT_PLAINTEXT_PORTS,
    MQTT_V2_FEATURE_NAMES,
    MQTT_V2_STRATEGY_VERSION,
    extract_mqtt_structural_features,
)
from iot_pcap_pipeline.modeling.baselines.data import reject_test_path
from iot_pcap_pipeline.modeling.view import DEFAULT_SPLIT_MANIFEST_PATH
from iot_pcap_pipeline.paths import PROJECT_ROOT, to_repo_relative
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE, frozen_window_policy
from iot_pcap_pipeline.windowing.stream import iter_windows

DEFAULT_MQTT_PROBE_DIR = (
    PROJECT_ROOT / "data" / "experiments" / "v2_mqtt" / "phase_v2m1b"
)

PROBE_GROUP_MQTT_MALFORMED = "mqtt_malformed"
PROBE_GROUP_PUBLISHER_BENIGN = "publisher benign"
PROBE_GROUP_PROFILING_BENIGN = "profiling benign"

# Stems / path markers for FIT benign PCAPs known to carry plaintext MQTT.
_BENIGN_MQTT_MARKERS = (
    "Benign_train",
    "ActiveBroker",
    "/Active/Active.pcap",
    "/Active/Active",
)


def _percentile(values: list[float], p: float) -> float:
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


def _is_benign_mqtt_pcap(row: dict[str, str]) -> bool:
    path = (row.get("pcap_path") or "").replace("\\", "/")
    pcap_id = row.get("pcap_id") or ""
    for marker in _BENIGN_MQTT_MARKERS:
        if marker in path or marker in pcap_id:
            return True
    return False


def probe_group_for_row(row: dict[str, str]) -> str | None:
    if row.get("modeling_split") != "fit":
        return None
    attack_type = (row.get("attack_type") or "").strip()
    label = (row.get("binary_label") or "").strip()
    benign_cat = (row.get("benign_category") or "").strip()
    group_kind = (row.get("group_kind") or "").strip()

    if attack_type == "MQTT_Malformed_Data":
        return PROBE_GROUP_MQTT_MALFORMED
    if label != "BENIGN":
        return None
    if not _is_benign_mqtt_pcap(row):
        return None
    if benign_cat == "publisher_benign" or group_kind == "publisher_benign":
        return PROBE_GROUP_PUBLISHER_BENIGN
    return PROBE_GROUP_PROFILING_BENIGN


def load_mqtt_probe_targets(
    split_manifest_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> list[dict[str, str]]:
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
            raise FileNotFoundError(f"FIT MQTT probe PCAP missing: {abs_path}")
        out = dict(row)
        out["probe_group"] = group
        selected.append(out)
    selected.sort(key=lambda r: (r["probe_group"], r["pcap_id"]))
    return selected


def iter_window_mqtt_rows(pcap_path: Path) -> Iterator[dict[str, float]]:
    reject_test_path(pcap_path)
    for window in iter_windows(iter_packets(pcap_path), frozen_window_policy()):
        feats = extract_mqtt_structural_features(window)
        yield {name: float(feats.to_feature_dict()[name]) for name in MQTT_V2_FEATURE_NAMES}


@dataclass
class _ValueBucket:
    values: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in MQTT_V2_FEATURE_NAMES}
    )

    def add(self, row: dict[str, float]) -> None:
        for name in MQTT_V2_FEATURE_NAMES:
            self.values[name].append(float(row[name]))

    def feature_stats(self, name: str) -> dict[str, float | int]:
        return _stats(self.values[name])

    def conditional_mqtt_stats(self, name: str) -> dict[str, float | int]:
        """Stats restricted to windows with mqtt_frame_count > 0."""
        counts = self.values["mqtt_frame_count"]
        vals = self.values[name]
        filtered = [v for c, v in zip(counts, vals, strict=True) if c > 0.0]
        return _stats(filtered)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_mqtt_fit_probe(
    *,
    split_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    progress_file: TextIO | None = None,
    max_windows_per_pcap: int | None = None,
) -> dict[str, Any]:
    root = project_root or PROJECT_ROOT
    out_dir = Path(output_dir or DEFAULT_MQTT_PROBE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_mqtt_probe_targets(split_manifest_path, project_root=root)
    if not targets:
        raise ValueError("no FIT MQTT malformed / benign-MQTT PCAPs selected")

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
        for feat_row in iter_window_mqtt_rows(abs_path):
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
        }
        if progress_file is not None:
            mqtt_windows = sum(1 for c in bucket.values["mqtt_frame_count"] if c > 0)
            print(
                f"  windows={n_windows} mqtt_windows={mqtt_windows}",
                file=progress_file,
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for group, bucket in sorted(group_buckets.items()):
        for name in MQTT_V2_FEATURE_NAMES:
            all_s = bucket.feature_stats(name)
            cond_s = bucket.conditional_mqtt_stats(name)
            summary_rows.append(
                {
                    "group": group,
                    "feature": name,
                    "scope": "all_windows",
                    **all_s,
                }
            )
            summary_rows.append(
                {
                    "group": group,
                    "feature": name,
                    "scope": "windows_with_mqtt",
                    "window_count": cond_s["window_count"],
                    "nonzero_count": cond_s["nonzero_count"],
                    "nonzero_rate": cond_s["nonzero_rate"],
                    "mean": cond_s["mean"],
                    "p50": cond_s["p50"],
                    "p95": cond_s["p95"],
                    "p99": cond_s["p99"],
                    "max": cond_s["max"],
                }
            )

    by_pcap_rows: list[dict[str, Any]] = []
    for pcap_id, bucket in sorted(pcap_buckets.items()):
        meta = pcap_meta[pcap_id]
        for name in MQTT_V2_FEATURE_NAMES:
            all_s = bucket.feature_stats(name)
            cond_s = bucket.conditional_mqtt_stats(name)
            by_pcap_rows.append(
                {
                    "group": meta["group"],
                    "pcap_path": meta["pcap_path"],
                    "pcap_id": pcap_id,
                    "feature": name,
                    "scope": "all_windows",
                    **all_s,
                }
            )
            by_pcap_rows.append(
                {
                    "group": meta["group"],
                    "pcap_path": meta["pcap_path"],
                    "pcap_id": pcap_id,
                    "feature": name,
                    "scope": "windows_with_mqtt",
                    "window_count": cond_s["window_count"],
                    "nonzero_count": cond_s["nonzero_count"],
                    "nonzero_rate": cond_s["nonzero_rate"],
                    "mean": cond_s["mean"],
                    "p50": cond_s["p50"],
                    "p95": cond_s["p95"],
                    "p99": cond_s["p99"],
                    "max": cond_s["max"],
                }
            )

    violation_features = (
        "mqtt_invalid_count",
        "mqtt_invalid_ratio",
        "mqtt_invalid_fixed_header_count",
        "mqtt_invalid_remaining_length_count",
        "mqtt_invalid_publish_qos_count",
        "mqtt_publish_wildcard_topic_count",
        "mqtt_invalid_publish_topic_count",
        "mqtt_invalid_connect_count",
        "mqtt_incomplete_count",
        "mqtt_incomplete_ratio",
    )
    violation_rows = [
        r
        for r in summary_rows
        if r["feature"] in violation_features and r["scope"] == "windows_with_mqtt"
    ]

    # Hypothesis check: invalid rate given MQTT should be high on malformed, low on benign.
    notes: list[str] = []
    def _cond_mean(group: str, feature: str) -> float:
        bucket = group_buckets.get(group)
        if bucket is None:
            return 0.0
        return float(bucket.conditional_mqtt_stats(feature)["mean"])

    def _cond_nz(group: str, feature: str) -> float:
        bucket = group_buckets.get(group)
        if bucket is None:
            return 0.0
        return float(bucket.conditional_mqtt_stats(feature)["nonzero_rate"])

    mal_inv = _cond_mean(PROBE_GROUP_MQTT_MALFORMED, "mqtt_invalid_ratio")
    pub_inv = _cond_mean(PROBE_GROUP_PUBLISHER_BENIGN, "mqtt_invalid_ratio")
    prof_inv = _cond_mean(PROBE_GROUP_PROFILING_BENIGN, "mqtt_invalid_ratio")
    benign_inv = max(pub_inv, prof_inv)
    if mal_inv > 0.2 and benign_inv < 0.05:
        notes.append(
            f"structural_signal_promising: mqtt_invalid_ratio|mqtt "
            f"malformed={mal_inv:.4f} >> benign_max={benign_inv:.4f}"
        )
        recommendation = "continue_candidate_selection"
    elif mal_inv <= benign_inv * 1.5:
        notes.append(
            f"hypothesis_failed: mqtt_invalid_ratio|mqtt malformed={mal_inv:.4f} "
            f"not clearly above benign_max={benign_inv:.4f}"
        )
        recommendation = "stop_mqtt_structural_features"
    else:
        notes.append(
            f"structural_signal_weak_or_partial: mqtt_invalid_ratio|mqtt "
            f"malformed={mal_inv:.4f} benign_max={benign_inv:.4f}"
        )
        recommendation = "review_before_training"

    summary_path = out_dir / "mqtt_feature_summary.csv"
    by_pcap_path = out_dir / "mqtt_feature_by_pcap.csv"
    viol_path = out_dir / "mqtt_violation_summary.csv"
    complete_path = out_dir / "mqtt_probe_complete.json"

    _write_csv(
        summary_path,
        [
            "group",
            "feature",
            "scope",
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
            "scope",
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
    _write_csv(
        viol_path,
        [
            "group",
            "feature",
            "scope",
            "window_count",
            "nonzero_count",
            "nonzero_rate",
            "mean",
            "p50",
            "p95",
            "p99",
            "max",
        ],
        violation_rows,
    )

    complete = {
        "status": "complete",
        "strategy_version": MQTT_V2_STRATEGY_VERSION,
        "phase": "v2m1b",
        "parent_attempt": "data/experiments/v2_mqtt/phase_v2m1/attempt1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "windowing": {
            "window_size": WINDOW_SIZE,
            "tcp_reassembly": False,
            "mqtt_plaintext_ports": sorted(MQTT_PLAINTEXT_PORTS),
        },
        "model_training": False,
        "data_access": {
            "development_data": "FIT only",
            "v1_final_test_access": False,
            "pcap_count": len(targets),
            "pcaps": [
                {
                    "probe_group": t["probe_group"],
                    "pcap_id": t["pcap_id"],
                    "pcap_path": t["pcap_path"],
                }
                for t in targets
            ],
        },
        "hypothesis": {
            "statement": (
                "MQTT_Malformed_Data contains protocol-structural violations "
                "observable from raw TCP payloads and uncommon in benign MQTT"
            ),
            "mqtt_invalid_ratio_given_mqtt": {
                "mqtt_malformed": mal_inv,
                "publisher_benign": pub_inv,
                "profiling_benign": prof_inv,
            },
            "mqtt_invalid_nonzero_rate_given_mqtt": {
                "mqtt_malformed": _cond_nz(
                    PROBE_GROUP_MQTT_MALFORMED, "mqtt_invalid_count"
                ),
                "publisher_benign": _cond_nz(
                    PROBE_GROUP_PUBLISHER_BENIGN, "mqtt_invalid_count"
                ),
                "profiling_benign": _cond_nz(
                    PROBE_GROUP_PROFILING_BENIGN, "mqtt_invalid_count"
                ),
            },
            "notes": notes,
            "recommended_next_step": recommendation,
        },
        "artifacts": {
            "mqtt_feature_summary": to_repo_relative(summary_path, project_root=root),
            "mqtt_feature_by_pcap": to_repo_relative(by_pcap_path, project_root=root),
            "mqtt_violation_summary": to_repo_relative(viol_path, project_root=root),
            "mqtt_probe_complete": to_repo_relative(complete_path, project_root=root),
        },
        "max_windows_per_pcap": max_windows_per_pcap,
    }
    complete_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    return complete


__all__ = [
    "DEFAULT_MQTT_PROBE_DIR",
    "PROBE_GROUP_MQTT_MALFORMED",
    "PROBE_GROUP_PROFILING_BENIGN",
    "PROBE_GROUP_PUBLISHER_BENIGN",
    "load_mqtt_probe_targets",
    "probe_group_for_row",
    "run_mqtt_fit_probe",
]
