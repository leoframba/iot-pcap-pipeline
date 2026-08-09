"""TRAIN-only feature characterization for Gate B smoke builds."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _pop_std(values: list[float]) -> float | None:
    if not values:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def summary_category(row: dict[str, Any]) -> str | None:
    source = (row.get("source") or "").strip()
    # inventory uses source; smoke rows may only have binary_label/attack_family
    label = (row.get("binary_label") or "").strip()
    family = (row.get("attack_family") or "").strip()
    profiling_type = (row.get("profiling_type") or "").strip()

    if label == "BENIGN" and not profiling_type and not family:
        return "publisher_benign"
    if profiling_type:
        return f"profiling_{profiling_type}"
    if label == "ATTACK" and family in {
        "DDoS",
        "DoS",
        "MQTT",
        "Recon",
        "Spoofing",
    }:
        return family
    # Fallback when source is present
    if source == "attacks" and label == "BENIGN":
        return "publisher_benign"
    if source == "profiling" and profiling_type:
        return f"profiling_{profiling_type}"
    return None


CHARACTERIZATION_COLUMNS: list[str] = [
    "scope",
    "feature",
    "count",
    "null_count",
    "nonfinite_count",
    "min",
    "max",
    "mean",
    "std",
    "p01",
    "p50",
    "p99",
]


def _feature_stats(
    rows: list[dict[str, Any]],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in V1_FEATURE_NAMES:
        values: list[float] = []
        null_count = 0
        nonfinite_count = 0
        for row in rows:
            raw = row.get(name, "")
            if raw is None or raw == "":
                null_count += 1
                continue
            value = float(raw)
            if not math.isfinite(value):
                nonfinite_count += 1
                continue
            values.append(value)
        out.append(
            {
                "scope": scope,
                "feature": name,
                "count": len(values),
                "null_count": null_count,
                "nonfinite_count": nonfinite_count,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": _mean(values),
                "std": _pop_std(values),
                "p01": _percentile(values, 1),
                "p50": _percentile(values, 50),
                "p99": _percentile(values, 99),
            }
        )
    return out


def characterize_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-feature stats overall and by summary category."""
    result = _feature_stats(rows, scope="all")
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cat = summary_category(row)
        if cat is not None:
            by_cat[cat].append(row)
    for cat in sorted(by_cat):
        result.extend(_feature_stats(by_cat[cat], scope=cat))
    return result


def write_characterization_csv(
    rows: list[dict[str, Any]],
    path: Path | str,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = characterize_feature_rows(rows)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHARACTERIZATION_COLUMNS)
        writer.writeheader()
        for row in stats:
            writer.writerow(
                {k: ("" if row.get(k) is None else row.get(k)) for k in CHARACTERIZATION_COLUMNS}
            )
    return out
