"""FIT-only hierarchical sampling characterization (Phase 2A).

Phase 2B must materialize rows with the same reservoir contract; 2A only
needs ``min(window_count, cap)`` for count simulation.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from iot_pcap_pipeline.modeling.seeds import reservoir_seed_for_pcap
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_SEED,
    MODELING_SPLIT_STRATEGY_VERSION,
)

SAMPLING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "plan_id",
    "binary_label",
    "category",
    "attack_family",
    "attack_type",
    "pcap_count",
    "windows_available",
    "cap_per_pcap",
    "windows_after_cap",
    "fraction_retained",
    "fraction_of_attack_sample",
    "attack_to_benign_ratio",
)

# Candidate characterization plans (not frozen).
# None cap => keep all windows for that family/category.
CANDIDATE_PLANS: tuple[dict[str, Any], ...] = (
    {
        "plan_id": "fullish",
        "description": (
            "High per-PCAP caps on floods; MQTT moderate; rare/BENIGN uncapped"
        ),
        "caps": {
            "DDoS": 50_000,
            "DoS": 50_000,
            "MQTT": 50_000,
            "Recon": None,
            "Spoofing": None,
            "BENIGN": None,
        },
    },
    {
        "plan_id": "family_balanced",
        "description": (
            "Tighter flood caps so DDoS/DoS/MQTT are similar order of magnitude"
        ),
        "caps": {
            "DDoS": 15_000,
            "DoS": 15_000,
            "MQTT": 30_000,
            "Recon": None,
            "Spoofing": None,
            "BENIGN": None,
        },
    },
    {
        "plan_id": "aggressive",
        "description": "Lower flood caps for a smaller FIT training view",
        "caps": {
            "DDoS": 8_000,
            "DoS": 8_000,
            "MQTT": 15_000,
            "Recon": None,
            "Spoofing": None,
            "BENIGN": None,
        },
    },
)


@dataclass(frozen=True)
class ReservoirContract:
    unit: str = "window"
    scope: str = "independently within each FIT PCAP"
    algorithm: str = "deterministic_reservoir_sample_without_replacement"
    seed_formula: str = (
        f"int(sha256('{MODELING_SPLIT_STRATEGY_VERSION}|{{base_seed}}|{{pcap_id}}')"
        ".hexdigest()[:16], 16)"
    )
    note: str = (
        "Phase 2A simulates counts via min(N, cap) only. "
        "Phase 2B must execute this reservoir algorithm when materializing views. "
        "TRAIN-validation is never sampled."
    )


def cap_for_row(row: dict[str, Any], caps: dict[str, int | None]) -> int | None:
    """Return per-PCAP cap for a FIT row (None = uncapped)."""
    if row.get("binary_label") == "BENIGN":
        return caps.get("BENIGN")
    family = (row.get("attack_family") or "").strip()
    return caps.get(family)


def windows_after_cap(window_count: int, cap: int | None) -> int:
    if cap is None:
        return int(window_count)
    return min(int(window_count), int(cap))


def simulate_plan(
    fit_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    base_seed: int = DEFAULT_MODELING_SEED,
) -> list[dict[str, Any]]:
    """Return sampling_summary rows for one candidate plan (FIT rows only)."""
    caps: dict[str, int | None] = plan["caps"]
    plan_id = plan["plan_id"]

    # Aggregate by category key.
    # category for attack: attack_type (or family if empty)
    # category for benign: benign_category
    buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    total_attack_after = 0
    total_benign_after = 0

    for row in fit_rows:
        # Record reservoir seed availability (contract check / future 2B).
        _ = reservoir_seed_for_pcap(str(row["pcap_id"]), base_seed=base_seed)
        label = row["binary_label"]
        family = row.get("attack_family") or ""
        attack_type = row.get("attack_type") or ""
        if label == "BENIGN":
            category = row.get("benign_category") or "publisher_benign"
        else:
            category = attack_type or family or "attack"
        key = (label, category, family, attack_type)
        if key not in buckets:
            buckets[key] = {
                "plan_id": plan_id,
                "binary_label": label,
                "category": category,
                "attack_family": family,
                "attack_type": attack_type,
                "pcap_count": 0,
                "windows_available": 0,
                "cap_per_pcap": "",
                "windows_after_cap": 0,
            }
        b = buckets[key]
        cap = cap_for_row(row, caps)
        avail = int(row["window_count"])
        after = windows_after_cap(avail, cap)
        b["pcap_count"] += 1
        b["windows_available"] += avail
        b["windows_after_cap"] += after
        # Cap displayed as family-level policy (same for all PCAPs in family).
        b["cap_per_pcap"] = "" if cap is None else str(cap)
        if label == "ATTACK":
            total_attack_after += after
        else:
            total_benign_after += after

    ratio = (
        (total_attack_after / total_benign_after) if total_benign_after > 0 else ""
    )
    out: list[dict[str, Any]] = []
    for b in sorted(
        buckets.values(),
        key=lambda r: (r["binary_label"], r["attack_family"], r["category"]),
    ):
        avail = int(b["windows_available"])
        after = int(b["windows_after_cap"])
        frac_ret = (after / avail) if avail else 0.0
        if b["binary_label"] == "ATTACK" and total_attack_after > 0:
            frac_attack = after / total_attack_after
        else:
            frac_attack = ""
        out.append(
            {
                "plan_id": plan_id,
                "binary_label": b["binary_label"],
                "category": b["category"],
                "attack_family": b["attack_family"],
                "attack_type": b["attack_type"],
                "pcap_count": b["pcap_count"],
                "windows_available": avail,
                "cap_per_pcap": b["cap_per_pcap"],
                "windows_after_cap": after,
                "fraction_retained": f"{frac_ret:.6f}",
                "fraction_of_attack_sample": (
                    "" if frac_attack == "" else f"{frac_attack:.6f}"
                ),
                "attack_to_benign_ratio": (
                    "" if ratio == "" else f"{float(ratio):.6f}"
                ),
            }
        )
    return out


def write_sampling_summary(path: Any, rows: list[dict[str, Any]]) -> Any:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLING_SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in SAMPLING_SUMMARY_COLUMNS})
    tmp.replace(out)
    return out


def build_split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate FIT/VAL summaries by logical groups (not just PCAP counts)."""

    def _side(split: str) -> dict[str, Any]:
        side_rows = [r for r in rows if r["modeling_split"] == split]
        groups = sorted({r["modeling_group_key"] for r in side_rows})
        by_family: dict[str, dict[str, int]] = defaultdict(
            lambda: {"pcaps": 0, "windows": 0, "groups": 0}
        )
        family_groups: dict[str, set[str]] = defaultdict(set)
        benign_by_cat: dict[str, dict[str, int]] = defaultdict(
            lambda: {"pcaps": 0, "windows": 0}
        )
        for r in side_rows:
            if r["binary_label"] == "ATTACK":
                fam = r.get("attack_family") or "UNKNOWN"
                by_family[fam]["pcaps"] += 1
                by_family[fam]["windows"] += int(r["window_count"])
                family_groups[fam].add(r["modeling_group_key"])
            else:
                cat = r.get("benign_category") or "publisher_benign"
                benign_by_cat[cat]["pcaps"] += 1
                benign_by_cat[cat]["windows"] += int(r["window_count"])
        for fam, gset in family_groups.items():
            by_family[fam]["groups"] = len(gset)
        return {
            "pcap_count": len(side_rows),
            "modeling_group_count": len(groups),
            "modeling_groups": groups,
            "windows_total": sum(int(r["window_count"]) for r in side_rows),
            "attack_by_family": {
                k: dict(v) for k, v in sorted(by_family.items())
            },
            "benign_by_category": {
                k: dict(v) for k, v in sorted(benign_by_cat.items())
            },
        }

    return {
        "fit": _side("fit"),
        "validation": _side("validation"),
        "validation_sampling": "none_all_windows_retained",
    }
