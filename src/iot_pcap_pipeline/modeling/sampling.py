"""FIT-only hierarchical sampling characterization (Phase 2A).

Attack hierarchy for group budgets:

  label → attack family → modeling_group_key → PCAP → windows

Per-PCAP plans keep a family-level cap applied independently to each PCAP.
Group plans give each modeling_group_key a total budget, allocated across
member PCAPs (deterministic proportional + largest-remainder).

Phase 2B must materialize rows with the reservoir contract; 2A only needs
counts after allocation.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from iot_pcap_pipeline.modeling.seeds import reservoir_seed_for_pcap
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_SEED,
    MODELING_SPLIT_STRATEGY_VERSION,
)

CapMode = Literal["per_pcap", "per_modeling_group"]

SAMPLING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "plan_id",
    "cap_mode",
    "binary_label",
    "category",
    "attack_family",
    "attack_type",
    "modeling_group_key",
    "pcap_count",
    "windows_available",
    "cap_per_pcap",
    "cap_per_modeling_group",
    "windows_after_cap",
    "fraction_retained",
    "fraction_of_attack_sample",
    "attack_to_benign_ratio",
)

# Candidate characterization plans (not frozen).
# None cap => keep all windows for that family.
CANDIDATE_PLANS: tuple[dict[str, Any], ...] = (
    {
        "plan_id": "fullish",
        "cap_mode": "per_pcap",
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
        "cap_mode": "per_pcap",
        "description": (
            "Tighter per-PCAP flood caps so DDoS/DoS/MQTT are similar order "
            "of magnitude (chunk-count sensitive)"
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
        "cap_mode": "per_pcap",
        "description": "Lower per-PCAP flood caps for a smaller FIT training view",
        "caps": {
            "DDoS": 8_000,
            "DoS": 8_000,
            "MQTT": 15_000,
            "Recon": None,
            "Spoofing": None,
            "BENIGN": None,
        },
    },
    {
        "plan_id": "group_balanced",
        "cap_mode": "per_modeling_group",
        "description": (
            "Budget per attack-type lineage (modeling_group_key), allocated "
            "across member PCAPs — avoids TCPDUMP chunk-count dominating weight"
        ),
        "caps": {
            "DDoS": 60_000,
            "DoS": 60_000,
            "MQTT": 30_000,
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
        "Phase 2A simulates counts via caps / group budgets only. "
        "Phase 2B must execute this reservoir algorithm when materializing "
        "views, respecting per-PCAP k after group-budget allocation. "
        "TRAIN-validation is never sampled."
    )


def family_cap(row: dict[str, Any], caps: dict[str, int | None]) -> int | None:
    """Return family/BENIGN policy cap (shared by per-PCAP and group modes)."""
    if row.get("binary_label") == "BENIGN":
        return caps.get("BENIGN")
    family = (row.get("attack_family") or "").strip()
    return caps.get(family)


def windows_after_cap(window_count: int, cap: int | None) -> int:
    if cap is None:
        return int(window_count)
    return min(int(window_count), int(cap))


def reservoir_indices(
    population_size: int,
    sample_size: int,
    seed: int,
) -> list[int]:
    """Deterministic Algorithm-R reservoir sample of row indices (sorted).

    Selects ``sample_size`` distinct indices from ``0 .. population_size-1``
    using ``random.Random(seed)``. Phase 2B must use this exact algorithm with
    ``seed=reservoir_seed_for_pcap(pcap_id)``.
    """
    n = int(population_size)
    k = int(sample_size)
    if n < 0 or k < 0:
        raise ValueError(
            f"population_size and sample_size must be >= 0; got n={n}, k={k}"
        )
    if k > n:
        raise ValueError(f"sample_size {k} exceeds population_size {n}")
    if k == 0:
        return []
    if k == n:
        return list(range(n))

    rng = random.Random(int(seed))
    reservoir = list(range(k))
    for i in range(k, n):
        j = rng.randint(0, i)
        if j < k:
            reservoir[j] = i
    return sorted(reservoir)


def allocate_fit_sample_sizes(
    fit_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    base_seed: int = DEFAULT_MODELING_SEED,
) -> dict[str, int]:
    """Return per-``pcap_path`` allocated sample sizes for a frozen plan."""
    caps: dict[str, int | None] = plan["caps"]
    cap_mode: CapMode = plan.get("cap_mode", "per_pcap")  # type: ignore[assignment]
    selected: dict[str, int] = {}
    if cap_mode == "per_pcap":
        for row in fit_rows:
            _ = reservoir_seed_for_pcap(str(row["pcap_id"]), base_seed=base_seed)
            cap = family_cap(row, caps)
            selected[str(row["pcap_path"])] = windows_after_cap(
                int(row["window_count"]), cap
            )
    elif cap_mode == "per_modeling_group":
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in fit_rows:
            _ = reservoir_seed_for_pcap(str(row["pcap_id"]), base_seed=base_seed)
            by_group[str(row["modeling_group_key"])].append(row)
        for _gkey, members in by_group.items():
            budget = family_cap(members[0], caps)
            selected.update(allocate_group_budget(members, budget))
    else:
        raise ValueError(f"unknown cap_mode: {cap_mode!r}")
    return selected


def allocate_group_budget(
    members: list[dict[str, Any]],
    budget: int | None,
) -> dict[str, int]:
    """Allocate a modeling-group budget across member PCAPs.

    Deterministic proportional allocation with largest-remainder, capped by
    each PCAP's available windows. If ``budget`` is None or covers all
    windows, every PCAP keeps its full window_count.
    """
    ordered = sorted(members, key=lambda m: str(m["pcap_path"]))
    if not ordered:
        return {}
    paths = [str(m["pcap_path"]) for m in ordered]
    avail = {str(m["pcap_path"]): int(m["window_count"]) for m in ordered}
    total = sum(avail.values())
    if budget is None or total <= int(budget):
        return dict(avail)

    budget_i = int(budget)
    floors: dict[str, int] = {}
    fracs: list[tuple[float, str]] = []
    for path in paths:
        exact = budget_i * (avail[path] / total)
        floor_v = min(avail[path], int(exact))
        floors[path] = floor_v
        fracs.append((exact - int(exact), path))
    remainder = budget_i - sum(floors.values())
    for _frac, path in sorted(fracs, key=lambda t: (-t[0], t[1])):
        if remainder <= 0:
            break
        if floors[path] < avail[path]:
            floors[path] += 1
            remainder -= 1
    return floors


def simulate_plan(
    fit_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    base_seed: int = DEFAULT_MODELING_SEED,
) -> list[dict[str, Any]]:
    """Return sampling_summary rows for one candidate plan (FIT rows only)."""
    caps: dict[str, int | None] = plan["caps"]
    plan_id = str(plan["plan_id"])
    cap_mode: CapMode = plan.get("cap_mode", "per_pcap")  # type: ignore[assignment]

    selected = allocate_fit_sample_sizes(fit_rows, plan, base_seed=base_seed)

    # Aggregate by (label, category, family, type, modeling_group_key).
    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    total_attack_after = 0
    total_benign_after = 0

    for row in fit_rows:
        path = str(row["pcap_path"])
        label = row["binary_label"]
        family = row.get("attack_family") or ""
        attack_type = row.get("attack_type") or ""
        gkey = row.get("modeling_group_key") or ""
        if label == "BENIGN":
            category = row.get("benign_category") or "publisher_benign"
        else:
            category = attack_type or family or "attack"
        key = (label, category, family, attack_type, gkey)
        if key not in buckets:
            fam_cap = family_cap(row, caps)
            buckets[key] = {
                "plan_id": plan_id,
                "cap_mode": cap_mode,
                "binary_label": label,
                "category": category,
                "attack_family": family,
                "attack_type": attack_type,
                "modeling_group_key": gkey,
                "pcap_count": 0,
                "windows_available": 0,
                "cap_per_pcap": "" if cap_mode != "per_pcap" or fam_cap is None else str(fam_cap),
                "cap_per_modeling_group": (
                    ""
                    if cap_mode != "per_modeling_group" or fam_cap is None
                    else str(fam_cap)
                ),
                "windows_after_cap": 0,
            }
        b = buckets[key]
        avail = int(row["window_count"])
        after = int(selected[path])
        b["pcap_count"] += 1
        b["windows_available"] += avail
        b["windows_after_cap"] += after
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
        key=lambda r: (
            r["binary_label"],
            r["attack_family"],
            r["modeling_group_key"],
            r["category"],
        ),
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
                "cap_mode": b["cap_mode"],
                "binary_label": b["binary_label"],
                "category": b["category"],
                "attack_family": b["attack_family"],
                "attack_type": b["attack_type"],
                "modeling_group_key": b["modeling_group_key"],
                "pcap_count": b["pcap_count"],
                "windows_available": avail,
                "cap_per_pcap": b["cap_per_pcap"],
                "cap_per_modeling_group": b["cap_per_modeling_group"],
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
