"""Deterministic whole-PCAP / device-group profiling split assignment."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Any

from iot_pcap_pipeline.dataset.schema import validate_inventory
from iot_pcap_pipeline.paths import DEFAULT_SPLIT_SEED, SPLIT_STRATEGY_VERSION

TARGET_TEST_FRACTION = 0.20
SINGLETON_PREFIX = "profiling:singleton:"
DEVICE_PREFIX = "profiling:device:"


def _profiling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("source") == "profiling" and r.get("binary_label") != "UNKNOWN"
    ]


def _group_profiling(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["grouping_key"]
        if key is None:
            raise ValueError(f"profiling row missing grouping_key: {row['pcap_path']}")
        groups[key].append(row)
    return dict(groups)


def _has_power_and_interaction(group_rows: list[dict[str, Any]]) -> bool:
    types = {r.get("profiling_type") for r in group_rows}
    return "power" in types and "interaction" in types


def _select_test_device_groups(
    device_groups: dict[str, list[dict[str, Any]]],
    total_profiling: int,
    seed: int,
) -> set[str]:
    """Select indivisible device groups for TEST.

    Hard constraints:
    - only device groups are candidates (singletons stay TRAIN)
    - prefer assignments whose TEST set includes power and interaction
    Soft objective:
    - among valid assignments, minimize |test_count/total - 0.20|
    Tie-break with seed 42 via shuffled enumeration order.
    """
    if total_profiling == 0:
        return set()

    keys = sorted(device_groups.keys())
    sizes = {k: len(device_groups[k]) for k in keys}
    capable = {k for k in keys if _has_power_and_interaction(device_groups[k])}

    rng = random.Random(seed)
    # Deterministic shuffle of combination search order for tie-breaking.
    search_order: list[tuple[str, ...]] = []
    for r in range(1, len(keys) + 1):
        combos = list(itertools.combinations(keys, r))
        rng.shuffle(combos)
        search_order.extend(combos)

    # Also consider empty TEST among device groups (all devices train) as last resort.
    search_order.append(())

    target = TARGET_TEST_FRACTION * total_profiling
    best: tuple[str, ...] | None = None
    best_key: tuple[int, float, int] | None = None

    for combo in search_order:
        test_count = sum(sizes[k] for k in combo)
        # Prefer combos that can provide power+interaction in TEST.
        # A combo provides both if any selected group has both, OR the union does.
        union_types: set[str] = set()
        for k in combo:
            union_types.update(
                r.get("profiling_type") for r in device_groups[k] if r.get("profiling_type")
            )
        has_both = "power" in union_types and "interaction" in union_types
        # Hard preference: if any capable groups exist, require has_both when
        # at least one device group is selected.
        if capable and combo and not has_both:
            continue

        distance = abs(test_count - target)
        # Sort key: prefer has_both (0), then closer distance, then fewer groups.
        quality = (0 if has_both else 1, distance, len(combo))
        if best_key is None or quality < best_key:
            best_key = quality
            best = combo

    if best is None:
        return set()
    return set(best)


def assign_profiling_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    strategy_version: str = SPLIT_STRATEGY_VERSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign profiling train/test splits and return (inventory, split_records).

    Attack / unknown rows keep existing splits. Profiling UNKNOWN rows keep
    their existing placeholder split and are not reassigned by the optimizer.
    """
    inventory = [dict(row) for row in rows]
    by_path = {r["pcap_path"]: r for r in inventory}

    profiling = _profiling_rows(inventory)
    groups = _group_profiling(profiling)

    singleton_keys = sorted(k for k in groups if k.startswith(SINGLETON_PREFIX))
    device_keys = sorted(k for k in groups if k.startswith(DEVICE_PREFIX))
    other_keys = sorted(
        k
        for k in groups
        if not k.startswith(SINGLETON_PREFIX) and not k.startswith(DEVICE_PREFIX)
    )
    if other_keys:
        raise ValueError(f"unexpected profiling grouping keys: {other_keys}")

    # Singletons always TRAIN.
    for key in singleton_keys:
        for row in groups[key]:
            row["split"] = "train"

    device_groups = {k: groups[k] for k in device_keys}
    test_keys = _select_test_device_groups(
        device_groups, total_profiling=len(profiling), seed=seed
    )

    for key in device_keys:
        assigned = "test" if key in test_keys else "train"
        for row in groups[key]:
            row["split"] = assigned

    # Build split records for all inventory rows.
    split_records: list[dict[str, Any]] = []
    for row in inventory:
        source = row["source"]
        if source == "attacks" or (
            source == "profiling" and row.get("binary_label") == "UNKNOWN"
        ):
            origin = "provided" if source == "attacks" else "unresolved"
            seed_value = None
        elif source == "profiling":
            origin = "assigned"
            seed_value = seed
        else:
            origin = "unresolved"
            seed_value = None

        # Ensure by_path reflects updated splits for profiling.
        updated = by_path[row["pcap_path"]]
        split_records.append(
            {
                "pcap_path": updated["pcap_path"],
                "source": updated["source"],
                "grouping_key": updated["grouping_key"],
                "split": updated["split"],
                "split_origin": origin,
                "strategy_version": strategy_version,
                "seed": seed_value,
            }
        )

    split_records.sort(key=lambda r: r["pcap_path"])
    inventory.sort(key=lambda r: r["pcap_path"])
    validate_inventory(inventory)

    # Integrity: device groups never cross splits.
    for key, group_rows in device_groups.items():
        splits = {r["split"] for r in group_rows}
        if len(splits) != 1:
            raise ValueError(f"device group {key} split across {splits}")

    return inventory, split_records


def profiling_split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiling = _profiling_rows(rows)
    train = sum(1 for r in profiling if r["split"] == "train")
    test = sum(1 for r in profiling if r["split"] == "test")
    return {
        "profiling_total": len(profiling),
        "profiling_train": train,
        "profiling_test": test,
        "test_fraction": (test / len(profiling)) if profiling else 0.0,
    }
