"""Tests for deterministic profiling split assignment."""

from __future__ import annotations

from pathlib import Path

from iot_pcap_pipeline.dataset.build import build_manifests
from iot_pcap_pipeline.dataset.inventory import build_inventory
from iot_pcap_pipeline.dataset.split import assign_profiling_splits


def test_singletons_always_train(synthetic_raw: Path, tmp_path: Path) -> None:
    inventory = build_inventory(synthetic_raw, project_root=tmp_path)
    inventory, _ = assign_profiling_splits(inventory, seed=42)
    by_name = {r["filename"]: r for r in inventory}
    for name in ("Idle.pcap", "Active.pcap", "ActiveBroker.pcap"):
        assert by_name[name]["split"] == "train"


def test_device_groups_indivisible(synthetic_raw: Path, tmp_path: Path) -> None:
    inventory = build_inventory(synthetic_raw, project_root=tmp_path)
    inventory, _ = assign_profiling_splits(inventory, seed=42)

    groups: dict[str, set[str]] = {}
    for row in inventory:
        if row["source"] != "profiling" or row["binary_label"] == "UNKNOWN":
            continue
        key = row["grouping_key"]
        groups.setdefault(key, set()).add(row["split"])

    for key, splits in groups.items():
        assert len(splits) == 1, f"{key} split across {splits}"


def test_test_has_power_and_interaction(synthetic_raw: Path, tmp_path: Path) -> None:
    inventory = build_inventory(synthetic_raw, project_root=tmp_path)
    inventory, _ = assign_profiling_splits(inventory, seed=42)
    test_types = {
        r["profiling_type"]
        for r in inventory
        if r["source"] == "profiling"
        and r["binary_label"] == "BENIGN"
        and r["split"] == "test"
    }
    assert "power" in test_types
    assert "interaction" in test_types


def test_seed_reproducibility(synthetic_raw: Path, tmp_path: Path) -> None:
    inv1 = build_inventory(synthetic_raw, project_root=tmp_path)
    inv2 = build_inventory(synthetic_raw, project_root=tmp_path)
    out1, split1 = assign_profiling_splits(inv1, seed=42)
    out2, split2 = assign_profiling_splits(inv2, seed=42)
    assert [r["split"] for r in out1] == [r["split"] for r in out2]
    assert split1 == split2

    out3, _ = assign_profiling_splits(
        build_inventory(synthetic_raw, project_root=tmp_path), seed=99
    )
    # Different seed may or may not change assignment; paths/order still stable.
    assert [r["pcap_path"] for r in out3] == [r["pcap_path"] for r in out1]


def test_attack_splits_immutable(synthetic_raw: Path, tmp_path: Path) -> None:
    before = build_inventory(synthetic_raw, project_root=tmp_path)
    before_attacks = {
        r["pcap_path"]: (r["split"], r["binary_label"])
        for r in before
        if r["source"] == "attacks"
    }
    after, records = assign_profiling_splits(before, seed=42)
    after_attacks = {
        r["pcap_path"]: (r["split"], r["binary_label"])
        for r in after
        if r["source"] == "attacks"
    }
    assert before_attacks == after_attacks
    for rec in records:
        if rec["source"] == "attacks":
            assert rec["split_origin"] == "provided"
            assert rec["seed"] is None


def test_group_integrity_over_exact_fraction(synthetic_raw: Path, tmp_path: Path) -> None:
    """Accept non-exact 20% rather than splitting a device group."""
    inventory = build_inventory(synthetic_raw, project_root=tmp_path)
    inventory, _ = assign_profiling_splits(inventory, seed=42)
    profiling = [
        r
        for r in inventory
        if r["source"] == "profiling" and r["binary_label"] == "BENIGN"
    ]
    test_n = sum(1 for r in profiling if r["split"] == "test")
    # Device groups are size 3 (or 2 for M1T in fixture: power+1 interaction).
    # Never a partial group: test count must be sum of whole group sizes.
    assert test_n >= 1
    by_group: dict[str, list] = {}
    for row in profiling:
        by_group.setdefault(row["grouping_key"], []).append(row)
    for rows in by_group.values():
        assert len({r["split"] for r in rows}) == 1


def test_manifest_build_idempotent(synthetic_raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "manifests"
    first = build_manifests(
        raw_root=synthetic_raw,
        output_dir=out,
        seed=42,
        project_root=tmp_path,
    )
    second = build_manifests(
        raw_root=synthetic_raw,
        output_dir=out,
        seed=42,
        project_root=tmp_path,
    )
    assert first["inventory_path"].read_text() == second["inventory_path"].read_text()
    assert first["split_path"].read_text() == second["split_path"].read_text()
    assert first["inventory"][0]["pcap_path"] <= first["inventory"][-1]["pcap_path"]
