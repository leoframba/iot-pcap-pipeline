"""Phase 1C.3b feature-dataset corpus orchestration tests."""

from __future__ import annotations

import csv
from pathlib import Path

import dpkt
import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.dataset import (
    EXPECTED_TRAIN_PCAP_COUNT,
    build_feature_dataset,
    build_one_pcap_job,
    logical_manifest_fingerprint,
    schedule_largest_first,
    select_train_rows,
    write_build_manifest,
)
from iot_pcap_pipeline.features.schema import write_feature_schema
from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> Path:
    columns = [
        "pcap_path",
        "filename",
        "dataset_scope",
        "source",
        "split",
        "binary_label",
        "attack_family",
        "attack_type",
        "profiling_type",
        "profiling_variant",
        "device",
        "capture_session",
        "file_size",
        "grouping_key",
        "unresolved_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _make_train_pcap(
    root: Path,
    rel: str,
    *,
    n_packets: int,
    binary_label: str = "BENIGN",
    wrong_size: bool | None = None,
) -> dict[str, str]:
    abs_path = root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        (1.0 + 0.01 * i, eth_ip_tcp(flags=dpkt.tcp.TH_ACK)) for i in range(n_packets)
    ]
    write_pcap(abs_path, frames)
    size = abs_path.stat().st_size
    manifest_size = size + 1 if wrong_size else size
    return {
        "pcap_path": rel,
        "filename": Path(rel).name,
        "dataset_scope": "wifi_mqtt",
        "source": "attacks",
        "split": "train",
        "binary_label": binary_label,
        "attack_family": "",
        "attack_type": "",
        "profiling_type": "",
        "profiling_variant": "",
        "device": "",
        "capture_session": "",
        "file_size": str(manifest_size),
        "grouping_key": "test",
        "unresolved_reason": "",
    }


def test_schedule_largest_first() -> None:
    rows = [
        {"pcap_path": "a.pcap", "file_size": "10"},
        {"pcap_path": "b.pcap", "file_size": "100"},
        {"pcap_path": "c.pcap", "file_size": "50"},
    ]
    ordered = schedule_largest_first(rows)
    assert [r["pcap_path"] for r in ordered] == ["b.pcap", "c.pcap", "a.pcap"]


def test_select_train_expects_85() -> None:
    inv = PROJECT_ROOT / "data" / "manifests" / "pcap_inventory.csv"
    if not inv.is_file():
        pytest.skip("inventory not present")
    rows = select_train_rows(inv, require_expected_count=True)
    assert len(rows) == EXPECTED_TRAIN_PCAP_COUNT


def test_size_mismatch_fails_pcap_only(tmp_path: Path) -> None:
    ok_rel = "data/raw/ok.pcap"
    bad_rel = "data/raw/bad.pcap"
    rows = [
        _make_train_pcap(tmp_path, ok_rel, n_packets=40, binary_label="BENIGN"),
        _make_train_pcap(
            tmp_path,
            bad_rel,
            n_packets=40,
            binary_label="ATTACK",
            wrong_size=True,
        ),
    ]
    inv = _write_inventory(tmp_path / "inv.csv", rows)
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    result = build_feature_dataset(
        inventory_path=inv,
        output_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / ".work",
        manifest_path=tmp_path / "build_manifest.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=1,
        resume=False,
        pcap_paths=[ok_rel, bad_rel],
    )
    assert result.ok_count == 1
    assert result.failed_count == 1
    assert result.results_by_path[ok_rel].status == "ok"
    assert result.results_by_path[bad_rel].status == "size_mismatch"
    # Successful shard must still exist after sibling failure.
    assert (tmp_path / "train" / f"{result.results_by_path[ok_rel].pcap_id}.parquet").is_file()


def test_workers_one_and_two_same_logical_result(tmp_path: Path) -> None:
    rels = [
        "data/raw/small_a.pcap",
        "data/raw/mid_b.pcap",
        "data/raw/tiny_c.pcap",
    ]
    rows = [
        _make_train_pcap(tmp_path, rels[0], n_packets=40, binary_label="BENIGN"),
        _make_train_pcap(tmp_path, rels[1], n_packets=80, binary_label="ATTACK"),
        _make_train_pcap(tmp_path, rels[2], n_packets=30, binary_label="BENIGN"),
    ]
    inv = _write_inventory(tmp_path / "inv.csv", rows)
    schema = write_feature_schema(tmp_path / "feature_schema.json")

    common = {
        "inventory_path": inv,
        "schema_path": schema,
        "project_root": tmp_path,
        "resume": False,
        "pcap_paths": rels,
    }
    one = build_feature_dataset(
        **common,
        output_dir=tmp_path / "out1",
        checkpoint_dir=tmp_path / "ck1",
        manifest_path=tmp_path / "m1.csv",
        workers=1,
    )
    two = build_feature_dataset(
        **common,
        output_dir=tmp_path / "out2",
        checkpoint_dir=tmp_path / "ck2",
        manifest_path=tmp_path / "m2.csv",
        workers=2,
    )
    assert one.ok_count == two.ok_count == 3
    assert logical_manifest_fingerprint(one.rows) == logical_manifest_fingerprint(
        two.rows
    )
    # Per-PCAP row counts and schema hashes must match.
    for rel in rels:
        assert (
            one.results_by_path[rel].output_row_count
            == two.results_by_path[rel].output_row_count
        )
        assert (
            one.results_by_path[rel].feature_schema_sha256
            == two.results_by_path[rel].feature_schema_sha256
        )


def test_worker_crash_does_not_erase_completed_shard(tmp_path: Path) -> None:
    ok_rel = "data/raw/keep.pcap"
    rows = [_make_train_pcap(tmp_path, ok_rel, n_packets=40)]
    inv = _write_inventory(tmp_path / "inv.csv", rows)
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    out_dir = tmp_path / "train"
    ckpt_dir = tmp_path / ".work"
    first = build_feature_dataset(
        inventory_path=inv,
        output_dir=out_dir,
        checkpoint_dir=ckpt_dir,
        manifest_path=tmp_path / "m.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=1,
        resume=False,
        pcap_paths=[ok_rel],
    )
    shard = out_dir / f"{first.results_by_path[ok_rel].pcap_id}.parquet"
    assert shard.is_file()
    before = shard.read_bytes()

    # Simulate an isolated crash result for a second PCAP while keep.pcap stays.
    crash = build_one_pcap_job(
        {
            "project_root": str(tmp_path),
            "pcap_path": "data/raw/missing.pcap",
            "binary_label": "ATTACK",
            "manifest_file_size": 123,
            "output_dir": str(out_dir),
            "checkpoint_dir": str(ckpt_dir),
            "schema_path": str(schema),
            "resume": False,
            "buffer_rows": 64,
            "split": "train",
        }
    )
    assert crash.status == "error"
    assert shard.read_bytes() == before


def test_build_manifest_sorted(tmp_path: Path) -> None:
    from iot_pcap_pipeline.features.dataset import PcapJobResult

    results = [
        PcapJobResult(
            pcap_path="z.pcap",
            pcap_id="z-1",
            binary_label="BENIGN",
            status="ok",
            output_row_count=1,
        ),
        PcapJobResult(
            pcap_path="a.pcap",
            pcap_id="a-1",
            binary_label="ATTACK",
            status="ok",
            output_row_count=2,
        ),
    ]
    path = write_build_manifest(tmp_path / "build_manifest.csv", results)
    with path.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert [r["pcap_path"] for r in loaded] == ["a.pcap", "z.pcap"]


def test_reject_non_train_split(tmp_path: Path) -> None:
    with pytest.raises(FeatureExtractionError, match="only supports --split train"):
        build_feature_dataset(
            split="test",  # type: ignore[arg-type]
            inventory_path=tmp_path / "missing.csv",
            project_root=tmp_path,
        )


def test_concurrent_workers_share_one_schema_hash(tmp_path: Path) -> None:
    """Several workers must hash the same parent-written schema (no truncate race)."""
    import json

    from iot_pcap_pipeline.features.parquet import feature_schema_sha256

    rels = [f"data/raw/w{i}.pcap" for i in range(6)]
    rows = [
        _make_train_pcap(
            tmp_path,
            rel,
            n_packets=40 + 5 * i,
            binary_label="BENIGN" if i % 2 == 0 else "ATTACK",
        )
        for i, rel in enumerate(rels)
    ]
    inv = _write_inventory(tmp_path / "inv.csv", rows)
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    expected_hash = feature_schema_sha256(schema)

    first = build_feature_dataset(
        inventory_path=inv,
        output_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / ".work",
        manifest_path=tmp_path / "build_manifest.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=4,
        resume=False,
        pcap_paths=rels,
    )
    assert first.ok_count == 6
    hashes = {r.feature_schema_sha256 for r in first.results_by_path.values()}
    assert hashes == {expected_hash}
    assert expected_hash != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    payload = json.loads(schema.read_text(encoding="utf-8"))
    assert payload.get("feature_strategy_version") == "phase1c2_v1"
    assert schema.stat().st_size > 0

    second = build_feature_dataset(
        inventory_path=inv,
        output_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / ".work",
        manifest_path=tmp_path / "build_manifest2.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=4,
        resume=True,
        pcap_paths=rels,
    )
    assert second.ok_count == 6
    assert second.resumed_count == 6
    assert {
        r.feature_schema_sha256 for r in second.results_by_path.values()
    } == {expected_hash}
