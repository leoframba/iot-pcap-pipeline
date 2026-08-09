"""Tests for parallel audit execution and checkpoint resume."""

from __future__ import annotations

import csv
from pathlib import Path

from pcap_synth import eth_ip_tcp, eth_lldp, write_pcap

from iot_pcap_pipeline.audit.checkpoints import CheckpointStore, checkpoint_id
from iot_pcap_pipeline.audit.live_progress import LiveProgressStore
from iot_pcap_pipeline.audit.scan import audit_corpus
from iot_pcap_pipeline.audit.worker import (
    AuditPolicy,
    scan_one_pcap_result,
    worker_crash_result,
)
from iot_pcap_pipeline.paths import to_repo_relative


def _write_inventory(path: Path, rows: list[dict]) -> None:
    cols = [
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
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _write_split(path: Path, rows: list[dict]) -> None:
    cols = [
        "pcap_path",
        "source",
        "grouping_key",
        "split",
        "split_origin",
        "strategy_version",
        "seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _build_corpus(tmp_path: Path, *, n: int = 4) -> tuple[Path, Path, Path, Path]:
    raw = tmp_path / "raw"
    project_root = tmp_path
    inv_rows = []
    split_rows = []
    # Vary sizes so largest-first scheduling is exercised.
    sizes = [3, 1, 8, 2, 5][:n]
    for i, packet_n in enumerate(sizes):
        packets = [(float(j), eth_ip_tcp() if j % 2 == 0 else eth_lldp()) for j in range(packet_n)]
        path = write_pcap(
            raw / "attacks" / "pcap" / "train" / f"A{i}_train.pcap",
            packets,
        )
        rel = to_repo_relative(path, project_root)
        inv_rows.append(
            {
                "pcap_path": rel,
                "filename": path.name,
                "dataset_scope": "wifi_mqtt",
                "source": "attacks",
                "split": "train",
                "binary_label": "ATTACK",
                "attack_family": "DDoS",
                "attack_type": "DDoS_UDP",
                "file_size": path.stat().st_size,
                "grouping_key": f"attacks:A{i}",
                "capture_session": f"A{i}",
            }
        )
        split_rows.append(
            {
                "pcap_path": rel,
                "source": "attacks",
                "grouping_key": f"attacks:A{i}",
                "split": "train",
                "split_origin": "provided",
                "strategy_version": "phase1a_v1",
            }
        )
    manifests = tmp_path / "manifests"
    inv = manifests / "pcap_inventory.csv"
    split = manifests / "dataset_split.csv"
    _write_inventory(inv, inv_rows)
    _write_split(split, split_rows)
    return project_root, raw, inv, split


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_workers_one_equals_workers_two(tmp_path: Path) -> None:
    project_root, raw, inv, split = _build_corpus(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    r1 = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out1,
        project_root=project_root,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    r2 = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out2,
        project_root=project_root,
        workers=2,
        resume=False,
        clear_checkpoints=True,
    )
    assert r1.hard_fail is False
    assert r2.hard_fail is False
    assert _read(out1 / "pcap_integrity.csv") == _read(out2 / "pcap_integrity.csv")
    assert _read(out1 / "training_characterization.csv") == _read(
        out2 / "training_characterization.csv"
    )
    assert _read(out1 / "audit_issues.csv") == _read(out2 / "audit_issues.csv")


def test_checkpoint_reuse_and_invalidation(tmp_path: Path) -> None:
    project_root, raw, inv, split = _build_corpus(tmp_path, n=2)
    out = tmp_path / "audit"
    first = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out,
        project_root=project_root,
        workers=1,
        resume=True,
        clear_checkpoints=True,
    )
    assert first.scanned_files == 2
    assert first.checkpoint_hits == 0

    second = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out,
        project_root=project_root,
        workers=1,
        resume=True,
        clear_checkpoints=False,
    )
    assert second.scanned_files == 0
    assert second.checkpoint_hits == 2
    assert _read(out / "pcap_integrity.csv") == _read(first.integrity_path)

    # Invalidate by changing IP cap policy.
    third = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out,
        project_root=project_root,
        workers=1,
        resume=True,
        ip_cardinality_cap=10,
        clear_checkpoints=False,
    )
    assert third.scanned_files == 2
    assert third.checkpoint_hits == 0


def test_tmp_checkpoint_not_valid(tmp_path: Path) -> None:
    project_root, raw, inv, split = _build_corpus(tmp_path, n=1)
    out = tmp_path / "audit"
    audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out,
        project_root=project_root,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    store = CheckpointStore(out / ".work")
    rows = list(csv.DictReader(inv.open()))
    pcap_path = rows[0]["pcap_path"]
    ckpt = store.path_for(pcap_path)
    assert ckpt.is_file()
    # Replace with only a .tmp file.
    tmp = ckpt.with_suffix(".json.tmp")
    tmp.write_text(ckpt.read_text(encoding="utf-8"), encoding="utf-8")
    ckpt.unlink()
    policy = AuditPolicy()
    loaded = store.load_valid(
        pcap_path=pcap_path,
        policy=policy,
        manifest_file_size=int(rows[0]["file_size"]),
        disk_file_size=int(rows[0]["file_size"]),
        split="train",
    )
    assert loaded is None


def test_worker_crash_result_not_checkpointed(tmp_path: Path) -> None:
    _project_root, _raw, inv, _split = _build_corpus(tmp_path, n=1)
    rows = list(csv.DictReader(inv.open()))
    result = worker_crash_result(rows[0], detail="boom")
    assert "file:worker_crash" in result.hard_failures
    assert result.integrity_row["parse_success"] is False


def test_largest_first_does_not_change_csv_order(tmp_path: Path) -> None:
    project_root, raw, inv, split = _build_corpus(tmp_path, n=4)
    out = tmp_path / "audit"
    result = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=out,
        project_root=project_root,
        workers=2,
        resume=False,
        clear_checkpoints=True,
    )
    paths = [r["pcap_path"] for r in result.integrity_rows]
    assert paths == sorted(paths)


def test_scan_one_pcap_result_direct(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "raw" / "x.pcap", [(1.0, eth_ip_tcp())])
    rel = to_repo_relative(path, tmp_path)
    row = {
        "pcap_path": rel,
        "filename": path.name,
        "source": "attacks",
        "split": "train",
        "binary_label": "ATTACK",
        "attack_family": "DDoS",
        "attack_type": "DDoS_UDP",
        "file_size": path.stat().st_size,
        "capture_session": "x",
    }
    result = scan_one_pcap_result(row, str(tmp_path), AuditPolicy())
    assert result.integrity_row["parse_success"] is True
    assert result.packet_count == 1
    assert result.training_row is not None
    assert checkpoint_id(rel)


def test_live_progress_store_roundtrip(tmp_path: Path) -> None:
    store = LiveProgressStore(tmp_path / "in_progress")
    store.write(
        pcap_path="data/raw/demo.pcap",
        packets=1_250_000,
        elapsed_seconds=12.5,
        status="running",
        file_size=1_000_000,
    )
    items = store.read_all()
    assert len(items) == 1
    assert items[0].filename == "demo.pcap"
    assert items[0].packets == 1_250_000
    assert items[0].status == "running"
    store.clear("data/raw/demo.pcap")
    assert store.read_all() == []


def test_worker_writes_live_packet_progress(tmp_path: Path) -> None:
    frames = [(float(i), eth_ip_tcp()) for i in range(1, 6)]
    path = write_pcap(tmp_path / "raw" / "prog.pcap", frames)
    rel = to_repo_relative(path, tmp_path)
    progress_dir = tmp_path / "live"
    row = {
        "pcap_path": rel,
        "filename": path.name,
        "source": "attacks",
        "split": "test",
        "binary_label": "ATTACK",
        "attack_family": "DDoS",
        "attack_type": "DDoS_UDP",
        "file_size": path.stat().st_size,
        "capture_session": "prog",
    }
    seen: list[int] = []
    original_write = LiveProgressStore.write

    def tracking_write(self, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(int(kwargs["packets"]))
        return original_write(self, **kwargs)

    LiveProgressStore.write = tracking_write  # type: ignore[method-assign]
    try:
        result = scan_one_pcap_result(
            row,
            str(tmp_path),
            AuditPolicy(progress_dir=str(progress_dir), progress_every_packets=2),
        )
    finally:
        LiveProgressStore.write = original_write  # type: ignore[method-assign]

    assert result.packet_count == 5
    assert 0 in seen  # starting
    assert 2 in seen
    assert 4 in seen
    assert list(progress_dir.glob("*.progress.json")) == []

