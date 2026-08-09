"""Synthetic tests for Phase 1B.2 corpus audit orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

from pcap_synth import (
    eth_ieee8023_llc,
    eth_ip_tcp,
    eth_lldp,
    eth_unknown_ethertype,
    write_pcap,
)

from iot_pcap_pipeline.audit.issues import IssueCollector, normalize_packet_issue_code
from iot_pcap_pipeline.audit.policy import (
    ISSUE_MANIFEST_EXTRA,
    ISSUE_MANIFEST_SIZE_MISMATCH,
    ISSUE_OPEN_FAILURE,
    SEVERITY_HARD_FAILURE,
)
from iot_pcap_pipeline.audit.reconcile import reconcile_manifests
from iot_pcap_pipeline.audit.scan import audit_corpus
from iot_pcap_pipeline.paths import to_repo_relative
from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus


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


def _make_corpus(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    raw = tmp_path / "raw"
    train_dir = raw / "attacks" / "pcap" / "train"
    test_dir = raw / "attacks" / "pcap" / "test"
    profiling = raw / "profiling" / "PCAP" / "Idle"

    train_pcap = write_pcap(
        train_dir / "TCP_IP-DDoS-UDP1_train.pcap",
        [
            (1.0, eth_ip_tcp()),
            (1.0, eth_ip_tcp()),  # duplicate timestamp
            (2.0, eth_lldp()),
            (3.0, eth_ieee8023_llc()),
        ],
    )
    # Many malformed short frames for catastrophic threshold tests elsewhere.
    test_pcap = write_pcap(
        test_dir / "Benign_test.pcap",
        [
            (10.0, eth_ip_tcp()),
            (11.0, eth_unknown_ethertype()),
        ],
    )
    idle_pcap = write_pcap(
        profiling / "Idle.pcap",
        [(100.0, eth_ip_tcp()), (101.0, eth_ip_tcp())],
    )

    project_root = tmp_path
    inv_rows = []
    split_rows = []
    for path, source, split, label, family, ptype in [
        (
            train_pcap,
            "attacks",
            "train",
            "ATTACK",
            "DDoS",
            "",
        ),
        (
            test_pcap,
            "attacks",
            "test",
            "BENIGN",
            "",
            "",
        ),
        (
            idle_pcap,
            "profiling",
            "train",
            "BENIGN",
            "",
            "idle",
        ),
    ]:
        rel = to_repo_relative(path, project_root)
        inv_rows.append(
            {
                "pcap_path": rel,
                "filename": path.name,
                "dataset_scope": "wifi_mqtt",
                "source": source,
                "split": split,
                "binary_label": label,
                "attack_family": family,
                "attack_type": "DDoS_UDP" if family else "",
                "profiling_type": ptype,
                "profiling_variant": "",
                "device": "",
                "capture_session": path.stem,
                "file_size": path.stat().st_size,
                "grouping_key": f"{source}:{path.stem}",
                "unresolved_reason": "",
            }
        )
        split_rows.append(
            {
                "pcap_path": rel,
                "source": source,
                "grouping_key": f"{source}:{path.stem}",
                "split": split,
                "split_origin": "provided",
                "strategy_version": "phase1a_v1",
                "seed": "",
            }
        )

    manifests = tmp_path / "manifests"
    inv_path = manifests / "pcap_inventory.csv"
    split_path = manifests / "dataset_split.csv"
    _write_inventory(inv_path, inv_rows)
    _write_split(split_path, split_rows)
    return project_root, raw, inv_path, split_path


def test_normalize_issue_codes() -> None:
    lldp = PacketRecord(
        packet_index=0,
        timestamp=1.0,
        frame_length=60,
        linktype=1,
        parse_status=ParseStatus.UNSUPPORTED,
        parse_detail="unsupported ethertype: 0x88cc",
        protocol_name="lldp",
    )
    assert normalize_packet_issue_code(lldp) == "unsupported:lldp"
    partial = PacketRecord(
        packet_index=1,
        timestamp=1.0,
        frame_length=40,
        linktype=1,
        parse_status=ParseStatus.PARTIAL,
        parse_detail="tcp decode failed: NeedData",
        protocol_name="ipv4",
    )
    assert normalize_packet_issue_code(partial) == "partial:tcp_truncated"


def test_audit_corpus_happy_path(tmp_path: Path) -> None:
    project_root, raw, inv_path, split_path = _make_corpus(tmp_path)
    out = tmp_path / "audit"
    result = audit_corpus(
        inventory_path=inv_path,
        split_path=split_path,
        raw_root=raw,
        output_dir=out,
        project_root=project_root,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    assert result.hard_fail is False
    assert len(result.integrity_rows) == 3
    assert len(result.train_rows) == 2  # train attack + idle
    assert (out / "pcap_integrity.csv").exists()
    assert (out / "training_characterization.csv").exists()
    assert (out / "audit_issues.csv").exists()

    train_paths = {r["pcap_path"] for r in result.train_rows}
    assert all("/train/" in p or "Idle.pcap" in p for p in train_paths)
    assert not any("Benign_test" in p for p in train_paths)

    # LLDP and LLC observed; unsupported present as warning issues, not hard fail.
    codes = {r["issue_code"] for r in result.issue_rows}
    assert "unsupported:lldp" in codes or "file:unsupported_present" in codes


def test_issue_cap_per_code(tmp_path: Path) -> None:
    project_root = tmp_path
    raw = tmp_path / "raw"
    pcap = write_pcap(
        raw / "attacks" / "pcap" / "train" / "x_train.pcap",
        [(float(i), eth_lldp()) for i in range(12)],
    )
    rel = to_repo_relative(pcap, project_root)
    inv = tmp_path / "manifests" / "pcap_inventory.csv"
    split = tmp_path / "manifests" / "dataset_split.csv"
    _write_inventory(
        inv,
        [
            {
                "pcap_path": rel,
                "filename": pcap.name,
                "dataset_scope": "wifi_mqtt",
                "source": "attacks",
                "split": "train",
                "binary_label": "ATTACK",
                "attack_family": "DDoS",
                "attack_type": "DDoS_UDP",
                "file_size": pcap.stat().st_size,
                "grouping_key": "attacks:x",
                "capture_session": "x",
            }
        ],
    )
    _write_split(
        split,
        [
            {
                "pcap_path": rel,
                "source": "attacks",
                "grouping_key": "attacks:x",
                "split": "train",
                "split_origin": "provided",
                "strategy_version": "phase1a_v1",
            }
        ],
    )
    result = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=tmp_path / "audit",
        project_root=project_root,
        issue_cap_per_code=5,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    packet_lldp = [
        r
        for r in result.issue_rows
        if r["scope"] == "packet" and r["issue_code"] == "unsupported:lldp"
    ]
    assert len(packet_lldp) == 5


def test_open_failure_and_extra_pcap(tmp_path: Path) -> None:
    project_root, raw, inv_path, split_path = _make_corpus(tmp_path)
    # Extra PCAP on disk
    write_pcap(raw / "extra.pcap", [(1.0, eth_ip_tcp())])
    # Corrupt inventory path
    rows = list(csv.DictReader(inv_path.open()))
    rows[0]["pcap_path"] = "raw/missing.pcap"
    _write_inventory(inv_path, rows)
    split_rows = list(csv.DictReader(split_path.open()))
    split_rows[0]["pcap_path"] = "raw/missing.pcap"
    _write_split(split_path, split_rows)

    result = audit_corpus(
        inventory_path=inv_path,
        split_path=split_path,
        raw_root=raw,
        output_dir=tmp_path / "audit",
        project_root=project_root,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    assert result.hard_fail is True
    codes = {r["issue_code"] for r in result.issue_rows}
    assert ISSUE_MANIFEST_EXTRA in codes
    assert ISSUE_OPEN_FAILURE in codes or "manifest:missing_pcap" in codes


def test_file_size_mismatch_hard_fail(tmp_path: Path) -> None:
    project_root, raw, inv_path, split_path = _make_corpus(tmp_path)
    rows = list(csv.DictReader(inv_path.open()))
    rows[0]["file_size"] = str(int(rows[0]["file_size"]) + 1)
    _write_inventory(inv_path, rows)
    result = audit_corpus(
        inventory_path=inv_path,
        split_path=split_path,
        raw_root=raw,
        output_dir=tmp_path / "audit",
        project_root=project_root,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    assert result.hard_fail is True
    assert any(
        r["issue_code"] == ISSUE_MANIFEST_SIZE_MISMATCH for r in result.issue_rows
    )


def test_catastrophic_malformed_hard_fail(tmp_path: Path) -> None:
    project_root = tmp_path
    raw = tmp_path / "raw"
    # 9 short malformed + 1 good => 90% malformed
    packets = [(float(i), b"\x00" * 8) for i in range(9)] + [(9.0, eth_ip_tcp())]
    pcap = write_pcap(raw / "attacks" / "pcap" / "train" / "bad_train.pcap", packets)
    rel = to_repo_relative(pcap, project_root)
    inv = tmp_path / "manifests" / "pcap_inventory.csv"
    split = tmp_path / "manifests" / "dataset_split.csv"
    _write_inventory(
        inv,
        [
            {
                "pcap_path": rel,
                "filename": pcap.name,
                "dataset_scope": "wifi_mqtt",
                "source": "attacks",
                "split": "train",
                "binary_label": "ATTACK",
                "attack_family": "DDoS",
                "attack_type": "DDoS_UDP",
                "file_size": pcap.stat().st_size,
                "grouping_key": "attacks:bad",
                "capture_session": "bad",
            }
        ],
    )
    _write_split(
        split,
        [
            {
                "pcap_path": rel,
                "source": "attacks",
                "grouping_key": "attacks:bad",
                "split": "train",
                "split_origin": "provided",
                "strategy_version": "phase1a_v1",
            }
        ],
    )
    result = audit_corpus(
        inventory_path=inv,
        split_path=split,
        raw_root=raw,
        output_dir=tmp_path / "audit",
        project_root=project_root,
        malformed_catastrophic_rate=0.80,
        workers=1,
        resume=False,
        clear_checkpoints=True,
    )
    assert result.hard_fail is True
    assert any(
        r["issue_code"] == "file:malformed_catastrophic"
        and r["severity"] == SEVERITY_HARD_FAILURE
        for r in result.issue_rows
    )


def test_reconcile_detects_extra(tmp_path: Path) -> None:
    project_root, raw, inv_path, split_path = _make_corpus(tmp_path)
    write_pcap(raw / "orphan.pcap", [(1.0, eth_ip_tcp())])
    issues = IssueCollector()
    result = reconcile_manifests(
        inventory_path=inv_path,
        split_path=split_path,
        raw_root=raw,
        project_root=project_root,
        issues=issues,
    )
    assert result.hard_fail is True
    assert any(i.issue_code == ISSUE_MANIFEST_EXTRA for i in issues.issues)
