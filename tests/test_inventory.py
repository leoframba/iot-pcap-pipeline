"""Tests for PCAP inventory classification."""

from __future__ import annotations

from pathlib import Path

from iot_pcap_pipeline.dataset.inventory import build_inventory, classify_pcap
from iot_pcap_pipeline.dataset.taxonomy import (
    classify_attack_stem,
    resolve_device_alias,
)
from iot_pcap_pipeline.paths import DATASET_SCOPE


def test_discover_and_scope(synthetic_raw: Path, tmp_path: Path) -> None:
    rows = build_inventory(synthetic_raw, project_root=tmp_path)
    assert rows
    assert all(r["dataset_scope"] == DATASET_SCOPE for r in rows)
    assert all(r["pcap_path"].startswith("WiFI_and_MQTT/") for r in rows)
    assert len({r["pcap_path"] for r in rows}) == len(rows)


def test_publisher_split_and_benign(synthetic_raw: Path, tmp_path: Path) -> None:
    rows = {r["filename"]: r for r in build_inventory(synthetic_raw, project_root=tmp_path)}

    benign_train = rows["Benign_train.pcap"]
    assert benign_train["binary_label"] == "BENIGN"
    assert benign_train["split"] == "train"
    assert benign_train["attack_family"] is None
    assert benign_train["source"] == "attacks"

    benign_test = rows["Benign_test.pcap"]
    assert benign_test["binary_label"] == "BENIGN"
    assert benign_test["split"] == "test"

    udp = rows["TCP_IP-DDoS-UDP1_train.pcap"]
    assert udp["binary_label"] == "ATTACK"
    assert udp["split"] == "train"
    assert udp["attack_family"] == "DDoS"
    assert udp["attack_type"] == "DDoS_UDP"
    assert udp["capture_session"] == "DDoS_UDP_1"
    assert udp["file_size"] == 100


def test_attack_family_type_normalization() -> None:
    a = classify_attack_stem("TCP_IP-DDoS-UDP3_train")
    b = classify_attack_stem("TCP_IP-DDoS-UDP_test")
    assert a is not None and b is not None
    assert a.attack_type == b.attack_type == "DDoS_UDP"
    assert a.family == b.family == "DDoS"
    assert a.capture_session == "DDoS_UDP_3"
    assert b.capture_session == "DDoS_UDP"

    recon = classify_attack_stem("Recon-Port_Scan_train")
    assert recon is not None
    assert recon.family == "Recon"
    assert recon.attack_type == "Port_Scan"


def test_contradictory_split_unknown(synthetic_raw: Path, tmp_path: Path) -> None:
    rows = {r["filename"]: r for r in build_inventory(synthetic_raw, project_root=tmp_path)}
    bad = rows["Recon-OS_Scan_test.pcap"]
    assert bad["binary_label"] == "UNKNOWN"
    assert "contradictory split" in (bad["unresolved_reason"] or "")


def test_profiling_variants_and_aliases(synthetic_raw: Path, tmp_path: Path) -> None:
    rows = {r["filename"]: r for r in build_inventory(synthetic_raw, project_root=tmp_path)}

    idle = rows["Idle.pcap"]
    assert idle["profiling_type"] == "idle"
    assert idle["profiling_variant"] is None
    assert idle["binary_label"] == "BENIGN"
    assert idle["device"] is None

    active = rows["Active.pcap"]
    assert active["profiling_type"] == "active"
    assert active["profiling_variant"] == "standard"

    broker = rows["ActiveBroker.pcap"]
    assert broker["profiling_type"] == "active"
    assert broker["profiling_variant"] == "active_broker"

    blink_power = rows["Blink_Mini_Camera_Power.pcap"]
    assert blink_power["device"] == "Blink_Camera"
    assert blink_power["grouping_key"] == "profiling:device:Blink_Camera"

    blink_int = rows["Blink_Camera_LAN_MIC.pcap"]
    assert blink_int["device"] == "Blink_Camera"
    assert blink_int["grouping_key"] == blink_power["grouping_key"]

    sense = rows["SenseUBaby_Power.pcap"]
    assert sense["device"] == "SenseU"

    precording = rows["M1T_Camera_LAN_PRECORDING.pcap"]
    assert precording["capture_session"] == "M1T_Camera_LAN_PRECORDING"
    assert precording["profiling_type"] == "interaction"


def test_unknown_device_not_fuzzy(synthetic_raw: Path, tmp_path: Path) -> None:
    assert resolve_device_alias("Blink_Mini_Camera") == "Blink_Camera"
    assert resolve_device_alias("BlinkMini") is None

    rows = {r["filename"]: r for r in build_inventory(synthetic_raw, project_root=tmp_path)}
    unknown = rows["UnknownGadget_Power.pcap"]
    assert unknown["binary_label"] == "UNKNOWN"
    assert "unknown power device alias" in (unknown["unresolved_reason"] or "")


def test_outside_tree_unknown(tmp_path: Path) -> None:
    lone = tmp_path / "other" / "x.pcap"
    lone.parent.mkdir(parents=True)
    lone.write_bytes(b"abc")
    row = classify_pcap(lone, project_root=tmp_path)
    assert row["binary_label"] == "UNKNOWN"
    assert row["source"] == "unknown"
