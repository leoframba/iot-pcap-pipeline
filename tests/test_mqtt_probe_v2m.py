"""V2M M4 unit tests for MQTT FIT probe selection (no full corpus run)."""

from __future__ import annotations

from pathlib import Path

import dpkt
import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.mqtt_v2 import extract_mqtt_structural_features
from iot_pcap_pipeline.features.mqtt_v2_probe import (
    PROBE_GROUP_MQTT_MALFORMED,
    PROBE_GROUP_PROFILING_BENIGN,
    PROBE_GROUP_PUBLISHER_BENIGN,
    load_mqtt_probe_targets,
    probe_group_for_row,
    run_mqtt_fit_probe,
)
from iot_pcap_pipeline.mqtt.parse import PKT_PUBLISH
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB, decode_frame
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.window import PacketWindow


def _encode_rl(value: int) -> bytes:
    out = bytearray()
    while True:
        encoded = value % 128
        value //= 128
        if value > 0:
            encoded |= 0x80
        out.append(encoded)
        if value == 0:
            break
    return bytes(out)


def _mqtt_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return len(raw).to_bytes(2, "big") + raw


def _publish(topic: str = "a/b") -> bytes:
    body = _mqtt_string(topic) + b"x"
    return bytes([PKT_PUBLISH << 4]) + _encode_rl(len(body)) + body


def test_probe_group_mapping() -> None:
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "attack_type": "MQTT_Malformed_Data",
                "binary_label": "ATTACK",
            }
        )
        == PROBE_GROUP_MQTT_MALFORMED
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "binary_label": "BENIGN",
                "benign_category": "publisher_benign",
                "group_kind": "publisher_benign",
                "pcap_path": "data/raw/.../Benign_train.pcap",
                "pcap_id": "Benign_train-x",
            }
        )
        == PROBE_GROUP_PUBLISHER_BENIGN
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "binary_label": "BENIGN",
                "benign_category": "profiling_active",
                "pcap_path": "data/raw/.../Broker/ActiveBroker.pcap",
                "pcap_id": "ActiveBroker-x",
            }
        )
        == PROBE_GROUP_PROFILING_BENIGN
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "binary_label": "BENIGN",
                "pcap_path": "data/raw/.../Blink_Camera_LAN_MIC.pcap",
                "pcap_id": "Blink-x",
            }
        )
        is None
    )


@pytest.mark.corpus
def test_load_mqtt_probe_targets_fit_only() -> None:
    targets = load_mqtt_probe_targets()
    assert any(t["probe_group"] == PROBE_GROUP_MQTT_MALFORMED for t in targets)
    assert any(t["probe_group"] == PROBE_GROUP_PUBLISHER_BENIGN for t in targets)
    assert any(t["probe_group"] == PROBE_GROUP_PROFILING_BENIGN for t in targets)
    assert sum(1 for t in targets if t["probe_group"] == PROBE_GROUP_MQTT_MALFORMED) == 1
    for t in targets:
        assert t["modeling_split"] == "fit"
        assert "/test/" not in t["pcap_path"].lower()


def test_extract_mqtt_structural_features_counts_invalid(tmp_path: Path) -> None:
    bad = _publish(topic="a/#")
    good = _publish(topic="a/b")
    bufs = [
        eth_ip_tcp(flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK, dport=1883, data=bad),
        eth_ip_tcp(flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK, dport=1883, data=good),
    ]
    while len(bufs) < WINDOW_SIZE:
        bufs.append(eth_ip_tcp(flags=dpkt.tcp.TH_SYN))
    packets = tuple(
        decode_frame(b, packet_index=i, timestamp=1.0 + 0.01 * i, linktype=DLT_EN10MB)
        for i, b in enumerate(bufs)
    )
    window = PacketWindow(
        segment_index=0,
        window_index=0,
        packet_index_start=0,
        packet_index_end=WINDOW_SIZE - 1,
        packets=packets,
    )
    feats = extract_mqtt_structural_features(window)
    assert feats.mqtt_control_packet_count == 2
    assert feats.mqtt_frame_count == 2
    assert feats.mqtt_invalid_count == 1
    assert feats.mqtt_valid_count == 1
    assert feats.mqtt_publish_wildcard_topic_count == 1
    assert feats.mqtt_invalid_ratio == 0.5
    assert feats.mqtt_frame_ratio == 2 / 25


def test_run_mqtt_fit_probe_smoke(tmp_path: Path) -> None:
    attack = tmp_path / "MQTT-Malformed_Data_train.pcap"
    benign = tmp_path / "Benign_train.pcap"
    write_pcap(
        attack,
        [
            (
                1.0 + 0.01 * i,
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=1883,
                    data=_publish("x/#"),
                ),
            )
            for i in range(50)
        ],
    )
    write_pcap(
        benign,
        [
            (
                1.0 + 0.01 * i,
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=1883,
                    data=_publish("ok/topic"),
                ),
            )
            for i in range(50)
        ],
    )
    manifest = tmp_path / "modeling_split_manifest.csv"
    manifest.write_text(
        "pcap_path,pcap_id,modeling_group_key,modeling_split,binary_label,"
        "attack_family,attack_type,profiling_type,device,window_count,"
        "feature_parquet_path,selection_reason,benign_category,group_kind\n"
        f"{attack.name},mal1,MQTT|MQTT_Malformed_Data,fit,ATTACK,MQTT,MQTT_Malformed_Data,"
        ",,2,,,,\n"
        f"{benign.name},ben1,benign|publisher|Benign_train,fit,BENIGN,,,,,2,,,"
        "publisher_benign,publisher_benign\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    payload = run_mqtt_fit_probe(
        split_manifest_path=manifest,
        output_dir=out,
        project_root=tmp_path,
        max_windows_per_pcap=2,
    )
    assert payload["status"] == "complete"
    assert payload["data_access"]["v1_final_test_access"] is False
    assert (out / "mqtt_feature_summary.csv").is_file()
    assert (out / "mqtt_feature_by_pcap.csv").is_file()
    assert (out / "mqtt_violation_summary.csv").is_file()
    assert (out / "mqtt_probe_complete.json").is_file()
