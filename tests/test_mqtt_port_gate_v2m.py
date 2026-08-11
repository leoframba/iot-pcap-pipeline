"""V2M M4b: port-gated MQTT extractor false-positive regressions."""

from __future__ import annotations

import dpkt
from pcap_synth import eth_ip_tcp

from iot_pcap_pipeline.features.mqtt_v2 import (
    MQTT_PLAINTEXT_PORTS,
    extract_mqtt_structural_features,
    is_plaintext_mqtt_candidate,
)
from iot_pcap_pipeline.mqtt.parse import PKT_PUBLISH, classify_mqtt_payload, MqttStatus
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


def _window_from_bufs(bufs: list[bytes]) -> PacketWindow:
    assert len(bufs) == WINDOW_SIZE
    packets = tuple(
        decode_frame(buf, packet_index=i, timestamp=1.0 + 0.01 * i, linktype=DLT_EN10MB)
        for i, buf in enumerate(bufs)
    )
    return PacketWindow(
        segment_index=0,
        window_index=0,
        packet_index_start=0,
        packet_index_end=WINDOW_SIZE - 1,
        packets=packets,
    )


def _pad(mqtt_bufs: list[bytes]) -> PacketWindow:
    bufs = list(mqtt_bufs)
    while len(bufs) < WINDOW_SIZE:
        bufs.append(eth_ip_tcp(flags=dpkt.tcp.TH_SYN, dport=9))
    return _window_from_bufs(bufs)


def test_plaintext_ports_pin_1883_only() -> None:
    assert MQTT_PLAINTEXT_PORTS == frozenset({1883})
    assert is_plaintext_mqtt_candidate(src_port=1234, dst_port=1883)
    assert is_plaintext_mqtt_candidate(src_port=1883, dst_port=5678)
    assert not is_plaintext_mqtt_candidate(src_port=1234, dst_port=8883)
    assert not is_plaintext_mqtt_candidate(src_port=443, dst_port=80)


def test_http_port_80_ignored_by_extractor() -> None:
    # 'G' = 0x47 → MQTT type nibble 4 (PUBACK-like) if ungated.
    http = b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n"
    assert ((http[0] >> 4) & 0x0F) == 4
    feats = extract_mqtt_structural_features(
        _pad(
            [
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=80,
                    data=http,
                )
            ]
        )
    )
    assert feats.mqtt_control_packet_count == 0
    assert feats.mqtt_frame_count == 0
    assert feats.mqtt_frame_ratio == 0.0


def test_tls_looking_payload_port_443_ignored() -> None:
    tls = b"\x16\x03\x01\x00\x01\x00"
    assert ((tls[0] >> 4) & 0x0F) == 1  # CONNECT-like nibble
    feats = extract_mqtt_structural_features(
        _pad(
            [
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=443,
                    data=tls,
                )
            ]
        )
    )
    assert feats.mqtt_control_packet_count == 0
    assert feats.mqtt_invalid_count == 0


def test_random_0x47_payload_not_mqtt_when_ungated_parser_alone() -> None:
    # Parser itself may attempt PUBACK-like parse; extractor must still ignore :80.
    payload = b"\x47" + b"\x00" * 20
    assert ((payload[0] >> 4) & 0x0F) == 4
    feats = extract_mqtt_structural_features(
        _pad(
            [
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=80,
                    data=payload,
                )
            ]
        )
    )
    assert feats.mqtt_frame_count == 0


def test_valid_mqtt_on_1883_counts_valid() -> None:
    good = _publish("sensors/temp")
    assert classify_mqtt_payload(good).status == MqttStatus.VALID
    feats = extract_mqtt_structural_features(
        _pad(
            [
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=1883,
                    data=good,
                )
            ]
        )
    )
    assert feats.mqtt_control_packet_count == 1
    assert feats.mqtt_frame_count == 1
    assert feats.mqtt_frame_ratio == 1 / 25
    assert feats.mqtt_valid_count == 1
    assert feats.mqtt_invalid_count == 0
    assert feats.mqtt_control_packets_per_frame == 1 / 25


def test_malformed_mqtt_on_1883_counts_invalid() -> None:
    bad = _publish("sensors/#")
    assert classify_mqtt_payload(bad).status == MqttStatus.INVALID
    feats = extract_mqtt_structural_features(
        _pad(
            [
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    sport=1883,
                    dport=50000,
                    data=bad,
                )
            ]
        )
    )
    assert feats.mqtt_invalid_count == 1
    assert feats.mqtt_publish_wildcard_topic_count == 1
    assert feats.mqtt_frame_ratio == 1 / 25


def test_mqtt_frame_ratio_bounded_when_many_controls_in_one_segment() -> None:
    # Two PUBLISH packets in one TCP segment → control count 2, frame count 1.
    two = _publish("a/b") + _publish("c/d")
    feats = extract_mqtt_structural_features(
        _pad(
            [
                eth_ip_tcp(
                    flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                    dport=1883,
                    data=two,
                )
            ]
        )
    )
    assert feats.mqtt_control_packet_count == 2
    assert feats.mqtt_control_packets_per_frame == 2 / 25
    assert feats.mqtt_frame_count == 1
    assert 0.0 <= feats.mqtt_frame_ratio <= 1.0
    assert feats.mqtt_frame_ratio == 1 / 25
