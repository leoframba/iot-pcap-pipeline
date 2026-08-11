"""V2M M3: MQTT structural parser fixture tests."""

from __future__ import annotations

from iot_pcap_pipeline.mqtt.parse import (
    MqttStatus,
    PKT_CONNECT,
    PKT_PUBLISH,
    classify_mqtt_payload,
    parse_mqtt_packet,
)


def _encode_remaining_length(value: int) -> bytes:
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


def _fixed(packet_type: int, flags: int, body: bytes) -> bytes:
    return bytes([(packet_type << 4) | (flags & 0x0F)]) + _encode_remaining_length(
        len(body)
    ) + body


def _valid_connect() -> bytes:
    # Protocol Name MQTT, level 4, connect flags 0, keep-alive 10, client id "c"
    body = _mqtt_string("MQTT") + bytes([4, 0x00, 0x00, 0x0A]) + _mqtt_string("c")
    return _fixed(PKT_CONNECT, 0x0, body)


def _valid_publish(qos: int = 0, topic: str = "sensors/temp", payload: bytes = b"22") -> bytes:
    flags = (qos & 0x03) << 1
    body = _mqtt_string(topic)
    if qos > 0:
        body += b"\x00\x01"  # packet id
    body += payload
    return _fixed(PKT_PUBLISH, flags, body)


def test_valid_connect() -> None:
    r = classify_mqtt_payload(_valid_connect())
    assert r.status == MqttStatus.VALID
    assert r.packet_type == PKT_CONNECT
    assert r.violations == frozenset()


def test_valid_publish_qos0() -> None:
    r = classify_mqtt_payload(_valid_publish(qos=0))
    assert r.status == MqttStatus.VALID
    assert r.packet_type == PKT_PUBLISH


def test_valid_publish_qos1() -> None:
    r = classify_mqtt_payload(_valid_publish(qos=1))
    assert r.status == MqttStatus.VALID


def test_publish_qos_bits_3_invalid() -> None:
    # QoS bits = 3 is illegal.
    body = _mqtt_string("a/b") + b"\x00\x01" + b"x"
    pkt = _fixed(PKT_PUBLISH, 0b0110, body)  # QoS=3, DUP=0, RETAIN=0
    r = classify_mqtt_payload(pkt)
    assert r.status == MqttStatus.INVALID
    assert "invalid_publish_qos" in r.violations


def test_publish_topic_contains_hash() -> None:
    r = classify_mqtt_payload(_valid_publish(topic="sensors/#"))
    assert r.status == MqttStatus.INVALID
    assert "publish_topic_contains_wildcard" in r.violations


def test_publish_topic_contains_plus() -> None:
    r = classify_mqtt_payload(_valid_publish(topic="sensors/+/temp"))
    assert r.status == MqttStatus.INVALID
    assert "publish_topic_contains_wildcard" in r.violations


def test_reserved_control_packet_flags_invalid() -> None:
    # CONNECT requires flags == 0.
    body = _mqtt_string("MQTT") + bytes([4, 0x00, 0x00, 0x0A]) + _mqtt_string("c")
    pkt = _fixed(PKT_CONNECT, 0x1, body)
    r = classify_mqtt_payload(pkt)
    assert r.status == MqttStatus.INVALID
    assert "invalid_fixed_header" in r.violations


def test_malformed_remaining_length_invalid() -> None:
    # Four continuation bytes (illegal Remaining Length encoding).
    pkt = bytes([PKT_PUBLISH << 4, 0xFF, 0xFF, 0xFF, 0xFF])
    r = classify_mqtt_payload(pkt)
    assert r.status == MqttStatus.INVALID
    assert "invalid_remaining_length" in r.violations


def test_truncated_remaining_length_incomplete() -> None:
    # Declares continuation but buffer ends.
    pkt = bytes([PKT_PUBLISH << 4, 0x80])
    r = classify_mqtt_payload(pkt)
    assert r.status == MqttStatus.INCOMPLETE
    assert r.violations == frozenset()


def test_truncated_mqtt_packet_incomplete() -> None:
    # Remaining Length says 10 bytes body; only 2 present.
    pkt = bytes([PKT_PUBLISH << 4, 0x0A, 0x00, 0x01])
    r = classify_mqtt_payload(pkt)
    assert r.status == MqttStatus.INCOMPLETE
    assert r.violations == frozenset()


def test_random_tcp_payload_not_mqtt() -> None:
    r = classify_mqtt_payload(b"\x00\xffGET / HTTP/1.1\r\n")
    assert r.status == MqttStatus.NOT_MQTT


def test_empty_tcp_payload_not_mqtt() -> None:
    r = classify_mqtt_payload(b"")
    assert r.status == MqttStatus.NOT_MQTT


def test_invalid_structure_vs_incomplete_distinction() -> None:
    invalid = classify_mqtt_payload(_valid_publish(topic="bad/#"))
    incomplete = parse_mqtt_packet(bytes([PKT_PUBLISH << 4, 0x05, 0x00]))
    not_mqtt = classify_mqtt_payload(b"\x00\x01\x02\x03")
    assert invalid.status == MqttStatus.INVALID
    assert incomplete.status == MqttStatus.INCOMPLETE
    assert not_mqtt.status == MqttStatus.NOT_MQTT
