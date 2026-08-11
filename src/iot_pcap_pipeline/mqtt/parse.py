"""Minimal MQTT 3.1.1 structural parser (V2M feasibility; no reassembly).

Parses one TCP application payload (or consecutive MQTT packets within it).
Does not implement a full MQTT stack. Truncation / short reads are INCOMPLETE,
never INVALID.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MQTT_V2_STRATEGY_VERSION = "v2m1_mqtt_structural_probe"

# MQTT 3.1.1 control packet types
PKT_CONNECT = 1
PKT_CONNACK = 2
PKT_PUBLISH = 3
PKT_PUBACK = 4
PKT_PUBREC = 5
PKT_PUBREL = 6
PKT_PUBCOMP = 7
PKT_SUBSCRIBE = 8
PKT_SUBACK = 9
PKT_UNSUBSCRIBE = 10
PKT_UNSUBACK = 11
PKT_PINGREQ = 12
PKT_PINGRESP = 13
PKT_DISCONNECT = 14

_VIOLATION_NAMES = (
    "invalid_fixed_header",
    "invalid_remaining_length",
    "invalid_publish_qos",
    "publish_topic_contains_wildcard",
    "invalid_publish_topic",
    "invalid_connect_structure",
)


class MqttStatus(str, Enum):
    NOT_MQTT = "NOT_MQTT"
    INCOMPLETE = "INCOMPLETE"
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class MqttParseResult:
    status: MqttStatus
    packet_type: int | None = None
    flags: int | None = None
    remaining_length: int | None = None
    bytes_consumed: int = 0
    violations: frozenset[str] = frozenset()

    @property
    def is_mqtt_attempt(self) -> bool:
        return self.status != MqttStatus.NOT_MQTT


def _result(
    status: MqttStatus,
    *,
    packet_type: int | None = None,
    flags: int | None = None,
    remaining_length: int | None = None,
    bytes_consumed: int = 0,
    violations: set[str] | frozenset[str] | None = None,
) -> MqttParseResult:
    viol = frozenset(violations or ())
    unknown = viol - set(_VIOLATION_NAMES)
    if unknown:
        raise ValueError(f"unknown MQTT violation flags: {sorted(unknown)}")
    return MqttParseResult(
        status=status,
        packet_type=packet_type,
        flags=flags,
        remaining_length=remaining_length,
        bytes_consumed=bytes_consumed,
        violations=viol,
    )


def _decode_remaining_length(buf: bytes, offset: int) -> tuple[str, int, int, int]:
    """Decode MQTT Remaining Length.

    Returns ``(status, value, nbytes, header_end_offset)`` where status is
    ``ok``, ``incomplete``, or ``invalid``.
    """
    multiplier = 1
    value = 0
    for i in range(4):
        idx = offset + i
        if idx >= len(buf):
            return "incomplete", 0, i, offset
        encoded = buf[idx]
        value += (encoded & 0x7F) * multiplier
        if multiplier > 128 * 128 * 128:
            return "invalid", 0, i + 1, offset + i + 1
        if (encoded & 0x80) == 0:
            return "ok", value, i + 1, offset + i + 1
        multiplier *= 128
    # A 5th continuation bit would be required — malformed encoding.
    return "invalid", 0, 4, offset + 4


def _fixed_header_flags_ok(packet_type: int, flags: int) -> bool:
    if packet_type == PKT_PUBLISH:
        return True  # DUP/QoS/RETAIN validated separately (QoS==3 → invalid)
    if packet_type in {PKT_PUBREL, PKT_SUBSCRIBE, PKT_UNSUBSCRIBE}:
        return flags == 0b0010
    return flags == 0b0000


def _read_mqtt_string(buf: bytes, offset: int, end: int) -> tuple[str, bytes | None, int]:
    """Read a 2-byte-length MQTT UTF-8 string within ``buf[offset:end]``.

    Returns ``(status, value_or_none, new_offset)`` with status in
    ``ok|incomplete|invalid``.
    """
    if offset + 2 > end:
        return "incomplete", None, offset
    length = (buf[offset] << 8) | buf[offset + 1]
    start = offset + 2
    stop = start + length
    if stop > end:
        return "incomplete", None, offset
    raw = buf[start:stop]
    # MQTT strings must be valid UTF-8 for topic/protocol name checks.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "invalid", None, stop
    if "\u0000" in text:
        return "invalid", None, stop
    return "ok", text, stop


def _validate_connect(vh_payload: bytes) -> set[str]:
    violations: set[str] = set()
    # Need at least: proto name len(2)+name + level(1) + flags(1) + keepalive(2)
    st, proto, off = _read_mqtt_string(vh_payload, 0, len(vh_payload))
    if st == "incomplete":
        # Caller already ensured Remaining Length bytes are present; short
        # variable header inside a fully-framed packet is structural invalid.
        violations.add("invalid_connect_structure")
        return violations
    if st == "invalid" or proto not in {"MQTT", "MQIsdp"}:
        violations.add("invalid_connect_structure")
        return violations
    if off + 4 > len(vh_payload):
        violations.add("invalid_connect_structure")
        return violations
    # Protocol level / connect flags / keep-alive present — enough for structural OK.
    return violations


def _validate_publish(flags: int, vh_payload: bytes) -> set[str]:
    violations: set[str] = set()
    qos = (flags >> 1) & 0x03
    if qos == 3:
        violations.add("invalid_publish_qos")

    st, topic, off = _read_mqtt_string(vh_payload, 0, len(vh_payload))
    if st == "incomplete":
        violations.add("invalid_publish_topic")
        return violations
    if st == "invalid" or topic is None or topic == "":
        violations.add("invalid_publish_topic")
        return violations
    if "#" in topic or "+" in topic:
        violations.add("publish_topic_contains_wildcard")
    if qos > 0:
        # Packet identifier required (2 bytes) after topic.
        if off + 2 > len(vh_payload):
            # Fully framed remaining-length body still missing packet id → invalid.
            violations.add("invalid_publish_topic")
    return violations


def parse_mqtt_packet(buf: bytes, offset: int = 0) -> MqttParseResult:
    """Parse one MQTT control packet starting at ``offset`` in ``buf``."""
    if offset >= len(buf):
        return _result(MqttStatus.NOT_MQTT, bytes_consumed=0)

    first = buf[offset]
    packet_type = (first >> 4) & 0x0F
    flags = first & 0x0F

    if packet_type < 1 or packet_type > 14:
        return _result(MqttStatus.NOT_MQTT, bytes_consumed=0)

    rl_status, remaining, _rl_nbytes, header_end = _decode_remaining_length(
        buf, offset + 1
    )
    if rl_status == "incomplete":
        return _result(
            MqttStatus.INCOMPLETE,
            packet_type=packet_type,
            flags=flags,
            bytes_consumed=0,
        )
    if rl_status == "invalid":
        return _result(
            MqttStatus.INVALID,
            packet_type=packet_type,
            flags=flags,
            bytes_consumed=header_end - offset,
            violations={"invalid_remaining_length"},
        )

    packet_end = header_end + remaining
    if packet_end > len(buf):
        return _result(
            MqttStatus.INCOMPLETE,
            packet_type=packet_type,
            flags=flags,
            remaining_length=remaining,
            bytes_consumed=0,
        )

    violations: set[str] = set()
    if not _fixed_header_flags_ok(packet_type, flags):
        violations.add("invalid_fixed_header")

    body = buf[header_end:packet_end]
    if packet_type == PKT_CONNECT:
        violations |= _validate_connect(body)
    elif packet_type == PKT_PUBLISH:
        violations |= _validate_publish(flags, body)

    consumed = packet_end - offset
    if violations:
        return _result(
            MqttStatus.INVALID,
            packet_type=packet_type,
            flags=flags,
            remaining_length=remaining,
            bytes_consumed=consumed,
            violations=violations,
        )
    return _result(
        MqttStatus.VALID,
        packet_type=packet_type,
        flags=flags,
        remaining_length=remaining,
        bytes_consumed=consumed,
    )


def iter_mqtt_packets(payload: bytes) -> list[MqttParseResult]:
    """Parse consecutive MQTT packets in one TCP payload (no stream reassembly)."""
    if not payload:
        return [_result(MqttStatus.NOT_MQTT, bytes_consumed=0)]

    results: list[MqttParseResult] = []
    offset = 0
    while offset < len(payload):
        result = parse_mqtt_packet(payload, offset)
        results.append(result)
        if result.status == MqttStatus.NOT_MQTT:
            break
        if result.status == MqttStatus.INCOMPLETE:
            break
        if result.bytes_consumed <= 0:
            break
        offset += result.bytes_consumed
        # After an INVALID framed packet we still advance (bytes_consumed set)
        # so subsequent packets in the same segment can be counted.
    return results


def classify_mqtt_payload(payload: bytes) -> MqttParseResult:
    """Classify a TCP payload by its first MQTT parse result."""
    results = iter_mqtt_packets(payload)
    return results[0]


__all__ = [
    "MQTT_V2_STRATEGY_VERSION",
    "MqttParseResult",
    "MqttStatus",
    "PKT_CONNECT",
    "PKT_PUBLISH",
    "classify_mqtt_payload",
    "iter_mqtt_packets",
    "parse_mqtt_packet",
]
