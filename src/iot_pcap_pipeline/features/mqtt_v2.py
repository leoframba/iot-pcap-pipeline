"""V2M MQTT structural window features (experiment; not part of frozen V1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from iot_pcap_pipeline.mqtt.parse import (
    MQTT_V2_STRATEGY_VERSION,
    MqttStatus,
    PKT_CONNECT,
    PKT_PUBLISH,
    iter_mqtt_packets,
)
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.window import PacketWindow

MQTT_V2_FEATURE_NAMES: tuple[str, ...] = (
    "mqtt_packet_count",
    "mqtt_packet_ratio",
    "mqtt_valid_count",
    "mqtt_invalid_count",
    "mqtt_incomplete_count",
    "mqtt_invalid_ratio",
    "mqtt_incomplete_ratio",
    "mqtt_invalid_fixed_header_count",
    "mqtt_invalid_remaining_length_count",
    "mqtt_publish_count",
    "mqtt_invalid_publish_qos_count",
    "mqtt_publish_wildcard_topic_count",
    "mqtt_invalid_publish_topic_count",
    "mqtt_connect_count",
    "mqtt_invalid_connect_count",
)

assert len(MQTT_V2_FEATURE_NAMES) == 15


@dataclass(frozen=True)
class MqttStructuralFeatures:
    mqtt_packet_count: int
    mqtt_packet_ratio: float
    mqtt_valid_count: int
    mqtt_invalid_count: int
    mqtt_incomplete_count: int
    mqtt_invalid_ratio: float
    mqtt_incomplete_ratio: float
    mqtt_invalid_fixed_header_count: int
    mqtt_invalid_remaining_length_count: int
    mqtt_publish_count: int
    mqtt_invalid_publish_qos_count: int
    mqtt_publish_wildcard_topic_count: int
    mqtt_invalid_publish_topic_count: int
    mqtt_connect_count: int
    mqtt_invalid_connect_count: int

    def to_ordered_values(self) -> tuple[float, ...]:
        data = asdict(self)
        return tuple(float(data[name]) for name in MQTT_V2_FEATURE_NAMES)

    def to_feature_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in MQTT_V2_FEATURE_NAMES}


def _empty() -> MqttStructuralFeatures:
    return MqttStructuralFeatures(
        mqtt_packet_count=0,
        mqtt_packet_ratio=0.0,
        mqtt_valid_count=0,
        mqtt_invalid_count=0,
        mqtt_incomplete_count=0,
        mqtt_invalid_ratio=0.0,
        mqtt_incomplete_ratio=0.0,
        mqtt_invalid_fixed_header_count=0,
        mqtt_invalid_remaining_length_count=0,
        mqtt_publish_count=0,
        mqtt_invalid_publish_qos_count=0,
        mqtt_publish_wildcard_topic_count=0,
        mqtt_invalid_publish_topic_count=0,
        mqtt_connect_count=0,
        mqtt_invalid_connect_count=0,
    )


def extract_mqtt_structural_features(window: PacketWindow) -> MqttStructuralFeatures:
    """Extract V2M MQTT structural counts/ratios from one full 25-packet window."""
    packets = window.packets
    n = len(packets)
    if n != WINDOW_SIZE:
        raise ValueError(
            f"V2M MQTT windows must contain exactly {WINDOW_SIZE} packets, got {n}"
        )

    valid = invalid = incomplete = 0
    inv_fixed = inv_rl = 0
    publish = inv_pub_qos = pub_wild = inv_pub_topic = 0
    connect = inv_connect = 0
    mqtt_packets = 0

    for packet in packets:
        if not packet.is_tcp:
            continue
        payload = packet.tcp_payload
        if payload is None:
            continue
        for result in iter_mqtt_packets(payload):
            if result.status == MqttStatus.NOT_MQTT:
                continue
            mqtt_packets += 1
            if result.status == MqttStatus.VALID:
                valid += 1
            elif result.status == MqttStatus.INVALID:
                invalid += 1
            elif result.status == MqttStatus.INCOMPLETE:
                incomplete += 1

            viol = result.violations
            if "invalid_fixed_header" in viol:
                inv_fixed += 1
            if "invalid_remaining_length" in viol:
                inv_rl += 1
            if "invalid_publish_qos" in viol:
                inv_pub_qos += 1
            if "publish_topic_contains_wildcard" in viol:
                pub_wild += 1
            if "invalid_publish_topic" in viol:
                inv_pub_topic += 1
            if "invalid_connect_structure" in viol:
                inv_connect += 1

            if result.packet_type == PKT_PUBLISH:
                publish += 1
            if result.packet_type == PKT_CONNECT:
                connect += 1

    if mqtt_packets == 0:
        return _empty()

    return MqttStructuralFeatures(
        mqtt_packet_count=mqtt_packets,
        mqtt_packet_ratio=mqtt_packets / float(WINDOW_SIZE),
        mqtt_valid_count=valid,
        mqtt_invalid_count=invalid,
        mqtt_incomplete_count=incomplete,
        mqtt_invalid_ratio=invalid / mqtt_packets,
        mqtt_incomplete_ratio=incomplete / mqtt_packets,
        mqtt_invalid_fixed_header_count=inv_fixed,
        mqtt_invalid_remaining_length_count=inv_rl,
        mqtt_publish_count=publish,
        mqtt_invalid_publish_qos_count=inv_pub_qos,
        mqtt_publish_wildcard_topic_count=pub_wild,
        mqtt_invalid_publish_topic_count=inv_pub_topic,
        mqtt_connect_count=connect,
        mqtt_invalid_connect_count=inv_connect,
    )


__all__ = [
    "MQTT_V2_FEATURE_NAMES",
    "MQTT_V2_STRATEGY_VERSION",
    "MqttStructuralFeatures",
    "extract_mqtt_structural_features",
]
