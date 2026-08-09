"""Invariant checks for V1 feature vectors."""

from __future__ import annotations

import math

from iot_pcap_pipeline.features.extractor import FeatureVector
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.window import PacketWindow

L3_SUM_TOLERANCE = 1e-9


class FeatureInvariantError(ValueError):
    """Raised when a feature vector violates the V1 contract."""


def validate_window_and_features(
    window: PacketWindow,
    features: FeatureVector,
) -> None:
    """Raise FeatureInvariantError on any contract violation."""
    if len(window.packets) != WINDOW_SIZE:
        raise FeatureInvariantError(
            f"window must contain {WINDOW_SIZE} packets, got {len(window.packets)}"
        )
    if window.packet_index_end < window.packet_index_start:
        raise FeatureInvariantError("packet_index_end < packet_index_start")

    values = features.to_ordered_values()
    if len(values) != len(V1_FEATURE_NAMES):
        raise FeatureInvariantError("feature arity mismatch")

    for name, value in zip(V1_FEATURE_NAMES, values, strict=True):
        if not math.isfinite(value):
            raise FeatureInvariantError(f"{name} is non-finite: {value!r}")

    if features.window_span_seconds < 0:
        raise FeatureInvariantError("window_span_seconds < 0")
    for name in (
        "iat_mean_seconds",
        "iat_std_seconds",
        "iat_p50_seconds",
        "iat_p95_seconds",
    ):
        if getattr(features, name) < 0:
            raise FeatureInvariantError(f"{name} < 0")

    if features.frame_length_min <= 0:
        raise FeatureInvariantError("frame_length_min must be > 0")
    if features.frame_length_max < features.frame_length_min:
        raise FeatureInvariantError("frame_length_max < frame_length_min")
    if features.frame_length_std < 0:
        raise FeatureInvariantError("frame_length_std < 0")

    for name in (
        "ipv4_ratio",
        "ipv6_ratio",
        "arp_ratio",
        "llc_ratio",
        "other_protocol_ratio",
        "tcp_ratio",
        "udp_ratio",
        "icmp_ratio",
        "icmpv6_ratio",
        "igmp_ratio",
        "tcp_syn_ratio",
        "tcp_ack_ratio",
        "tcp_fin_ratio",
        "tcp_rst_ratio",
        "tcp_psh_ratio",
        "tcp_urg_ratio",
    ):
        value = getattr(features, name)
        if value < 0.0 or value > 1.0:
            raise FeatureInvariantError(f"{name} out of [0,1]: {value}")

    l3_sum = (
        features.ipv4_ratio
        + features.ipv6_ratio
        + features.arp_ratio
        + features.llc_ratio
        + features.other_protocol_ratio
    )
    if abs(l3_sum - 1.0) > L3_SUM_TOLERANCE:
        raise FeatureInvariantError(f"L3 ratios sum to {l3_sum}, expected 1.0")

    if not (0 <= features.unique_ip_count <= 50):
        raise FeatureInvariantError(
            f"unique_ip_count out of bounds: {features.unique_ip_count}"
        )
    if not (0 <= features.unique_port_count <= 50):
        raise FeatureInvariantError(
            f"unique_port_count out of bounds: {features.unique_port_count}"
        )
