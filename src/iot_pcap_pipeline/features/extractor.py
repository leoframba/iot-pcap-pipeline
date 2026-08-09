"""PacketWindow → V1 FeatureVector extractor (Phase 1C.2).

Label-/split-/path-independent. Offline and inference must call this same
function for production parity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.window import PacketWindow


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100). Exact for small n."""
    if not values:
        raise ValueError("percentile requires a non-empty list")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    weight = rank - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: list[float]) -> float:
    """Population standard deviation (ddof=0)."""
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _l3_category(packet: Any) -> str:
    if packet.is_ipv4:
        return "ipv4"
    if packet.is_ipv6:
        return "ipv6"
    if packet.is_arp:
        return "arp"
    if packet.is_llc:
        return "llc"
    return "other"


@dataclass(frozen=True)
class FeatureVector:
    """Canonical V1 model input (27 numeric features)."""

    window_span_seconds: float
    iat_mean_seconds: float
    iat_std_seconds: float
    iat_p50_seconds: float
    iat_p95_seconds: float
    frame_length_mean: float
    frame_length_std: float
    frame_length_min: float
    frame_length_max: float
    ipv4_ratio: float
    ipv6_ratio: float
    arp_ratio: float
    llc_ratio: float
    other_protocol_ratio: float
    tcp_ratio: float
    udp_ratio: float
    icmp_ratio: float
    icmpv6_ratio: float
    igmp_ratio: float
    tcp_syn_ratio: float
    tcp_ack_ratio: float
    tcp_fin_ratio: float
    tcp_rst_ratio: float
    tcp_psh_ratio: float
    tcp_urg_ratio: float
    unique_ip_count: int
    unique_port_count: int

    def to_ordered_values(self) -> tuple[float, ...]:
        data = asdict(self)
        return tuple(float(data[name]) for name in V1_FEATURE_NAMES)

    def to_feature_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in V1_FEATURE_NAMES}


def extract_features(window: PacketWindow) -> FeatureVector:
    """Extract the frozen 27 V1 features from one full packet window.

    Does not accept labels, splits, paths, or other dataset metadata.
    """
    packets = window.packets
    n = len(packets)
    if n != WINDOW_SIZE:
        raise ValueError(
            f"V1 windows must contain exactly {WINDOW_SIZE} packets, got {n}"
        )

    timestamps = [p.timestamp for p in packets]
    span = max(timestamps) - min(timestamps)
    if span < 0:
        raise ValueError(f"window_span_seconds must be >= 0, got {span}")

    iats: list[float] = []
    for i in range(1, n):
        iats.append(max(timestamps[i] - timestamps[i - 1], 0.0))
    if len(iats) != WINDOW_SIZE - 1:
        raise ValueError("expected 24 IAT observations")

    lengths = [float(p.frame_length) for p in packets]

    ipv4 = ipv6 = arp = llc = other = 0
    tcp = udp = icmp = icmpv6 = igmp = 0
    syn = ack = fin = rst = psh = urg = 0
    ips: set[str] = set()
    ports: set[int] = set()

    for p in packets:
        cat = _l3_category(p)
        if cat == "ipv4":
            ipv4 += 1
        elif cat == "ipv6":
            ipv6 += 1
        elif cat == "arp":
            arp += 1
        elif cat == "llc":
            llc += 1
        else:
            other += 1

        if p.is_tcp:
            tcp += 1
            if p.tcp_flag_syn:
                syn += 1
            if p.tcp_flag_ack:
                ack += 1
            if p.tcp_flag_fin:
                fin += 1
            if p.tcp_flag_rst:
                rst += 1
            if p.tcp_flag_psh:
                psh += 1
            if p.tcp_flag_urg:
                urg += 1
        if p.is_udp:
            udp += 1
        if p.is_icmp:
            icmp += 1
        if p.is_icmpv6:
            icmpv6 += 1
        if p.is_igmp:
            igmp += 1

        if p.src_ip is not None:
            ips.add(p.src_ip)
        if p.dst_ip is not None:
            ips.add(p.dst_ip)
        if p.src_port is not None:
            ports.add(p.src_port)
        if p.dst_port is not None:
            ports.add(p.dst_port)

    denom = float(WINDOW_SIZE)
    if tcp > 0:
        tcp_den = float(tcp)
        tcp_syn_ratio = syn / tcp_den
        tcp_ack_ratio = ack / tcp_den
        tcp_fin_ratio = fin / tcp_den
        tcp_rst_ratio = rst / tcp_den
        tcp_psh_ratio = psh / tcp_den
        tcp_urg_ratio = urg / tcp_den
    else:
        tcp_syn_ratio = tcp_ack_ratio = tcp_fin_ratio = 0.0
        tcp_rst_ratio = tcp_psh_ratio = tcp_urg_ratio = 0.0

    return FeatureVector(
        window_span_seconds=span,
        iat_mean_seconds=_mean(iats),
        iat_std_seconds=_population_std(iats),
        iat_p50_seconds=_percentile(iats, 50),
        iat_p95_seconds=_percentile(iats, 95),
        frame_length_mean=_mean(lengths),
        frame_length_std=_population_std(lengths),
        frame_length_min=min(lengths),
        frame_length_max=max(lengths),
        ipv4_ratio=ipv4 / denom,
        ipv6_ratio=ipv6 / denom,
        arp_ratio=arp / denom,
        llc_ratio=llc / denom,
        other_protocol_ratio=other / denom,
        tcp_ratio=tcp / denom,
        udp_ratio=udp / denom,
        icmp_ratio=icmp / denom,
        icmpv6_ratio=icmpv6 / denom,
        igmp_ratio=igmp / denom,
        tcp_syn_ratio=tcp_syn_ratio,
        tcp_ack_ratio=tcp_ack_ratio,
        tcp_fin_ratio=tcp_fin_ratio,
        tcp_rst_ratio=tcp_rst_ratio,
        tcp_psh_ratio=tcp_psh_ratio,
        tcp_urg_ratio=tcp_urg_ratio,
        unique_ip_count=len(ips),
        unique_port_count=len(ports),
    )
