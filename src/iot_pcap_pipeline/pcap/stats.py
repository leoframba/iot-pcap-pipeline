"""Streaming integrity and TRAIN characterization accumulators."""

from __future__ import annotations

import socket
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus

DEFAULT_IP_CARDINALITY_CAP = 100_000


def _pack_ip(ip: str | None) -> bytes | None:
    if not ip:
        return None
    try:
        if ":" in ip:
            return socket.inet_pton(socket.AF_INET6, ip)
        return socket.inet_pton(socket.AF_INET, ip)
    except OSError:
        # Fall back to UTF-8 bytes so unknown forms still contribute to cardinality.
        return ip.encode("utf-8")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


@dataclass
class IntegrityStats:
    """Label-independent integrity metrics for one PCAP."""

    packet_count: int = 0
    total_frame_bytes: int = 0
    min_frame_length: int | None = None
    max_frame_length: int | None = None

    ok_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    malformed_count: int = 0
    error_count: int = 0

    ipv4_count: int = 0
    ipv6_count: int = 0
    arp_count: int = 0
    llc_count: int = 0
    tcp_count: int = 0
    udp_count: int = 0
    icmp_count: int = 0
    icmpv6_count: int = 0
    igmp_count: int = 0
    vlan_count: int = 0

    by_protocol: Counter[str] = field(default_factory=Counter)

    first_timestamp: float | None = None
    last_timestamp: float | None = None
    min_timestamp: float | None = None
    max_timestamp: float | None = None
    duplicate_timestamp_count: int = 0
    negative_delta_count: int = 0
    non_monotonic_timestamp_count: int = 0
    _prev_timestamp: float | None = field(default=None, repr=False)

    linktype: int | None = None
    accounting_errors: list[str] = field(default_factory=list)

    def observe(self, record: PacketRecord) -> None:
        self.packet_count += 1
        length = int(record.frame_length)
        self.total_frame_bytes += length
        if self.min_frame_length is None or length < self.min_frame_length:
            self.min_frame_length = length
        if self.max_frame_length is None or length > self.max_frame_length:
            self.max_frame_length = length

        status = record.parse_status
        if status == ParseStatus.OK:
            self.ok_count += 1
        elif status == ParseStatus.PARTIAL:
            self.partial_count += 1
        elif status == ParseStatus.UNSUPPORTED:
            self.unsupported_count += 1
        elif status == ParseStatus.MALFORMED:
            self.malformed_count += 1
        elif status == ParseStatus.ERROR:
            self.error_count += 1

        proto = record.protocol_name or "unknown"
        self.by_protocol[proto] += 1

        if record.is_ipv4:
            self.ipv4_count += 1
        if record.is_ipv6:
            self.ipv6_count += 1
        if record.is_arp:
            self.arp_count += 1
        if record.is_llc:
            self.llc_count += 1
        if record.is_tcp:
            self.tcp_count += 1
        if record.is_udp:
            self.udp_count += 1
        if record.is_icmp:
            self.icmp_count += 1
        if record.is_icmpv6:
            self.icmpv6_count += 1
        if record.is_igmp:
            self.igmp_count += 1
        if record.vlan_ids:
            self.vlan_count += 1

        ts = float(record.timestamp)
        if self.first_timestamp is None:
            self.first_timestamp = ts
        self.last_timestamp = ts
        if self.min_timestamp is None or ts < self.min_timestamp:
            self.min_timestamp = ts
        if self.max_timestamp is None or ts > self.max_timestamp:
            self.max_timestamp = ts

        if self._prev_timestamp is not None:
            delta = ts - self._prev_timestamp
            if delta == 0:
                self.duplicate_timestamp_count += 1
                self.non_monotonic_timestamp_count += 1
            elif delta < 0:
                self.negative_delta_count += 1
                self.non_monotonic_timestamp_count += 1
        self._prev_timestamp = ts

        if self.linktype is None:
            self.linktype = record.linktype

    @property
    def mean_frame_length(self) -> float | None:
        if self.packet_count <= 0:
            return None
        return self.total_frame_bytes / self.packet_count

    @property
    def capture_order_duration(self) -> float | None:
        if self.first_timestamp is None or self.last_timestamp is None:
            return None
        return self.last_timestamp - self.first_timestamp

    @property
    def capture_timestamp_span(self) -> float | None:
        if self.min_timestamp is None or self.max_timestamp is None:
            return None
        return self.max_timestamp - self.min_timestamp

    @property
    def malformed_rate(self) -> float | None:
        return _safe_ratio(self.malformed_count, self.packet_count)

    def validate_invariants(self) -> list[str]:
        errors: list[str] = []
        status_sum = (
            self.ok_count
            + self.partial_count
            + self.unsupported_count
            + self.malformed_count
            + self.error_count
        )
        if status_sum != self.packet_count:
            errors.append(
                f"status counts {status_sum} != packet_count {self.packet_count}"
            )
        if self.tcp_count + self.udp_count > self.packet_count:
            errors.append("tcp+udp exceeds packet_count")
        if self.vlan_count > self.packet_count:
            errors.append("vlan_count exceeds packet_count")
        for name, count in (
            ("ipv4", self.ipv4_count),
            ("ipv6", self.ipv6_count),
            ("arp", self.arp_count),
            ("llc", self.llc_count),
            ("tcp", self.tcp_count),
            ("udp", self.udp_count),
            ("icmp", self.icmp_count),
            ("icmpv6", self.icmpv6_count),
            ("igmp", self.igmp_count),
        ):
            if count < 0 or count > self.packet_count:
                errors.append(f"{name}_count out of bounds: {count}")
        self.accounting_errors = errors
        return errors

    def to_integrity_fields(self) -> dict[str, Any]:
        return {
            "packet_count": self.packet_count,
            "total_frame_bytes": self.total_frame_bytes,
            "min_frame_length": self.min_frame_length,
            "max_frame_length": self.max_frame_length,
            "mean_frame_length": self.mean_frame_length,
            "ok_count": self.ok_count,
            "partial_count": self.partial_count,
            "unsupported_count": self.unsupported_count,
            "malformed_count": self.malformed_count,
            "error_count": self.error_count,
            "ipv4_count": self.ipv4_count,
            "ipv6_count": self.ipv6_count,
            "arp_count": self.arp_count,
            "llc_count": self.llc_count,
            "tcp_count": self.tcp_count,
            "udp_count": self.udp_count,
            "icmp_count": self.icmp_count,
            "icmpv6_count": self.icmpv6_count,
            "igmp_count": self.igmp_count,
            "vlan_count": self.vlan_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "min_timestamp": self.min_timestamp,
            "max_timestamp": self.max_timestamp,
            "capture_order_duration": self.capture_order_duration,
            "capture_timestamp_span": self.capture_timestamp_span,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "negative_delta_count": self.negative_delta_count,
            "non_monotonic_timestamp_count": self.non_monotonic_timestamp_count,
            "linktype": self.linktype,
        }


@dataclass
class TrainCharacterizationStats:
    """TRAIN-only behavioral characterization for one PCAP."""

    integrity: IntegrityStats
    ip_cardinality_cap: int = DEFAULT_IP_CARDINALITY_CAP

    tcp_syn_count: int = 0
    tcp_ack_count: int = 0
    tcp_fin_count: int = 0
    tcp_rst_count: int = 0
    tcp_psh_count: int = 0

    _src_ips: set[bytes] = field(default_factory=set, repr=False)
    _dst_ips: set[bytes] = field(default_factory=set, repr=False)
    _src_ports: set[int] = field(default_factory=set, repr=False)
    _dst_ports: set[int] = field(default_factory=set, repr=False)
    unique_src_ips_capped: bool = False
    unique_dst_ips_capped: bool = False

    def observe(self, record: PacketRecord) -> None:
        if record.is_tcp:
            if record.tcp_flag_syn:
                self.tcp_syn_count += 1
            if record.tcp_flag_ack:
                self.tcp_ack_count += 1
            if record.tcp_flag_fin:
                self.tcp_fin_count += 1
            if record.tcp_flag_rst:
                self.tcp_rst_count += 1
            if record.tcp_flag_psh:
                self.tcp_psh_count += 1

        src_packed = _pack_ip(record.src_ip)
        if src_packed is not None:
            if len(self._src_ips) < self.ip_cardinality_cap:
                self._src_ips.add(src_packed)
            elif src_packed not in self._src_ips:
                self.unique_src_ips_capped = True

        dst_packed = _pack_ip(record.dst_ip)
        if dst_packed is not None:
            if len(self._dst_ips) < self.ip_cardinality_cap:
                self._dst_ips.add(dst_packed)
            elif dst_packed not in self._dst_ips:
                self.unique_dst_ips_capped = True

        if record.src_port is not None:
            self._src_ports.add(int(record.src_port))
        if record.dst_port is not None:
            self._dst_ports.add(int(record.dst_port))

    def to_characterization_fields(self) -> dict[str, Any]:
        n = self.integrity.packet_count
        tcp_n = self.integrity.tcp_count
        span = self.integrity.capture_timestamp_span
        packets_per_second = None
        bytes_per_second = None
        if span is not None and span > 0:
            packets_per_second = n / span
            bytes_per_second = self.integrity.total_frame_bytes / span

        return {
            "packet_count": n,
            "total_frame_bytes": self.integrity.total_frame_bytes,
            "min_frame_length": self.integrity.min_frame_length,
            "max_frame_length": self.integrity.max_frame_length,
            "mean_frame_length": self.integrity.mean_frame_length,
            "capture_order_duration": self.integrity.capture_order_duration,
            "capture_timestamp_span": span,
            "packets_per_second": packets_per_second,
            "bytes_per_second": bytes_per_second,
            "tcp_count": self.integrity.tcp_count,
            "udp_count": self.integrity.udp_count,
            "arp_count": self.integrity.arp_count,
            "llc_count": self.integrity.llc_count,
            "ipv4_count": self.integrity.ipv4_count,
            "ipv6_count": self.integrity.ipv6_count,
            "icmp_count": self.integrity.icmp_count,
            "icmpv6_count": self.integrity.icmpv6_count,
            "igmp_count": self.integrity.igmp_count,
            "vlan_count": self.integrity.vlan_count,
            # Protocol proportions use packet_count as denominator; they need not sum to 1.
            "tcp_ratio": _safe_ratio(self.integrity.tcp_count, n),
            "udp_ratio": _safe_ratio(self.integrity.udp_count, n),
            "arp_ratio": _safe_ratio(self.integrity.arp_count, n),
            "llc_ratio": _safe_ratio(self.integrity.llc_count, n),
            "ipv4_ratio": _safe_ratio(self.integrity.ipv4_count, n),
            "ipv6_ratio": _safe_ratio(self.integrity.ipv6_count, n),
            "icmp_ratio": _safe_ratio(self.integrity.icmp_count, n),
            "icmpv6_ratio": _safe_ratio(self.integrity.icmpv6_count, n),
            "igmp_ratio": _safe_ratio(self.integrity.igmp_count, n),
            "vlan_ratio": _safe_ratio(self.integrity.vlan_count, n),
            "tcp_syn_count": self.tcp_syn_count,
            "tcp_ack_count": self.tcp_ack_count,
            "tcp_fin_count": self.tcp_fin_count,
            "tcp_rst_count": self.tcp_rst_count,
            "tcp_psh_count": self.tcp_psh_count,
            # TCP flag proportions use tcp_count as denominator.
            "tcp_syn_ratio": _safe_ratio(self.tcp_syn_count, tcp_n),
            "tcp_ack_ratio": _safe_ratio(self.tcp_ack_count, tcp_n),
            "tcp_fin_ratio": _safe_ratio(self.tcp_fin_count, tcp_n),
            "tcp_rst_ratio": _safe_ratio(self.tcp_rst_count, tcp_n),
            "tcp_psh_ratio": _safe_ratio(self.tcp_psh_count, tcp_n),
            "unique_src_ips_count": len(self._src_ips),
            "unique_src_ips_capped": self.unique_src_ips_capped,
            "unique_dst_ips_count": len(self._dst_ips),
            "unique_dst_ips_capped": self.unique_dst_ips_capped,
            "unique_src_ports_count": len(self._src_ports),
            "unique_dst_ports_count": len(self._dst_ports),
        }
