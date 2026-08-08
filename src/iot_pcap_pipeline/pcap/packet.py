"""Normalized packet records for Phase 1B PCAP decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ParseStatus(str, Enum):
    """Outcome of decoding a single PCAP record."""

    OK = "ok"
    PARTIAL = "partial"
    MALFORMED = "malformed"
    UNSUPPORTED_LINKTYPE = "unsupported_linktype"
    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    ERROR = "error"


@dataclass(frozen=True)
class PacketRecord:
    """Label-independent view of one captured frame.

    Timestamps and frame lengths are retained even when deeper decoding fails.
    Addresses and ports are available for later window features but are never
    derived from filenames, labels, or split metadata.
    """

    packet_index: int
    timestamp: float
    frame_length: int
    linktype: int
    parse_status: ParseStatus
    parse_detail: str | None = None

    ethertype: int | None = None
    vlan_ids: tuple[int, ...] = ()

    is_arp: bool = False
    is_ipv4: bool = False
    is_ipv6: bool = False
    is_tcp: bool = False
    is_udp: bool = False
    is_icmp: bool = False
    is_icmpv6: bool = False
    is_igmp: bool = False

    ip_version: int | None = None
    ip_protocol: int | None = None
    src_ip: str | None = None
    dst_ip: str | None = None

    src_port: int | None = None
    dst_port: int | None = None

    tcp_flags: int | None = None
    tcp_flag_syn: bool = False
    tcp_flag_ack: bool = False
    tcp_flag_fin: bool = False
    tcp_flag_rst: bool = False
    tcp_flag_psh: bool = False
    tcp_flag_urg: bool = False

    protocol_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parse_status"] = self.parse_status.value
        data["vlan_ids"] = list(self.vlan_ids)
        return data
