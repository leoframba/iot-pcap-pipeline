"""Normalized packet records for Phase 1B PCAP decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ParseStatus(str, Enum):
    """Outcome of decoding a single PCAP record.

    These statuses are not interchangeable with "success/failure":

    - ``ok``: known framing decoded to a supported L3/L4 view (or LLC recognized)
    - ``partial``: valid frame, but some expected layer could not be recovered
    - ``unsupported``: structurally valid packet/protocol we intentionally do not
      deep-decode (e.g. LLDP, exotic IP proto). Not a parse failure.
    - ``malformed``: truncated or structurally invalid frame/headers
    - ``error``: unexpected decoder exception
    """

    OK = "ok"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    ERROR = "error"


# Statuses that indicate a real decode problem (not merely "other protocol").
FAILURE_STATUSES = frozenset({ParseStatus.MALFORMED, ParseStatus.ERROR})


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

    # Raw 16-bit Ethernet type/length field when linktype is Ethernet.
    ethernet_type_or_length: int | None = None
    # True when type/length >= 0x0600 (Ethernet II). False for IEEE 802.3 length.
    is_ethernet_ii: bool | None = None
    # Effective EtherType after VLAN unwrap for Ethernet II; None for 802.3/LLC.
    ethertype: int | None = None
    vlan_ids: tuple[int, ...] = ()

    is_llc: bool = False
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

    # Internal only (V2M): TCP segment application bytes. Never a model feature /
    # never written to feature Parquet. None when not TCP or payload unavailable.
    tcp_payload: bytes | None = None

    protocol_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.parse_status in FAILURE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parse_status"] = self.parse_status.value
        data["vlan_ids"] = list(self.vlan_ids)
        data["is_failure"] = self.is_failure
        # Never serialize raw payload bytes (internal experimental field only).
        payload = data.pop("tcp_payload", None)
        data["tcp_payload_len"] = len(payload) if isinstance(payload, (bytes, bytearray)) else None
        return data
