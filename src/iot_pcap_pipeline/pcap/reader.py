"""Streaming PCAP reader yielding normalized PacketRecords."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import dpkt

from iot_pcap_pipeline.pcap.decode import decode_frame
from iot_pcap_pipeline.pcap.packet import FAILURE_STATUSES, PacketRecord, ParseStatus


@dataclass
class PcapReadStats:
    """Aggregate counters for one PCAP inspection pass."""

    path: str
    packets_total: int = 0
    packets_ok: int = 0
    packets_unsupported: int = 0
    packets_partial: int = 0
    packets_failed: int = 0
    by_parse_status: Counter[str] = field(default_factory=Counter)
    by_protocol: Counter[str] = field(default_factory=Counter)
    tcp: int = 0
    udp: int = 0
    icmp: int = 0
    icmpv6: int = 0
    igmp: int = 0
    arp: int = 0
    llc: int = 0
    ipv4: int = 0
    ipv6: int = 0
    vlan_frames: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    linktype: int | None = None

    def observe(self, record: PacketRecord) -> None:
        self.packets_total += 1
        self.by_parse_status[record.parse_status.value] += 1
        if record.parse_status == ParseStatus.OK:
            self.packets_ok += 1
        elif record.parse_status == ParseStatus.UNSUPPORTED:
            self.packets_unsupported += 1
        elif record.parse_status == ParseStatus.PARTIAL:
            self.packets_partial += 1
        if record.parse_status in FAILURE_STATUSES:
            self.packets_failed += 1
        proto = record.protocol_name or "unknown"
        self.by_protocol[proto] += 1
        if record.is_tcp:
            self.tcp += 1
        if record.is_udp:
            self.udp += 1
        if record.is_icmp:
            self.icmp += 1
        if record.is_icmpv6:
            self.icmpv6 += 1
        if record.is_igmp:
            self.igmp += 1
        if record.is_arp:
            self.arp += 1
        if record.is_llc:
            self.llc += 1
        if record.is_ipv4:
            self.ipv4 += 1
        if record.is_ipv6:
            self.ipv6 += 1
        if record.vlan_ids:
            self.vlan_frames += 1
        if self.first_timestamp is None:
            self.first_timestamp = record.timestamp
        self.last_timestamp = record.timestamp
        if self.linktype is None:
            self.linktype = record.linktype


def iter_packets(
    pcap_path: Path | str,
    *,
    max_packets: int | None = None,
) -> Iterator[PacketRecord]:
    """Yield PacketRecords in original capture order.

    Read-only: opens the PCAP for binary reading and never writes back.
    Parsing failures become PacketRecords with an explicit parse_status.
    """
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP not found: {path}")

    with path.open("rb") as handle:
        try:
            reader = dpkt.pcap.Reader(handle)
        except (ValueError, OSError, dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError) as exc:
            raise ValueError(f"failed to open PCAP {path}: {exc}") from exc

        linktype = int(reader.datalink())
        for index, (timestamp, buf) in enumerate(reader):
            if max_packets is not None and index >= max_packets:
                break
            yield decode_frame(
                buf,
                packet_index=index,
                timestamp=float(timestamp),
                linktype=linktype,
            )


def summarize_pcap(
    pcap_path: Path | str,
    *,
    max_packets: int | None = None,
) -> PcapReadStats:
    """Stream a PCAP and return aggregate parse/protocol counters."""
    path = Path(pcap_path)
    stats = PcapReadStats(path=str(path))
    for record in iter_packets(path, max_packets=max_packets):
        stats.observe(record)
    return stats
