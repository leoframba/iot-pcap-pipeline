"""Immutable packet window for Phase 1C.2 feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

from iot_pcap_pipeline.pcap.packet import PacketRecord


@dataclass(frozen=True)
class PacketWindow:
    """One full fixed-size packet window within a single PCAP segment.

    ``window_index`` increases globally per PCAP (not per segment) each time a
    full window is emitted. ``segment_index`` starts at 0 and increments on
    every accepted segment reset.
    """

    segment_index: int
    window_index: int
    packet_index_start: int
    packet_index_end: int
    packets: tuple[PacketRecord, ...]

    def __post_init__(self) -> None:
        if not self.packets:
            raise ValueError("PacketWindow must contain at least one packet")
        if self.packet_index_start > self.packet_index_end:
            raise ValueError("packet_index_end must be >= packet_index_start")
        if self.packets[0].packet_index != self.packet_index_start:
            raise ValueError("packet_index_start does not match first packet")
        if self.packets[-1].packet_index != self.packet_index_end:
            raise ValueError("packet_index_end does not match last packet")
