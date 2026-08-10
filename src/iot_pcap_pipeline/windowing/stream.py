"""Production PacketRecord → fixed-window streaming engine (Phase 1C.2).

Consumes an ordered ``Iterable[PacketRecord]`` (typically from ``iter_packets``)
and emits full windows under the frozen Gate-A policy. Incomplete windows are
dropped at inactivity / backward-discontinuity boundaries and at EOF.

Decoder ``ParseStatus.ERROR`` aborts extraction. ``MALFORMED``, ``UNSUPPORTED``,
and ``PARTIAL`` packets remain in the stream.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus
from iot_pcap_pipeline.windowing.policy import WindowPolicy, frozen_window_policy
from iot_pcap_pipeline.windowing.window import PacketWindow


class FeatureExtractionError(RuntimeError):
    """Raised when a PCAP cannot be transformed under the V1 contract."""


@dataclass
class WindowStreamStats:
    """Accounting for one PCAP / PacketRecord stream."""

    packets_seen: int = 0
    segment_count: int = 0
    full_window_count: int = 0
    dropped_partial_window_count: int = 0
    dropped_partial_packet_count: int = 0
    by_parse_status: Counter[str] = field(default_factory=Counter)

    def observe_status(self, status: ParseStatus) -> None:
        self.by_parse_status[status.value] += 1


def iter_windows(
    packets: Iterable[PacketRecord],
    policy: WindowPolicy | None = None,
    *,
    stats: WindowStreamStats | None = None,
    max_windows: int | None = None,
) -> Iterator[PacketWindow]:
    """Yield full non-overlapping windows from an ordered packet stream.

    Boundary comparisons (exact):

    - ``delta > inactivity_timeout`` → new segment (drop incomplete)
    - ``delta < -backward_reset`` → new segment (drop incomplete)

    Packets with ``parse_status == ERROR`` abort the entire stream.
    """
    pol = policy if policy is not None else frozen_window_policy()
    acc = _WindowAccumulator(policy=pol, stats=stats)
    emitted = 0

    for packet in packets:
        if packet.parse_status == ParseStatus.ERROR:
            detail = packet.parse_detail or "decoder error"
            raise FeatureExtractionError(
                f"ParseStatus.ERROR at packet_index={packet.packet_index}: {detail}"
            )
        window = acc.observe(packet)
        if window is not None:
            yield window
            emitted += 1
            if max_windows is not None and emitted >= max_windows:
                return

    # EOF: drop incomplete; do not emit.
    acc.finalize()


def count_full_windows(
    packets: Iterable[PacketRecord],
    policy: WindowPolicy | None = None,
) -> int:
    """Count full windows under the (frozen) policy without feature extraction."""
    n = 0
    for _ in iter_windows(packets, policy=policy):
        n += 1
    return n


@dataclass
class _WindowAccumulator:
    policy: WindowPolicy
    stats: WindowStreamStats | None = None
    _segment_index: int = -1
    _window_index: int = -1
    _prev: PacketRecord | None = None
    _buf: list[PacketRecord] = field(default_factory=list)

    def observe(self, packet: PacketRecord) -> PacketWindow | None:
        if self.stats is not None:
            self.stats.packets_seen += 1
            self.stats.observe_status(packet.parse_status)

        if self._prev is None:
            self._start_segment(packet)
            return self._maybe_emit()

        delta = packet.timestamp - self._prev.timestamp
        if delta > self.policy.inactivity_timeout_seconds or delta < -self.policy.backward_reset_seconds:
            self._drop_partial()
            self._start_segment(packet)
        else:
            self._buf.append(packet)

        self._prev = packet
        return self._maybe_emit()

    def finalize(self) -> None:
        self._drop_partial()

    def _start_segment(self, packet: PacketRecord) -> None:
        self._segment_index += 1
        if self.stats is not None:
            self.stats.segment_count = self._segment_index + 1
        self._buf = [packet]
        self._prev = packet

    def _drop_partial(self) -> None:
        if not self._buf:
            return
        if self.stats is not None:
            self.stats.dropped_partial_window_count += 1
            self.stats.dropped_partial_packet_count += len(self._buf)
        self._buf = []

    def _maybe_emit(self) -> PacketWindow | None:
        if len(self._buf) < self.policy.window_size:
            return None
        if len(self._buf) > self.policy.window_size:
            raise FeatureExtractionError(
                "internal window buffer exceeded window_size"
            )
        packets = tuple(self._buf)
        self._buf = []
        self._window_index += 1
        if self.stats is not None:
            self.stats.full_window_count += 1
        return PacketWindow(
            segment_index=self._segment_index,
            window_index=self._window_index,
            packet_index_start=packets[0].packet_index,
            packet_index_end=packets[-1].packet_index,
            packets=packets,
        )
