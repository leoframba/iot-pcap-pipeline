"""Unit tests for streaming integrity / TRAIN characterization stats."""

from __future__ import annotations

from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus
from iot_pcap_pipeline.pcap.stats import IntegrityStats, TrainCharacterizationStats


def _pkt(
    *,
    index: int,
    ts: float,
    length: int = 100,
    status: ParseStatus = ParseStatus.OK,
    is_tcp: bool = False,
    is_udp: bool = False,
    is_ipv4: bool = False,
    src_ip: str | None = None,
    dst_ip: str | None = None,
    src_port: int | None = None,
    dst_port: int | None = None,
    syn: bool = False,
    ack: bool = False,
    protocol_name: str = "tcp",
    detail: str | None = None,
) -> PacketRecord:
    return PacketRecord(
        packet_index=index,
        timestamp=ts,
        frame_length=length,
        linktype=1,
        parse_status=status,
        parse_detail=detail,
        is_tcp=is_tcp,
        is_udp=is_udp,
        is_ipv4=is_ipv4,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        tcp_flag_syn=syn,
        tcp_flag_ack=ack,
        protocol_name=protocol_name,
    )


def test_frame_and_timestamp_durations() -> None:
    stats = IntegrityStats()
    stats.observe(_pkt(index=0, ts=10.0, length=50))
    stats.observe(_pkt(index=1, ts=10.0, length=150))  # duplicate
    stats.observe(_pkt(index=2, ts=9.0, length=100))  # reversal
    stats.observe(_pkt(index=3, ts=12.0, length=100))

    assert stats.packet_count == 4
    assert stats.total_frame_bytes == 400
    assert stats.min_frame_length == 50
    assert stats.max_frame_length == 150
    assert stats.mean_frame_length == 100.0
    assert stats.first_timestamp == 10.0
    assert stats.last_timestamp == 12.0
    assert stats.min_timestamp == 9.0
    assert stats.max_timestamp == 12.0
    assert stats.capture_order_duration == 2.0
    assert stats.capture_timestamp_span == 3.0
    assert stats.duplicate_timestamp_count == 1
    assert stats.negative_delta_count == 1
    assert stats.non_monotonic_timestamp_count == 2
    assert stats.validate_invariants() == []


def test_status_separation_and_invariants() -> None:
    stats = IntegrityStats()
    stats.observe(_pkt(index=0, ts=1.0, status=ParseStatus.OK, is_tcp=True, is_ipv4=True))
    stats.observe(
        _pkt(index=1, ts=2.0, status=ParseStatus.UNSUPPORTED, protocol_name="lldp")
    )
    stats.observe(
        _pkt(index=2, ts=3.0, status=ParseStatus.MALFORMED, detail="ethernet short")
    )
    stats.observe(_pkt(index=3, ts=4.0, status=ParseStatus.PARTIAL, is_ipv4=True))
    assert stats.ok_count == 1
    assert stats.unsupported_count == 1
    assert stats.malformed_count == 1
    assert stats.partial_count == 1
    assert stats.error_count == 0
    assert stats.validate_invariants() == []


def test_train_ratios_and_ip_cap() -> None:
    integrity = IntegrityStats()
    train = TrainCharacterizationStats(integrity=integrity, ip_cardinality_cap=2)
    for i in range(3):
        rec = _pkt(
            index=i,
            ts=float(i),
            is_tcp=True,
            is_ipv4=True,
            src_ip=f"10.0.0.{i}",
            dst_ip=f"10.1.0.{i}",
            src_port=1000 + i,
            dst_port=80,
            syn=True,
        )
        integrity.observe(rec)
        train.observe(rec)

    fields = train.to_characterization_fields()
    assert fields["tcp_ratio"] == 1.0
    assert fields["tcp_syn_ratio"] == 1.0
    assert fields["unique_src_ips_count"] == 2
    assert fields["unique_src_ips_capped"] is True
    assert fields["unique_dst_ips_count"] == 2
    assert fields["unique_dst_ips_capped"] is True
    assert fields["unique_src_ports_count"] == 3
    assert fields["unique_dst_ports_count"] == 1
    assert fields["packets_per_second"] == 1.5  # 3 packets / span 2.0
