"""Phase 1C.2 V1 windowing + feature extraction tests."""

from __future__ import annotations

from pathlib import Path

import dpkt
import pytest
from pcap_synth import (
    eth_arp,
    eth_ieee8023_llc,
    eth_ip_icmp,
    eth_ip_igmp,
    eth_ip_tcp,
    eth_ip_udp,
    eth_ipv6_icmp,
    eth_ipv6_tcp,
    eth_lldp,
    write_pcap,
)

from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES, write_feature_schema
from iot_pcap_pipeline.features.validate import validate_window_and_features
from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import frozen_window_policy
from iot_pcap_pipeline.windowing.stream import (
    FeatureExtractionError,
    WindowStreamStats,
    iter_windows,
)


def _pkt(
    index: int,
    ts: float,
    *,
    length: int = 100,
    status: ParseStatus = ParseStatus.OK,
    is_ipv4: bool = True,
    is_ipv6: bool = False,
    is_arp: bool = False,
    is_llc: bool = False,
    is_tcp: bool = True,
    is_udp: bool = False,
    is_icmp: bool = False,
    is_icmpv6: bool = False,
    is_igmp: bool = False,
    src_ip: str | None = "10.0.0.1",
    dst_ip: str | None = "10.0.0.2",
    src_port: int | None = 1234,
    dst_port: int | None = 80,
    syn: bool = False,
    ack: bool = False,
    fin: bool = False,
    rst: bool = False,
    psh: bool = False,
    urg: bool = False,
    detail: str | None = None,
) -> PacketRecord:
    return PacketRecord(
        packet_index=index,
        timestamp=ts,
        frame_length=length,
        linktype=1,
        parse_status=status,
        parse_detail=detail,
        is_ipv4=is_ipv4,
        is_ipv6=is_ipv6,
        is_arp=is_arp,
        is_llc=is_llc,
        is_tcp=is_tcp,
        is_udp=is_udp,
        is_icmp=is_icmp,
        is_icmpv6=is_icmpv6,
        is_igmp=is_igmp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        tcp_flag_syn=syn,
        tcp_flag_ack=ack,
        tcp_flag_fin=fin,
        tcp_flag_rst=rst,
        tcp_flag_psh=psh,
        tcp_flag_urg=urg,
    )


def _steady(n: int, start: float = 0.0, step: float = 0.01) -> list[PacketRecord]:
    return [_pkt(i, start + i * step) for i in range(n)]


def test_frozen_policy_unchanged() -> None:
    p = frozen_window_policy()
    assert p.window_size == 25
    assert p.inactivity_timeout_seconds == 5.0
    assert p.backward_reset_seconds == 1.0


def test_boundary_exactly_5s_no_reset() -> None:
    packets = _steady(24)
    packets.append(_pkt(24, packets[-1].timestamp + 5.0))  # delta == 5.0 exactly
    windows = list(iter_windows(packets, frozen_window_policy()))
    assert len(windows) == 1
    assert windows[0].segment_index == 0


def test_boundary_gt_5s_resets_clean() -> None:
    first = [_pkt(i, float(i)) for i in range(10)]
    boundary = _pkt(10, 10.0 + 5.01)
    rest = [_pkt(11 + i, boundary.timestamp + 0.01 * (i + 1)) for i in range(24)]
    packets = first + [boundary] + rest
    stats = WindowStreamStats()
    windows = list(iter_windows(packets, frozen_window_policy(), stats=stats))
    assert stats.dropped_partial_packet_count == 10
    assert len(windows) == 1
    assert windows[0].segment_index == 1
    assert windows[0].packets[0].packet_index == 10


def test_boundary_exactly_minus_1s_no_reset() -> None:
    packets = [_pkt(i, float(i)) for i in range(24)] + [_pkt(24, 23.0 - 1.0)]
    windows = list(iter_windows(packets, frozen_window_policy()))
    assert len(windows) == 1
    assert windows[0].segment_index == 0


def test_boundary_lt_minus_1s_resets() -> None:
    first = [_pkt(i, float(i)) for i in range(5)]
    boundary = _pkt(5, 4.0 - 1.01)
    rest = [_pkt(6 + i, boundary.timestamp + 0.01 * (i + 1)) for i in range(24)]
    windows = list(
        iter_windows(first + [boundary] + rest, frozen_window_policy())
    )
    assert len(windows) == 1
    assert windows[0].segment_index == 1
    assert windows[0].packets[0].packet_index == 5


def test_duplicate_and_tiny_negative_no_reset() -> None:
    packets = [_pkt(0, 1.0)]
    packets += [_pkt(1, 1.0)]  # duplicate
    packets += [_pkt(2, 0.999999)]  # tiny negative
    packets += [_pkt(i, 1.0 + 0.01 * (i - 2)) for i in range(3, 25)]
    windows = list(iter_windows(packets, frozen_window_policy()))
    assert len(windows) == 1
    assert windows[0].segment_index == 0


def test_window_counts_24_25_26_50_51() -> None:
    policy = frozen_window_policy()
    assert list(iter_windows(_steady(24), policy)) == []
    assert len(list(iter_windows(_steady(25), policy))) == 1
    assert len(list(iter_windows(_steady(26), policy))) == 1  # 1 full + drop 1
    assert len(list(iter_windows(_steady(50), policy))) == 2
    assert len(list(iter_windows(_steady(51), policy))) == 2


def test_eof_drops_partial() -> None:
    stats = WindowStreamStats()
    windows = list(iter_windows(_steady(30), frozen_window_policy(), stats=stats))
    assert len(windows) == 1
    assert stats.dropped_partial_packet_count == 5


def test_window_span_max_min_and_iat_sanitization() -> None:
    # first == last with larger middle; tiny negative IAT sanitized
    packets = [
        _pkt(0, 1.0, length=60),
        _pkt(1, 1.1, length=70),
        _pkt(2, 1.0, length=80),  # negative raw delta vs prev
    ] + [_pkt(3 + i, 1.2 + 0.01 * i, length=90) for i in range(22)]
    windows = list(iter_windows(packets, frozen_window_policy()))
    assert len(windows) == 1
    feats = extract_features(windows[0])
    assert feats.window_span_seconds == pytest.approx(max(p.timestamp for p in packets) - min(p.timestamp for p in packets))
    assert feats.window_span_seconds >= 0
    assert feats.iat_mean_seconds >= 0
    assert feats.iat_std_seconds >= 0
    validate_window_and_features(windows[0], feats)


def test_all_equal_timestamps_zero_span_zero_iat() -> None:
    packets = [_pkt(i, 5.0) for i in range(25)]
    feats = extract_features(next(iter_windows(packets)))
    assert feats.window_span_seconds == 0.0
    assert feats.iat_mean_seconds == 0.0
    assert feats.iat_std_seconds == 0.0
    assert feats.iat_p50_seconds == 0.0


def test_frame_stats_hand_check() -> None:
    lengths = [10, 20, 30, 40]
    # pad to 25 with 50
    packets = [_pkt(i, float(i), length=lengths[i] if i < 4 else 50) for i in range(25)]
    feats = extract_features(next(iter_windows(packets)))
    expected_mean = (10 + 20 + 30 + 40 + 50 * 21) / 25
    assert feats.frame_length_mean == pytest.approx(expected_mean)
    assert feats.frame_length_min == 10
    assert feats.frame_length_max == 50
    # population std
    vals = [float(p.frame_length) for p in packets]
    m = sum(vals) / 25
    expected_std = (sum((x - m) ** 2 for x in vals) / 25) ** 0.5
    assert feats.frame_length_std == pytest.approx(expected_std)


def test_l3_partition_sums_to_one() -> None:
    packets = (
        [_pkt(i, float(i), is_ipv4=True, is_tcp=True) for i in range(10)]
        + [
            _pkt(
                10 + i,
                float(10 + i),
                is_ipv4=False,
                is_ipv6=True,
                is_tcp=True,
                src_ip="2001:db8::1",
                dst_ip="2001:db8::2",
            )
            for i in range(5)
        ]
        + [
            _pkt(
                15 + i,
                float(15 + i),
                is_ipv4=False,
                is_tcp=False,
                is_arp=True,
                src_ip=None,
                dst_ip=None,
                src_port=None,
                dst_port=None,
            )
            for i in range(4)
        ]
        + [
            _pkt(
                19 + i,
                float(19 + i),
                is_ipv4=False,
                is_tcp=False,
                is_llc=True,
                src_ip=None,
                dst_ip=None,
                src_port=None,
                dst_port=None,
            )
            for i in range(3)
        ]
        + [
            _pkt(
                22 + i,
                float(22 + i),
                is_ipv4=False,
                is_tcp=False,
                src_ip=None,
                dst_ip=None,
                src_port=None,
                dst_port=None,
                status=ParseStatus.UNSUPPORTED,
            )
            for i in range(3)
        ]
    )
    assert len(packets) == 25
    feats = extract_features(next(iter_windows(packets)))
    assert feats.ipv4_ratio == pytest.approx(10 / 25)
    assert feats.ipv6_ratio == pytest.approx(5 / 25)
    assert feats.arp_ratio == pytest.approx(4 / 25)
    assert feats.llc_ratio == pytest.approx(3 / 25)
    assert feats.other_protocol_ratio == pytest.approx(3 / 25)
    assert (
        feats.ipv4_ratio
        + feats.ipv6_ratio
        + feats.arp_ratio
        + feats.llc_ratio
        + feats.other_protocol_ratio
    ) == pytest.approx(1.0)


def test_transport_and_tcp_flags() -> None:
    packets = [
        _pkt(0, 0.0, is_tcp=True, syn=True),
        _pkt(1, 0.1, is_tcp=True, syn=True, ack=True),
        _pkt(2, 0.2, is_tcp=True, ack=True),
        _pkt(3, 0.3, is_tcp=True, fin=True, ack=True),
        _pkt(4, 0.4, is_tcp=True, rst=True),
        _pkt(5, 0.5, is_tcp=True, psh=True, ack=True),
        _pkt(6, 0.6, is_tcp=True, urg=True),
        _pkt(7, 0.7, is_tcp=False, is_udp=True, src_port=53, dst_port=53),
        _pkt(8, 0.8, is_tcp=False, is_icmp=True, src_port=None, dst_port=None),
        _pkt(
            9,
            0.9,
            is_ipv4=False,
            is_ipv6=True,
            is_tcp=False,
            is_icmpv6=True,
            src_ip="2001:db8::1",
            dst_ip="2001:db8::2",
            src_port=None,
            dst_port=None,
        ),
        _pkt(10, 1.0, is_tcp=False, is_igmp=True, src_port=None, dst_port=None),
    ] + [_pkt(11 + i, 1.1 + 0.01 * i, is_tcp=True, ack=True) for i in range(14)]
    assert len(packets) == 25
    feats = extract_features(next(iter_windows(packets)))
    assert feats.tcp_ratio == pytest.approx(21 / 25)
    assert feats.udp_ratio == pytest.approx(1 / 25)
    assert feats.icmp_ratio == pytest.approx(1 / 25)
    assert feats.icmpv6_ratio == pytest.approx(1 / 25)
    assert feats.igmp_ratio == pytest.approx(1 / 25)
    # SYN on packets 0 and 1 among 21 TCP
    assert feats.tcp_syn_ratio == pytest.approx(2 / 21)


def test_no_tcp_flag_ratios_zero() -> None:
    packets = [
        _pkt(
            i,
            float(i),
            is_tcp=False,
            is_udp=True,
            src_port=1000 + i,
            dst_port=53,
        )
        for i in range(25)
    ]
    feats = extract_features(next(iter_windows(packets)))
    assert feats.tcp_ratio == 0.0
    assert feats.tcp_syn_ratio == 0.0
    assert feats.tcp_ack_ratio == 0.0
    assert feats.tcp_fin_ratio == 0.0
    assert feats.tcp_rst_ratio == 0.0
    assert feats.tcp_psh_ratio == 0.0
    assert feats.tcp_urg_ratio == 0.0


def test_endpoint_diversity() -> None:
    packets = [
        _pkt(0, 0.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=80),
        _pkt(1, 0.1, src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1),
        _pkt(2, 0.2, src_ip="10.0.0.1", dst_ip="10.0.0.1", src_port=1, dst_port=1),
        _pkt(
            3,
            0.3,
            is_ipv4=False,
            is_ipv6=True,
            src_ip="2001:db8::1",
            dst_ip="2001:db8::2",
            src_port=443,
            dst_port=443,
        ),
        _pkt(
            4,
            0.4,
            is_tcp=False,
            is_arp=True,
            is_ipv4=False,
            src_ip=None,
            dst_ip=None,
            src_port=None,
            dst_port=None,
        ),
        ] + [
            _pkt(
                5 + i,
                0.5 + 0.01 * i,
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                src_port=1,
                dst_port=80,
            )
            for i in range(20)
        ]
    feats = extract_features(next(iter_windows(packets)))
    assert feats.unique_ip_count == 4  # 10.0.0.1, 10.0.0.2, 2001:db8::1, ::2
    assert feats.unique_port_count == 3  # 1, 80, 443


def test_malformed_unsupported_partial_retained() -> None:
    for status in (
        ParseStatus.UNSUPPORTED,
        ParseStatus.PARTIAL,
        ParseStatus.MALFORMED,
    ):
        packets = _steady(24)
        packets.append(_pkt(24, packets[-1].timestamp + 0.01, status=status))
        windows = list(iter_windows(packets, frozen_window_policy()))
        assert len(windows) == 1
        assert windows[0].packets[-1].parse_status == status


def test_error_aborts_not_is_failure() -> None:
    packets = _steady(10) + [
        _pkt(10, 10.0, status=ParseStatus.ERROR, detail="boom")
    ]
    with pytest.raises(FeatureExtractionError, match="ParseStatus.ERROR"):
        list(iter_windows(packets, frozen_window_policy()))


def test_determinism() -> None:
    packets = _steady(50)
    a = [
        (w.segment_index, w.window_index, w.packet_index_start, extract_features(w).to_ordered_values())
        for w in iter_windows(packets, frozen_window_policy())
    ]
    b = [
        (w.segment_index, w.window_index, w.packet_index_start, extract_features(w).to_ordered_values())
        for w in iter_windows(packets, frozen_window_policy())
    ]
    assert a == b


def test_ordered_feature_contract() -> None:
    feats = extract_features(next(iter_windows(_steady(25))))
    assert list(feats.to_feature_dict().keys()) == list(V1_FEATURE_NAMES)
    assert len(feats.to_ordered_values()) == 27


def test_production_parity_via_pcap(tmp_path: Path) -> None:
    """Path A (PCAP→iter_packets→windows→features) == Path B (same records streamed)."""
    frames: list[tuple[float, bytes]] = []
    # normal increasing
    for i in range(20):
        frames.append((1.0 + 0.01 * i, eth_ip_tcp(flags=dpkt.tcp.TH_SYN)))
    # duplicate
    frames.append((frames[-1][0], eth_ip_udp()))
    # tiny negative jitter
    frames.append((frames[-1][0] - 1e-6, eth_ip_icmp()))
    # >5s gap then fill to windows
    t = frames[-1][0] + 5.01
    frames.append((t, eth_arp()))
    frames.append((t + 0.01, eth_ieee8023_llc()))
    frames.append((t + 0.02, eth_lldp()))
    frames.append((t + 0.03, eth_ipv6_tcp()))
    frames.append((t + 0.04, eth_ipv6_icmp()))
    frames.append((t + 0.05, eth_ip_igmp()))
    # <-1s jump then enough packets for a full window
    t2 = frames[-1][0] - 1.5
    frames.append((t2, eth_ip_tcp(flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH)))
    while len(frames) < 20 + 2 + 6 + 1 + 24:
        last = frames[-1][0]
        frames.append((last + 0.01, eth_ip_tcp(flags=dpkt.tcp.TH_ACK)))

    path = write_pcap(tmp_path / "parity.pcap", frames)
    records = list(iter_packets(path))

    path_a = [
        extract_features(w).to_ordered_values()
        for w in iter_windows(iter_packets(path), frozen_window_policy())
    ]
    path_b = [
        extract_features(w).to_ordered_values()
        for w in iter_windows(records, frozen_window_policy())
    ]
    assert path_a == path_b
    assert len(path_a) >= 1


def test_write_feature_schema(tmp_path: Path) -> None:
    out = write_feature_schema(tmp_path / "feature_schema.json")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "phase1c2_v1" in text
    assert "window_span_seconds" in text
    assert '"window_size": 25' in text


def test_max_windows_cap() -> None:
    packets = _steady(100)
    windows = list(
        iter_windows(packets, frozen_window_policy(), max_windows=2)
    )
    assert len(windows) == 2
    assert windows[0].window_index == 0
    assert windows[1].window_index == 1


def test_packets_not_shared_across_windows() -> None:
    windows = list(iter_windows(_steady(50), frozen_window_policy()))
    ids0 = {id(p) for p in windows[0].packets}
    ids1 = {id(p) for p in windows[1].packets}
    assert ids0.isdisjoint(ids1)
    assert windows[0].packet_index_end < windows[1].packet_index_start
