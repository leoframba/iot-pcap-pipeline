"""Synthetic tests for the Phase 1B.1 PCAP reader/decoder."""

from __future__ import annotations

from pathlib import Path

import dpkt
from pcap_synth import (
    eth_arp,
    eth_ip_icmp,
    eth_ip_igmp,
    eth_ip_tcp,
    eth_ip_udp,
    eth_ipv6_icmp,
    eth_ipv6_tcp,
    eth_unknown_ethertype,
    eth_vlan_ip_udp,
    write_pcap,
)

from iot_pcap_pipeline.pcap.packet import ParseStatus
from iot_pcap_pipeline.pcap.reader import iter_packets, summarize_pcap


def test_supported_protocols_and_order(tmp_path: Path) -> None:
    packets = [
        (1.0, eth_ip_tcp(flags=dpkt.tcp.TH_SYN)),
        (1.5, eth_ip_udp()),
        (2.0, eth_ip_icmp()),
        (2.5, eth_arp()),
        (3.0, eth_ipv6_tcp()),
        (3.5, eth_ipv6_icmp()),
        (4.0, eth_ip_igmp()),
    ]
    path = write_pcap(tmp_path / "mixed.pcap", packets)
    records = list(iter_packets(path))

    assert [r.packet_index for r in records] == list(range(len(packets)))
    assert [r.timestamp for r in records] == [p[0] for p in packets]
    assert all(r.frame_length == len(buf) for (_, buf), r in zip(packets, records, strict=True))
    assert all(r.parse_status == ParseStatus.OK for r in records)

    tcp, udp, icmp, arp, tcp6, icmp6, igmp = records
    assert tcp.is_tcp and tcp.tcp_flag_syn and tcp.src_port == 12345
    assert udp.is_udp and udp.dst_port == 53
    assert icmp.is_icmp and icmp.is_ipv4
    assert arp.is_arp and arp.src_ip == "10.0.0.1"
    assert tcp6.is_tcp and tcp6.is_ipv6 and tcp6.src_ip == "2001:db8::1"
    assert icmp6.is_icmpv6 and icmp6.is_ipv6
    assert igmp.is_igmp


def test_vlan_frame(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "vlan.pcap", [(10.0, eth_vlan_ip_udp(vlan_id=42))])
    record = next(iter_packets(path))
    assert record.parse_status == ParseStatus.OK
    assert record.vlan_ids == (42,)
    assert record.is_udp
    assert record.ethertype == dpkt.ethernet.ETH_TYPE_IP


def test_malformed_ethernet_short_frame(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "short.pcap", [(1.0, b"\x00" * 8)])
    record = next(iter_packets(path))
    assert record.timestamp == 1.0
    assert record.frame_length == 8
    assert record.parse_status == ParseStatus.MALFORMED
    assert record.src_ip is None
    assert record.is_tcp is False


def test_malformed_ipv4_header(tmp_path: Path) -> None:
    # Valid Ethernet header + truncated IPv4 body.
    dest = b"\xaa\xbb\xcc\xdd\xee\xff"
    src = b"\x11\x22\x33\x44\x55\x66"
    ethertype = dpkt.ethernet.ETH_TYPE_IP.to_bytes(2, "big")
    buf = dest + src + ethertype + b"\x45\x00"  # incomplete IP header
    path = write_pcap(tmp_path / "bad_ip.pcap", [(2.0, buf)])
    record = next(iter_packets(path))
    assert record.timestamp == 2.0
    assert record.frame_length == len(buf)
    assert record.parse_status in {ParseStatus.PARTIAL, ParseStatus.MALFORMED}
    assert record.is_tcp is False


def test_unknown_ethertype(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "unknown.pcap", [(1.0, eth_unknown_ethertype())])
    record = next(iter_packets(path))
    assert record.parse_status == ParseStatus.UNSUPPORTED_PROTOCOL
    assert record.ethertype == 0x88B5
    assert record.frame_length > 0


def test_unsupported_linktype(tmp_path: Path) -> None:
    # DLT_LINUX_SLL = 113
    path = write_pcap(tmp_path / "sll.pcap", [(1.0, b"\x00" * 32)], linktype=113)
    record = next(iter_packets(path))
    assert record.parse_status == ParseStatus.UNSUPPORTED_LINKTYPE
    assert record.linktype == 113
    assert "unsupported linktype" in (record.parse_detail or "")


def test_max_packets_and_summary(tmp_path: Path) -> None:
    packets = [(float(i), eth_ip_tcp()) for i in range(5)]
    path = write_pcap(tmp_path / "many.pcap", packets)
    records = list(iter_packets(path, max_packets=3))
    assert len(records) == 3
    stats = summarize_pcap(path, max_packets=3)
    assert stats.packets_total == 3
    assert stats.tcp == 3
    assert stats.by_parse_status["ok"] == 3


def test_decoder_ignores_path_labels(tmp_path: Path) -> None:
    """Filename/label tokens must not influence decode fields."""
    labeled = tmp_path / "ATTACK_DDoS_UDP_train.pcap"
    path = write_pcap(labeled, [(1.0, eth_ip_udp())])
    record = next(iter_packets(path))
    assert record.is_udp
    assert record.parse_status == ParseStatus.OK
    # No label-derived fields exist on PacketRecord.
    assert not hasattr(record, "binary_label")
    assert not hasattr(record, "attack_family")
