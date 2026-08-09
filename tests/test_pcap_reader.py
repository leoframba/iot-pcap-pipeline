"""Synthetic tests for the Phase 1B.1 PCAP reader/decoder."""

from __future__ import annotations

import socket
from pathlib import Path

import dpkt
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


def test_ieee8023_llc_not_unsupported_ethertype(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "llc.pcap", [(1.0, eth_ieee8023_llc(length=6))])
    record = next(iter_packets(path))
    assert record.parse_status == ParseStatus.OK
    assert record.is_llc is True
    assert record.is_ethernet_ii is False
    assert record.ethernet_type_or_length == 0x0006
    assert record.ethertype is None
    assert record.protocol_name == "llc"
    assert record.extra.get("llc_dsap") == 0
    assert record.extra.get("llc_ssap") == 1
    assert record.is_failure is False


def test_lldp_is_unsupported_not_failure(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "lldp.pcap", [(1.0, eth_lldp())])
    record = next(iter_packets(path))
    assert record.parse_status == ParseStatus.UNSUPPORTED
    assert record.protocol_name == "lldp"
    assert record.is_failure is False
    stats = summarize_pcap(path)
    assert stats.packets_unsupported == 1
    assert stats.packets_failed == 0


def test_malformed_ethernet_short_frame(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "short.pcap", [(1.0, b"\x00" * 8)])
    record = next(iter_packets(path))
    assert record.timestamp == 1.0
    assert record.frame_length == 8
    assert record.parse_status == ParseStatus.MALFORMED
    assert record.src_ip is None
    assert record.is_tcp is False
    assert record.is_failure is True


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
    assert record.parse_status == ParseStatus.UNSUPPORTED
    assert record.ethertype == 0x88B5
    assert record.frame_length > 0
    assert record.is_failure is False


def test_unsupported_linktype(tmp_path: Path) -> None:
    # DLT_LINUX_SLL = 113
    path = write_pcap(tmp_path / "sll.pcap", [(1.0, b"\x00" * 32)], linktype=113)
    record = next(iter_packets(path))
    assert record.parse_status == ParseStatus.UNSUPPORTED
    assert record.linktype == 113
    assert "unsupported linktype" in (record.parse_detail or "")
    assert record.is_failure is False


def test_ipv6_hop_by_hop_uses_final_protocol(tmp_path: Path) -> None:
    """Hop-by-Hop next-header 0 must not be reported as ip_proto_0."""
    # Build IPv6 + Hop-by-Hop + ICMPv6 similar to Benign_test samples.
    icmp6 = dpkt.icmp6.ICMP6(type=143, code=0, data=b"\x00" * 8)
    # Hop-by-hop options header: nxt=ICMPv6, hdrlen=0 → 8 bytes total
    hbh = b"\x3a\x00\x05\x02\x00\x00\x01\x00"
    ip6 = dpkt.ip6.IP6(
        src=socket.inet_pton(socket.AF_INET6, "fe80::1"),
        dst=socket.inet_pton(socket.AF_INET6, "ff02::16"),
        nxt=dpkt.ip.IP_PROTO_HOPOPTS,
        data=hbh + bytes(icmp6),
    )
    ip6.plen = len(hbh) + len(icmp6)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\x33\x33\x00\x00\x00\x16",
        src=b"\x74\x78\x27\x81\xc6\x6f",
        type=dpkt.ethernet.ETH_TYPE_IP6,
        data=ip6,
    )
    path = write_pcap(tmp_path / "hbh.pcap", [(1.0, bytes(eth))])
    record = next(iter_packets(path))
    assert record.is_ipv6
    assert record.ip_protocol == dpkt.ip.IP_PROTO_ICMP6
    assert record.is_icmpv6
    assert record.protocol_name == "icmpv6"
    assert record.parse_status == ParseStatus.OK
    assert record.extra.get("ipv6_first_next_header") == dpkt.ip.IP_PROTO_HOPOPTS


def test_max_packets_and_summary(tmp_path: Path) -> None:
    packets = [(float(i), eth_ip_tcp()) for i in range(5)]
    path = write_pcap(tmp_path / "many.pcap", packets)
    records = list(iter_packets(path, max_packets=3))
    assert len(records) == 3
    stats = summarize_pcap(path, max_packets=3)
    assert stats.packets_total == 3
    assert stats.tcp == 3
    assert stats.by_parse_status["ok"] == 3
    assert stats.packets_failed == 0


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
