"""V2A A3: hard ARP semantic feature tests on synthetic 25-packet windows."""

from __future__ import annotations

import dpkt
from pcap_synth import eth_arp, eth_ip_tcp

from iot_pcap_pipeline.features.arp_v2 import extract_arp_semantic_features
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB, decode_frame
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.window import PacketWindow

MAC_A = "aa:aa:aa:aa:aa:aa"
MAC_B = "bb:bb:bb:bb:bb:bb"


def _window_from_bufs(bufs: list[bytes]) -> PacketWindow:
    assert len(bufs) == WINDOW_SIZE
    packets = tuple(
        decode_frame(buf, packet_index=i, timestamp=1.0 + 0.01 * i, linktype=DLT_EN10MB)
        for i, buf in enumerate(bufs)
    )
    return PacketWindow(
        segment_index=0,
        window_index=0,
        packet_index_start=0,
        packet_index_end=WINDOW_SIZE - 1,
        packets=packets,
    )


def _pad(arp_bufs: list[bytes]) -> PacketWindow:
    bufs = list(arp_bufs)
    while len(bufs) < WINDOW_SIZE:
        bufs.append(eth_ip_tcp(flags=dpkt.tcp.TH_SYN))
    return _window_from_bufs(bufs)


def test_a3_normal_arp_no_conflict() -> None:
    """10.0.0.1→A, 10.0.0.1→A, 10.0.0.2→B → no conflict."""
    bufs = [
        eth_arp(spa="10.0.0.1", tpa="10.0.0.254", sha=MAC_A),
        eth_arp(spa="10.0.0.1", tpa="10.0.0.254", sha=MAC_A),
        eth_arp(spa="10.0.0.2", tpa="10.0.0.254", sha=MAC_B),
    ]
    feats = extract_arp_semantic_features(_pad(bufs))
    assert feats.arp_sender_ip_conflict_count == 0
    assert feats.arp_max_macs_per_sender_ip == 1
    assert feats.arp_mapping_change_count == 0
    assert feats.arp_sender_ip_conflict_ratio == 0.0
    assert feats.arp_unique_sender_ip_count == 2
    assert feats.arp_unique_sender_mac_count == 2


def test_a3_spoof_like_conflict() -> None:
    """10.0.0.1→A,A,B,B → one conflict IP, max_macs=2, one novel MAC claim."""
    bufs = [
        eth_arp(spa="10.0.0.1", tpa="10.0.0.254", sha=MAC_A),
        eth_arp(spa="10.0.0.1", tpa="10.0.0.254", sha=MAC_A),
        eth_arp(spa="10.0.0.1", tpa="10.0.0.254", sha=MAC_B),
        eth_arp(spa="10.0.0.1", tpa="10.0.0.254", sha=MAC_B),
    ]
    feats = extract_arp_semantic_features(_pad(bufs))
    assert feats.arp_sender_ip_conflict_count == 1
    assert feats.arp_max_macs_per_sender_ip == 2
    assert feats.arp_mapping_change_count == 1
    assert feats.arp_sender_ip_conflict_ratio == 1.0
    assert feats.arp_unique_sender_ip_count == 1
    assert feats.arp_unique_sender_mac_count == 2


def test_a3_multiple_legitimate_arp_probes_not_conflicts() -> None:
    """0.0.0.0→A and 0.0.0.0→B are probes, not IP ownership conflicts."""
    bufs = [
        eth_arp(spa="0.0.0.0", tpa="10.0.0.10", sha=MAC_A),
        eth_arp(spa="0.0.0.0", tpa="10.0.0.20", sha=MAC_B),
    ]
    feats = extract_arp_semantic_features(_pad(bufs))
    assert feats.arp_probe_ratio > 0.0
    assert feats.arp_probe_ratio == 1.0
    assert feats.arp_sender_ip_conflict_count == 0
    assert feats.arp_mapping_change_count == 0
    assert feats.arp_sender_ip_conflict_ratio == 0.0
    assert feats.arp_max_macs_per_sender_ip == 0
    assert feats.arp_unique_sender_ip_count == 0
    assert feats.arp_unique_sender_mac_count == 0
