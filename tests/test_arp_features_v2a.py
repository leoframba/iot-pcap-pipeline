"""V2A A2: stateless ARP semantic feature extractor tests."""

from __future__ import annotations

from pathlib import Path

import dpkt
import pytest
from pcap_synth import eth_arp, eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.arp_v2 import (
    ARP_V2_FEATURE_NAMES,
    ArpSemanticFeatures,
    extract_arp_semantic_features,
)
from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB, decode_frame
from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE, frozen_window_policy
from iot_pcap_pipeline.windowing.stream import iter_windows
from iot_pcap_pipeline.windowing.window import PacketWindow


def _window_from_bufs(bufs: list[bytes]) -> PacketWindow:
    packets: list[PacketRecord] = []
    for i, buf in enumerate(bufs):
        packets.append(
            decode_frame(buf, packet_index=i, timestamp=1.0 + 0.01 * i, linktype=DLT_EN10MB)
        )
    return PacketWindow(
        segment_index=0,
        window_index=0,
        packet_index_start=0,
        packet_index_end=len(packets) - 1,
        packets=tuple(packets),
    )


def _pad_to_window(arp_bufs: list[bytes], *, n: int = 25) -> PacketWindow:
    bufs = list(arp_bufs)
    while len(bufs) < n:
        bufs.append(eth_ip_tcp(flags=dpkt.tcp.TH_SYN))
    assert len(bufs) == n
    return _window_from_bufs(bufs)


def test_rejects_partial_and_oversized_windows() -> None:
    with pytest.raises(ValueError, match="exactly 25 packets, got 24"):
        extract_arp_semantic_features(_pad_to_window([], n=24))
    feats = extract_arp_semantic_features(_pad_to_window([], n=25))
    assert feats.arp_request_ratio == 0.0
    with pytest.raises(ValueError, match="exactly 25 packets, got 26"):
        extract_arp_semantic_features(_pad_to_window([], n=26))
    assert WINDOW_SIZE == 25


def test_no_arp_returns_zeros() -> None:
    feats = extract_arp_semantic_features(_pad_to_window([]))
    assert feats == ArpSemanticFeatures(
        arp_request_ratio=0.0,
        arp_reply_ratio=0.0,
        arp_probe_ratio=0.0,
        arp_gratuitous_ratio=0.0,
        arp_sender_ip_conflict_count=0,
        arp_sender_ip_conflict_ratio=0.0,
        arp_max_macs_per_sender_ip=0,
        arp_mapping_change_count=0,
        arp_eth_src_sha_mismatch_ratio=0.0,
        arp_unique_sender_ip_count=0,
        arp_unique_sender_mac_count=0,
    )
    assert list(feats.to_feature_dict().keys()) == list(ARP_V2_FEATURE_NAMES)
    assert len(feats.to_ordered_values()) == 11


def test_basic_request_reply_probe_gratuitous_ratios() -> None:
    # 4 ARP: normal request, reply, probe, GARP request.
    # request includes normal + probe + GARP → 3/4; reply 1/4; probe 1/4; GARP 1/4.
    bufs = [
        eth_arp(spa="10.0.0.1", tpa="10.0.0.2", op=dpkt.arp.ARP_OP_REQUEST),
        eth_arp(
            spa="10.0.0.2",
            tpa="10.0.0.1",
            op=dpkt.arp.ARP_OP_REPLY,
            sha="aa:bb:cc:dd:ee:01",
            tha="11:22:33:44:55:66",
        ),
        eth_arp(spa="0.0.0.0", tpa="10.0.0.50", op=dpkt.arp.ARP_OP_REQUEST),
        eth_arp(
            spa="10.0.0.9",
            tpa="10.0.0.9",
            op=dpkt.arp.ARP_OP_REQUEST,
            sha="de:ad:be:ef:00:01",
        ),
    ]
    feats = extract_arp_semantic_features(_pad_to_window(bufs))
    assert feats.arp_request_ratio == 0.75
    assert feats.arp_reply_ratio == 0.25
    assert feats.arp_probe_ratio == 0.25
    assert feats.arp_gratuitous_ratio == 0.25


def test_probe_excluded_from_conflict_and_unique_ip() -> None:
    # Two probes for 0.0.0.0 with different SHAs must not invent a conflict.
    bufs = [
        eth_arp(spa="0.0.0.0", tpa="10.0.0.1", sha="11:22:33:44:55:66"),
        eth_arp(spa="0.0.0.0", tpa="10.0.0.2", sha="aa:bb:cc:dd:ee:ff"),
        eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="11:22:33:44:55:66"),
    ]
    feats = extract_arp_semantic_features(_pad_to_window(bufs))
    assert feats.arp_probe_ratio == 2 / 3
    assert feats.arp_sender_ip_conflict_count == 0
    assert feats.arp_sender_ip_conflict_ratio == 0.0
    assert feats.arp_unique_sender_ip_count == 1
    assert feats.arp_unique_sender_mac_count == 1
    assert feats.arp_max_macs_per_sender_ip == 1
    assert feats.arp_mapping_change_count == 0


def test_sender_ip_conflict_count_and_ratio() -> None:
    # Example from plan: .10→AA, .10→AA, .10→BB, .20→CC → conflict_count=1
    # Observations on conflicting IP: 3/4 → ratio 0.75
    aa = "aa:aa:aa:aa:aa:aa"
    bb = "bb:bb:bb:bb:bb:bb"
    cc = "cc:cc:cc:cc:cc:cc"
    bufs = [
        eth_arp(spa="192.168.1.10", tpa="192.168.1.1", sha=aa),
        eth_arp(spa="192.168.1.10", tpa="192.168.1.1", sha=aa),
        eth_arp(spa="192.168.1.10", tpa="192.168.1.1", sha=bb),
        eth_arp(spa="192.168.1.20", tpa="192.168.1.1", sha=cc),
    ]
    feats = extract_arp_semantic_features(_pad_to_window(bufs))
    assert feats.arp_sender_ip_conflict_count == 1
    assert feats.arp_sender_ip_conflict_ratio == 0.75
    assert feats.arp_max_macs_per_sender_ip == 2
    assert feats.arp_mapping_change_count == 1
    assert feats.arp_unique_sender_ip_count == 2
    assert feats.arp_unique_sender_mac_count == 3


def test_mapping_change_counts_within_window_order() -> None:
    # Novel additional MAC claims (not every transition): AA → BB → AA → CC
    # BB new → +1; AA already known → +0; CC new → +1. Final cardinality=3.
    aa = "aa:aa:aa:aa:aa:aa"
    bb = "bb:bb:bb:bb:bb:bb"
    cc = "cc:cc:cc:cc:cc:cc"
    bufs = [
        eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=aa),
        eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=bb),
        eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=aa),
        eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=cc),
    ]
    feats = extract_arp_semantic_features(_pad_to_window(bufs))
    assert feats.arp_max_macs_per_sender_ip == 3
    assert feats.arp_mapping_change_count == 2
    assert feats.arp_sender_ip_conflict_count == 1
    assert feats.arp_sender_ip_conflict_ratio == 1.0


def test_eth_src_sha_mismatch_ratio() -> None:
    bufs = [
        eth_arp(
            spa="10.0.0.1",
            tpa="10.0.0.2",
            sha="11:22:33:44:55:66",
            eth_src="11:22:33:44:55:66",
        ),
        eth_arp(
            spa="10.0.0.3",
            tpa="10.0.0.2",
            sha="11:22:33:44:55:66",
            eth_src="99:88:77:66:55:44",
        ),
    ]
    feats = extract_arp_semantic_features(_pad_to_window(bufs))
    assert feats.arp_eth_src_sha_mismatch_ratio == 0.5
    assert feats.arp_unique_sender_ip_count == 2
    assert feats.arp_unique_sender_mac_count == 1


def test_eth_src_sha_mismatch_skips_missing_src_mac() -> None:
    """Missing Ethernet src MAC is not in the mismatch denominator."""
    packets: list[PacketRecord] = []
    for i in range(WINDOW_SIZE):
        if i == 0:
            packets.append(
                PacketRecord(
                    packet_index=i,
                    timestamp=1.0 + 0.01 * i,
                    frame_length=42,
                    linktype=DLT_EN10MB,
                    parse_status=ParseStatus.OK,
                    is_arp=True,
                    src_ip="10.0.0.1",
                    dst_ip="10.0.0.2",
                    protocol_name="arp",
                    extra={
                        "arp_op": dpkt.arp.ARP_OP_REQUEST,
                        "arp_sha": "11:22:33:44:55:66",
                        "arp_tha": "00:00:00:00:00:00",
                        # no src_mac
                    },
                )
            )
        elif i == 1:
            packets.append(
                PacketRecord(
                    packet_index=i,
                    timestamp=1.0 + 0.01 * i,
                    frame_length=42,
                    linktype=DLT_EN10MB,
                    parse_status=ParseStatus.OK,
                    is_arp=True,
                    src_ip="10.0.0.3",
                    dst_ip="10.0.0.2",
                    protocol_name="arp",
                    extra={
                        "arp_op": dpkt.arp.ARP_OP_REQUEST,
                        "arp_sha": "11:22:33:44:55:66",
                        "arp_tha": "00:00:00:00:00:00",
                        "src_mac": "99:88:77:66:55:44",
                    },
                )
            )
        else:
            packets.append(
                decode_frame(
                    eth_ip_tcp(flags=dpkt.tcp.TH_SYN),
                    packet_index=i,
                    timestamp=1.0 + 0.01 * i,
                    linktype=DLT_EN10MB,
                )
            )
    window = PacketWindow(
        segment_index=0,
        window_index=0,
        packet_index_start=0,
        packet_index_end=WINDOW_SIZE - 1,
        packets=tuple(packets),
    )
    feats = extract_arp_semantic_features(window)
    # Only the second identity is mismatch-eligible → ratio 1.0 (not 0.5).
    assert feats.arp_eth_src_sha_mismatch_ratio == 1.0
    assert feats.arp_unique_sender_ip_count == 2


def test_same_ip_same_mac_no_conflict() -> None:
    bufs = [
        eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="11:22:33:44:55:66"),
        eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="11:22:33:44:55:66"),
        eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="11:22:33:44:55:66"),
    ]
    feats = extract_arp_semantic_features(_pad_to_window(bufs))
    assert feats.arp_sender_ip_conflict_count == 0
    assert feats.arp_sender_ip_conflict_ratio == 0.0
    assert feats.arp_max_macs_per_sender_ip == 1
    assert feats.arp_mapping_change_count == 0
    assert feats.arp_unique_sender_ip_count == 1
    assert feats.arp_unique_sender_mac_count == 1


def test_does_not_alter_v1_feature_vector(tmp_path: Path) -> None:
    frames = [
        (1.0 + 0.01 * i, eth_arp(spa="10.0.0.1", tpa="10.0.0.2") if i % 5 == 0 else eth_ip_tcp())
        for i in range(25)
    ]
    path = write_pcap(tmp_path / "mix.pcap", frames)
    records = list(iter_packets(path))
    window = next(iter_windows(records, frozen_window_policy()))
    v1 = extract_features(window)
    arp = extract_arp_semantic_features(window)
    assert len(v1.to_ordered_values()) == 27
    assert len(arp.to_ordered_values()) == 11
    # Distinct representations; ARP semantic fields are not on FeatureVector.
    assert not hasattr(v1, "arp_sender_ip_conflict_count")
    assert arp.arp_request_ratio > 0.0


def test_contract_lists_eleven_candidates() -> None:
    import json

    contract_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "experiments"
        / "v2_arp"
        / "phase_v2a1"
        / "arp_feature_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["candidate_features"]["feature_names"] == list(ARP_V2_FEATURE_NAMES)
    assert contract["candidate_features"]["feature_count"] == 11
    assert contract["candidate_features"]["required_window_size"] == 25
    assert "novel additional MAC" in contract["candidate_features"]["arp_mapping_change_count"]
    assert (
        contract["candidate_features"]["arp_eth_src_sha_mismatch_denominator"]
        == "valid ARP identity observations with a valid Ethernet source MAC"
    )
