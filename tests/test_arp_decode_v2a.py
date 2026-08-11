"""V2A A1: ARP identity extras + V1 27-feature regression."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import dpkt
from pcap_synth import eth_arp, eth_arp_truncated, eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB, _decode_arp, decode_frame
from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import frozen_window_policy
from iot_pcap_pipeline.windowing.stream import iter_windows

# Golden V1 27-feature ordered values for the fixed fixture in
# test_v1_27_feature_vector_unchanged_with_arp_decode_extras. Captured before
# A1 landed; ARP extras must not change extractor outputs.
_V1_ARP_MIXED_GOLDEN = (
    0.24,
    0.01,
    7.795969021526131e-17,
    0.010000000000000009,
    0.010000000000000009,
    51.6,
    4.8,
    42.0,
    54.0,
    0.8,
    0.0,
    0.2,
    0.0,
    0.0,
    0.8,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0,
    2.0,
)


def _decode(buf: bytes) -> PacketRecord:
    return decode_frame(buf, packet_index=0, timestamp=1.0, linktype=DLT_EN10MB)


def _base_record(**extra: object) -> PacketRecord:
    return PacketRecord(
        packet_index=0,
        timestamp=1.0,
        frame_length=60,
        linktype=DLT_EN10MB,
        parse_status=ParseStatus.OK,
        extra=dict(extra) if extra else {},
    )


def test_normal_arp_request() -> None:
    rec = _decode(eth_arp(spa="10.0.0.1", tpa="10.0.0.2", op=dpkt.arp.ARP_OP_REQUEST))
    assert rec.is_arp
    assert rec.src_ip == "10.0.0.1"
    assert rec.dst_ip == "10.0.0.2"
    assert rec.extra["arp_op"] == dpkt.arp.ARP_OP_REQUEST
    assert rec.extra["arp_sha"] == "11:22:33:44:55:66"
    assert rec.extra["arp_tha"] == "00:00:00:00:00:00"
    assert rec.extra["src_mac"] == "11:22:33:44:55:66"
    assert rec.extra["dst_mac"] == "ff:ff:ff:ff:ff:ff"


def test_normal_arp_reply() -> None:
    rec = _decode(
        eth_arp(
            spa="10.0.0.2",
            tpa="10.0.0.1",
            op=dpkt.arp.ARP_OP_REPLY,
            sha="aa:bb:cc:dd:ee:ff",
            tha="11:22:33:44:55:66",
            eth_dst="11:22:33:44:55:66",
        )
    )
    assert rec.is_arp
    assert rec.extra["arp_op"] == dpkt.arp.ARP_OP_REPLY
    assert rec.extra["arp_sha"] == "aa:bb:cc:dd:ee:ff"
    assert rec.extra["arp_tha"] == "11:22:33:44:55:66"
    assert rec.src_ip == "10.0.0.2"
    assert rec.dst_ip == "10.0.0.1"


def test_gratuitous_arp() -> None:
    # spa == tpa announces ownership of that IP.
    rec = _decode(
        eth_arp(
            spa="192.168.1.20",
            tpa="192.168.1.20",
            op=dpkt.arp.ARP_OP_REQUEST,
            sha="de:ad:be:ef:00:01",
        )
    )
    assert rec.is_arp
    assert rec.src_ip == rec.dst_ip == "192.168.1.20"
    assert rec.extra["arp_sha"] == "de:ad:be:ef:00:01"
    assert rec.extra["arp_op"] == dpkt.arp.ARP_OP_REQUEST


def test_arp_probe() -> None:
    # Probe: spa = 0.0.0.0, asking who-has tpa.
    rec = _decode(eth_arp(spa="0.0.0.0", tpa="10.0.0.50", op=dpkt.arp.ARP_OP_REQUEST))
    assert rec.is_arp
    assert rec.src_ip == "0.0.0.0"
    assert rec.dst_ip == "10.0.0.50"
    assert rec.extra["arp_op"] == dpkt.arp.ARP_OP_REQUEST
    assert rec.extra["arp_sha"] == "11:22:33:44:55:66"


def test_same_ip_same_mac_repeated(tmp_path: Path) -> None:
    frames = [
        (1.0 + 0.01 * i, eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="11:22:33:44:55:66"))
        for i in range(3)
    ]
    path = write_pcap(tmp_path / "same.pcap", frames)
    records = list(iter_packets(path))
    assert len(records) == 3
    triples = {(r.src_ip, r.extra["arp_sha"], r.extra["arp_op"]) for r in records}
    assert triples == {("10.0.0.7", "11:22:33:44:55:66", dpkt.arp.ARP_OP_REQUEST)}


def test_same_ip_two_macs(tmp_path: Path) -> None:
    frames = [
        (1.0, eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="11:22:33:44:55:66")),
        (1.01, eth_arp(spa="10.0.0.7", tpa="10.0.0.1", sha="aa:bb:cc:dd:ee:ff")),
    ]
    path = write_pcap(tmp_path / "conflict.pcap", frames)
    records = list(iter_packets(path))
    assert records[0].src_ip == records[1].src_ip == "10.0.0.7"
    assert records[0].extra["arp_sha"] == "11:22:33:44:55:66"
    assert records[1].extra["arp_sha"] == "aa:bb:cc:dd:ee:ff"
    assert records[0].extra["arp_sha"] != records[1].extra["arp_sha"]


def test_ethernet_src_ne_arp_sha() -> None:
    rec = _decode(
        eth_arp(
            spa="10.0.0.1",
            tpa="10.0.0.2",
            sha="11:22:33:44:55:66",
            eth_src="99:88:77:66:55:44",
        )
    )
    assert rec.extra["src_mac"] == "99:88:77:66:55:44"
    assert rec.extra["arp_sha"] == "11:22:33:44:55:66"
    assert rec.extra["src_mac"] != rec.extra["arp_sha"]


def test_truncated_hardware_addresses_set_none() -> None:
    # Internal parsing path: non-6-byte SHA/THA must not invent MAC strings.
    arp = SimpleNamespace(
        spa=b"\x0a\x00\x00\x01",
        tpa=b"\x0a\x00\x00\x02",
        op=dpkt.arp.ARP_OP_REQUEST,
        sha=b"\xaa\xbb\xcc",
        tha=b"\x00\x00",
    )
    rec = _decode_arp(_base_record(src_mac="11:22:33:44:55:66"), arp)  # type: ignore[arg-type]
    assert rec.is_arp
    assert rec.src_ip == "10.0.0.1"
    assert rec.extra["arp_op"] == dpkt.arp.ARP_OP_REQUEST
    assert rec.extra["arp_sha"] is None
    assert rec.extra["arp_tha"] is None


def test_wire_truncated_arp_is_partial_not_arp() -> None:
    rec = _decode(eth_arp_truncated())
    assert rec.parse_status == ParseStatus.PARTIAL
    assert rec.is_arp is False
    assert "arp_op" not in rec.extra
    assert "arp_sha" not in rec.extra
    assert "arp_tha" not in rec.extra


def test_non_arp_packet_has_no_arp_identity_extras() -> None:
    rec = _decode(eth_ip_tcp(flags=dpkt.tcp.TH_SYN))
    assert rec.is_arp is False
    assert "arp_op" not in rec.extra
    assert "arp_sha" not in rec.extra
    assert "arp_tha" not in rec.extra
    assert "src_mac" in rec.extra


def test_v1_27_feature_vector_unchanged_with_arp_decode_extras(tmp_path: Path) -> None:
    """A1 ARP extras must not change the frozen V1 27-feature extractor outputs."""
    frames: list[tuple[float, bytes]] = []
    t = 1.0
    for i in range(25):
        if i % 5 == 0:
            frames.append((t, eth_arp(spa="10.0.0.1", tpa="10.0.0.2")))
        else:
            frames.append((t, eth_ip_tcp(flags=dpkt.tcp.TH_SYN)))
        t += 0.01

    path = write_pcap(tmp_path / "v1_reg.pcap", frames)
    records = list(iter_packets(path))
    # Sanity: ARP packets now carry identity extras.
    arp_recs = [r for r in records if r.is_arp]
    assert arp_recs
    assert all("arp_op" in r.extra and "arp_sha" in r.extra for r in arp_recs)

    windows = list(iter_windows(records, frozen_window_policy()))
    assert len(windows) == 1
    feats = extract_features(windows[0])
    assert list(feats.to_feature_dict().keys()) == list(V1_FEATURE_NAMES)
    assert len(feats.to_ordered_values()) == 27
    assert feats.to_ordered_values() == _V1_ARP_MIXED_GOLDEN


def test_arp_feature_contract_pins() -> None:
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
    assert contract["strategy_version"] == "v2a1_arp_stateless"
    assert contract["windowing"]["window_size"] == 25
    assert contract["windowing"]["policy"] == "existing_frozen_v1"
    assert contract["extraction_constraints"]["state_across_windows"] is False
    assert contract["extraction_constraints"]["raw_mac_model_features"] is False
    assert contract["extraction_constraints"]["label_dependent_extraction"] is False
    assert contract["data_access"]["development_data"] == "FIT only"
    assert contract["data_access"]["v1_final_test_access"] is False
