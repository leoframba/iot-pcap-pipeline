"""V2M M1: TCP payload exposure + V1 27-feature regression."""

from __future__ import annotations

from pathlib import Path

import dpkt
from pcap_synth import eth_arp, eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB, decode_frame
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import frozen_window_policy
from iot_pcap_pipeline.windowing.stream import iter_windows

# Golden captured against the same fixture used in ARP V1 regression, after
# tcp_payload exposure (V1 extractor ignores payload).
_V1_TCP_PAYLOAD_GOLDEN = (
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


def test_tcp_payload_retained_internally() -> None:
    payload = b"\x10\x0ahelloMQTT"
    buf = eth_ip_tcp(
        flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
        dport=1883,
        data=payload,
    )
    rec = decode_frame(buf, packet_index=0, timestamp=1.0, linktype=DLT_EN10MB)
    assert rec.is_tcp
    assert rec.tcp_payload == payload
    d = rec.to_dict()
    assert "tcp_payload" not in d
    assert d["tcp_payload_len"] == len(payload)


def test_empty_tcp_payload_is_empty_bytes() -> None:
    rec = decode_frame(
        eth_ip_tcp(flags=dpkt.tcp.TH_SYN),
        packet_index=0,
        timestamp=1.0,
        linktype=DLT_EN10MB,
    )
    assert rec.is_tcp
    assert rec.tcp_payload == b""


def test_non_tcp_has_no_tcp_payload() -> None:
    rec = decode_frame(
        eth_arp(),
        packet_index=0,
        timestamp=1.0,
        linktype=DLT_EN10MB,
    )
    assert not rec.is_tcp
    assert rec.tcp_payload is None


def test_v1_27_feature_vector_unchanged_with_tcp_payload(tmp_path: Path) -> None:
    """tcp_payload exposure must not change V1 outputs for the fixed ARP/TCP fixture."""
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
    assert all(r.tcp_payload == b"" for r in records if r.is_tcp)
    window = next(iter_windows(records, frozen_window_policy()))
    feats = extract_features(window)
    assert list(feats.to_feature_dict().keys()) == list(V1_FEATURE_NAMES)
    assert feats.to_ordered_values() == _V1_TCP_PAYLOAD_GOLDEN
