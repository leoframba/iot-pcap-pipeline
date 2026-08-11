"""V2A A6: whole-PCAP ARP stateful feasibility probe tests."""

from __future__ import annotations

from pathlib import Path

import dpkt
from pcap_synth import eth_arp, eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.arp_v2_stateful_probe import (
    analyze_pcap_stateful_arp,
    run_arp_stateful_feasibility_probe,
)


def test_stateful_conflict_across_long_gap(tmp_path: Path) -> None:
    """Conflict spanning >>25 packets is visible whole-PCAP but would miss windows."""
    mac_a = "aa:aa:aa:aa:aa:aa"
    mac_b = "bb:bb:bb:bb:bb:bb"
    frames: list[tuple[float, bytes]] = []
    t = 1.0
    frames.append((t, eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=mac_a)))
    t += 0.01
    for _ in range(100):
        frames.append((t, eth_ip_tcp(flags=dpkt.tcp.TH_SYN)))
        t += 0.01
    frames.append((t, eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=mac_b)))

    path = write_pcap(tmp_path / "gap.pcap", frames)
    stats = analyze_pcap_stateful_arp(path)
    assert stats.valid_identity_obs == 2
    assert stats.conflict_ip_count == 1
    assert stats.conflict_obs_count == 2
    assert stats.conflict_obs_ratio == 1.0
    assert stats.novel_mac_claim_count == 1
    assert stats.mapping_transition_count == 1
    assert stats.first_conflict_event_count == 1
    assert stats.packet_distance_samples == (101.0,)
    assert abs(stats.time_distance_samples[0] - 1.01) < 1e-9


def test_stateful_conflict_packet_and_time_distance(tmp_path: Path) -> None:
    mac_a = "aa:aa:aa:aa:aa:aa"
    mac_b = "bb:bb:bb:bb:bb:bb"
    frames = [
        (10.0, eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=mac_a)),
        (10.5, eth_ip_tcp()),
        (11.0, eth_arp(spa="10.0.0.5", tpa="10.0.0.1", sha=mac_b)),
    ]
    path = write_pcap(tmp_path / "dist.pcap", frames)
    stats = analyze_pcap_stateful_arp(path)
    assert stats.packet_distance_samples == (2.0,)
    assert abs(stats.time_distance_samples[0] - 1.0) < 1e-9


def test_stateful_probes_do_not_create_conflicts(tmp_path: Path) -> None:
    frames = [
        (1.0, eth_arp(spa="0.0.0.0", tpa="10.0.0.1", sha="aa:aa:aa:aa:aa:aa")),
        (1.1, eth_arp(spa="0.0.0.0", tpa="10.0.0.2", sha="bb:bb:bb:bb:bb:bb")),
        (1.2, eth_arp(spa="10.0.0.9", tpa="10.0.0.1", sha="cc:cc:cc:cc:cc:cc")),
    ]
    path = write_pcap(tmp_path / "probe.pcap", frames)
    stats = analyze_pcap_stateful_arp(path)
    assert stats.valid_identity_obs == 1
    assert stats.conflict_ip_count == 0
    assert stats.conflict_obs_ratio == 0.0
    assert stats.novel_mac_claim_count == 0
    assert stats.mapping_transition_count == 0


def test_stateful_stable_mapping_no_conflict(tmp_path: Path) -> None:
    mac = "aa:aa:aa:aa:aa:aa"
    frames = [
        (1.0 + 0.01 * i, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha=mac))
        for i in range(20)
    ]
    path = write_pcap(tmp_path / "stable.pcap", frames)
    stats = analyze_pcap_stateful_arp(path)
    assert stats.valid_identity_obs == 20
    assert stats.conflict_ip_count == 0
    assert stats.conflict_obs_ratio == 0.0
    assert stats.novel_mac_claim_count == 0
    assert stats.mapping_transition_count == 0


def test_stateful_flip_flops_count_transitions(tmp_path: Path) -> None:
    aa = "aa:aa:aa:aa:aa:aa"
    bb = "bb:bb:bb:bb:bb:bb"
    frames = [
        (1.0, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha=aa)),
        (1.1, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha=bb)),
        (1.2, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha=aa)),
        (1.3, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha=bb)),
    ]
    path = write_pcap(tmp_path / "flip.pcap", frames)
    stats = analyze_pcap_stateful_arp(path)
    assert stats.conflict_ip_count == 1
    assert stats.novel_mac_claim_count == 1  # only BB is novel once
    assert stats.mapping_transition_count == 3  # A→B, B→A, A→B
    assert stats.first_conflict_event_count == 1
    assert stats.conflict_obs_ratio == 1.0


def test_run_stateful_probe_smoke(tmp_path: Path) -> None:
    spoof = tmp_path / "ARP_Spoofing_train.pcap"
    benign = tmp_path / "Benign_train.pcap"
    write_pcap(
        spoof,
        [
            (1.0, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha="aa:aa:aa:aa:aa:aa")),
            (2.0, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha="bb:bb:bb:bb:bb:bb")),
        ],
    )
    write_pcap(
        benign,
        [
            (1.0, eth_arp(spa="10.0.0.3", tpa="10.0.0.2", sha="cc:cc:cc:cc:cc:cc")),
            (2.0, eth_arp(spa="10.0.0.3", tpa="10.0.0.2", sha="cc:cc:cc:cc:cc:cc")),
        ],
    )
    manifest = tmp_path / "modeling_split_manifest.csv"
    manifest.write_text(
        "pcap_path,pcap_id,modeling_group_key,modeling_split,binary_label,"
        "attack_family,attack_type,profiling_type,device,window_count,"
        "feature_parquet_path,selection_reason,benign_category,group_kind\n"
        f"{spoof.name},spoof1,Spoofing|ARP_Spoofing,fit,ATTACK,Spoofing,ARP_Spoofing,"
        ",,1,,,spoofing\n"
        f"{benign.name},benign1,benign|publisher|Benign_train,fit,BENIGN,,,,,1,,,"
        "publisher_benign,publisher_benign\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    payload = run_arp_stateful_feasibility_probe(
        split_manifest_path=manifest,
        output_dir=out,
        project_root=tmp_path,
    )
    assert payload["status"] == "complete"
    assert payload["data_access"]["v1_final_test_access"] is False
    assert (out / "arp_stateful_by_pcap.csv").is_file()
    assert (out / "arp_stateful_by_group.csv").is_file()
    assert (out / "arp_stateful_feasibility_complete.json").is_file()
    spoof_row = payload["group_summary"]["spoofing"]
    assert float(spoof_row["conflict_obs_ratio"]) == 1.0
