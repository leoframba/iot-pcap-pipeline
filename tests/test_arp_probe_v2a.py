"""Unit tests for V2A A4 FIT ARP probe selection (no full corpus run)."""

from __future__ import annotations

from pathlib import Path

import dpkt
import pytest
from pcap_synth import eth_arp, eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.arp_v2_probe import (
    PROBE_GROUP_PROFILING_BENIGN,
    PROBE_GROUP_PUBLISHER_BENIGN,
    PROBE_GROUP_SPOOFING,
    load_arp_probe_targets,
    probe_group_for_row,
    run_arp_fit_probe,
)
from iot_pcap_pipeline.modeling.baselines.data import reject_test_path
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError


def test_probe_group_mapping() -> None:
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "attack_family": "Spoofing",
                "binary_label": "ATTACK",
                "group_kind": "spoofing",
            }
        )
        == PROBE_GROUP_SPOOFING
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "binary_label": "BENIGN",
                "benign_category": "publisher_benign",
                "group_kind": "publisher_benign",
            }
        )
        == PROBE_GROUP_PUBLISHER_BENIGN
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "binary_label": "BENIGN",
                "benign_category": "profiling_interaction",
                "group_kind": "profiling_device",
            }
        )
        == PROBE_GROUP_PROFILING_BENIGN
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "fit",
                "binary_label": "ATTACK",
                "attack_family": "MQTT",
            }
        )
        is None
    )
    assert (
        probe_group_for_row(
            {
                "modeling_split": "validation",
                "attack_family": "Spoofing",
                "group_kind": "spoofing",
            }
        )
        is None
    )


@pytest.mark.corpus
def test_load_arp_probe_targets_fit_only() -> None:
    targets = load_arp_probe_targets()
    assert len(targets) == 26  # 1 spoof + 25 benign
    groups = {t["probe_group"] for t in targets}
    assert groups == {
        PROBE_GROUP_SPOOFING,
        PROBE_GROUP_PUBLISHER_BENIGN,
        PROBE_GROUP_PROFILING_BENIGN,
    }
    assert sum(1 for t in targets if t["probe_group"] == PROBE_GROUP_SPOOFING) == 1
    assert (
        sum(1 for t in targets if t["probe_group"] == PROBE_GROUP_PUBLISHER_BENIGN) == 1
    )
    assert (
        sum(1 for t in targets if t["probe_group"] == PROBE_GROUP_PROFILING_BENIGN) == 24
    )
    for t in targets:
        reject_test_path(t["pcap_path"])
        assert "/test/" not in t["pcap_path"].lower()
        assert t["modeling_split"] == "fit"


def test_reject_test_path_blocks_spoof_test() -> None:
    with pytest.raises(FeatureExtractionError, match="TEST path rejected"):
        reject_test_path(
            "data/raw/WiFI_and_MQTT/attacks/pcap/test/ARP_Spoofing_test.pcap"
        )


def test_run_arp_fit_probe_smoke(tmp_path: Path) -> None:
    """Tiny synthetic FIT manifest → probe artifacts (capped windows)."""
    spoof = tmp_path / "ARP_Spoofing_train.pcap"
    benign = tmp_path / "Benign_train.pcap"
    frames_s = [
        (1.0 + 0.01 * i, eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha="aa:aa:aa:aa:aa:aa"))
        if i % 2 == 0
        else (
            1.0 + 0.01 * i,
            eth_arp(spa="10.0.0.1", tpa="10.0.0.2", sha="bb:bb:bb:bb:bb:bb"),
        )
        for i in range(50)
    ]
    frames_b = [
        (1.0 + 0.01 * i, eth_ip_tcp(flags=dpkt.tcp.TH_SYN)) for i in range(50)
    ]
    write_pcap(spoof, frames_s)
    write_pcap(benign, frames_b)

    manifest = tmp_path / "modeling_split_manifest.csv"
    manifest.write_text(
        "pcap_path,pcap_id,modeling_group_key,modeling_split,binary_label,"
        "attack_family,attack_type,profiling_type,device,window_count,"
        "feature_parquet_path,selection_reason,benign_category,group_kind\n"
        f"{spoof.name},spoof1,Spoofing|ARP_Spoofing,fit,ATTACK,Spoofing,ARP_Spoofing,"
        ",,2,,,spoofing\n"
        f"{benign.name},benign1,benign|publisher|Benign_train,fit,BENIGN,,,,,2,,,"
        "publisher_benign,publisher_benign\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    payload = run_arp_fit_probe(
        split_manifest_path=manifest,
        output_dir=out,
        project_root=tmp_path,
        max_windows_per_pcap=2,
    )
    assert payload["status"] == "complete"
    assert (out / "arp_feature_summary.csv").is_file()
    assert (out / "arp_feature_by_pcap.csv").is_file()
    assert (out / "arp_feature_nonzero_rates.csv").is_file()
    assert (out / "arp_vs_arp_ratio.csv").is_file()
    assert (out / "arp_probe_complete.json").is_file()
    assert payload["data_access"]["v1_final_test_access"] is False
