"""Tests for TRAIN per-group Parquet characterization."""

from __future__ import annotations

import csv
from pathlib import Path

import dpkt
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.characterize_dataset import (
    characterize_train_feature_groups,
)
from iot_pcap_pipeline.features.dataset import build_feature_dataset
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES, write_feature_schema


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _make_pcap(root: Path, rel: str, n_packets: int) -> Path:
    abs_path = root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        (1.0 + 0.01 * i, eth_ip_tcp(flags=dpkt.tcp.TH_ACK)) for i in range(n_packets)
    ]
    write_pcap(abs_path, frames)
    return abs_path


def test_group_characterization_smoke(tmp_path: Path) -> None:
    rel_a = "data/raw/a.pcap"
    rel_b = "data/raw/b.pcap"
    p_a = _make_pcap(tmp_path, rel_a, 50)
    p_b = _make_pcap(tmp_path, rel_b, 50)
    inv_cols = [
        "pcap_path",
        "filename",
        "dataset_scope",
        "source",
        "split",
        "binary_label",
        "attack_family",
        "attack_type",
        "profiling_type",
        "profiling_variant",
        "device",
        "capture_session",
        "file_size",
        "grouping_key",
        "unresolved_reason",
    ]
    inv = _write_csv(
        tmp_path / "inv.csv",
        [
            {
                "pcap_path": rel_a,
                "filename": "a.pcap",
                "dataset_scope": "wifi_mqtt",
                "source": "attacks",
                "split": "train",
                "binary_label": "BENIGN",
                "attack_family": "",
                "attack_type": "",
                "profiling_type": "",
                "profiling_variant": "",
                "device": "",
                "capture_session": "",
                "file_size": str(p_a.stat().st_size),
                "grouping_key": "x",
                "unresolved_reason": "",
            },
            {
                "pcap_path": rel_b,
                "filename": "b.pcap",
                "dataset_scope": "wifi_mqtt",
                "source": "attacks",
                "split": "train",
                "binary_label": "ATTACK",
                "attack_family": "Recon",
                "attack_type": "Port_Scan",
                "profiling_type": "",
                "profiling_variant": "",
                "device": "",
                "capture_session": "",
                "file_size": str(p_b.stat().st_size),
                "grouping_key": "y",
                "unresolved_reason": "",
            },
        ],
        inv_cols,
    )
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    build_feature_dataset(
        inventory_path=inv,
        output_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / ".work",
        manifest_path=tmp_path / "build_manifest.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=1,
        resume=False,
        pcap_paths=[rel_a, rel_b],
    )
    result = characterize_train_feature_groups(
        manifest_path=tmp_path / "build_manifest.csv",
        inventory_path=inv,
        group_summary_output=tmp_path / "groups.csv",
        pcap_diagnostics_output=tmp_path / "diag.csv",
        summary_json_output=tmp_path / "summary.json",
        project_root=tmp_path,
        percentile_sample_cap=1_000,
    )
    assert result.pcap_count == 2
    assert result.group_count >= 3  # all_train + publisher_benign + attack groups
    with result.group_summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {r["group"] for r in rows}
    assert "all_train" in groups
    assert "publisher_benign" in groups
    assert "attack_family_Recon" in groups
    assert "attack_type_Port_Scan" in groups
    assert len(rows) == result.group_count * len(V1_FEATURE_NAMES)
    for key in ("zero_count", "nonzero_count", "p01", "p50", "p95", "p99", "p99_9"):
        assert key in rows[0]
