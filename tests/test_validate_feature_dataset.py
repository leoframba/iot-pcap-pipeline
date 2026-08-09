"""Read-only feature-dataset validation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import dpkt
import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.dataset import (
    EXPECTED_TRAIN_PCAP_COUNT,
    build_feature_dataset,
)
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES, write_feature_schema
from iot_pcap_pipeline.features.validate_dataset import (
    validate_feature_dataset,
)
from iot_pcap_pipeline.paths import (
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
)
from iot_pcap_pipeline.windowing.policy import (
    BACKWARD_RESET_SECONDS,
    INACTIVITY_TIMEOUT_SECONDS,
    WINDOW_SIZE,
)


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


def _inventory_row(rel: str, size: int, label: str = "BENIGN") -> dict[str, str]:
    return {
        "pcap_path": rel,
        "filename": Path(rel).name,
        "dataset_scope": "wifi_mqtt",
        "source": "attacks",
        "split": "train",
        "binary_label": label,
        "attack_family": "",
        "attack_type": "",
        "profiling_type": "",
        "profiling_variant": "",
        "device": "",
        "capture_session": "",
        "file_size": str(size),
        "grouping_key": "test",
        "unresolved_reason": "",
    }


def test_validate_passes_on_consistent_mini_build(tmp_path: Path) -> None:
    rel = "data/raw/mini.pcap"
    pcap = _make_pcap(tmp_path, rel, n_packets=50)
    inv = _write_csv(
        tmp_path / "inv.csv",
        [_inventory_row(rel, pcap.stat().st_size)],
        list(_inventory_row(rel, 0).keys()),
    )
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    built = build_feature_dataset(
        inventory_path=inv,
        output_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / ".work",
        manifest_path=tmp_path / "build_manifest.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=1,
        resume=False,
        pcap_paths=[rel],
    )
    assert built.ok_count == 1
    job = built.results_by_path[rel]
    assert job.output_row_count is not None

    # Pad manifest to 85 identical logical rows would be wrong; override count
    # check by monkeypatching EXPECTED via writing a single-row validation with
    # a custom expected count is not supported. Instead patch module constant
    # through validate call after rewriting manifest? Simpler: patch EXPECTED.
    integrity = _write_csv(
        tmp_path / "integrity.csv",
        [
            {
                "pcap_path": rel,
                "packet_count": str(job.packets_processed),
                "split": "train",
            }
        ],
        ["pcap_path", "packet_count", "split"],
    )
    characterization = _write_csv(
        tmp_path / "windowing.csv",
        [
            {
                "pcap_path": rel,
                "window_size": str(WINDOW_SIZE),
                "inactivity_timeout_seconds": str(INACTIVITY_TIMEOUT_SECONDS),
                "backward_reset_seconds": str(BACKWARD_RESET_SECONDS),
                "full_window_count": str(job.output_row_count),
            }
        ],
        [
            "pcap_path",
            "window_size",
            "inactivity_timeout_seconds",
            "backward_reset_seconds",
            "full_window_count",
        ],
    )

    import iot_pcap_pipeline.features.validate_dataset as vd

    original = vd.EXPECTED_TRAIN_PCAP_COUNT
    vd.EXPECTED_TRAIN_PCAP_COUNT = 1
    try:
        result = validate_feature_dataset(
            manifest_path=tmp_path / "build_manifest.csv",
            integrity_path=integrity,
            characterization_path=characterization,
            schema_path=schema,
            summary_output=tmp_path / "summary.csv",
            constant_output=tmp_path / "constants.csv",
            complete_output=tmp_path / "complete.json",
            project_root=tmp_path,
        )
    finally:
        vd.EXPECTED_TRAIN_PCAP_COUNT = original

    assert result.passed
    assert result.complete_path is not None
    assert result.complete_path.is_file()
    payload = json.loads(result.complete_path.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "passed"
    assert payload["pcap_count"] == 1
    assert payload["total_feature_rows"] == job.output_row_count
    assert payload["feature_strategy_version"] == FEATURE_STRATEGY_VERSION
    assert payload["feature_build_strategy_version"] == FEATURE_BUILD_STRATEGY_VERSION
    assert result.summary_path is not None
    with result.summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == len(V1_FEATURE_NAMES)
    assert all(int(r["nonfinite_count"]) == 0 for r in summary)


def test_validate_fails_on_packet_count_mismatch(tmp_path: Path) -> None:
    rel = "data/raw/bad_packets.pcap"
    pcap = _make_pcap(tmp_path, rel, n_packets=40)
    inv = _write_csv(
        tmp_path / "inv.csv",
        [_inventory_row(rel, pcap.stat().st_size)],
        list(_inventory_row(rel, 0).keys()),
    )
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    built = build_feature_dataset(
        inventory_path=inv,
        output_dir=tmp_path / "train",
        checkpoint_dir=tmp_path / ".work",
        manifest_path=tmp_path / "build_manifest.csv",
        schema_path=schema,
        project_root=tmp_path,
        workers=1,
        resume=False,
        pcap_paths=[rel],
    )
    job = built.results_by_path[rel]
    integrity = _write_csv(
        tmp_path / "integrity.csv",
        [{"pcap_path": rel, "packet_count": str(int(job.packets_processed or 0) + 1)}],
        ["pcap_path", "packet_count"],
    )
    characterization = _write_csv(
        tmp_path / "windowing.csv",
        [
            {
                "pcap_path": rel,
                "window_size": str(WINDOW_SIZE),
                "inactivity_timeout_seconds": str(INACTIVITY_TIMEOUT_SECONDS),
                "backward_reset_seconds": str(BACKWARD_RESET_SECONDS),
                "full_window_count": str(job.output_row_count),
            }
        ],
        [
            "pcap_path",
            "window_size",
            "inactivity_timeout_seconds",
            "backward_reset_seconds",
            "full_window_count",
        ],
    )

    import iot_pcap_pipeline.features.validate_dataset as vd

    original = vd.EXPECTED_TRAIN_PCAP_COUNT
    vd.EXPECTED_TRAIN_PCAP_COUNT = 1
    try:
        result = validate_feature_dataset(
            manifest_path=tmp_path / "build_manifest.csv",
            integrity_path=integrity,
            characterization_path=characterization,
            schema_path=schema,
            summary_output=tmp_path / "summary.csv",
            constant_output=tmp_path / "constants.csv",
            complete_output=tmp_path / "complete.json",
            project_root=tmp_path,
        )
    finally:
        vd.EXPECTED_TRAIN_PCAP_COUNT = original

    assert not result.passed
    assert result.complete_path is None
    assert not (tmp_path / "complete.json").exists()
    assert any(i.code == "packet_count_mismatch" for i in result.issues)


def test_validate_rejects_non_train_split() -> None:
    from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

    with pytest.raises(FeatureExtractionError, match="train only"):
        validate_feature_dataset(split="test")  # type: ignore[arg-type]


def test_expected_train_count_constant() -> None:
    assert EXPECTED_TRAIN_PCAP_COUNT == 85
