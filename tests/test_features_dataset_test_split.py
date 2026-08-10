"""Phase 1C.3c TEST feature-dataset build / validate guards."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import dpkt
import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.dataset import (
    EXPECTED_TEST_PCAP_COUNT,
    EXPECTED_TRAIN_PCAP_COUNT,
    build_feature_dataset,
    require_train_build_complete,
    select_split_rows,
)
from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import write_feature_schema
from iot_pcap_pipeline.features.validate_dataset import validate_feature_dataset
from iot_pcap_pipeline.paths import (
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    WINDOWING_STRATEGY_VERSION,
)
from iot_pcap_pipeline.windowing.policy import (
    BACKWARD_RESET_SECONDS,
    INACTIVITY_TIMEOUT_SECONDS,
    WINDOW_SIZE,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError


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


def _inventory_row(
    rel: str,
    size: int,
    *,
    split: str,
    label: str = "BENIGN",
) -> dict[str, str]:
    return {
        "pcap_path": rel,
        "filename": Path(rel).name,
        "dataset_scope": "wifi_mqtt",
        "source": "attacks",
        "split": split,
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


def _train_complete_payload(schema_hash: str) -> dict:
    return {
        "validation_status": "passed",
        "split": "train",
        "pcap_count": EXPECTED_TRAIN_PCAP_COUNT,
        "feature_strategy_version": FEATURE_STRATEGY_VERSION,
        "feature_build_strategy_version": FEATURE_BUILD_STRATEGY_VERSION,
        "feature_schema_sha256": schema_hash,
        "windowing_strategy_version": WINDOWING_STRATEGY_VERSION,
        "windowing": {
            "window_size": WINDOW_SIZE,
            "inactivity_timeout_seconds": INACTIVITY_TIMEOUT_SECONDS,
            "backward_reset_seconds": BACKWARD_RESET_SECONDS,
        },
    }


def test_select_test_expects_29() -> None:
    inv = PROJECT_ROOT / "data" / "manifests" / "pcap_inventory.csv"
    if not inv.is_file():
        pytest.skip("inventory not present")
    rows = select_split_rows(inv, split="test", require_expected_count=True)
    assert len(rows) == EXPECTED_TEST_PCAP_COUNT
    assert all(r["split"] == "test" for r in rows)


def test_require_train_complete_missing(tmp_path: Path) -> None:
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    with pytest.raises(FeatureExtractionError, match="TRAIN completion marker"):
        require_train_build_complete(
            tmp_path / "missing.json",
            project_root=tmp_path,
            schema_path=schema,
        )


def test_test_build_refuses_without_train_marker(tmp_path: Path) -> None:
    rel = "data/raw/test_only.pcap"
    pcap = _make_pcap(tmp_path, rel, n_packets=40)
    inv = _write_csv(
        tmp_path / "inv.csv",
        [_inventory_row(rel, pcap.stat().st_size, split="test")],
        list(_inventory_row(rel, 0, split="test").keys()),
    )
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    with pytest.raises(FeatureExtractionError, match="TRAIN completion marker"):
        build_feature_dataset(
            split="test",
            inventory_path=inv,
            output_dir=tmp_path / "test",
            checkpoint_dir=tmp_path / ".work" / "test",
            manifest_path=tmp_path / "test_build_manifest.csv",
            schema_path=schema,
            project_root=tmp_path,
            workers=1,
            resume=False,
            pcap_paths=[rel],
        )


def test_test_build_and_validate_structural(tmp_path: Path) -> None:
    rel = "data/raw/heldout.pcap"
    pcap = _make_pcap(tmp_path, rel, n_packets=50)
    inv = _write_csv(
        tmp_path / "inv.csv",
        [_inventory_row(rel, pcap.stat().st_size, split="test", label="ATTACK")],
        list(_inventory_row(rel, 0, split="test").keys()),
    )
    schema = write_feature_schema(tmp_path / "feature_schema.json")
    schema_hash = feature_schema_sha256(schema)
    (tmp_path / "data" / "features" / "v1").mkdir(parents=True)
    marker = tmp_path / "data" / "features" / "v1" / "train_build_complete.json"
    marker.write_text(
        json.dumps(_train_complete_payload(schema_hash), indent=2) + "\n",
        encoding="utf-8",
    )

    built = build_feature_dataset(
        split="test",
        inventory_path=inv,
        output_dir=tmp_path / "test",
        checkpoint_dir=tmp_path / ".work" / "test",
        manifest_path=tmp_path / "test_build_manifest.csv",
        schema_path=schema,
        train_complete_path=marker,
        project_root=tmp_path,
        workers=1,
        resume=False,
        pcap_paths=[rel],
    )
    assert built.ok_count == 1
    assert built.split == "test"
    job = built.results_by_path[rel]
    assert job.output_row_count is not None
    assert not any((tmp_path / "train").glob("*.parquet")) if (tmp_path / "train").exists() else True
    assert str(built.output_dir).endswith("test")

    integrity = _write_csv(
        tmp_path / "integrity.csv",
        [
            {
                "pcap_path": rel,
                "packet_count": str(job.packets_processed),
                "split": "test",
            }
        ],
        ["pcap_path", "packet_count", "split"],
    )

    import iot_pcap_pipeline.features.validate_dataset as vd

    original = vd.EXPECTED_TEST_PCAP_COUNT
    vd.EXPECTED_TEST_PCAP_COUNT = 1
    try:
        result = validate_feature_dataset(
            split="test",
            manifest_path=tmp_path / "test_build_manifest.csv",
            integrity_path=integrity,
            schema_path=schema,
            complete_output=tmp_path / "test_build_complete.json",
            train_complete_path=marker,
            project_root=tmp_path,
        )
    finally:
        vd.EXPECTED_TEST_PCAP_COUNT = original

    assert result.passed
    assert result.train_contract_verified
    assert result.summary_path is None
    assert result.constant_path is None
    assert result.complete_path is not None
    payload = json.loads(result.complete_path.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "passed"
    assert payload["split"] == "test"
    assert payload["pcap_count"] == 1
    assert payload["train_contract_verified"] is True
    assert payload["feature_schema_sha256"] == schema_hash
