"""Phase 2B.1 TRAIN-fit view materialization tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from iot_pcap_pipeline.features.parquet import (
    StreamingFeatureParquetWriter,
    feature_parquet_arrow_schema,
    feature_schema_sha256,
)
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES, write_feature_schema
from iot_pcap_pipeline.modeling.sampling import reservoir_indices
from iot_pcap_pipeline.modeling.seeds import reservoir_seed_for_pcap
from iot_pcap_pipeline.modeling.view import (
    file_sha256,
    load_and_verify_training_view_contract,
    materialize_fit_shard,
    write_training_view_contract,
)
from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError


def test_reservoir_indices_deterministic_and_sized() -> None:
    a = reservoir_indices(1000, 50, seed=123)
    b = reservoir_indices(1000, 50, seed=123)
    c = reservoir_indices(1000, 50, seed=124)
    assert a == b
    assert a != c
    assert len(a) == 50
    assert a == sorted(a)
    assert len(set(a)) == 50
    assert a[0] >= 0 and a[-1] < 1000
    assert reservoir_indices(10, 10, seed=1) == list(range(10))
    assert reservoir_indices(10, 0, seed=1) == []


def test_reservoir_seed_matches_contract_formula() -> None:
    seed = reservoir_seed_for_pcap("ARP_Spoofing_train-96f95da2dd4aca68", base_seed=42)
    import hashlib

    expected = int(
        hashlib.sha256(b"phase2a_v1|42|ARP_Spoofing_train-96f95da2dd4aca68")
        .hexdigest()[:16],
        16,
    )
    assert seed == expected


def _write_tiny_feature_parquet(path: Path, *, n: int, pcap_id: str) -> None:
    schema = feature_parquet_arrow_schema()
    writer = StreamingFeatureParquetWriter(path, buffer_rows=32, schema=schema)
    for i in range(n):
        row: dict = {
            "pcap_id": pcap_id,
            "binary_label": "ATTACK",
            "segment_index": 0,
            "window_index": i,
            "packet_index_start": i * 25,
            "packet_index_end": i * 25 + 24,
        }
        for name in V1_FEATURE_NAMES:
            row[name] = 0.0
        row["unique_ip_count"] = 1
        row["unique_port_count"] = 1
        writer.append(row)
    writer.close()


def test_materialize_fit_shard_reservoir_and_resume(tmp_path: Path) -> None:
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    schema_sha = feature_schema_sha256(schema_path)

    pcap_id = "tiny-ddos-aaaaaaaaaaaaaaaa"
    source = tmp_path / "src" / f"{pcap_id}.parquet"
    source.parent.mkdir(parents=True)
    _write_tiny_feature_parquet(source, n=100, pcap_id=pcap_id)

    out = tmp_path / "fit" / f"{pcap_id}.parquet"
    ckpt = tmp_path / "ckpt" / f"{pcap_id}.json"
    row = {
        "pcap_id": pcap_id,
        "pcap_path": f"data/raw/{pcap_id}.pcap",
        "feature_parquet_path": str(source),
        "window_count": "100",
        "modeling_group_key": "DDoS|DDoS_ICMP",
        "binary_label": "ATTACK",
        "attack_family": "DDoS",
        "attack_type": "DDoS_ICMP",
        "benign_category": "",
    }
    seed = reservoir_seed_for_pcap(pcap_id, base_seed=42)
    expected_idx = reservoir_indices(100, 20, seed)

    first = materialize_fit_shard(
        row=row,
        allocated_k=20,
        group_budget=60_000,
        output_path=out,
        checkpoint_path=ckpt,
        base_seed=42,
        schema_sha256=schema_sha,
        contract_sha256="deadbeef",
        project_root=tmp_path,
        resume=False,
    )
    assert first.status == "ok"
    assert first.resumed is False
    assert first.output_row_count == 20
    assert first.sampling_mode == "reservoir"
    table = pq.read_table(out)
    assert table.num_rows == 20
    assert set(table.column_names) == set(feature_parquet_arrow_schema().names)
    assert "attack_family" not in table.column_names
    got_windows = table.column("window_index").to_pylist()
    assert got_windows == expected_idx

    second = materialize_fit_shard(
        row=row,
        allocated_k=20,
        group_budget=60_000,
        output_path=out,
        checkpoint_path=ckpt,
        base_seed=42,
        schema_sha256=schema_sha,
        contract_sha256="deadbeef",
        project_root=tmp_path,
        resume=True,
    )
    assert second.resumed is True
    assert second.selection_sha256 == first.selection_sha256


def test_contract_refuses_manifest_drift(tmp_path: Path) -> None:
    split = tmp_path / "modeling_split_manifest.csv"
    split.write_text("pcap_path\nx\n", encoding="utf-8")
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    contract_path = write_training_view_contract(
        tmp_path / "training_view_contract.json",
        split_manifest_path=split,
        feature_schema_path=schema_path,
        project_root=tmp_path,
    )
    split.write_text("pcap_path\nx\ny\n", encoding="utf-8")
    with pytest.raises(FeatureExtractionError, match="hash mismatch"):
        load_and_verify_training_view_contract(
            contract_path=contract_path,
            split_manifest_path=split,
            feature_schema_path=schema_path,
            project_root=tmp_path,
        )


def test_pinned_training_view_contract_matches_repo() -> None:
    contract_path = (
        PROJECT_ROOT / "data" / "modeling" / "v1" / "training_view_contract.json"
    )
    split_path = (
        PROJECT_ROOT / "data" / "modeling" / "v1" / "modeling_split_manifest.csv"
    )
    if not (contract_path.is_file() and split_path.is_file()):
        pytest.skip("frozen modeling artifacts missing")
    contract = load_and_verify_training_view_contract()
    assert contract["sampling_plan_id"] == "group_balanced"
    assert contract["expected_fit_rows"] == 704_305
    assert contract["modeling_split_manifest_sha256"] == file_sha256(split_path)
    assert contract["feature_schema_sha256"] == feature_schema_sha256()
