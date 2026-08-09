"""Phase 1C.3a streaming Parquet storage contract tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import dpkt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.features.parquet import (
    FEATURE_BUILD_STRATEGY_VERSION,
    StreamingFeatureParquetWriter,
    build_pcap_parquet,
    checkpoint_is_reusable,
    feature_parquet_arrow_schema,
    feature_schema_sha256,
    load_build_checkpoint,
    pcap_id_from_path,
    read_feature_rows_from_parquet,
)
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES, write_feature_schema
from iot_pcap_pipeline.paths import FEATURE_STRATEGY_VERSION
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import frozen_window_policy
from iot_pcap_pipeline.windowing.stream import iter_windows


def _steady_pcap(path: Path, n_packets: int = 80) -> Path:
    frames = [
        (1.0 + 0.01 * i, eth_ip_tcp(flags=dpkt.tcp.TH_SYN)) for i in range(n_packets)
    ]
    return write_pcap(path, frames)


def _expected_from_pcap(pcap_path: Path) -> list[dict]:
    rows = []
    pcap_id = pcap_id_from_path(pcap_path)
    for window in iter_windows(iter_packets(pcap_path), frozen_window_policy()):
        feats = extract_features(window)
        row = {
            "pcap_id": pcap_id,
            "binary_label": "BENIGN",
            "segment_index": window.segment_index,
            "window_index": window.window_index,
            "packet_index_start": window.packet_index_start,
            "packet_index_end": window.packet_index_end,
            **feats.to_feature_dict(),
        }
        rows.append(row)
    return rows


def _assert_rows_equal(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got["pcap_id"] == want["pcap_id"]
        assert got["binary_label"] == want["binary_label"]
        assert got["segment_index"] == want["segment_index"]
        assert got["window_index"] == want["window_index"]
        assert got["packet_index_start"] == want["packet_index_start"]
        assert got["packet_index_end"] == want["packet_index_end"]
        for name in V1_FEATURE_NAMES:
            if name in ("unique_ip_count", "unique_port_count"):
                assert int(got[name]) == int(want[name])
            else:
                assert float(got[name]) == float(want[name])


def test_feature_strategy_version_unchanged() -> None:
    assert FEATURE_STRATEGY_VERSION == "phase1c2_v1"
    assert FEATURE_BUILD_STRATEGY_VERSION == "phase1c3_v1"


def test_arrow_schema_matches_contract() -> None:
    schema = feature_parquet_arrow_schema()
    assert [f.name for f in schema] == [
        "pcap_id",
        "binary_label",
        "segment_index",
        "window_index",
        "packet_index_start",
        "packet_index_end",
        *V1_FEATURE_NAMES,
    ]
    assert schema.field("window_span_seconds").type == pa.float64()
    assert schema.field("unique_ip_count").type == pa.int64()
    assert schema.field("unique_port_count").type == pa.int64()


def test_parquet_round_trip_matches_extractor(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "roundtrip.pcap", n_packets=80)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "roundtrip.parquet"
    ckpt = tmp_path / "roundtrip.json"

    result = build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        buffer_rows=7,
        resume=False,
    )
    assert result.row_count > 0
    assert result.resumed is False
    assert out.is_file()
    assert not out.with_suffix(".parquet.tmp").exists()

    expected = _expected_from_pcap(pcap)
    actual = read_feature_rows_from_parquet(out)
    assert result.row_count == len(expected)
    _assert_rows_equal(actual, expected)


def test_multiple_batches_one_valid_file(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "batches.pcap", n_packets=120)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "batches.parquet"

    result = build_pcap_parquet(
        pcap,
        {"binary_label": "ATTACK"},
        out,
        checkpoint_path=tmp_path / "batches.json",
        schema_path=schema_path,
        project_root=tmp_path,
        buffer_rows=3,
        resume=False,
    )
    assert result.row_count >= 3
    table = pq.read_table(out)
    assert table.num_rows == result.row_count
    assert table.schema.equals(feature_parquet_arrow_schema(), check_metadata=False)


def test_temp_renamed_only_after_success(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "atomic.pcap", n_packets=50)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "atomic.parquet"
    tmp = out.with_suffix(".parquet.tmp")

    result = build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=tmp_path / "atomic.json",
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    assert out.is_file()
    assert not tmp.exists()
    assert result.output_file_size == out.stat().st_size


def test_failed_write_leaves_no_final_shard(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "fail.pcap", n_packets=60)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "fail.parquet"
    tmp = out.with_suffix(".parquet.tmp")

    with (
        patch.object(
            StreamingFeatureParquetWriter,
            "_flush",
            side_effect=RuntimeError("simulated write failure"),
        ),
        pytest.raises(RuntimeError, match="simulated write failure"),
    ):
        build_pcap_parquet(
            pcap,
            {"binary_label": "BENIGN"},
            out,
            checkpoint_path=tmp_path / "fail.json",
            schema_path=schema_path,
            project_root=tmp_path,
            buffer_rows=1,
            resume=False,
        )

    assert not out.exists()
    assert not tmp.exists()
    assert not (tmp_path / "fail.json").exists()


def test_valid_checkpoint_causes_resume_skip(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "resume.pcap", n_packets=55)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "resume.parquet"
    ckpt = tmp_path / "resume.json"

    first = build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    mtime = out.stat().st_mtime_ns
    second = build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=True,
    )
    assert second.resumed is True
    assert second.row_count == first.row_count
    assert out.stat().st_mtime_ns == mtime


def test_missing_parquet_invalidates_checkpoint(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "missing.pcap", n_packets=50)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "missing.parquet"
    ckpt = tmp_path / "missing.json"

    build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    out.unlink()
    assert (
        checkpoint_is_reusable(
            checkpoint_path=ckpt,
            output_path=out,
            pcap_path=pcap,
            input_file_size=pcap.stat().st_size,
            schema_sha256=feature_schema_sha256(schema_path),
        )
        is False
    )

    rebuilt = build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=True,
    )
    assert rebuilt.resumed is False
    assert out.is_file()


def test_corrupted_parquet_invalidates_checkpoint(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "corrupt.pcap", n_packets=50)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "corrupt.parquet"
    ckpt = tmp_path / "corrupt.json"

    build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    out.write_bytes(b"not-a-parquet-file")
    assert (
        checkpoint_is_reusable(
            checkpoint_path=ckpt,
            output_path=out,
            pcap_path=pcap,
            input_file_size=pcap.stat().st_size,
            schema_sha256=feature_schema_sha256(schema_path),
        )
        is False
    )


def test_changed_schema_hash_invalidates_checkpoint(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "hash.pcap", n_packets=50)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "hash.parquet"
    ckpt = tmp_path / "hash.json"

    build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    assert (
        checkpoint_is_reusable(
            checkpoint_path=ckpt,
            output_path=out,
            pcap_path=pcap,
            input_file_size=pcap.stat().st_size,
            schema_sha256="0" * 64,
        )
        is False
    )


def test_changed_input_size_invalidates_checkpoint(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "size.pcap", n_packets=50)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "size.parquet"
    ckpt = tmp_path / "size.json"

    build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    assert (
        checkpoint_is_reusable(
            checkpoint_path=ckpt,
            output_path=out,
            pcap_path=pcap,
            input_file_size=pcap.stat().st_size + 1,
            schema_sha256=feature_schema_sha256(schema_path),
        )
        is False
    )


def test_checkpoint_payload_fields(tmp_path: Path) -> None:
    pcap = _steady_pcap(tmp_path / "fields.pcap", n_packets=40)
    schema_path = tmp_path / "feature_schema.json"
    write_feature_schema(schema_path)
    out = tmp_path / "fields.parquet"
    ckpt = tmp_path / "fields.json"

    result = build_pcap_parquet(
        pcap,
        {"binary_label": "BENIGN"},
        out,
        checkpoint_path=ckpt,
        schema_path=schema_path,
        project_root=tmp_path,
        resume=False,
    )
    payload = load_build_checkpoint(ckpt)
    assert payload is not None
    for key in (
        "pcap_id",
        "pcap_path",
        "input_file_size",
        "feature_strategy_version",
        "feature_build_strategy_version",
        "feature_schema_sha256",
        "output_path",
        "output_row_count",
        "output_file_size",
    ):
        assert key in payload
    assert payload["feature_strategy_version"] == "phase1c2_v1"
    assert payload["feature_build_strategy_version"] == "phase1c3_v1"
    assert payload["output_row_count"] == result.row_count


def test_writer_buffer_clears_between_batches(tmp_path: Path) -> None:
    schema = feature_parquet_arrow_schema()
    path = tmp_path / "buf.parquet.tmp"
    writer = StreamingFeatureParquetWriter(path, buffer_rows=2, schema=schema)
    base = {
        "pcap_id": "x",
        "binary_label": "BENIGN",
        "segment_index": 0,
        "window_index": 0,
        "packet_index_start": 0,
        "packet_index_end": 24,
        **{
            name: 0.0
            for name in V1_FEATURE_NAMES
            if name not in ("unique_ip_count", "unique_port_count")
        },
        "unique_ip_count": 1,
        "unique_port_count": 2,
    }
    for i in range(5):
        row = dict(base)
        row["window_index"] = i
        writer.append(row)
    assert len(writer._buffer) == 1
    assert writer.rows_written == 4
    writer.close()
    final = tmp_path / "buf.parquet"
    path.replace(final)
    assert pq.read_table(final).num_rows == 5
