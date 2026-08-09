"""Synthetic tests for Phase 1B.3 timestamp ordering probe."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.pcap.timestamps import (
    DEFAULT_EXAMPLE_LIMIT,
    PROBE_COLUMNS,
    probe_timestamps,
    write_probe_artifacts,
)


def _pcap(tmp_path: Path, name: str, timestamps: list[float]) -> Path:
    packets = [(ts, eth_ip_tcp()) for ts in timestamps]
    return write_pcap(tmp_path / name, packets)


def test_strictly_increasing(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "inc.pcap", [1.0, 1.1, 1.2])
    result = probe_timestamps(path)

    assert result.packet_count == 3
    assert result.adjacent_delta_count == 2
    assert result.positive_delta_count == 2
    assert result.duplicate_delta_count == 0
    assert result.negative_delta_count == 0
    assert result.negative_delta_ratio == 0.0
    assert result.duplicate_ratio == 0.0
    assert result.examples == []
    assert result.negative_run_count == 0
    assert result.positive_percentile_method == "exact"
    assert result.positive_delta_min == pytest.approx(0.1)
    assert result.positive_delta_max == pytest.approx(0.1)
    assert result.positive_delta_mean == pytest.approx(0.1)
    assert result.positive_delta_p50 == pytest.approx(0.1)


def test_duplicate_timestamps(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "dup.pcap", [1.0, 1.0, 1.1])
    result = probe_timestamps(path)

    assert result.adjacent_delta_count == 2
    assert result.duplicate_delta_count == 1
    assert result.negative_delta_count == 0
    assert result.positive_delta_count == 1
    assert result.duplicate_ratio == pytest.approx(0.5)
    assert result.negative_delta_ratio == pytest.approx(0.0)


def test_negative_timestamp(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "neg.pcap", [1.0, 1.1, 1.05, 1.2])
    result = probe_timestamps(path)

    assert result.adjacent_delta_count == 3
    assert result.positive_delta_count == 2
    assert result.duplicate_delta_count == 0
    assert result.negative_delta_count == 1
    assert result.negative_delta_min_magnitude == pytest.approx(0.05)
    assert result.negative_delta_max_magnitude == pytest.approx(0.05)
    assert result.negative_delta_mean_magnitude == pytest.approx(0.05)
    assert result.negative_delta_p50_magnitude == pytest.approx(0.05)
    assert result.negative_delta_p95_magnitude == pytest.approx(0.05)
    assert result.negative_delta_p99_magnitude == pytest.approx(0.05)
    assert len(result.examples) == 1
    ex = result.examples[0]
    assert ex.packet_index_previous == 1
    assert ex.packet_index_current == 2
    assert ex.previous_timestamp == pytest.approx(1.1)
    assert ex.current_timestamp == pytest.approx(1.05)
    assert ex.delta_seconds == pytest.approx(-0.05)
    assert ex.delta_magnitude_seconds == pytest.approx(0.05)
    assert result.negative_run_count == 1
    assert result.negative_run_max_length == 1
    assert result.negative_run_mean_length == pytest.approx(1.0)


def test_mixed_sequence_counts_ratios_percentiles_examples(tmp_path: Path) -> None:
    # deltas: +0.1, 0, -0.05, -0.01, +0.2, +0.3
    path = _pcap(
        tmp_path,
        "mixed.pcap",
        [1.0, 1.1, 1.1, 1.05, 1.04, 1.24, 1.54],
    )
    result = probe_timestamps(path, example_limit=10)

    assert result.packet_count == 7
    assert result.adjacent_delta_count == 6
    assert result.positive_delta_count == 3
    assert result.duplicate_delta_count == 1
    assert result.negative_delta_count == 2
    assert result.duplicate_ratio == pytest.approx(1 / 6)
    assert result.negative_delta_ratio == pytest.approx(2 / 6)

    assert result.negative_delta_min_magnitude == pytest.approx(0.01)
    assert result.negative_delta_max_magnitude == pytest.approx(0.05)
    assert result.negative_delta_mean_magnitude == pytest.approx(0.03)
    assert result.negative_delta_p50_magnitude == pytest.approx(0.03)

    assert result.positive_delta_min == pytest.approx(0.1)
    assert result.positive_delta_max == pytest.approx(0.3)
    assert result.positive_delta_mean == pytest.approx(0.2)
    assert result.positive_delta_p50 == pytest.approx(0.2)
    assert result.positive_delta_p95 == pytest.approx(0.29)
    assert result.positive_delta_p99 == pytest.approx(0.298)

    assert len(result.examples) == 2
    assert result.examples[0].delta_magnitude_seconds == pytest.approx(0.05)
    assert result.examples[1].delta_magnitude_seconds == pytest.approx(0.01)

    # two consecutive negatives form one run of length 2
    assert result.negative_run_count == 1
    assert result.negative_run_max_length == 2
    assert result.negative_run_mean_length == pytest.approx(2.0)


def test_single_packet_pcap(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "one.pcap", [1.0])
    result = probe_timestamps(path)

    assert result.packet_count == 1
    assert result.adjacent_delta_count == 0
    assert result.positive_delta_count == 0
    assert result.duplicate_delta_count == 0
    assert result.negative_delta_count == 0
    assert result.duplicate_ratio is None
    assert result.negative_delta_ratio is None
    assert result.examples == []


def test_empty_pcap(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "empty.pcap", [])
    result = probe_timestamps(path)

    assert result.packet_count == 0
    assert result.adjacent_delta_count == 0
    assert result.negative_delta_count == 0
    assert result.positive_delta_min is None
    assert result.negative_delta_p99_magnitude is None


def test_all_duplicate_timestamps(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "all_dup.pcap", [5.0, 5.0, 5.0, 5.0])
    result = probe_timestamps(path)

    assert result.adjacent_delta_count == 3
    assert result.duplicate_delta_count == 3
    assert result.positive_delta_count == 0
    assert result.negative_delta_count == 0
    assert result.duplicate_ratio == pytest.approx(1.0)
    assert result.positive_delta_mean is None


def test_all_increasing_timestamps(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "all_inc.pcap", [0.0, 1.0, 2.0, 4.0])
    result = probe_timestamps(path)

    assert result.positive_delta_count == 3
    assert result.duplicate_delta_count == 0
    assert result.negative_delta_count == 0
    assert result.positive_delta_min == pytest.approx(1.0)
    assert result.positive_delta_max == pytest.approx(2.0)
    assert result.positive_delta_mean == pytest.approx(4.0 / 3.0)


def test_bounded_example_limit(tmp_path: Path) -> None:
    # five consecutive reversals
    path = _pcap(tmp_path, "burst.pcap", [10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    result = probe_timestamps(path, example_limit=2)

    assert result.negative_delta_count == 5
    assert len(result.examples) == 2
    assert result.examples[0].packet_index_current == 1
    assert result.examples[1].packet_index_current == 2
    assert result.negative_run_count == 1
    assert result.negative_run_max_length == 5


def test_positive_reservoir_method_documented(tmp_path: Path) -> None:
    timestamps = [float(i) for i in range(20)]
    path = _pcap(tmp_path, "res.pcap", timestamps)
    result = probe_timestamps(path, positive_sample_cap=5)

    assert result.positive_delta_count == 19
    assert result.positive_percentile_method.startswith("reservoir_n5_seed")
    assert result.positive_percentile_sample_size == 5
    assert result.positive_delta_min == pytest.approx(1.0)  # exact streaming
    assert result.positive_delta_max == pytest.approx(1.0)
    assert result.positive_delta_p50 is not None


def test_write_probe_artifacts(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "out.pcap", [1.0, 1.1, 1.05])
    result = probe_timestamps(path, example_limit=DEFAULT_EXAMPLE_LIMIT)
    probe_csv = tmp_path / "timestamp_probe.csv"
    examples_csv = tmp_path / "timestamp_reversal_examples.csv"

    written = write_probe_artifacts(
        [result],
        output_path=probe_csv,
        examples_path=examples_csv,
    )
    assert written["probe_path"] == probe_csv
    assert written["examples_path"] == examples_csv

    with probe_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == PROBE_COLUMNS
    assert int(rows[0]["negative_delta_count"]) == 1

    with examples_csv.open(newline="", encoding="utf-8") as handle:
        examples = list(csv.DictReader(handle))
    assert len(examples) == 1
    assert float(examples[0]["delta_magnitude_seconds"]) == pytest.approx(0.05)


def test_max_packets_cap(tmp_path: Path) -> None:
    path = _pcap(tmp_path, "cap.pcap", [1.0, 1.1, 1.05, 1.2, 1.3])
    result = probe_timestamps(path, max_packets=3)

    assert result.packet_count == 3
    assert result.adjacent_delta_count == 2
    assert result.negative_delta_count == 1
