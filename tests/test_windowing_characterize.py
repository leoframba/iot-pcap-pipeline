"""Synthetic tests for Phase 1C.1 windowing-policy characterization."""

from __future__ import annotations

from pathlib import Path

import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.windowing.characterize import (
    CHARACTERIZATION_COLUMNS,
    PolicyAccumulator,
    characterize_pcap,
    characterize_timestamps,
    format_characterization_summary,
    summary_category,
)
from iot_pcap_pipeline.windowing.policy import WindowPolicy, candidate_policies


def _acc(policy: WindowPolicy, timestamps: list[float]) -> PolicyAccumulator:
    return characterize_timestamps(timestamps, [policy])[0]


def test_candidate_grid_has_six_configs() -> None:
    policies = candidate_policies()
    assert len(policies) == 6
    assert {(p.window_size, p.inactivity_timeout_seconds) for p in policies} == {
        (25, 5.0),
        (25, 30.0),
        (50, 5.0),
        (50, 30.0),
        (100, 5.0),
        (100, 30.0),
    }
    assert all(p.backward_reset_seconds == 1.0 for p in policies)


def test_exact_full_windows_no_gaps() -> None:
    # 100 packets, 0.01s apart → exactly 4 windows of size 25, no drops
    timestamps = [i * 0.01 for i in range(100)]
    acc = _acc(WindowPolicy(25, 5.0), timestamps)

    assert acc.packet_count == 100
    assert acc.full_window_count == 4
    assert acc.emitted_packet_count == 100
    assert acc.dropped_partial_packet_count == 0
    assert acc.segment_count == 1
    assert acc.positive_gap_boundary_count == 0
    assert acc.backward_discontinuity_boundary_count == 0
    row = acc.to_row({})
    assert row["packet_retention_ratio"] == pytest.approx(1.0)
    assert row["window_span_min"] == pytest.approx(0.24)  # 24 gaps × 0.01
    assert row["window_span_min"] >= 0
    assert row["window_span_percentile_method"] == "exact"


def test_partial_tail_dropped_at_eof() -> None:
    timestamps = [float(i) for i in range(30)]  # 30 packets, window 25
    acc = _acc(WindowPolicy(25, 5.0), timestamps)

    assert acc.full_window_count == 1
    assert acc.emitted_packet_count == 25
    assert acc.dropped_partial_window_count == 1
    assert acc.dropped_partial_packet_count == 5
    assert acc.to_row({})["packet_retention_ratio"] == pytest.approx(25 / 30)


def test_inactivity_boundary_drops_partial_and_starts_new_segment() -> None:
    # 3 packets, then 10s gap (>5s), then 25 packets → first 3 dropped
    timestamps = [0.0, 0.1, 0.2] + [10.0 + i * 0.01 for i in range(25)]
    acc = _acc(WindowPolicy(25, 5.0), timestamps)

    assert acc.positive_gap_boundary_count == 1
    assert acc.segment_count == 2
    assert acc.dropped_partial_packet_count == 3
    assert acc.full_window_count == 1
    assert acc.emitted_packet_count == 25


def test_small_negative_jitter_stays_in_segment() -> None:
    # -0.5s jitter is within [-1.0, 0) → keep same segment
    timestamps = [0.0, 0.1, 0.2, -0.3, 0.4] + [1.0 + i * 0.01 for i in range(20)]
    # total 25 packets
    assert len(timestamps) == 25
    acc = _acc(WindowPolicy(25, 5.0, backward_reset_seconds=1.0), timestamps)

    assert acc.backward_discontinuity_boundary_count == 0
    assert acc.segment_count == 1
    assert acc.full_window_count == 1
    assert acc.dropped_partial_packet_count == 0


def test_tiny_negative_jitter_never_creates_negative_span() -> None:
    timestamps = [1.0, 0.999999, 1.000001]
    acc = _acc(WindowPolicy(3, 5.0), timestamps)
    row = acc.to_row({})

    expected = max(timestamps) - min(timestamps)
    assert row["window_span_min"] == pytest.approx(expected)
    assert row["window_span_max"] == pytest.approx(expected)
    assert row["window_span_min"] >= 0
    assert row["window_span_max"] >= 0
    assert acc.zero_span_window_count == 0


def test_first_equals_last_does_not_imply_zero_span() -> None:
    timestamps = [1.0, 1.1, 1.0]
    acc = _acc(WindowPolicy(3, 5.0), timestamps)
    row = acc.to_row({})

    assert row["window_span_min"] == pytest.approx(0.1)
    assert row["window_span_max"] == pytest.approx(0.1)
    assert acc.zero_span_window_count == 0
    assert row["zero_span_window_ratio"] == pytest.approx(0.0)


def test_large_backward_discontinuity_boundary() -> None:
    # -12.82s like Benign_train outlier → new segment; drop partial before it
    timestamps = [0.0, 1.0, 2.0, 2.0 - 12.82] + [
        (2.0 - 12.82) + 0.01 * (i + 1) for i in range(24)
    ]
    # 3 before boundary + 1 boundary packet + 24 more = 28
    assert len(timestamps) == 28
    acc = _acc(WindowPolicy(25, 30.0, backward_reset_seconds=1.0), timestamps)

    assert acc.backward_discontinuity_boundary_count == 1
    assert acc.segment_count == 2
    assert acc.dropped_partial_packet_count == 3  # first segment incomplete
    assert acc.full_window_count == 1  # 25 packets in second segment
    assert acc.emitted_packet_count == 25


def test_duplicate_timestamps_stay_in_segment() -> None:
    timestamps = [0.0] * 25
    acc = _acc(WindowPolicy(25, 5.0), timestamps)
    row = acc.to_row({})

    assert acc.segment_count == 1
    assert acc.full_window_count == 1
    assert acc.zero_span_window_count == 1
    assert row["window_span_min"] == 0.0
    assert row["window_span_max"] == 0.0
    assert row["zero_span_window_ratio"] == pytest.approx(1.0)


def test_span_reservoir_method_when_over_cap() -> None:
    # 12 windows of size 5 → sample cap 5 forces reservoir method
    timestamps = [float(i) for i in range(60)]
    acc = characterize_timestamps(
        timestamps,
        [WindowPolicy(5, 30.0)],
        span_sample_cap=5,
        span_sample_seed=42,
    )[0]
    row = acc.to_row({})

    assert acc.full_window_count == 12
    assert row["window_span_percentile_method"].startswith("reservoir_n5_seed42")
    assert row["window_span_percentile_sample_size"] == 5
    assert row["window_span_min"] == pytest.approx(4.0)  # exact streaming
    assert row["window_span_max"] == pytest.approx(4.0)


def test_multi_config_single_pass_differs_by_window_size() -> None:
    timestamps = [i * 0.001 for i in range(100)]
    policies = [
        WindowPolicy(25, 5.0),
        WindowPolicy(50, 5.0),
        WindowPolicy(100, 5.0),
    ]
    accs = characterize_timestamps(timestamps, policies)

    assert [a.full_window_count for a in accs] == [4, 2, 1]
    assert all(a.packet_count == 100 for a in accs)
    assert all(a.dropped_partial_packet_count == 0 for a in accs)


def test_metadata_does_not_affect_segmentation() -> None:
    timestamps = [float(i) for i in range(50)]
    policy = WindowPolicy(25, 5.0)
    a = characterize_timestamps(timestamps, [policy])[0]
    b = characterize_timestamps(timestamps, [policy])[0]
    assert a.full_window_count == b.full_window_count == 2
    assert a.emitted_packet_count == b.emitted_packet_count


def test_characterize_pcap_writes_rows(tmp_path: Path) -> None:
    path = write_pcap(
        tmp_path / "t.pcap",
        [(float(i), eth_ip_tcp()) for i in range(50)],
    )
    rows = characterize_pcap(
        path,
        policies=[WindowPolicy(25, 5.0), WindowPolicy(50, 30.0)],
        meta={"source": "attacks", "binary_label": "BENIGN"},
        project_root=tmp_path,
    )
    assert len(rows) == 2
    assert rows[0]["full_window_count"] == 2
    assert rows[1]["full_window_count"] == 1
    assert set(rows[0].keys()) >= set(CHARACTERIZATION_COLUMNS)


def test_summary_category_mapping() -> None:
    assert (
        summary_category({"source": "attacks", "binary_label": "BENIGN"})
        == "publisher_benign"
    )
    assert (
        summary_category(
            {"source": "profiling", "profiling_type": "interaction"}
        )
        == "profiling_interaction"
    )
    assert (
        summary_category(
            {"source": "attacks", "binary_label": "ATTACK", "attack_family": "DDoS"}
        )
        == "DDoS"
    )


def test_format_summary_includes_gate_a(tmp_path: Path) -> None:
    rows = [
        {
            "pcap_path": "a.pcap",
            "source": "attacks",
            "binary_label": "BENIGN",
            "attack_family": "",
            "profiling_type": "",
            "window_size": 25,
            "inactivity_timeout_seconds": 5.0,
            "backward_reset_seconds": 1.0,
            "packet_count": 100,
            "full_window_count": 4,
            "emitted_packet_count": 100,
            "dropped_partial_packet_count": 0,
            "segment_count": 1,
            "zero_span_window_count": 0,
            "window_span_p50": 0.1,
            "window_span_p95": 0.2,
            "window_span_p99": 0.3,
        }
    ]
    text = format_characterization_summary(rows)
    assert "GATE A" in text
    assert "publisher_benign" in text
    assert "window_size=25" in text


def test_30s_timeout_does_not_split_where_5s_does() -> None:
    # 10s gap: splits under 5s timeout, continues under 30s
    timestamps = [0.0, 1.0, 2.0, 12.0, 13.0] + [14.0 + i for i in range(20)]
    assert len(timestamps) == 25
    tight = _acc(WindowPolicy(25, 5.0), timestamps)
    loose = _acc(WindowPolicy(25, 30.0), timestamps)

    assert tight.positive_gap_boundary_count == 1
    assert tight.segment_count == 2
    assert tight.full_window_count == 0  # neither segment reaches 25

    assert loose.positive_gap_boundary_count == 0
    assert loose.segment_count == 1
    assert loose.full_window_count == 1
