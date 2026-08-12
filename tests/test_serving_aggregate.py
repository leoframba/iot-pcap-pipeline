"""Unit tests for pure PCAP window-score aggregation (D0)."""

from __future__ import annotations

import pytest

from iot_pcap_pipeline.serving.aggregate import (
    StreamingWindowAggregator,
    aggregate_window_scores,
    window_is_attack,
)
from iot_pcap_pipeline.serving.candidates import WINDOW_ATTACK_THRESHOLD
from iot_pcap_pipeline.serving.contract import (
    FROZEN_ATTACK_RATE_THRESHOLD,
    FROZEN_MIN_ATTACK_WINDOWS,
    FROZEN_MIN_COMPLETE_WINDOWS,
)
from iot_pcap_pipeline.serving.errors import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
)

THR = WINDOW_ATTACK_THRESHOLD


def test_window_score_below_threshold_is_benign() -> None:
    assert window_is_attack(THR - 1e-12) is False


def test_window_score_equal_threshold_is_attack() -> None:
    assert window_is_attack(THR) is True


def test_window_score_above_threshold_is_attack() -> None:
    assert window_is_attack(THR + 1e-12) is True


def test_zero_windows_insufficient_data() -> None:
    result = aggregate_window_scores([])
    assert result.status == STATUS_INSUFFICIENT_DATA
    assert result.prediction is None
    assert result.pcap_attack_score is None
    assert result.window_summary.total_windows == 0


def test_one_or_two_windows_insufficient_even_if_extreme_scores() -> None:
    for n in (1, 2):
        result = aggregate_window_scores([0.99999] * n)
        assert result.status == STATUS_INSUFFICIENT_DATA
        assert result.prediction is None
        assert result.pcap_attack_score is None
        assert result.window_summary.total_windows == n
        assert result.window_summary.attack_windows == n


def test_all_benign_windows_predict_benign() -> None:
    scores = [0.1] * 10
    result = aggregate_window_scores(scores)
    assert result.status == STATUS_OK
    assert result.prediction == "BENIGN"
    assert result.pcap_attack_score == 0.0
    assert result.window_summary.attack_windows == 0


def test_all_attack_windows_predict_attack() -> None:
    scores = [0.99] * 10
    result = aggregate_window_scores(scores)
    assert result.status == STATUS_OK
    assert result.prediction == "ATTACK"
    assert result.pcap_attack_score == 1.0


def test_exact_rate_threshold_obeys_gte() -> None:
    # 3 attack / 600 windows = 0.005 exactly; K=3 met.
    scores = [0.99] * 3 + [0.1] * 597
    result = aggregate_window_scores(scores)
    assert result.pcap_attack_score == pytest.approx(FROZEN_ATTACK_RATE_THRESHOLD)
    assert result.prediction == "ATTACK"


def test_exact_min_attack_windows_obeys_gte() -> None:
    # Exactly K=3 attack windows, rate above R on a modest PCAP.
    scores = [0.99] * FROZEN_MIN_ATTACK_WINDOWS + [0.1] * 97
    result = aggregate_window_scores(scores)
    assert result.window_summary.attack_windows == FROZEN_MIN_ATTACK_WINDOWS
    assert result.prediction == "ATTACK"


def test_rate_just_below_threshold_is_benign() -> None:
    # 3 / 601 < 0.005
    scores = [0.99] * 3 + [0.1] * 598
    result = aggregate_window_scores(scores)
    assert result.pcap_attack_score is not None
    assert result.pcap_attack_score < FROZEN_ATTACK_RATE_THRESHOLD
    assert result.prediction == "BENIGN"


def test_count_below_k_is_benign_even_if_rate_high() -> None:
    # 2 attack / 3 windows ≈ 0.667 >= R but K not met → still insufficient? 
    # total=3 meets min windows; attack_windows=2 < K → BENIGN
    scores = [0.99, 0.99, 0.1]
    result = aggregate_window_scores(scores)
    assert result.status == STATUS_OK
    assert result.window_summary.attack_windows == 2
    assert result.prediction == "BENIGN"


def test_one_extreme_score_does_not_bypass_aggregation() -> None:
    scores = [0.99999] + [0.01] * 999
    result = aggregate_window_scores(scores)
    assert result.window_summary.max_window_attack_score == pytest.approx(0.99999)
    assert result.window_summary.attack_windows == 1
    assert result.prediction == "BENIGN"


def test_max_and_mean_scores() -> None:
    scores = [0.2, 0.4, 0.6]
    result = aggregate_window_scores(scores)
    assert result.window_summary.max_window_attack_score == pytest.approx(0.6)
    assert result.window_summary.mean_window_attack_score == pytest.approx(0.4)


def test_pcap_attack_score_bounded_unit_interval() -> None:
    for scores in ([0.1] * 5, [0.99] * 5, [0.99, 0.1, 0.1, 0.1, 0.1]):
        result = aggregate_window_scores(scores)
        assert result.status == STATUS_OK
        assert result.pcap_attack_score is not None
        assert 0.0 <= result.pcap_attack_score <= 1.0


def test_raising_score_benign_to_attack_cannot_reduce_rate() -> None:
    base = [0.1] * 10
    raised = list(base)
    raised[0] = THR  # promote one benign → attack
    a = aggregate_window_scores(base)
    b = aggregate_window_scores(raised)
    assert a.pcap_attack_score is not None and b.pcap_attack_score is not None
    assert b.pcap_attack_score >= a.pcap_attack_score
    assert b.window_summary.attack_windows == a.window_summary.attack_windows + 1


def test_decision_block_echoes_frozen_knobs() -> None:
    result = aggregate_window_scores([0.1] * FROZEN_MIN_COMPLETE_WINDOWS)
    assert result.decision.minimum_complete_windows == FROZEN_MIN_COMPLETE_WINDOWS
    assert result.decision.pcap_min_attack_windows == FROZEN_MIN_ATTACK_WINDOWS
    assert result.decision.pcap_attack_rate_threshold == FROZEN_ATTACK_RATE_THRESHOLD
    assert result.decision.window_attack_threshold == THR


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.1],
        [0.99, 0.99],
        [0.1] * 10,
        [0.99] * 10,
        [0.99] * 3 + [0.1] * 597,
        [0.99999] + [0.01] * 999,
        [0.2, 0.4, 0.6, THR, 0.1, 0.1],
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 64])
def test_streaming_aggregator_matches_aggregate_window_scores(
    scores: list[float],
    batch_size: int,
) -> None:
    expected = aggregate_window_scores(scores)
    agg = StreamingWindowAggregator()
    for i in range(0, len(scores), batch_size):
        agg.observe_many(scores[i : i + batch_size])
    got = agg.finalize()
    assert got.status == expected.status
    assert got.prediction == expected.prediction
    assert got.pcap_attack_score == expected.pcap_attack_score
    assert got.window_summary == expected.window_summary
    assert got.decision == expected.decision
