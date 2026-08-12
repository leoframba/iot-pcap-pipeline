"""Pure PCAP aggregation over window attack scores (no decode / joblib / HTTP)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from iot_pcap_pipeline.serving.contract import (
    FROZEN_ATTACK_RATE_THRESHOLD,
    FROZEN_MIN_ATTACK_WINDOWS,
    FROZEN_MIN_COMPLETE_WINDOWS,
    WINDOW_ATTACK_THRESHOLD,
)
from iot_pcap_pipeline.serving.errors import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
)

PREDICTION_ATTACK = "ATTACK"
PREDICTION_BENIGN = "BENIGN"


@dataclass(frozen=True)
class WindowSummary:
    total_windows: int
    attack_windows: int
    benign_windows: int
    max_window_attack_score: float | None
    mean_window_attack_score: float | None


@dataclass(frozen=True)
class AggregationDecision:
    window_attack_threshold: float
    minimum_complete_windows: int
    pcap_min_attack_windows: int
    pcap_attack_rate_threshold: float


@dataclass(frozen=True)
class AggregationResult:
    """PCAP-level aggregation result from window scores only."""

    status: str
    prediction: str | None
    pcap_attack_score: float | None
    window_summary: WindowSummary
    decision: AggregationDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "prediction": self.prediction,
            "pcap_attack_score": self.pcap_attack_score,
            "window_summary": asdict(self.window_summary),
            "decision": asdict(self.decision),
        }


def window_is_attack(
    score: float,
    *,
    window_threshold: float = WINDOW_ATTACK_THRESHOLD,
) -> bool:
    """ATTACK iff score >= window threshold (frozen >= semantics)."""
    return float(score) >= float(window_threshold)


@dataclass
class StreamingWindowAggregator:
    """Constant-memory aggregator for complete-window attack scores."""

    window_threshold: float = WINDOW_ATTACK_THRESHOLD
    minimum_complete_windows: int = FROZEN_MIN_COMPLETE_WINDOWS
    min_attack_windows: int = FROZEN_MIN_ATTACK_WINDOWS
    attack_rate_threshold: float = FROZEN_ATTACK_RATE_THRESHOLD
    _total: int = 0
    _attack: int = 0
    _score_sum: float = 0.0
    _score_max: float = float("-inf")

    def observe(self, score: float) -> None:
        s = float(score)
        self._total += 1
        self._score_sum += s
        if s > self._score_max:
            self._score_max = s
        if window_is_attack(s, window_threshold=self.window_threshold):
            self._attack += 1

    def observe_many(self, scores: Sequence[float]) -> None:
        for score in scores:
            self.observe(score)

    def finalize(self) -> AggregationResult:
        decision = AggregationDecision(
            window_attack_threshold=float(self.window_threshold),
            minimum_complete_windows=int(self.minimum_complete_windows),
            pcap_min_attack_windows=int(self.min_attack_windows),
            pcap_attack_rate_threshold=float(self.attack_rate_threshold),
        )
        total = int(self._total)
        attack_windows = int(self._attack)
        if total == 0:
            summary = WindowSummary(
                total_windows=0,
                attack_windows=0,
                benign_windows=0,
                max_window_attack_score=None,
                mean_window_attack_score=None,
            )
            return AggregationResult(
                status=STATUS_INSUFFICIENT_DATA,
                prediction=None,
                pcap_attack_score=None,
                window_summary=summary,
                decision=decision,
            )

        summary = WindowSummary(
            total_windows=total,
            attack_windows=attack_windows,
            benign_windows=total - attack_windows,
            max_window_attack_score=self._score_max,
            mean_window_attack_score=self._score_sum / total,
        )
        if total < int(self.minimum_complete_windows):
            return AggregationResult(
                status=STATUS_INSUFFICIENT_DATA,
                prediction=None,
                pcap_attack_score=None,
                window_summary=summary,
                decision=decision,
            )

        rate = attack_windows / total
        if attack_windows >= int(self.min_attack_windows) and rate >= float(
            self.attack_rate_threshold
        ):
            prediction = PREDICTION_ATTACK
        else:
            prediction = PREDICTION_BENIGN
        return AggregationResult(
            status=STATUS_OK,
            prediction=prediction,
            pcap_attack_score=rate,
            window_summary=summary,
            decision=decision,
        )


def aggregate_window_scores(
    scores: Sequence[float],
    *,
    window_threshold: float = WINDOW_ATTACK_THRESHOLD,
    minimum_complete_windows: int = FROZEN_MIN_COMPLETE_WINDOWS,
    min_attack_windows: int = FROZEN_MIN_ATTACK_WINDOWS,
    attack_rate_threshold: float = FROZEN_ATTACK_RATE_THRESHOLD,
) -> AggregationResult:
    """Aggregate complete-window scores into a PCAP-level decision.

    ``scores`` must contain one uncalibrated window_attack_score per complete
    window. Incomplete trailing windows must not be included by the caller.
    """
    agg = StreamingWindowAggregator(
        window_threshold=window_threshold,
        minimum_complete_windows=minimum_complete_windows,
        min_attack_windows=min_attack_windows,
        attack_rate_threshold=attack_rate_threshold,
    )
    agg.observe_many(scores)
    return agg.finalize()
