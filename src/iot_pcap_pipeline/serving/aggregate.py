"""Pure PCAP aggregation over window attack scores (no decode / joblib / HTTP)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from iot_pcap_pipeline.serving.contract import (
    FROZEN_ATTACK_RATE_THRESHOLD,
    FROZEN_MIN_ATTACK_WINDOWS,
    FROZEN_MIN_COMPLETE_WINDOWS,
)
from iot_pcap_pipeline.serving.candidates import WINDOW_ATTACK_THRESHOLD

STATUS_OK = "OK"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
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
    vals = [float(s) for s in scores]
    total = len(vals)
    decision = AggregationDecision(
        window_attack_threshold=float(window_threshold),
        minimum_complete_windows=int(minimum_complete_windows),
        pcap_min_attack_windows=int(min_attack_windows),
        pcap_attack_rate_threshold=float(attack_rate_threshold),
    )

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

    attack_flags = [window_is_attack(s, window_threshold=window_threshold) for s in vals]
    attack_windows = int(sum(1 for flag in attack_flags if flag))
    benign_windows = total - attack_windows
    score_max = max(vals)
    score_mean = sum(vals) / total
    rate = attack_windows / total

    summary = WindowSummary(
        total_windows=total,
        attack_windows=attack_windows,
        benign_windows=benign_windows,
        max_window_attack_score=score_max,
        mean_window_attack_score=score_mean,
    )

    if total < int(minimum_complete_windows):
        return AggregationResult(
            status=STATUS_INSUFFICIENT_DATA,
            prediction=None,
            pcap_attack_score=None,
            window_summary=summary,
            decision=decision,
        )

    if attack_windows >= int(min_attack_windows) and rate >= float(attack_rate_threshold):
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
