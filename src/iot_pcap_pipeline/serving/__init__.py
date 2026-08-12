"""D0 serving semantics (aggregation + contract). Import-light; no CLI/FastAPI."""

from __future__ import annotations

from iot_pcap_pipeline.serving.aggregate import (
    AggregationResult,
    aggregate_window_scores,
    window_is_attack,
)

__all__ = [
    "AggregationResult",
    "aggregate_window_scores",
    "window_is_attack",
]
