"""Phase 1C windowing policy constants and candidate configurations.

Segmentation and windowing depend only on packet order, timestamps, and these
fixed configuration values — never on labels or other dataset metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from iot_pcap_pipeline.paths import WINDOWING_STRATEGY_VERSION

# Fixed V1 backward discontinuity threshold (not a candidate).
DEFAULT_BACKWARD_RESET_SECONDS = 1.0

CANDIDATE_WINDOW_SIZES: tuple[int, ...] = (25, 50, 100)
CANDIDATE_INACTIVITY_TIMEOUTS: tuple[float, ...] = (5.0, 30.0)

__all__ = [
    "CANDIDATE_INACTIVITY_TIMEOUTS",
    "CANDIDATE_WINDOW_SIZES",
    "DEFAULT_BACKWARD_RESET_SECONDS",
    "WINDOWING_STRATEGY_VERSION",
    "WindowPolicy",
    "candidate_policies",
]


@dataclass(frozen=True)
class WindowPolicy:
    """Fixed packet-window + timestamp-boundary configuration."""

    window_size: int
    inactivity_timeout_seconds: float
    backward_reset_seconds: float = DEFAULT_BACKWARD_RESET_SECONDS

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.inactivity_timeout_seconds <= 0:
            raise ValueError(
                "inactivity_timeout_seconds must be > 0, "
                f"got {self.inactivity_timeout_seconds}"
            )
        if self.backward_reset_seconds <= 0:
            raise ValueError(
                "backward_reset_seconds must be > 0, "
                f"got {self.backward_reset_seconds}"
            )

    @property
    def config_id(self) -> str:
        return (
            f"w{self.window_size}"
            f"_gap{self.inactivity_timeout_seconds:g}"
            f"_bw{self.backward_reset_seconds:g}"
        )


def candidate_policies(
    *,
    window_sizes: tuple[int, ...] = CANDIDATE_WINDOW_SIZES,
    inactivity_timeouts: tuple[float, ...] = CANDIDATE_INACTIVITY_TIMEOUTS,
    backward_reset_seconds: float = DEFAULT_BACKWARD_RESET_SECONDS,
) -> list[WindowPolicy]:
    """Return the Phase 1C.1 candidate grid (default: 6 configs)."""
    return [
        WindowPolicy(
            window_size=size,
            inactivity_timeout_seconds=timeout,
            backward_reset_seconds=backward_reset_seconds,
        )
        for size in window_sizes
        for timeout in inactivity_timeouts
    ]
