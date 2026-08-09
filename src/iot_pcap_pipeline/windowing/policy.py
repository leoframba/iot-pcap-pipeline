"""Phase 1C windowing policy constants and frozen V1 configuration.

Segmentation and windowing depend only on packet order, timestamps, and these
fixed configuration values — never on labels or other dataset metadata.

Gate A (Phase 1C.1) reviewed the TRAIN characterization grid and froze:

    WINDOW_SIZE = 25
    INACTIVITY_TIMEOUT_SECONDS = 5.0
    BACKWARD_RESET_SECONDS = 1.0

See ``frozen_window_policy()`` for the V1 contract used by Phase 1C.2+.
"""

from __future__ import annotations

from dataclasses import dataclass

from iot_pcap_pipeline.paths import WINDOWING_STRATEGY_VERSION

# ---------------------------------------------------------------------------
# Gate A — frozen V1 windowing policy (do not change without a new gate)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 25
INACTIVITY_TIMEOUT_SECONDS = 5.0
BACKWARD_RESET_SECONDS = 1.0

# Alias retained for callers / CLI defaults.
DEFAULT_BACKWARD_RESET_SECONDS = BACKWARD_RESET_SECONDS

GATE_A_STATUS = "passed"
GATE_A_DECISION = (
    "Freeze WINDOW_SIZE=25, INACTIVITY_TIMEOUT_SECONDS=5.0, "
    "BACKWARD_RESET_SECONDS=1.0 after TRAIN characterization review "
    "(phase1c1_v2; 85 PCAPs × 6 policies)."
)

# Phase 1C.1 characterization candidate grid (historical; not the V1 runtime).
CANDIDATE_WINDOW_SIZES: tuple[int, ...] = (25, 50, 100)
CANDIDATE_INACTIVITY_TIMEOUTS: tuple[float, ...] = (5.0, 30.0)

__all__ = [
    "BACKWARD_RESET_SECONDS",
    "CANDIDATE_INACTIVITY_TIMEOUTS",
    "CANDIDATE_WINDOW_SIZES",
    "DEFAULT_BACKWARD_RESET_SECONDS",
    "GATE_A_DECISION",
    "GATE_A_STATUS",
    "INACTIVITY_TIMEOUT_SECONDS",
    "WINDOWING_STRATEGY_VERSION",
    "WINDOW_SIZE",
    "WindowPolicy",
    "candidate_policies",
    "frozen_window_policy",
]


@dataclass(frozen=True)
class WindowPolicy:
    """Fixed packet-window + timestamp-boundary configuration."""

    window_size: int
    inactivity_timeout_seconds: float
    backward_reset_seconds: float = BACKWARD_RESET_SECONDS

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


def frozen_window_policy() -> WindowPolicy:
    """Return the Gate-A-frozen V1 windowing policy (25 / 5s / 1s)."""
    return WindowPolicy(
        window_size=WINDOW_SIZE,
        inactivity_timeout_seconds=INACTIVITY_TIMEOUT_SECONDS,
        backward_reset_seconds=BACKWARD_RESET_SECONDS,
    )


def candidate_policies(
    *,
    window_sizes: tuple[int, ...] = CANDIDATE_WINDOW_SIZES,
    inactivity_timeouts: tuple[float, ...] = CANDIDATE_INACTIVITY_TIMEOUTS,
    backward_reset_seconds: float = BACKWARD_RESET_SECONDS,
) -> list[WindowPolicy]:
    """Return the Phase 1C.1 candidate grid (default: 6 configs).

    Characterization only. Runtime extraction must use ``frozen_window_policy()``.
    """
    return [
        WindowPolicy(
            window_size=size,
            inactivity_timeout_seconds=timeout,
            backward_reset_seconds=backward_reset_seconds,
        )
        for size in window_sizes
        for timeout in inactivity_timeouts
    ]
