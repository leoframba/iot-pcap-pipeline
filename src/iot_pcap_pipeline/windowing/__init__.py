"""Phase 1C segmentation / fixed-window utilities."""

from iot_pcap_pipeline.windowing.characterize import (
    CharacterizationResult,
    characterize_pcap,
    characterize_timestamps,
    characterize_train_windowing,
    format_characterization_summary,
)
from iot_pcap_pipeline.windowing.policy import (
    BACKWARD_RESET_SECONDS,
    DEFAULT_BACKWARD_RESET_SECONDS,
    GATE_A_DECISION,
    GATE_A_STATUS,
    INACTIVITY_TIMEOUT_SECONDS,
    WINDOW_SIZE,
    WINDOWING_STRATEGY_VERSION,
    WindowPolicy,
    candidate_policies,
    frozen_window_policy,
)
from iot_pcap_pipeline.windowing.stream import (
    FeatureExtractionError,
    WindowStreamStats,
    iter_windows,
)
from iot_pcap_pipeline.windowing.window import PacketWindow

__all__ = [
    "BACKWARD_RESET_SECONDS",
    "DEFAULT_BACKWARD_RESET_SECONDS",
    "GATE_A_DECISION",
    "GATE_A_STATUS",
    "INACTIVITY_TIMEOUT_SECONDS",
    "WINDOWING_STRATEGY_VERSION",
    "WINDOW_SIZE",
    "CharacterizationResult",
    "FeatureExtractionError",
    "PacketWindow",
    "WindowPolicy",
    "WindowStreamStats",
    "candidate_policies",
    "characterize_pcap",
    "characterize_timestamps",
    "characterize_train_windowing",
    "format_characterization_summary",
    "frozen_window_policy",
    "iter_windows",
]
