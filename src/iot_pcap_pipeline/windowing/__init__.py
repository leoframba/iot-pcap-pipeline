"""Phase 1C segmentation / fixed-window utilities."""

from iot_pcap_pipeline.windowing.characterize import (
    CharacterizationResult,
    characterize_pcap,
    characterize_timestamps,
    characterize_train_windowing,
    format_characterization_summary,
)
from iot_pcap_pipeline.windowing.policy import (
    DEFAULT_BACKWARD_RESET_SECONDS,
    WINDOWING_STRATEGY_VERSION,
    WindowPolicy,
    candidate_policies,
)

__all__ = [
    "DEFAULT_BACKWARD_RESET_SECONDS",
    "WINDOWING_STRATEGY_VERSION",
    "CharacterizationResult",
    "WindowPolicy",
    "candidate_policies",
    "characterize_pcap",
    "characterize_timestamps",
    "characterize_train_windowing",
    "format_characterization_summary",
]
