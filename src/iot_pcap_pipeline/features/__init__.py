"""Phase 1C.2 V1 feature extraction package."""

from iot_pcap_pipeline.features.extractor import FeatureVector, extract_features
from iot_pcap_pipeline.features.schema import (
    GATE_B_DECISION,
    GATE_B_STATUS,
    V1_FEATURE_NAMES,
    write_feature_schema,
)
from iot_pcap_pipeline.features.validate import (
    FeatureInvariantError,
    validate_window_and_features,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

__all__ = [
    "GATE_B_DECISION",
    "GATE_B_STATUS",
    "V1_FEATURE_NAMES",
    "FeatureExtractionError",
    "FeatureInvariantError",
    "FeatureVector",
    "extract_features",
    "validate_window_and_features",
    "write_feature_schema",
]
