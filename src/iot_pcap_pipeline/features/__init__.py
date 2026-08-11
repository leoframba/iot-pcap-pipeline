"""V1 feature extraction package.

Import-light surface for serving and unit tests. Research/storage helpers
(Parquet builders, ARP/MQTT experiment modules) live in sibling modules and
must be imported explicitly.
"""

from iot_pcap_pipeline.features.extractor import FeatureVector, extract_features
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.features.validate import (
    FeatureInvariantError,
    validate_window_and_features,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

__all__ = [
    "FeatureExtractionError",
    "FeatureInvariantError",
    "FeatureVector",
    "V1_FEATURE_NAMES",
    "extract_features",
    "validate_window_and_features",
]
