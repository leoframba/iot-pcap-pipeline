"""Phase 1C.2 V1 feature extraction package (+ 1C.3a Parquet storage).

V2A ARP semantic features live in ``arp_v2`` and are intentionally separate
from the frozen V1 ``FeatureVector``.
"""

from iot_pcap_pipeline.features.arp_v2 import (
    ARP_V2_FEATURE_NAMES,
    ARP_V2_STRATEGY_VERSION,
    ArpSemanticFeatures,
    extract_arp_semantic_features,
)
from iot_pcap_pipeline.features.extractor import FeatureVector, extract_features
from iot_pcap_pipeline.features.parquet import (
    BuildResult,
    build_pcap_parquet,
    feature_parquet_arrow_schema,
)
from iot_pcap_pipeline.features.schema import (
    GATE_B_DECISION,
    GATE_B_STATUS,
    GATE_C_DECISION,
    GATE_C_STATUS,
    V1_FEATURE_NAMES,
    write_feature_schema,
    write_train_feature_contract,
)
from iot_pcap_pipeline.features.validate import (
    FeatureInvariantError,
    validate_window_and_features,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

__all__ = [
    "ARP_V2_FEATURE_NAMES",
    "ARP_V2_STRATEGY_VERSION",
    "GATE_B_DECISION",
    "GATE_B_STATUS",
    "GATE_C_DECISION",
    "GATE_C_STATUS",
    "V1_FEATURE_NAMES",
    "ArpSemanticFeatures",
    "BuildResult",
    "FeatureExtractionError",
    "FeatureInvariantError",
    "FeatureVector",
    "build_pcap_parquet",
    "extract_arp_semantic_features",
    "extract_features",
    "feature_parquet_arrow_schema",
    "validate_window_and_features",
    "write_feature_schema",
    "write_train_feature_contract",
]
