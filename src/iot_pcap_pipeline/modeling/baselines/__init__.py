"""Phase 2B.2 unweighted binary IDS baselines (FIT view → TRAIN-validation)."""

from iot_pcap_pipeline.modeling.baselines.constants import (
    ATTACK_VAL_GROUPS,
    BASELINE_STRATEGY_VERSION,
    DECISION_THRESHOLD,
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_PCAPS,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_ATTACK,
    EXPECTED_VAL_BENIGN,
    EXPECTED_VAL_PCAPS,
    EXPECTED_VAL_ROWS,
    FORBIDDEN_MODEL_COLUMNS,
    IDENTITY_COLUMNS,
    LABEL_MAPPING,
    POSITIVE_CLASS,
    SMOKE_ROWS_PER_GROUP,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    format_prepare_baseline_summary,
    prepare_baseline_run,
)
from iot_pcap_pipeline.modeling.baselines.run import (
    format_baselines_summary,
    train_baselines,
)

__all__ = [
    "ATTACK_VAL_GROUPS",
    "BASELINE_STRATEGY_VERSION",
    "DECISION_THRESHOLD",
    "EXPECTED_FIT_ATTACK",
    "EXPECTED_FIT_BENIGN",
    "EXPECTED_FIT_PCAPS",
    "EXPECTED_FIT_ROWS",
    "EXPECTED_VAL_ATTACK",
    "EXPECTED_VAL_BENIGN",
    "EXPECTED_VAL_PCAPS",
    "EXPECTED_VAL_ROWS",
    "FORBIDDEN_MODEL_COLUMNS",
    "IDENTITY_COLUMNS",
    "LABEL_MAPPING",
    "POSITIVE_CLASS",
    "SMOKE_ROWS_PER_GROUP",
    "format_baselines_summary",
    "format_prepare_baseline_summary",
    "prepare_baseline_run",
    "train_baselines",
]
