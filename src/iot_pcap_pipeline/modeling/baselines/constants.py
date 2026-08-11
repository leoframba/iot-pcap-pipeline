"""Phase 2B.2 constants (no heavy imports)."""

from __future__ import annotations

BASELINE_STRATEGY_VERSION = "phase2b2_v1"

LABEL_MAPPING: dict[str, int] = {
    "BENIGN": 0,
    "ATTACK": 1,
}
POSITIVE_CLASS = "ATTACK"
DECISION_THRESHOLD = 0.5

EXPECTED_FIT_PCAPS = 65
EXPECTED_FIT_ROWS = 704_305
EXPECTED_FIT_ATTACK = 493_235
EXPECTED_FIT_BENIGN = 211_070
EXPECTED_VAL_PCAPS = 20
EXPECTED_VAL_ROWS = 4_944_060
EXPECTED_VAL_ATTACK = 4_921_556
EXPECTED_VAL_BENIGN = 22_504

IDENTITY_COLUMNS: tuple[str, ...] = (
    "pcap_id",
    "binary_label",
    "segment_index",
    "window_index",
    "packet_index_start",
    "packet_index_end",
)

FORBIDDEN_MODEL_COLUMNS: frozenset[str] = frozenset(
    {
        "pcap_id",
        "binary_label",
        "segment_index",
        "window_index",
        "packet_index_start",
        "packet_index_end",
        "attack_family",
        "attack_type",
        "device",
        "modeling_group_key",
        "pcap_path",
        "split",
        "profiling_type",
        "benign_category",
        "group_kind",
    }
)

ATTACK_VAL_GROUPS: tuple[str, ...] = (
    "DDoS|DDoS_TCP",
    "DoS|DoS_TCP",
    "MQTT|MQTT_DoS_Publish_Flood",
    "Recon|OS_Scan",
)

# Real-corpus smoke: small fixed slices per FIT/VAL group (not a global row cap).
SMOKE_ROWS_PER_GROUP = 750
SMOKE_FIT_BUCKETS: tuple[str, ...] = (
    "benign",
    "DDoS",
    "DoS",
    "MQTT",
    "Recon",
    "Spoofing",
)
SMOKE_VAL_ATTACK_GROUPS: tuple[str, ...] = ATTACK_VAL_GROUPS
SMOKE_VAL_BENIGN_GROUPS: tuple[str, ...] = (
    "profiling_idle",
    "owltron_interaction",
    "owltron_power",
)
