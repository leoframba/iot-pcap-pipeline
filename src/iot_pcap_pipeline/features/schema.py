"""Phase 1C.2 V1 feature contract (27 ordered numeric features)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from iot_pcap_pipeline.paths import FEATURE_STRATEGY_VERSION, PROJECT_ROOT
from iot_pcap_pipeline.windowing.policy import (
    BACKWARD_RESET_SECONDS,
    INACTIVITY_TIMEOUT_SECONDS,
    WINDOW_SIZE,
)

DType = Literal["float64", "int64"]

GATE_B_STATUS = "passed"
GATE_B_DECISION = (
    "Freeze FEATURE_STRATEGY_VERSION=phase1c2_v1 with all 27 ordered features "
    "after TRAIN smoke review. Keep tcp_urg_ratio despite smoke-constant zero; "
    "after full TRAIN extraction, report globally constant features and exclude "
    "from model input only via an explicit pre-training schema/model-contract "
    "decision (TEST must not be consulted)."
)

V1_FEATURE_NAMES: tuple[str, ...] = (
    "window_span_seconds",
    "iat_mean_seconds",
    "iat_std_seconds",
    "iat_p50_seconds",
    "iat_p95_seconds",
    "frame_length_mean",
    "frame_length_std",
    "frame_length_min",
    "frame_length_max",
    "ipv4_ratio",
    "ipv6_ratio",
    "arp_ratio",
    "llc_ratio",
    "other_protocol_ratio",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "icmpv6_ratio",
    "igmp_ratio",
    "tcp_syn_ratio",
    "tcp_ack_ratio",
    "tcp_fin_ratio",
    "tcp_rst_ratio",
    "tcp_psh_ratio",
    "tcp_urg_ratio",
    "unique_ip_count",
    "unique_port_count",
)

assert len(V1_FEATURE_NAMES) == 27

DEFAULT_FEATURE_SCHEMA_PATH = (
    PROJECT_ROOT / "data" / "features" / "v1" / "feature_schema.json"
)

METADATA_COLUMN_NAMES: tuple[str, ...] = (
    "pcap_path",
    "split",
    "binary_label",
    "attack_family",
    "attack_type",
    "profiling_type",
    "profiling_variant",
    "device",
    "capture_session",
    "segment_index",
    "window_index",
    "packet_index_start",
    "packet_index_end",
    "feature_strategy_version",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    index: int
    dtype: DType
    unit: str
    definition: str
    denominator: str | None = None


V1_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "window_span_seconds",
        0,
        "float64",
        "seconds",
        "max(timestamps) - min(timestamps) over the 25 packets; always >= 0",
    ),
    FeatureSpec(
        "iat_mean_seconds",
        1,
        "float64",
        "seconds",
        "Mean of 24 sanitized adjacent IATs; IAT = max(ts[i]-ts[i-1], 0)",
    ),
    FeatureSpec(
        "iat_std_seconds",
        2,
        "float64",
        "seconds",
        "Population std (ddof=0) of the 24 sanitized IATs",
    ),
    FeatureSpec(
        "iat_p50_seconds",
        3,
        "float64",
        "seconds",
        "Median (linear-interpolation p50) of the 24 sanitized IATs",
    ),
    FeatureSpec(
        "iat_p95_seconds",
        4,
        "float64",
        "seconds",
        "Linear-interpolation p95 of the 24 sanitized IATs",
    ),
    FeatureSpec(
        "frame_length_mean",
        5,
        "float64",
        "bytes",
        "Mean captured frame length (PacketRecord.frame_length)",
    ),
    FeatureSpec(
        "frame_length_std",
        6,
        "float64",
        "bytes",
        "Population std (ddof=0) of captured frame lengths",
    ),
    FeatureSpec(
        "frame_length_min",
        7,
        "float64",
        "bytes",
        "Minimum captured frame length",
    ),
    FeatureSpec(
        "frame_length_max",
        8,
        "float64",
        "bytes",
        "Maximum captured frame length",
    ),
    FeatureSpec(
        "ipv4_ratio",
        9,
        "float64",
        "ratio",
        "IPv4 packet count / 25 (exclusive L3 partition)",
        "window_size",
    ),
    FeatureSpec(
        "ipv6_ratio",
        10,
        "float64",
        "ratio",
        "IPv6 packet count / 25 (exclusive L3 partition)",
        "window_size",
    ),
    FeatureSpec(
        "arp_ratio",
        11,
        "float64",
        "ratio",
        "ARP packet count / 25 (exclusive L3 partition)",
        "window_size",
    ),
    FeatureSpec(
        "llc_ratio",
        12,
        "float64",
        "ratio",
        "LLC / IEEE 802.3 packet count / 25 (exclusive L3 partition)",
        "window_size",
    ),
    FeatureSpec(
        "other_protocol_ratio",
        13,
        "float64",
        "ratio",
        "Other / unsupported / unrecognized L3 count / 25",
        "window_size",
    ),
    FeatureSpec(
        "tcp_ratio",
        14,
        "float64",
        "ratio",
        "TCP packet count / 25",
        "window_size",
    ),
    FeatureSpec(
        "udp_ratio",
        15,
        "float64",
        "ratio",
        "UDP packet count / 25",
        "window_size",
    ),
    FeatureSpec(
        "icmp_ratio",
        16,
        "float64",
        "ratio",
        "ICMPv4 packet count / 25",
        "window_size",
    ),
    FeatureSpec(
        "icmpv6_ratio",
        17,
        "float64",
        "ratio",
        "ICMPv6 packet count / 25",
        "window_size",
    ),
    FeatureSpec(
        "igmp_ratio",
        18,
        "float64",
        "ratio",
        "IGMP packet count / 25",
        "window_size",
    ),
    FeatureSpec(
        "tcp_syn_ratio",
        19,
        "float64",
        "ratio",
        "TCP packets with SYN / tcp_packet_count (0 if no TCP)",
        "tcp_packet_count",
    ),
    FeatureSpec(
        "tcp_ack_ratio",
        20,
        "float64",
        "ratio",
        "TCP packets with ACK / tcp_packet_count (0 if no TCP)",
        "tcp_packet_count",
    ),
    FeatureSpec(
        "tcp_fin_ratio",
        21,
        "float64",
        "ratio",
        "TCP packets with FIN / tcp_packet_count (0 if no TCP)",
        "tcp_packet_count",
    ),
    FeatureSpec(
        "tcp_rst_ratio",
        22,
        "float64",
        "ratio",
        "TCP packets with RST / tcp_packet_count (0 if no TCP)",
        "tcp_packet_count",
    ),
    FeatureSpec(
        "tcp_psh_ratio",
        23,
        "float64",
        "ratio",
        "TCP packets with PSH / tcp_packet_count (0 if no TCP)",
        "tcp_packet_count",
    ),
    FeatureSpec(
        "tcp_urg_ratio",
        24,
        "float64",
        "ratio",
        (
            "TCP packets with URG / tcp_packet_count (0 if no TCP). "
            "Retained in phase1c2_v1 even if smoke-constant; defer drop "
            "decision until full TRAIN constant-feature report"
        ),
        "tcp_packet_count",
    ),
    FeatureSpec(
        "unique_ip_count",
        25,
        "int64",
        "count",
        "Cardinality of {src_ip, dst_ip} ignoring None; undirected",
    ),
    FeatureSpec(
        "unique_port_count",
        26,
        "int64",
        "count",
        "Cardinality of {src_port, dst_port} ignoring None; undirected",
    ),
)

assert tuple(s.name for s in V1_FEATURE_SPECS) == V1_FEATURE_NAMES


def build_feature_schema_document() -> dict[str, Any]:
    """Canonical schema document for data/features/v1/feature_schema.json."""
    return {
        "feature_strategy_version": FEATURE_STRATEGY_VERSION,
        "gate_b_status": GATE_B_STATUS,
        "gate_b_decision": GATE_B_DECISION,
        "windowing": {
            "window_size": WINDOW_SIZE,
            "inactivity_timeout_seconds": INACTIVITY_TIMEOUT_SECONDS,
            "backward_reset_seconds": BACKWARD_RESET_SECONDS,
        },
        "globals": {
            "window_span": "max(timestamps) - min(timestamps)",
            "iat_sanitization": "iat = max(raw_delta, 0.0)",
            "std": "population (ddof=0)",
            "percentile_method": "linear_interpolation",
            "tcp_zero_denominator": "all six TCP flag ratios = 0.0",
            "malformed_policy": "include in windows",
            "unsupported_policy": "include in windows",
            "partial_policy": "include in windows",
            "error_policy": "abort feature extraction for the PCAP",
            "partial_windows": "dropped at boundaries and EOF",
            "feature_count": len(V1_FEATURE_NAMES),
            "constant_feature_policy": (
                "After full TRAIN feature build, report globally constant "
                "features. Exclusion from model input is allowed only before "
                "model training via an explicit schema/model-contract decision. "
                "TEST must not be consulted for that decision. Extraction still "
                "emits all 27 V1 columns unless/until the contract is revised."
            ),
            "full_build_note": (
                "Phase 1C.3 must stream windows to Parquet; do not accumulate "
                "all FeatureVector rows in memory via the smoke list wrapper."
            ),
        },
        "features": [asdict(spec) for spec in V1_FEATURE_SPECS],
        "metadata_columns_not_features": list(METADATA_COLUMN_NAMES),
    }


def write_feature_schema(path: Path | str | None = None) -> Path:
    out = Path(path or DEFAULT_FEATURE_SCHEMA_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = build_feature_schema_document()
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out
