"""Command-line interface for iot-pcap-pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from iot_pcap_pipeline.audit.live_progress import DEFAULT_PROGRESS_EVERY_PACKETS
from iot_pcap_pipeline.audit.policy import (
    DEFAULT_ISSUE_CAP_PER_CODE,
    DEFAULT_MALFORMED_CATASTROPHIC_RATE,
    DEFAULT_MALFORMED_HIGH_WARNING_RATE,
    DEFAULT_WORKERS,
)
from iot_pcap_pipeline.audit.scan import audit_corpus
from iot_pcap_pipeline.dataset.build import build_manifests
from iot_pcap_pipeline.features.arp_v2_probe import (
    DEFAULT_ARP_PROBE_DIR,
    run_arp_fit_probe,
)
from iot_pcap_pipeline.features.arp_v2_stateful_probe import (
    run_arp_stateful_feasibility_probe,
)
from iot_pcap_pipeline.features.mqtt_v2_probe import (
    DEFAULT_MQTT_PROBE_DIR,
    run_mqtt_fit_probe,
)
from iot_pcap_pipeline.features.build import (
    DEFAULT_MAX_WINDOWS_PER_PCAP,
    DEFAULT_SMOKE_FEATURES_CSV,
    extract_pcap_feature_rows,
    format_smoke_summary,
    load_inventory_index,
    run_smoke_extraction,
)
from iot_pcap_pipeline.features.characterize import write_characterization_csv
from iot_pcap_pipeline.features.characterize_dataset import (
    DEFAULT_GROUP_CHARACTERIZATION_JSON,
    DEFAULT_GROUP_SUMMARY_CSV,
    DEFAULT_PCAP_DIAGNOSTICS_CSV,
    DEFAULT_PERCENTILE_SAMPLE_CAP,
    characterize_train_feature_groups,
    format_group_characterization_summary,
)
from iot_pcap_pipeline.features.dataset import (
    DEFAULT_BUILD_MANIFEST_PATH,
    DEFAULT_FEATURE_DATASET_WORKERS,
    DEFAULT_SMOKE_BUILD_MANIFEST_PATH,
    DEFAULT_SMOKE_CHECKPOINT_DIR,
    DEFAULT_SMOKE_DATASET_DIR,
    DEFAULT_SMOKE_TEST_BUILD_MANIFEST_PATH,
    DEFAULT_TEST_BUILD_MANIFEST_PATH,
    DEFAULT_TEST_CHECKPOINT_DIR,
    DEFAULT_TEST_PARQUET_DIR,
    DEFAULT_TRAIN_BUILD_COMPLETE_JSON,
    DEFAULT_TRAIN_PARQUET_DIR,
    EXPECTED_TEST_PCAP_COUNT,
    EXPECTED_TRAIN_PCAP_COUNT,
    build_feature_dataset,
    format_feature_dataset_summary,
)
from iot_pcap_pipeline.features.parquet import (
    DEFAULT_BUFFER_ROWS,
    DEFAULT_FEATURE_CHECKPOINT_DIR,
    DEFAULT_PARQUET_SMOKE_DIR,
    build_pcap_parquet,
    format_parquet_smoke_summary,
    pcap_id_from_path,
    run_parquet_smoke,
)
from iot_pcap_pipeline.features.schema import (
    METADATA_COLUMN_NAMES,
    V1_FEATURE_NAMES,
    write_feature_schema,
)
from iot_pcap_pipeline.features.validate import FeatureInvariantError
from iot_pcap_pipeline.features.validate_dataset import (
    DEFAULT_INTEGRITY_CSV,
    DEFAULT_TEST_BUILD_COMPLETE_JSON,
    DEFAULT_TRAIN_CONSTANT_FEATURES_CSV,
    DEFAULT_TRAIN_FEATURE_SUMMARY_CSV,
    format_validation_summary,
    validate_feature_dataset,
)
from iot_pcap_pipeline.modeling.characterize import (
    DEFAULT_MODELING_V1_DIR,
    DEFAULT_SAMPLING_PLAN_PATH,
    DEFAULT_SAMPLING_SUMMARY_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    characterize_modeling_split,
    format_modeling_characterization_summary,
)
from iot_pcap_pipeline.modeling.freeze import (
    FROZEN_SAMPLING_PLAN_ID,
    format_gate_2a_freeze_summary,
    freeze_gate_2a,
)
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_FIT_VIEW_ROOT,
    DEFAULT_TRAINING_VIEW_CONTRACT_PATH,
    build_modeling_fit_view,
    format_fit_view_summary,
)
from iot_pcap_pipeline.modeling.baselines import (
    format_baselines_summary,
    train_baselines,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    DEFAULT_BASELINE_CONTRACT_PATH,
    format_prepare_baseline_summary,
    prepare_baseline_run,
)
from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
    format_threshold_sweep_summary,
    run_threshold_sweep,
)
from iot_pcap_pipeline.modeling.baselines.ablations import (
    format_ablation_summary,
    run_hgb_ablations,
)
from iot_pcap_pipeline.modeling.baselines.c_threshold_refine import (
    format_c_threshold_refine_summary,
    run_c_threshold_refine,
)
from iot_pcap_pipeline.modeling.baselines.model_family import (
    format_model_family_summary,
    format_prepare_model_family_summary,
    prepare_model_family_bakeoff,
    run_model_family_bakeoff,
)
from iot_pcap_pipeline.modeling.baselines.extratrees import (
    format_extratrees_summary,
    format_prepare_extratrees_summary,
    prepare_extratrees_challenger,
    run_extratrees_challenger,
)
from iot_pcap_pipeline.modeling.baselines.external_boosting import (
    format_external_boost_summary,
    format_prepare_external_boost_summary,
    prepare_external_boost_challengers,
    run_external_boost_challengers,
)
from iot_pcap_pipeline.modeling.baselines.feature22_boost import (
    format_feature22_boost_summary,
    format_prepare_feature22_boost_summary,
    prepare_feature22_boost_rematch,
    run_feature22_boost_rematch,
)
from iot_pcap_pipeline.modeling.baselines.v1_candidate_freeze import (
    format_v1_candidate_freeze_summary,
    freeze_v1_candidate,
)
from iot_pcap_pipeline.modeling.baselines.hgb_sensitivity import (
    format_hgb_sensitivity_summary,
    format_prepare_hgb_sensitivity_summary,
    prepare_hgb_sensitivity,
    run_hgb_sensitivity,
)
from iot_pcap_pipeline.modeling.baselines.phase2c_freeze import (
    format_phase2c_freeze_summary,
    freeze_phase2c,
)
from iot_pcap_pipeline.modeling.baselines.final_test import (
    format_final_test_summary,
    format_prepare_final_test_summary,
    format_preflight_final_test_summary,
    prepare_final_test,
    preflight_final_test,
    run_final_test,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MODELING_SEED,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_SEED,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.pcap.reader import summarize_pcap
from iot_pcap_pipeline.pcap.stats import DEFAULT_IP_CARDINALITY_CAP
from iot_pcap_pipeline.pcap.timestamps import (
    DEFAULT_EXAMPLE_LIMIT,
    DEFAULT_EXAMPLES_CSV,
    DEFAULT_LARGEST_EXAMPLE_LIMIT,
    DEFAULT_POSITIVE_SAMPLE_CAP,
    DEFAULT_PROBE_CSV,
    format_probe_summary,
    probe_timestamps,
    resolve_pcap_path,
    write_probe_artifacts,
)
from iot_pcap_pipeline.windowing.characterize import (
    DEFAULT_CHARACTERIZATION_CSV,
    DEFAULT_SPAN_SAMPLE_CAP,
    characterize_train_windowing,
    format_characterization_summary,
)
from iot_pcap_pipeline.windowing.characterize import (
    DEFAULT_WORKERS as WINDOWING_DEFAULT_WORKERS,
)
from iot_pcap_pipeline.windowing.policy import (
    DEFAULT_BACKWARD_RESET_SECONDS,
    candidate_policies,
)
from iot_pcap_pipeline.serving.classify import DEFAULT_SCORE_BATCH_SIZE, classify_pcap
from iot_pcap_pipeline.serving.evaluate_aggregation import (
    DEFAULT_SERVING_DIR,
    format_review_summary,
    write_aggregation_evaluation,
)
from iot_pcap_pipeline.serving.model import V1InferenceEngine
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iot-pcap-pipeline",
        description="CICIoMT2024 Wi-Fi/MQTT PCAP preprocessing utilities",
    )
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser(
        "build-manifests",
        help="Build Phase 1A PCAP inventory and deterministic split manifests",
    )
    build.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Raw CICIoMT Wi-Fi/MQTT root (default: {DEFAULT_RAW_ROOT})",
    )
    build.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=f"Manifest output directory (default: {DEFAULT_MANIFEST_DIR})",
    )
    build.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"RNG seed for profiling split tie-breaks (default: {DEFAULT_SPLIT_SEED})",
    )

    inspect_cmd = subparsers.add_parser(
        "inspect-pcaps",
        help="Stream-decode one or more PCAPs and print packet/protocol/error counts",
    )
    inspect_cmd.add_argument(
        "pcaps",
        nargs="+",
        type=Path,
        help="PCAP paths (absolute or repo-relative)",
    )
    inspect_cmd.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Optional per-file packet cap (useful for large flood captures)",
    )

    audit_cmd = subparsers.add_parser(
        "audit-corpus",
        help="Phase 1B.2 full-corpus integrity audit and TRAIN characterization",
    )
    audit_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="Path to pcap_inventory.csv",
    )
    audit_cmd.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "dataset_split.csv",
        help="Path to dataset_split.csv",
    )
    audit_cmd.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Raw CICIoMT Wi-Fi/MQTT root (default: {DEFAULT_RAW_ROOT})",
    )
    audit_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help=f"Audit output directory (default: {DEFAULT_AUDIT_DIR})",
    )
    audit_cmd.add_argument(
        "--ip-cardinality-cap",
        type=int,
        default=DEFAULT_IP_CARDINALITY_CAP,
        help=f"Exact unique IP cap per field per TRAIN PCAP (default: {DEFAULT_IP_CARDINALITY_CAP})",
    )
    audit_cmd.add_argument(
        "--issue-cap-per-code",
        type=int,
        default=DEFAULT_ISSUE_CAP_PER_CODE,
        help=f"Max packet examples per (pcap, issue_code) (default: {DEFAULT_ISSUE_CAP_PER_CODE})",
    )
    audit_cmd.add_argument(
        "--malformed-high-rate",
        type=float,
        default=DEFAULT_MALFORMED_HIGH_WARNING_RATE,
        help="Malformed rate for high warning (default: 0.01)",
    )
    audit_cmd.add_argument(
        "--malformed-catastrophic-rate",
        type=float,
        default=DEFAULT_MALFORMED_CATASTROPHIC_RATE,
        help="Malformed rate for hard failure (default: 0.80)",
    )
    audit_cmd.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Process pool size across PCAPs (default: {DEFAULT_WORKERS}; use 1 for sequential)",
    )
    resume_group = audit_cmd.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse valid per-PCAP checkpoints (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and rescan all PCAPs",
    )
    audit_cmd.set_defaults(resume=True)
    audit_cmd.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: <output-dir>/.work)",
    )
    audit_cmd.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Delete existing checkpoints before scanning",
    )
    audit_cmd.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_PACKETS,
        dest="progress_every_packets",
        help=(
            "Emit live per-file packet progress every N packets "
            f"(default: {DEFAULT_PROGRESS_EVERY_PACKETS:,}; 0 disables mid-file updates)"
        ),
    )

    probe_cmd = subparsers.add_parser(
        "probe-timestamps",
        help=(
            "Phase 1B.3 timestamp-only ordering probe "
            "(adjacent deltas; no frame decoding)"
        ),
    )
    probe_cmd.add_argument(
        "pcaps",
        nargs="+",
        type=Path,
        help="PCAP paths (absolute or repo-relative)",
    )
    probe_cmd.add_argument(
        "--example-limit",
        type=int,
        default=DEFAULT_EXAMPLE_LIMIT,
        help=(
            "Max first-seen reversal examples retained per PCAP "
            f"(default: {DEFAULT_EXAMPLE_LIMIT})"
        ),
    )
    probe_cmd.add_argument(
        "--largest-example-limit",
        type=int,
        default=DEFAULT_LARGEST_EXAMPLE_LIMIT,
        help=(
            "Max largest-by-magnitude reversal examples retained per PCAP "
            f"(default: {DEFAULT_LARGEST_EXAMPLE_LIMIT})"
        ),
    )
    probe_cmd.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROBE_CSV,
        help=f"Summary CSV path (default: {DEFAULT_PROBE_CSV})",
    )
    probe_cmd.add_argument(
        "--examples-output",
        type=Path,
        default=DEFAULT_EXAMPLES_CSV,
        help=f"Reversal examples CSV path (default: {DEFAULT_EXAMPLES_CSV})",
    )
    probe_cmd.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip writing the reversal examples CSV",
    )
    probe_cmd.add_argument(
        "--positive-sample-cap",
        type=int,
        default=DEFAULT_POSITIVE_SAMPLE_CAP,
        help=(
            "Reservoir sample size for positive-delta percentiles "
            f"(default: {DEFAULT_POSITIVE_SAMPLE_CAP:,}; exact when count <= cap)"
        ),
    )
    probe_cmd.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Optional per-file packet cap (for smoke tests on large floods)",
    )

    win_cmd = subparsers.add_parser(
        "characterize-windowing",
        help=(
            "Phase 1C.1 TRAIN-only windowing-policy characterization "
            "(timestamp-only; Gate A — stop for review)"
        ),
    )
    win_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="Path to pcap_inventory.csv",
    )
    win_cmd.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHARACTERIZATION_CSV,
        help=f"Characterization CSV path (default: {DEFAULT_CHARACTERIZATION_CSV})",
    )
    win_cmd.add_argument(
        "--workers",
        type=int,
        default=WINDOWING_DEFAULT_WORKERS,
        help=(
            "Process pool size across TRAIN PCAPs "
            f"(default: {WINDOWING_DEFAULT_WORKERS}; use 1 for sequential)"
        ),
    )
    win_cmd.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Optional per-file packet cap (smoke tests only)",
    )
    win_cmd.add_argument(
        "--backward-reset",
        type=float,
        default=DEFAULT_BACKWARD_RESET_SECONDS,
        help=(
            "Backward discontinuity threshold in seconds "
            f"(default: {DEFAULT_BACKWARD_RESET_SECONDS})"
        ),
    )
    win_cmd.add_argument(
        "--span-sample-cap",
        type=int,
        default=DEFAULT_SPAN_SAMPLE_CAP,
        help=(
            "Reservoir sample size for window-span percentiles "
            f"(default: {DEFAULT_SPAN_SAMPLE_CAP:,}; exact when count <= cap)"
        ),
    )

    feat_cmd = subparsers.add_parser(
        "extract-features",
        help=(
            "Phase 1C.2 V1 windowing + feature extraction "
            "(Gate B smoke / arbitrary PCAPs)"
        ),
    )
    feat_cmd.add_argument(
        "pcaps",
        nargs="*",
        type=Path,
        help="PCAP paths (absolute or repo-relative). Omit with --smoke.",
    )
    feat_cmd.add_argument(
        "--smoke",
        action="store_true",
        help="Run the representative TRAIN-only Gate B smoke set",
    )
    feat_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="Inventory for optional metadata join (not required for extraction)",
    )
    feat_cmd.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SMOKE_FEATURES_CSV,
        help=f"Feature CSV path (default: {DEFAULT_SMOKE_FEATURES_CSV})",
    )
    feat_cmd.add_argument(
        "--characterization-output",
        type=Path,
        default=None,
        help="Optional feature characterization CSV path (default under smoke dir)",
    )
    feat_cmd.add_argument(
        "--max-windows-per-pcap",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_PCAP,
        help=(
            "Stop after N emitted windows per PCAP "
            f"(default: {DEFAULT_MAX_WINDOWS_PER_PCAP:,}; use 0 for uncapped)"
        ),
    )
    feat_cmd.add_argument(
        "--write-schema-only",
        action="store_true",
        help="Only write data/features/v1/feature_schema.json and exit",
    )

    pq_cmd = subparsers.add_parser(
        "build-feature-parquet",
        help=(
            "Phase 1C.3a streaming Parquet shards for frozen V1 features "
            "(atomic write + resume checkpoints)"
        ),
    )
    pq_cmd.add_argument(
        "pcaps",
        nargs="*",
        type=Path,
        help="PCAP paths (absolute or repo-relative). Omit with --smoke.",
    )
    pq_cmd.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the 1C.3a TRAIN-only Parquet smoke "
            "(Benign_train, Idle, Recon-VulScan_train)"
        ),
    )
    pq_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="Inventory for binary_label join",
    )
    pq_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PARQUET_SMOKE_DIR,
        help=f"Directory for <pcap-id>.parquet shards (default: {DEFAULT_PARQUET_SMOKE_DIR})",
    )
    pq_cmd.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_FEATURE_CHECKPOINT_DIR,
        help=(
            "Per-PCAP checkpoint directory "
            f"(default: {DEFAULT_FEATURE_CHECKPOINT_DIR})"
        ),
    )
    pq_cmd.add_argument(
        "--buffer-rows",
        type=int,
        default=DEFAULT_BUFFER_ROWS,
        help=f"In-memory Parquet write buffer size (default: {DEFAULT_BUFFER_ROWS:,})",
    )
    pq_cmd.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoints and rebuild every PCAP",
    )

    ds_cmd = subparsers.add_parser(
        "build-feature-dataset",
        help=(
            "Phase 1C.3 corpus orchestration: inventory → largest-first "
            "process pool → Parquet shards + build manifest "
            "(TEST requires passed train_build_complete.json)"
        ),
    )
    ds_cmd.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to build (train or test)",
    )
    ds_cmd.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_FEATURE_DATASET_WORKERS,
        help=(
            "Process pool size (default: "
            f"{DEFAULT_FEATURE_DATASET_WORKERS}; use 1 for sequential)"
        ),
    )
    ds_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="PCAP inventory CSV",
    )
    ds_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Per-PCAP Parquet shard directory "
            f"(train → {DEFAULT_TRAIN_PARQUET_DIR}; "
            f"test → {DEFAULT_TEST_PARQUET_DIR}; "
            f"train --smoke → {DEFAULT_SMOKE_DATASET_DIR})"
        ),
    )
    ds_cmd.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Per-PCAP checkpoint directory "
            f"(train → {DEFAULT_FEATURE_CHECKPOINT_DIR}; "
            f"test → {DEFAULT_TEST_CHECKPOINT_DIR}; "
            f"train --smoke → {DEFAULT_SMOKE_CHECKPOINT_DIR})"
        ),
    )
    ds_cmd.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "Build manifest CSV path "
            f"(train → {DEFAULT_BUILD_MANIFEST_PATH}; "
            f"test → {DEFAULT_TEST_BUILD_MANIFEST_PATH}; "
            f"train --smoke → {DEFAULT_SMOKE_BUILD_MANIFEST_PATH}; "
            f"test --smoke → {DEFAULT_SMOKE_TEST_BUILD_MANIFEST_PATH})"
        ),
    )
    ds_cmd.add_argument(
        "--buffer-rows",
        type=int,
        default=DEFAULT_BUFFER_ROWS,
        help=f"Parquet write buffer size (default: {DEFAULT_BUFFER_ROWS:,})",
    )
    ds_cmd.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Orchestration smoke: train → 6 modest TRAIN PCAPs; "
            "test → Benign_test + MQTT-DoS-Publish_Flood_test + "
            f"SenseUBaby_Power (skips full {EXPECTED_TRAIN_PCAP_COUNT}/"
            f"{EXPECTED_TEST_PCAP_COUNT} count assertion). "
            "TEST smoke writes canonical test/ + .work/test/ so --resume "
            "can reuse shards."
        ),
    )
    resume_group = ds_cmd.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse valid per-PCAP checkpoints (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoints and rebuild every PCAP",
    )
    ds_cmd.set_defaults(resume=True)

    val_cmd = subparsers.add_parser(
        "validate-feature-dataset",
        help=(
            "Read-only Parquet build validation "
            "(train: summaries + joins; test: structural only)"
        ),
    )
    val_cmd.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to validate",
    )
    val_cmd.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Build manifest CSV "
            f"(train default: {DEFAULT_BUILD_MANIFEST_PATH}; "
            f"test default: {DEFAULT_TEST_BUILD_MANIFEST_PATH})"
        ),
    )
    val_cmd.add_argument(
        "--integrity",
        type=Path,
        default=DEFAULT_INTEGRITY_CSV,
        help=f"pcap_integrity.csv (default: {DEFAULT_INTEGRITY_CSV})",
    )
    val_cmd.add_argument(
        "--characterization",
        type=Path,
        default=DEFAULT_CHARACTERIZATION_CSV,
        help=(
            "TRAIN windowing characterization CSV "
            f"(default: {DEFAULT_CHARACTERIZATION_CSV}; ignored for --split test)"
        ),
    )
    val_cmd.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_TRAIN_FEATURE_SUMMARY_CSV,
        help=(
            "TRAIN feature summary CSV "
            f"(default: {DEFAULT_TRAIN_FEATURE_SUMMARY_CSV}; ignored for test)"
        ),
    )
    val_cmd.add_argument(
        "--constant-output",
        type=Path,
        default=DEFAULT_TRAIN_CONSTANT_FEATURES_CSV,
        help=(
            "TRAIN constant-feature report CSV "
            f"(default: {DEFAULT_TRAIN_CONSTANT_FEATURES_CSV}; ignored for test)"
        ),
    )
    val_cmd.add_argument(
        "--complete-output",
        type=Path,
        default=None,
        help=(
            "Pass marker JSON written only when all checks pass "
            f"(train default: {DEFAULT_TRAIN_BUILD_COMPLETE_JSON}; "
            f"test default: {DEFAULT_TEST_BUILD_COMPLETE_JSON})"
        ),
    )

    char_ds_cmd = subparsers.add_parser(
        "characterize-feature-dataset",
        help=(
            "TRAIN-only read-only per-group feature characterization "
            "over existing Parquet shards (no PCAP decode)"
        ),
    )
    char_ds_cmd.add_argument(
        "--split",
        choices=["train"],
        default="train",
        help="Dataset split to characterize (only train for now)",
    )
    char_ds_cmd.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_BUILD_MANIFEST_PATH,
        help=f"Build manifest CSV (default: {DEFAULT_BUILD_MANIFEST_PATH})",
    )
    char_ds_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="Inventory for group labels",
    )
    char_ds_cmd.add_argument(
        "--group-summary-output",
        type=Path,
        default=DEFAULT_GROUP_SUMMARY_CSV,
        help=f"Per-group feature summary CSV (default: {DEFAULT_GROUP_SUMMARY_CSV})",
    )
    char_ds_cmd.add_argument(
        "--pcap-diagnostics-output",
        type=Path,
        default=DEFAULT_PCAP_DIAGNOSTICS_CSV,
        help=(
            "Per-PCAP diagnostic CSV "
            f"(default: {DEFAULT_PCAP_DIAGNOSTICS_CSV})"
        ),
    )
    char_ds_cmd.add_argument(
        "--summary-json-output",
        type=Path,
        default=DEFAULT_GROUP_CHARACTERIZATION_JSON,
        help=(
            "Compact diagnostic JSON "
            f"(default: {DEFAULT_GROUP_CHARACTERIZATION_JSON})"
        ),
    )
    char_ds_cmd.add_argument(
        "--percentile-sample-cap",
        type=int,
        default=DEFAULT_PERCENTILE_SAMPLE_CAP,
        help=(
            "Bounded sample size for percentiles "
            f"(default: {DEFAULT_PERCENTILE_SAMPLE_CAP:,})"
        ),
    )

    model_cmd = subparsers.add_parser(
        "characterize-modeling-split",
        help=(
            "Phase 2A: TRAIN-only modeling_group fit/validation split + "
            "FIT sampling-cap characterization (no model training)"
        ),
    )
    model_cmd.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "pcap_inventory.csv",
        help="PCAP inventory CSV",
    )
    model_cmd.add_argument(
        "--build-manifest",
        type=Path,
        default=DEFAULT_BUILD_MANIFEST_PATH,
        help=f"TRAIN feature build manifest (default: {DEFAULT_BUILD_MANIFEST_PATH})",
    )
    model_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODELING_V1_DIR,
        help=f"Modeling artifact directory (default: {DEFAULT_MODELING_V1_DIR})",
    )
    model_cmd.add_argument(
        "--split-manifest-output",
        type=Path,
        default=None,
        help=f"Canonical 85-row split CSV (default: {DEFAULT_SPLIT_MANIFEST_PATH})",
    )
    model_cmd.add_argument(
        "--sampling-plan-output",
        type=Path,
        default=None,
        help=f"Sampling plan JSON (default: {DEFAULT_SAMPLING_PLAN_PATH})",
    )
    model_cmd.add_argument(
        "--sampling-summary-output",
        type=Path,
        default=None,
        help=f"Sampling summary CSV (default: {DEFAULT_SAMPLING_SUMMARY_PATH})",
    )
    model_cmd.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MODELING_SEED,
        help=f"Base seed for SHA-256-derived RNG (default: {DEFAULT_MODELING_SEED})",
    )

    freeze_cmd = subparsers.add_parser(
        "freeze-modeling-split",
        help=(
            "Gate 2A: freeze TRAIN modeling split + chosen sampling plan "
            f"(default plan_id={FROZEN_SAMPLING_PLAN_ID})"
        ),
    )
    freeze_cmd.add_argument(
        "--plan-id",
        default=FROZEN_SAMPLING_PLAN_ID,
        help=f"Sampling plan to freeze (default: {FROZEN_SAMPLING_PLAN_ID})",
    )
    freeze_cmd.add_argument(
        "--sampling-plan",
        type=Path,
        default=DEFAULT_SAMPLING_PLAN_PATH,
        help=f"sampling_plan.json to update (default: {DEFAULT_SAMPLING_PLAN_PATH})",
    )
    freeze_cmd.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help=f"modeling_split_manifest.csv (default: {DEFAULT_SPLIT_MANIFEST_PATH})",
    )

    fit_view_cmd = subparsers.add_parser(
        "build-modeling-fit-view",
        help=(
            "Phase 2B.1: materialize frozen TRAIN-fit view "
            f"({FROZEN_SAMPLING_PLAN_ID}) as per-PCAP Parquet shards"
        ),
    )
    fit_view_cmd.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-PCAP checkpoints when valid (default: true)",
    )
    fit_view_cmd.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help=f"Frozen modeling split CSV (default: {DEFAULT_SPLIT_MANIFEST_PATH})",
    )
    fit_view_cmd.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_TRAINING_VIEW_CONTRACT_PATH,
        help=(
            "Pinned training view contract JSON "
            f"(default: {DEFAULT_TRAINING_VIEW_CONTRACT_PATH})"
        ),
    )
    fit_view_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIT_VIEW_ROOT,
        help=f"View root directory (default: {DEFAULT_FIT_VIEW_ROOT})",
    )

    baselines_cmd = subparsers.add_parser(
        "train-baselines",
        help=(
            "Phase 2B.2: train unweighted LR + HistGradientBoosting on FIT view; "
            "evaluate on unsampled TRAIN-validation (TEST sealed). "
            "Full runs require a frozen baseline_contract.json from "
            "prepare-baseline-run."
        ),
    )
    baselines_cmd.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Stratified tiny FIT/VAL slices for implementation verification only "
            "(marks smoke_only=true; not real baseline results)"
        ),
    )

    prepare_base_cmd = subparsers.add_parser(
        "prepare-baseline-run",
        help=(
            "Freeze and write data/modeling/v1/baselines/phase2b2_v1/"
            "baseline_contract.json (pins hashes; no training)"
        ),
    )
    prepare_base_cmd.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_BASELINE_CONTRACT_PATH,
        help=f"Output contract path (default: {DEFAULT_BASELINE_CONTRACT_PATH})",
    )

    sweep_cmd = subparsers.add_parser(
        "threshold-sweep-baselines",
        help=(
            "Phase 2B.3A: rescore TRAIN-validation with frozen 2B.2 models and "
            "sweep operating thresholds (no retraining, no TEST)"
        ),
    )
    sweep_cmd.add_argument(
        "--no-cache-scores",
        action="store_true",
        help="Force rescoring even if predictions/*.npz caches exist",
    )

    abl_cmd = subparsers.add_parser(
        "run-hgb-ablations",
        help=(
            "Phase 2B.3B: HGB 27/22 × unweighted/balanced ablations + "
            "low-FPR threshold comparison (no TEST)"
        ),
    )
    abl_cmd.add_argument(
        "--no-cache-scores",
        action="store_true",
        help="Force rescoring even if predictions/*.npz caches exist",
    )

    refine_cmd = subparsers.add_parser(
        "refine-c-thresholds",
        help=(
            "Phase 2B.3C: focused benign-FPR threshold refine on C "
            "(22-feature unweighted HGB; reuses 2B.3B scores; no TEST)"
        ),
    )

    prep_fam_cmd = subparsers.add_parser(
        "prepare-model-family-bakeoff",
        help=(
            "Phase 2B.4: freeze model_family_contract.json before training "
            "(feature selection deferred; no TEST)"
        ),
    )

    fam_cmd = subparsers.add_parser(
        "run-model-family-bakeoff",
        help=(
            "Phase 2B.4: HGB vs AdaBoost vs Random Forest on 27 features "
            "(full TRAIN-validation; no TEST; no auto-winner)"
        ),
    )
    fam_cmd.add_argument(
        "--no-cache-scores",
        action="store_true",
        help="Force rescoring even if predictions/*.npz caches exist",
    )

    prep_et_cmd = subparsers.add_parser(
        "prepare-extratrees-challenger",
        help=(
            "Phase 2B.4B: freeze extratrees_contract.json before fitting "
            "(reuses 2B.4 pins; feature selection deferred; no TEST)"
        ),
    )

    et_cmd = subparsers.add_parser(
        "run-extratrees-challenger",
        help=(
            "Phase 2B.4B: ExtraTrees final challenger vs HGB/AdaBoost/RF "
            "(full TRAIN-validation; no TEST; no auto-advance)"
        ),
    )
    et_cmd.add_argument(
        "--no-cache-scores",
        action="store_true",
        help="Force rescoring even if predictions/*.npz caches exist",
    )

    prep_ext_cmd = subparsers.add_parser(
        "prepare-external-boost-challengers",
        help=(
            "Phase 2B.4C: freeze external_boost_contract.json before fitting "
            "XGBoost/CatBoost (no early stopping; no TEST)"
        ),
    )

    ext_cmd = subparsers.add_parser(
        "run-external-boost-challengers",
        help=(
            "Phase 2B.4C: fixed XGBoost + CatBoost challengers "
            "(full TRAIN-validation; no early stopping; no TEST)"
        ),
    )
    ext_cmd.add_argument(
        "--no-cache-scores",
        action="store_true",
        help="Force rescoring even if predictions/*.npz caches exist",
    )

    prep_f22_cmd = subparsers.add_parser(
        "prepare-feature22-boost-rematch",
        help=(
            "Phase 2B.4D: freeze 22-feature rematch contract "
            "(HGB-C vs XGBoost/CatBoost; no TEST)"
        ),
    )

    f22_cmd = subparsers.add_parser(
        "run-feature22-boost-rematch",
        help=(
            "Phase 2B.4D: HGB-22 vs XGBoost-22 vs CatBoost-22 "
            "(full TRAIN-validation; no early stopping; no TEST)"
        ),
    )
    f22_cmd.add_argument(
        "--no-cache-scores",
        action="store_true",
        help="Force rescoring even if predictions/*.npz caches exist",
    )

    freeze_cand_cmd = subparsers.add_parser(
        "freeze-v1-candidate",
        help=(
            "Phase 2B.5: close model exploration; freeze HGB-22 as V1 candidate "
            "and resolve 22-feature model input (threshold still unfrozen; no TEST)"
        ),
    )

    prep_sens_cmd = subparsers.add_parser(
        "prepare-hgb-sensitivity",
        help=(
            "Phase 2C.1: freeze HGB sensitivity contract + group-aware FIT CV folds "
            "(no fitting; no main VAL; no TEST)"
        ),
    )

    sens_cmd = subparsers.add_parser(
        "run-hgb-sensitivity",
        help=(
            "Phase 2C.1: FIT-only HGB sensitivity (12 configs × 3 folds), then "
            "one baseline-vs-winner TRAIN-validation compare (no TEST)"
        ),
    )

    freeze_2c_cmd = subparsers.add_parser(
        "freeze-phase2c",
        help=(
            "Phase 2C close: freeze V1 model package (HGB-22 H0 + threshold); "
            "close hyperparameter and threshold tuning (TEST sealed)"
        ),
    )

    prep_2d_cmd = subparsers.add_parser(
        "prepare-final-test",
        help=(
            "Phase 2D.0: freeze pre-TEST contract (pins model/threshold/hashes); "
            "does not open TEST feature shards"
        ),
    )

    preflight_2d_cmd = subparsers.add_parser(
        "preflight-final-test",
        help=(
            "Phase 2D.2: verify contracts + TEST inventory metadata; "
            "no predictions or metrics (final stop before one-shot TEST)"
        ),
    )

    run_2d_cmd = subparsers.add_parser(
        "run-final-test",
        help=(
            "Phase 2D: one-shot sealed TEST evaluation of the frozen V1 package "
            "(no --model/--threshold/--features overrides; measurement only)"
        ),
    )

    arp_probe_cmd = subparsers.add_parser(
        "probe-arp-features-fit",
        help=(
            "V2A A4/A5: FIT-only ARP semantic feature probe "
            "(ARP Spoofing FIT + all BENIGN FIT; no TEST; no training)"
        ),
    )
    arp_probe_cmd.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help=f"modeling_split_manifest.csv (default: {DEFAULT_SPLIT_MANIFEST_PATH})",
    )
    arp_probe_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARP_PROBE_DIR,
        help=f"Experiment output directory (default: {DEFAULT_ARP_PROBE_DIR})",
    )
    arp_probe_cmd.add_argument(
        "--max-windows-per-pcap",
        type=int,
        default=None,
        help="Optional per-PCAP window cap (smoke only; omit for full FIT probe)",
    )

    arp_stateful_cmd = subparsers.add_parser(
        "probe-arp-stateful-fit",
        help=(
            "V2A A6: FIT-only whole-PCAP ARP conflict feasibility probe "
            "(no production stateful extractor; no TEST; no training)"
        ),
    )
    arp_stateful_cmd.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help=f"modeling_split_manifest.csv (default: {DEFAULT_SPLIT_MANIFEST_PATH})",
    )
    arp_stateful_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARP_PROBE_DIR,
        help=f"Experiment output directory (default: {DEFAULT_ARP_PROBE_DIR})",
    )

    mqtt_probe_cmd = subparsers.add_parser(
        "probe-mqtt-features-fit",
        help=(
            "V2M M4: FIT-only MQTT structural feature probe "
            "(MQTT_Malformed_Data FIT + benign MQTT FIT; no TEST; no training)"
        ),
    )
    mqtt_probe_cmd.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help=f"modeling_split_manifest.csv (default: {DEFAULT_SPLIT_MANIFEST_PATH})",
    )
    mqtt_probe_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MQTT_PROBE_DIR,
        help=f"Experiment output directory (default: {DEFAULT_MQTT_PROBE_DIR})",
    )
    mqtt_probe_cmd.add_argument(
        "--max-windows-per-pcap",
        type=int,
        default=None,
        help="Optional per-PCAP window cap (smoke only; omit for full FIT probe)",
    )

    agg_cmd = subparsers.add_parser(
        "evaluate-pcap-aggregation",
        help=(
            "D0: score frozen H0 on TRAIN-validation PCAPs and evaluate the "
            "predeclared (K,R) PCAP aggregation grid (does not freeze K/R; "
            "never reads TEST)"
        ),
    )
    agg_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SERVING_DIR,
        help=f"Selection evidence directory (default: {DEFAULT_SERVING_DIR})",
    )

    classify_cmd = subparsers.add_parser(
        "classify-pcap",
        help=(
            "D1 local smoke: run frozen V1 inference on one Ethernet PCAP "
            "(no HTTP; prints JSON prediction)"
        ),
    )
    classify_cmd.add_argument(
        "pcap",
        type=Path,
        help="Path to a classic libpcap capture (DLT_EN10MB / Ethernet)",
    )
    classify_cmd.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SCORE_BATCH_SIZE,
        help=f"Window scoring batch size (default: {DEFAULT_SCORE_BATCH_SIZE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "build-manifests":
        result = build_manifests(
            raw_root=args.raw_root,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(f"Wrote {result['inventory_path']}")
        print(f"Wrote {result['split_path']}")
        return 0

    if args.command == "inspect-pcaps":
        for raw in args.pcaps:
            path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
            stats = summarize_pcap(path, max_packets=args.max_packets)
            duration = None
            if stats.first_timestamp is not None and stats.last_timestamp is not None:
                duration = stats.last_timestamp - stats.first_timestamp
            print(f"\n=== {stats.path} ===")
            print(f"linktype: {stats.linktype}")
            print(f"packets_total: {stats.packets_total}")
            print(f"packets_ok: {stats.packets_ok}")
            print(f"packets_unsupported: {stats.packets_unsupported}")
            print(f"packets_partial: {stats.packets_partial}")
            print(f"packets_failed: {stats.packets_failed}")
            if duration is not None:
                print(f"duration_s: {duration:.6f}")
            print(
                "tcp/udp/icmp/icmpv6/igmp/arp/llc: "
                f"{stats.tcp}/{stats.udp}/{stats.icmp}/"
                f"{stats.icmpv6}/{stats.igmp}/{stats.arp}/{stats.llc}"
            )
            print(f"ipv4/ipv6/vlan_frames: {stats.ipv4}/{stats.ipv6}/{stats.vlan_frames}")
            print("parse_status:", dict(stats.by_parse_status))
            print("protocols:", dict(stats.by_protocol))
        return 0

    if args.command == "audit-corpus":
        result = audit_corpus(
            inventory_path=args.inventory,
            split_path=args.split_manifest,
            raw_root=args.raw_root,
            output_dir=args.output_dir,
            ip_cardinality_cap=args.ip_cardinality_cap,
            issue_cap_per_code=args.issue_cap_per_code,
            malformed_high_rate=args.malformed_high_rate,
            malformed_catastrophic_rate=args.malformed_catastrophic_rate,
            workers=args.workers,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            clear_checkpoints=args.clear_checkpoints,
            progress_every_packets=args.progress_every_packets,
            progress_file=sys.stderr,
        )
        print(f"Wrote {result.integrity_path}")
        print(f"Wrote {result.train_path}")
        print(f"Wrote {result.issues_path}")
        return 1 if result.hard_fail else 0

    if args.command == "probe-timestamps":
        results = []
        for raw in args.pcaps:
            path = resolve_pcap_path(raw)
            print(f"Probing {path} ...", file=sys.stderr)
            probe = probe_timestamps(
                path,
                example_limit=args.example_limit,
                largest_example_limit=args.largest_example_limit,
                positive_sample_cap=args.positive_sample_cap,
                max_packets=args.max_packets,
            )
            results.append(probe)
            print(format_probe_summary(probe))
        written = write_probe_artifacts(
            results,
            output_path=args.output,
            examples_path=None if args.no_examples else args.examples_output,
        )
        print(f"\nWrote {written['probe_path']}")
        if "examples_path" in written:
            print(f"Wrote {written['examples_path']}")
        return 0

    if args.command == "characterize-windowing":
        policies = candidate_policies(backward_reset_seconds=args.backward_reset)
        result = characterize_train_windowing(
            inventory_path=args.inventory,
            output_path=args.output,
            policies=policies,
            workers=args.workers,
            max_packets=args.max_packets,
            span_sample_cap=args.span_sample_cap,
            progress_file=sys.stderr,
        )
        print(format_characterization_summary(result.rows))
        print(f"\nWrote {result.output_path}")
        print(
            f"Rows: {len(result.rows)} "
            f"({result.train_pcap_count} TRAIN PCAPs × {result.policy_count} policies)"
        )
        print("\n*** GATE A COMPLETE — awaiting human config freeze ***")
        return 0

    if args.command == "extract-features":
        if args.write_schema_only:
            path = write_feature_schema()
            print(f"Wrote {path}")
            return 0

        max_windows = args.max_windows_per_pcap
        if max_windows == 0:
            max_windows = None

        if args.smoke:
            result = run_smoke_extraction(
                inventory_path=args.inventory,
                output_path=args.output,
                characterization_path=args.characterization_output,
                max_windows_per_pcap=max_windows,
                progress_file=sys.stderr,
            )
            print(format_smoke_summary(result))
            return 0

        if not args.pcaps:
            parser.error("provide PCAP paths or --smoke")

        inv = args.inventory
        inv_path = inv if inv.is_absolute() else (PROJECT_ROOT / inv)
        index = load_inventory_index(inv_path)
        all_rows: list[dict] = []
        for raw in args.pcaps:
            path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
            rel = to_repo_relative(path)
            meta = dict(index.get(rel, {}))
            try:
                rows, _stats = extract_pcap_feature_rows(
                    path,
                    meta=meta,
                    max_windows=max_windows,
                    validate=True,
                )
            except (FeatureExtractionError, FeatureInvariantError) as exc:
                print(f"FAILED {path}: {exc}", file=sys.stderr)
                return 1
            all_rows.extend(rows)
            print(f"{path}: {len(rows)} windows", file=sys.stderr)

        out = args.output
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        columns = list(METADATA_COLUMN_NAMES) + list(V1_FEATURE_NAMES)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        print(f"Wrote {out} ({len(all_rows)} windows)")
        if args.characterization_output is not None:
            char_path = write_characterization_csv(
                all_rows, args.characterization_output
            )
            print(f"Wrote {char_path}")
        write_feature_schema()
        return 0

    if args.command == "build-feature-parquet":
        resume = not args.no_resume
        if args.smoke:
            payload = run_parquet_smoke(
                inventory_path=args.inventory,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                resume=resume,
                buffer_rows=args.buffer_rows,
            )
            print(format_parquet_smoke_summary(payload))
            # Second pass verifies resume hits when resume is enabled.
            if resume:
                payload2 = run_parquet_smoke(
                    inventory_path=args.inventory,
                    output_dir=args.output_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    resume=True,
                    buffer_rows=args.buffer_rows,
                )
                hits = sum(1 for r in payload2["results"] if r.resumed)
                print(f"\nResume verification: {hits}/{len(payload2['results'])} hits")
            return 0

        if not args.pcaps:
            parser.error("provide PCAP paths or --smoke")

        inv = args.inventory
        inv_path = inv if inv.is_absolute() else (PROJECT_ROOT / inv)
        index = load_inventory_index(inv_path)
        out_dir = args.output_dir
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        ckpt_dir = args.checkpoint_dir
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir

        for raw in args.pcaps:
            path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
            rel = to_repo_relative(path)
            meta = dict(index.get(rel, {}))
            pcap_id = pcap_id_from_path(path)
            try:
                result = build_pcap_parquet(
                    path,
                    meta,
                    out_dir / f"{pcap_id}.parquet",
                    checkpoint_path=ckpt_dir / f"{pcap_id}.json",
                    resume=resume,
                    buffer_rows=args.buffer_rows,
                )
            except (FeatureExtractionError, FeatureInvariantError, OSError) as exc:
                print(f"FAILED {path}: {exc}", file=sys.stderr)
                return 1
            print(
                f"{result.pcap_id}: rows={result.row_count} "
                f"resumed={result.resumed} elapsed={result.elapsed_seconds:.3f}s"
            )
        return 0

    if args.command == "build-feature-dataset":
        try:
            result = build_feature_dataset(
                split=args.split,
                inventory_path=args.inventory,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                manifest_path=args.manifest_output,
                workers=args.workers,
                resume=args.resume,
                buffer_rows=args.buffer_rows,
                smoke=args.smoke,
                progress_file=sys.stderr,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_feature_dataset_summary(result))
        if result.failed_count:
            print(
                f"Completed with {result.failed_count} failed PCAP(s); "
                f"see {result.manifest_path}",
                file=sys.stderr,
            )
            return 1
        return 0

    if args.command == "validate-feature-dataset":
        try:
            result = validate_feature_dataset(
                split=args.split,
                manifest_path=args.manifest,
                integrity_path=args.integrity,
                characterization_path=args.characterization,
                summary_output=args.summary_output,
                constant_output=args.constant_output,
                complete_output=args.complete_output,
                progress_file=sys.stderr,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_validation_summary(result))
        return 0 if result.passed else 1

    if args.command == "characterize-feature-dataset":
        if args.split != "train":
            print("Only --split train is supported", file=sys.stderr)
            return 1
        try:
            result = characterize_train_feature_groups(
                manifest_path=args.manifest,
                inventory_path=args.inventory,
                group_summary_output=args.group_summary_output,
                pcap_diagnostics_output=args.pcap_diagnostics_output,
                summary_json_output=args.summary_json_output,
                percentile_sample_cap=args.percentile_sample_cap,
                progress_file=sys.stderr,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_group_characterization_summary(result))
        return 0

    if args.command == "characterize-modeling-split":
        try:
            result = characterize_modeling_split(
                inventory_path=args.inventory,
                build_manifest_path=args.build_manifest,
                output_dir=args.output_dir,
                split_manifest_path=args.split_manifest_output,
                sampling_plan_path=args.sampling_plan_output,
                sampling_summary_path=args.sampling_summary_output,
                base_seed=args.seed,
                progress_file=sys.stderr,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_modeling_characterization_summary(result))
        return 0 if result.passed else 1

    if args.command == "freeze-modeling-split":
        try:
            payload = freeze_gate_2a(
                plan_id=args.plan_id,
                sampling_plan_path=args.sampling_plan,
                split_manifest_path=args.split_manifest,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_gate_2a_freeze_summary(payload))
        return 0

    if args.command == "build-modeling-fit-view":
        try:
            result = build_modeling_fit_view(
                resume=args.resume,
                split_manifest_path=args.split_manifest,
                contract_path=args.contract,
                output_dir=args.output_dir,
                progress_file=sys.stderr,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_fit_view_summary(result))
        return 0 if result.passed else 1

    if args.command == "prepare-baseline-run":
        try:
            payload = prepare_baseline_run(contract_path=args.contract)
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_baseline_summary(payload, Path(args.contract)))
        return 0

    if args.command == "train-baselines":
        try:
            result = train_baselines(
                smoke=args.smoke,
                progress_file=sys.stderr,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_baselines_summary(result))
        return 0 if result.passed else 1

    if args.command == "threshold-sweep-baselines":
        try:
            payload = run_threshold_sweep(
                progress_file=sys.stderr,
                cache_scores=not args.no_cache_scores,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_threshold_sweep_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "run-hgb-ablations":
        try:
            payload = run_hgb_ablations(
                progress_file=sys.stderr,
                cache_scores=not args.no_cache_scores,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_ablation_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "refine-c-thresholds":
        try:
            payload = run_c_threshold_refine(progress_file=sys.stderr)
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_c_threshold_refine_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "prepare-model-family-bakeoff":
        try:
            payload = prepare_model_family_bakeoff()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_model_family_summary(payload))
        return 0

    if args.command == "run-model-family-bakeoff":
        try:
            payload = run_model_family_bakeoff(
                progress_file=sys.stderr,
                cache_scores=not args.no_cache_scores,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_model_family_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "prepare-extratrees-challenger":
        try:
            payload = prepare_extratrees_challenger()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_extratrees_summary(payload))
        return 0

    if args.command == "run-extratrees-challenger":
        try:
            payload = run_extratrees_challenger(
                progress_file=sys.stderr,
                cache_scores=not args.no_cache_scores,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_extratrees_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "prepare-external-boost-challengers":
        try:
            payload = prepare_external_boost_challengers()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_external_boost_summary(payload))
        return 0

    if args.command == "run-external-boost-challengers":
        try:
            payload = run_external_boost_challengers(
                progress_file=sys.stderr,
                cache_scores=not args.no_cache_scores,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_external_boost_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "prepare-feature22-boost-rematch":
        try:
            payload = prepare_feature22_boost_rematch()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_feature22_boost_summary(payload))
        return 0

    if args.command == "run-feature22-boost-rematch":
        try:
            payload = run_feature22_boost_rematch(
                progress_file=sys.stderr,
                cache_scores=not args.no_cache_scores,
            )
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_feature22_boost_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "freeze-v1-candidate":
        try:
            payload = freeze_v1_candidate()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_v1_candidate_freeze_summary(payload))
        return 0 if payload.get("status") == "frozen" else 1

    if args.command == "prepare-hgb-sensitivity":
        try:
            payload = prepare_hgb_sensitivity()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_hgb_sensitivity_summary(payload))
        return 0

    if args.command == "run-hgb-sensitivity":
        try:
            payload = run_hgb_sensitivity(progress_file=sys.stderr)
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_hgb_sensitivity_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "freeze-phase2c":
        try:
            payload = freeze_phase2c()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_phase2c_freeze_summary(payload))
        return 0 if payload.get("status") == "frozen" else 1

    if args.command == "prepare-final-test":
        try:
            payload = prepare_final_test()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_prepare_final_test_summary(payload))
        return 0 if payload.get("gate_2d0_status") == "passed" else 1

    if args.command == "preflight-final-test":
        try:
            payload = preflight_final_test()
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_preflight_final_test_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "run-final-test":
        try:
            payload = run_final_test(progress_file=sys.stderr)
        except FeatureExtractionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(format_final_test_summary(payload))
        return 0 if payload.get("status") == "passed" else 1

    if args.command == "probe-arp-features-fit":
        try:
            payload = run_arp_fit_probe(
                split_manifest_path=args.split_manifest,
                output_dir=args.output_dir,
                progress_file=sys.stderr,
                max_windows_per_pcap=args.max_windows_per_pcap,
            )
        except (FeatureExtractionError, FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        arts = payload["artifacts"]
        print("ARP FIT probe complete")
        print(f"  PCAPs: {payload['data_access']['pcap_count']}")
        print(f"  groups: {payload['data_access']['group_counts']}")
        print(f"  windows_by_group: {payload['window_counts_by_group']}")
        for note in payload.get("a5_semantic_signal_notes") or []:
            print(f"  note: {note}")
        for key in (
            "arp_feature_summary",
            "arp_feature_by_pcap",
            "arp_feature_nonzero_rates",
            "arp_vs_arp_ratio",
            "arp_probe_complete",
        ):
            print(f"Wrote {arts[key]}")
        return 0

    if args.command == "probe-arp-stateful-fit":
        try:
            payload = run_arp_stateful_feasibility_probe(
                split_manifest_path=args.split_manifest,
                output_dir=args.output_dir,
                progress_file=sys.stderr,
            )
        except (FeatureExtractionError, FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        arts = payload["artifacts"]
        print("ARP stateful feasibility probe complete")
        print(f"  PCAPs: {payload['data_access']['pcap_count']}")
        print(f"  verdict: {payload['verdict']}")
        print(f"  next: {payload['recommended_next_step']}")
        for g, row in (payload.get("group_summary") or {}).items():
            print(
                f"  {g}: conflict_obs_ratio={float(row['conflict_obs_ratio']):.4f} "
                f"conflict_ips={row['conflict_ip_count']} "
                f"valid_arp={row['valid_identity_obs']} "
                f"transitions={row['mapping_transition_count']}"
            )
        for key in (
            "arp_stateful_by_pcap",
            "arp_stateful_by_group",
            "arp_stateful_feasibility_complete",
        ):
            print(f"Wrote {arts[key]}")
        return 0

    if args.command == "probe-mqtt-features-fit":
        try:
            payload = run_mqtt_fit_probe(
                split_manifest_path=args.split_manifest,
                output_dir=args.output_dir,
                progress_file=sys.stderr,
                max_windows_per_pcap=args.max_windows_per_pcap,
            )
        except (FeatureExtractionError, FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        arts = payload["artifacts"]
        hyp = payload.get("hypothesis") or {}
        print("MQTT FIT structural probe complete")
        print(f"  PCAPs: {payload['data_access']['pcap_count']}")
        print(f"  next: {hyp.get('recommended_next_step')}")
        for note in hyp.get("notes") or []:
            print(f"  note: {note}")
        for key in (
            "mqtt_feature_summary",
            "mqtt_feature_by_pcap",
            "mqtt_violation_summary",
            "mqtt_probe_complete",
        ):
            print(f"Wrote {arts[key]}")
        return 0

    if args.command == "evaluate-pcap-aggregation":
        payload = write_aggregation_evaluation(out_dir=args.output_dir)
        print(format_review_summary(payload))
        arts = payload.get("artifacts") or {}
        for key in ("by_pcap", "summary", "review"):
            if arts.get(key):
                print(f"Wrote {arts[key]}")
        return 0

    if args.command == "classify-pcap":
        pcap_path = args.pcap if args.pcap.is_absolute() else (PROJECT_ROOT / args.pcap)
        engine = V1InferenceEngine.load_default()
        result = classify_pcap(
            pcap_path,
            engine=engine,
            batch_size=args.batch_size,
        )
        payload = result.to_dict()
        payload["pcap"] = str(pcap_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        # Non-zero only for input failures; INSUFFICIENT_DATA / ATTACK / BENIGN → 0.
        if result.status in {"INVALID_INPUT", "UNSUPPORTED_INPUT"}:
            return 1
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
