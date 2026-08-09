"""Command-line interface for iot-pcap-pipeline."""

from __future__ import annotations

import argparse
import csv
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
from iot_pcap_pipeline.features.build import (
    DEFAULT_MAX_WINDOWS_PER_PCAP,
    DEFAULT_SMOKE_FEATURES_CSV,
    extract_pcap_feature_rows,
    format_smoke_summary,
    load_inventory_index,
    run_smoke_extraction,
)
from iot_pcap_pipeline.features.characterize import write_characterization_csv
from iot_pcap_pipeline.features.dataset import (
    DEFAULT_BUILD_MANIFEST_PATH,
    DEFAULT_FEATURE_DATASET_WORKERS,
    DEFAULT_SMOKE_BUILD_MANIFEST_PATH,
    DEFAULT_SMOKE_CHECKPOINT_DIR,
    DEFAULT_SMOKE_DATASET_DIR,
    DEFAULT_TRAIN_PARQUET_DIR,
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
    DEFAULT_TRAIN_BUILD_COMPLETE_JSON,
    DEFAULT_TRAIN_CONSTANT_FEATURES_CSV,
    DEFAULT_TRAIN_FEATURE_SUMMARY_CSV,
    format_validation_summary,
    validate_feature_dataset,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_MANIFEST_DIR,
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
            "Phase 1C.3b corpus orchestration: TRAIN inventory → "
            "largest-first process pool → Parquet shards + build_manifest.csv"
        ),
    )
    ds_cmd.add_argument(
        "--split",
        choices=["train"],
        default="train",
        help="Dataset split to build (only train in 1C.3b step 1)",
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
            f"(default: {DEFAULT_TRAIN_PARQUET_DIR}; "
            f"smoke → {DEFAULT_SMOKE_DATASET_DIR})"
        ),
    )
    ds_cmd.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Per-PCAP checkpoint directory "
            f"(default: {DEFAULT_FEATURE_CHECKPOINT_DIR}; "
            f"smoke → {DEFAULT_SMOKE_CHECKPOINT_DIR})"
        ),
    )
    ds_cmd.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "Build manifest CSV path "
            f"(default: {DEFAULT_BUILD_MANIFEST_PATH}; "
            f"smoke → {DEFAULT_SMOKE_BUILD_MANIFEST_PATH})"
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
            "Orchestration smoke: 6 modest TRAIN PCAPs only "
            f"(skips full {EXPECTED_TRAIN_PCAP_COUNT}-PCAP assertion)"
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
            "Read-only TRAIN Parquet build validation "
            "(shard schema/rows + audit/windowing joins + feature summaries)"
        ),
    )
    val_cmd.add_argument(
        "--split",
        choices=["train"],
        default="train",
        help="Dataset split to validate (only train for now)",
    )
    val_cmd.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_BUILD_MANIFEST_PATH,
        help=f"Build manifest CSV (default: {DEFAULT_BUILD_MANIFEST_PATH})",
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
            "Windowing characterization CSV "
            f"(default: {DEFAULT_CHARACTERIZATION_CSV})"
        ),
    )
    val_cmd.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_TRAIN_FEATURE_SUMMARY_CSV,
        help=f"Feature summary CSV (default: {DEFAULT_TRAIN_FEATURE_SUMMARY_CSV})",
    )
    val_cmd.add_argument(
        "--constant-output",
        type=Path,
        default=DEFAULT_TRAIN_CONSTANT_FEATURES_CSV,
        help=(
            "Constant-feature report CSV "
            f"(default: {DEFAULT_TRAIN_CONSTANT_FEATURES_CSV})"
        ),
    )
    val_cmd.add_argument(
        "--complete-output",
        type=Path,
        default=DEFAULT_TRAIN_BUILD_COMPLETE_JSON,
        help=(
            "Pass marker JSON written only when all checks pass "
            f"(default: {DEFAULT_TRAIN_BUILD_COMPLETE_JSON})"
        ),
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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
