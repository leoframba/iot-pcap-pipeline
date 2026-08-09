"""Command-line interface for iot-pcap-pipeline."""

from __future__ import annotations

import argparse
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
from iot_pcap_pipeline.paths import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_SEED,
    PROJECT_ROOT,
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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
