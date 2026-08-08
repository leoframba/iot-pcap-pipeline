"""Command-line interface for iot-pcap-pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iot_pcap_pipeline.dataset.build import build_manifests
from iot_pcap_pipeline.paths import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_SEED,
    PROJECT_ROOT,
)
from iot_pcap_pipeline.pcap.reader import summarize_pcap


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
            if duration is not None:
                print(f"duration_s: {duration:.6f}")
            print(
                "tcp/udp/icmp/icmpv6/igmp/arp: "
                f"{stats.tcp}/{stats.udp}/{stats.icmp}/"
                f"{stats.icmpv6}/{stats.igmp}/{stats.arp}"
            )
            print(f"ipv4/ipv6/vlan_frames: {stats.ipv4}/{stats.ipv6}/{stats.vlan_frames}")
            print("parse_status:", dict(stats.by_parse_status))
            print("protocols:", dict(stats.by_protocol))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
