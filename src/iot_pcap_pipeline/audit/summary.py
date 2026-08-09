"""Terminal summary for Phase 1B.2 corpus audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.audit.policy import (
    SEVERITY_HARD_FAILURE,
    SEVERITY_HIGH_WARNING,
    SEVERITY_WARNING,
)
from iot_pcap_pipeline.audit.worker import PcapAuditResult


def _train_bucket(row: dict[str, Any]) -> str | None:
    source = row.get("source")
    label = row.get("binary_label")
    if source == "attacks" and label == "BENIGN":
        return "publisher_benign"
    if source == "profiling":
        profiling_type = row.get("profiling_type")
        if profiling_type:
            return f"profiling_{profiling_type}"
        return "profiling_other"
    family = row.get("attack_family")
    if family:
        return str(family)
    return None


def _bucket_stats(rows: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = _train_bucket(row)
        if bucket:
            groups[bucket].append(row)

    lines: list[str] = []
    preferred = [
        "publisher_benign",
        "profiling_idle",
        "profiling_active",
        "profiling_power",
        "profiling_interaction",
        "DDoS",
        "DoS",
        "MQTT",
        "Recon",
        "Spoofing",
    ]
    keys = [k for k in preferred if k in groups] + sorted(
        k for k in groups if k not in preferred
    )
    for key in keys:
        items = groups[key]
        n = len(items)
        tcp_ratios = [r["tcp_ratio"] for r in items if r.get("tcp_ratio") is not None]
        pps = [
            r["packets_per_second"]
            for r in items
            if r.get("packets_per_second") is not None
        ]
        uniq = [
            r["unique_src_ips_count"]
            for r in items
            if r.get("unique_src_ips_count") is not None
        ]
        mean_tcp = sum(tcp_ratios) / len(tcp_ratios) if tcp_ratios else None
        mean_pps = sum(pps) / len(pps) if pps else None
        mean_uniq = sum(uniq) / len(uniq) if uniq else None
        lines.append(
            f"{key}: n={n}"
            + (f" mean_tcp_ratio={mean_tcp:.3f}" if mean_tcp is not None else "")
            + (f" mean_pps={mean_pps:.3f}" if mean_pps is not None else "")
            + (f" mean_unique_src_ips={mean_uniq:.1f}" if mean_uniq is not None else "")
        )
    return lines


def format_audit_summary(
    *,
    integrity_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    by_protocol_by_path: dict[str, dict[str, int]],
    hard_fail: bool,
    checkpoint_hits: int = 0,
    scanned_files: int = 0,
    total_scan_seconds: float | None = None,
    slowest: list[PcapAuditResult] | None = None,
    workers: int = 1,
) -> str:
    total_files = len(integrity_rows)
    open_failures = sum(1 for r in integrity_rows if not r.get("parse_success"))
    processed = total_files - open_failures

    totals = Counter()
    linktypes: Counter[str] = Counter()
    unsupported_packets: Counter[str] = Counter()
    unsupported_pcaps: dict[str, set[str]] = defaultdict(set)
    llc_total = 0
    ipv6_total = 0
    vlan_total = 0
    ts_issue_files = 0
    malformed_files = 0
    packet_total = 0
    byte_total = 0

    for row in integrity_rows:
        if not row.get("parse_success"):
            continue
        packet_total += int(row.get("packet_count") or 0)
        byte_total += int(row.get("total_frame_bytes") or 0)
        for key in (
            "ok_count",
            "partial_count",
            "unsupported_count",
            "malformed_count",
            "error_count",
        ):
            totals[key] += int(row.get(key) or 0)
        linktype = row.get("linktype")
        if linktype is not None:
            linktypes[str(linktype)] += 1
        llc_total += int(row.get("llc_count") or 0)
        ipv6_total += int(row.get("ipv6_count") or 0)
        vlan_total += int(row.get("vlan_count") or 0)
        if int(row.get("malformed_count") or 0) > 0:
            malformed_files += 1
        if (
            int(row.get("duplicate_timestamp_count") or 0) > 0
            or int(row.get("negative_delta_count") or 0) > 0
        ):
            ts_issue_files += 1

        for proto, count in (by_protocol_by_path.get(row["pcap_path"]) or {}).items():
            if proto == "lldp" or proto.startswith(
                ("ethertype_", "ip_proto_", "linktype_")
            ):
                unsupported_packets[proto] += count
                unsupported_pcaps[proto].add(row["pcap_path"])

    hard_issues = [
        r for r in issue_rows if r.get("severity") == SEVERITY_HARD_FAILURE
    ]
    high_warnings = [
        r for r in issue_rows if r.get("severity") == SEVERITY_HIGH_WARNING
    ]
    warnings = [r for r in issue_rows if r.get("severity") == SEVERITY_WARNING]

    lines = [
        "Phase 1B.2 Corpus Audit Summary",
        "===============================",
        f"PCAPs processed:             {processed} / {total_files}",
        f"PCAP open failures:          {open_failures}",
        f"Workers:                     {workers}",
        f"Checkpoint hits:             {checkpoint_hits}",
        f"Newly scanned files:         {scanned_files}",
        f"Hard fail:                   {hard_fail}",
        "",
        f"Total packets:               {packet_total}",
        f"OK:                          {totals['ok_count']}",
        f"Partial:                     {totals['partial_count']}",
        f"Unsupported:                 {totals['unsupported_count']}",
        f"Malformed:                   {totals['malformed_count']}",
        f"Errors:                      {totals['error_count']}",
        "",
        "Linktypes encountered:",
    ]
    for linktype, count in sorted(linktypes.items()):
        label = "Ethernet (1)" if linktype == "1" else f"linktype_{linktype}"
        lines.append(f"  {label}:              {count}")

    lines.extend(["", "Unsupported protocols:"])
    if unsupported_packets:
        for proto, count in sorted(
            unsupported_packets.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            pcaps = len(unsupported_pcaps[proto])
            lines.append(f"  {proto}: {count} packets across {pcaps} PCAPs")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            f"802.3 / LLC packets:         {llc_total}",
            f"IPv6 packets:                {ipv6_total}",
            f"VLAN frames:                 {vlan_total}",
            "",
            f"PCAPs with timestamp issues: {ts_issue_files}",
            f"PCAPs with malformed frames: {malformed_files}",
            "",
            f"Hard failure issues:         {len(hard_issues)}",
            f"High warnings:               {len(high_warnings)}",
            f"Warnings:                    {len(warnings)}",
        ]
    )

    if total_scan_seconds is not None and scanned_files:
        lines.extend(
            [
                "",
                "Runtime",
                "-------",
                f"New scan wall-ish seconds:   {total_scan_seconds:.1f}",
            ]
        )
        if total_scan_seconds > 0:
            lines.append(
                f"Packets/sec (new scans):     {packet_total / total_scan_seconds:,.1f}"
            )
            lines.append(
                f"MB/sec (new scans):          {(byte_total / 1e6) / total_scan_seconds:,.2f}"
            )
        if slowest:
            lines.append("Slowest PCAPs:")
            for item in slowest:
                elapsed = item.scan_elapsed_seconds or 0.0
                lines.append(
                    f"  {Path(item.pcap_path).name}  {elapsed:.1f}s  "
                    f"packets={item.packet_count}"
                )

    lines.extend(
        [
            "",
            "TRAIN characterization by metadata",
            "----------------------------------",
        ]
    )
    lines.extend(_bucket_stats(train_rows) or ["  (no train rows)"])
    lines.append("")
    return "\n".join(lines) + "\n"
