"""Picklable per-PCAP audit worker for Phase 1B.2 parallel execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.audit.issues import IssueCollector
from iot_pcap_pipeline.audit.policy import (
    AUDIT_STRATEGY_VERSION,
    DEFAULT_ISSUE_CAP_PER_CODE,
    DEFAULT_MALFORMED_CATASTROPHIC_RATE,
    DEFAULT_MALFORMED_HIGH_WARNING_RATE,
    EXPECTED_LINKTYPE,
    ISSUE_ACCOUNTING_INVARIANT,
    ISSUE_DECODER_ERROR,
    ISSUE_IP_CARDINALITY_CAPPED,
    ISSUE_MALFORMED_CATASTROPHIC,
    ISSUE_MALFORMED_HIGH,
    ISSUE_MALFORMED_PRESENT,
    ISSUE_OPEN_FAILURE,
    ISSUE_PARTIAL_PRESENT,
    ISSUE_TIMESTAMP_DUPLICATE,
    ISSUE_TIMESTAMP_REVERSAL,
    ISSUE_UNSUPPORTED_LINKTYPE,
    ISSUE_UNSUPPORTED_PRESENT,
    ISSUE_WORKER_CRASH,
    ISSUE_ZERO_DURATION,
    ISSUE_ZERO_PACKETS,
    SEVERITY_HARD_FAILURE,
    SEVERITY_HIGH_WARNING,
    SEVERITY_WARNING,
    SUPPORTED_LINKTYPES,
)
from iot_pcap_pipeline.audit.live_progress import (
    DEFAULT_PROGRESS_EVERY_PACKETS,
    LiveProgressStore,
)
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.pcap.stats import (
    DEFAULT_IP_CARDINALITY_CAP,
    IntegrityStats,
    TrainCharacterizationStats,
)


@dataclass(frozen=True)
class AuditPolicy:
    """Immutable audit settings shared with worker processes."""

    ip_cardinality_cap: int = DEFAULT_IP_CARDINALITY_CAP
    issue_cap_per_code: int = DEFAULT_ISSUE_CAP_PER_CODE
    malformed_high_rate: float = DEFAULT_MALFORMED_HIGH_WARNING_RATE
    malformed_catastrophic_rate: float = DEFAULT_MALFORMED_CATASTROPHIC_RATE
    audit_strategy_version: str = AUDIT_STRATEGY_VERSION
    # Observability only — not part of checkpoint validity.
    progress_dir: str | None = None
    progress_every_packets: int = DEFAULT_PROGRESS_EVERY_PACKETS


@dataclass
class PcapAuditResult:
    """Self-contained serializable result for one PCAP."""

    pcap_path: str
    integrity_row: dict[str, Any]
    training_row: dict[str, Any] | None
    issue_rows: list[dict[str, Any]] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    by_protocol: dict[str, int] = field(default_factory=dict)
    scan_elapsed_seconds: float | None = None
    packet_count: int | None = None
    file_size: int | None = None
    from_checkpoint: bool = False


def _empty_integrity_packet_fields() -> dict[str, Any]:
    return {
        "linktype": None,
        "packet_count": None,
        "total_frame_bytes": None,
        "min_frame_length": None,
        "max_frame_length": None,
        "mean_frame_length": None,
        "ok_count": None,
        "partial_count": None,
        "unsupported_count": None,
        "malformed_count": None,
        "error_count": None,
        "ipv4_count": None,
        "ipv6_count": None,
        "arp_count": None,
        "llc_count": None,
        "tcp_count": None,
        "udp_count": None,
        "icmp_count": None,
        "icmpv6_count": None,
        "igmp_count": None,
        "vlan_count": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "min_timestamp": None,
        "max_timestamp": None,
        "capture_order_duration": None,
        "capture_timestamp_span": None,
        "duplicate_timestamp_count": None,
        "negative_delta_count": None,
        "non_monotonic_timestamp_count": None,
    }


def _apply_file_level_observations(
    *,
    issues: IssueCollector,
    pcap_path: str,
    stats: IntegrityStats,
    train: TrainCharacterizationStats | None,
    malformed_high_rate: float,
    malformed_catastrophic_rate: float,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    hard_failures: list[str] = []

    if stats.linktype is not None and stats.linktype not in SUPPORTED_LINKTYPES:
        issues.add_file(
            ISSUE_UNSUPPORTED_LINKTYPE,
            f"unsupported linktype={stats.linktype} (expected {EXPECTED_LINKTYPE})",
            pcap_path=pcap_path,
        )
        hard_failures.append(ISSUE_UNSUPPORTED_LINKTYPE)

    if stats.error_count > 0:
        issues.add_file(
            ISSUE_DECODER_ERROR,
            f"decoder error packets: {stats.error_count}",
            pcap_path=pcap_path,
        )
        hard_failures.append(ISSUE_DECODER_ERROR)

    invariants = stats.validate_invariants()
    if invariants:
        issues.add_file(
            ISSUE_ACCOUNTING_INVARIANT,
            "; ".join(invariants),
            pcap_path=pcap_path,
        )
        hard_failures.append(ISSUE_ACCOUNTING_INVARIANT)

    if stats.packet_count == 0:
        issues.add_file(
            ISSUE_ZERO_PACKETS,
            "PCAP contains zero packets",
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_ZERO_PACKETS)

    if stats.malformed_count > 0:
        rate = stats.malformed_rate or 0.0
        if rate >= malformed_catastrophic_rate:
            issues.add_file(
                ISSUE_MALFORMED_CATASTROPHIC,
                f"malformed rate {rate:.4%} >= {malformed_catastrophic_rate:.0%}",
                pcap_path=pcap_path,
            )
            hard_failures.append(ISSUE_MALFORMED_CATASTROPHIC)
        elif rate >= malformed_high_rate:
            issues.add_file(
                ISSUE_MALFORMED_HIGH,
                f"malformed rate {rate:.4%} >= {malformed_high_rate:.0%}",
                pcap_path=pcap_path,
                severity=SEVERITY_HIGH_WARNING,
            )
            warnings.append(ISSUE_MALFORMED_HIGH)
        else:
            issues.add_file(
                ISSUE_MALFORMED_PRESENT,
                f"malformed packets: {stats.malformed_count} ({rate:.4%})",
                pcap_path=pcap_path,
                severity=SEVERITY_WARNING,
            )
            warnings.append(ISSUE_MALFORMED_PRESENT)

    if stats.partial_count > 0:
        issues.add_file(
            ISSUE_PARTIAL_PRESENT,
            f"partial packets: {stats.partial_count}",
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_PARTIAL_PRESENT)

    if stats.unsupported_count > 0:
        issues.add_file(
            ISSUE_UNSUPPORTED_PRESENT,
            f"unsupported packets: {stats.unsupported_count}",
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_UNSUPPORTED_PRESENT)

    if stats.duplicate_timestamp_count > 0:
        issues.add_file(
            ISSUE_TIMESTAMP_DUPLICATE,
            f"duplicate adjacent timestamps: {stats.duplicate_timestamp_count}",
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_TIMESTAMP_DUPLICATE)

    if stats.negative_delta_count > 0:
        issues.add_file(
            ISSUE_TIMESTAMP_REVERSAL,
            f"negative timestamp deltas: {stats.negative_delta_count}",
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_TIMESTAMP_REVERSAL)

    span = stats.capture_timestamp_span
    if span is not None and span <= 0 and stats.packet_count > 1:
        issues.add_file(
            ISSUE_ZERO_DURATION,
            "non-positive capture_timestamp_span with multiple packets",
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_ZERO_DURATION)

    if train is not None and (
        train.unique_src_ips_capped or train.unique_dst_ips_capped
    ):
        issues.add_file(
            ISSUE_IP_CARDINALITY_CAPPED,
            (
                f"IP cardinality capped "
                f"(src={train.unique_src_ips_capped}, dst={train.unique_dst_ips_capped})"
            ),
            pcap_path=pcap_path,
            severity=SEVERITY_WARNING,
        )
        warnings.append(ISSUE_IP_CARDINALITY_CAPPED)

    return warnings, hard_failures


def _manifest_base(manifest_row: dict[str, Any], disk_size: int | None) -> dict[str, Any]:
    return {
        "pcap_path": manifest_row["pcap_path"],
        "filename": manifest_row.get("filename"),
        "source": manifest_row.get("source"),
        "split": manifest_row.get("split"),
        "binary_label": manifest_row.get("binary_label"),
        "attack_family": manifest_row.get("attack_family"),
        "attack_type": manifest_row.get("attack_type"),
        "profiling_type": manifest_row.get("profiling_type"),
        "profiling_variant": manifest_row.get("profiling_variant"),
        "device": manifest_row.get("device"),
        "capture_session": manifest_row.get("capture_session"),
        "file_size_manifest": manifest_row.get("file_size"),
        "file_size_bytes": disk_size,
        "audit_strategy_version": AUDIT_STRATEGY_VERSION,
    }


def scan_one_pcap_result(
    manifest_row: dict[str, Any],
    project_root: str,
    policy: AuditPolicy,
) -> PcapAuditResult:
    """Scan one PCAP and return a picklable result.

    Top-level function suitable for ProcessPoolExecutor.
    """
    started = time.perf_counter()
    root = Path(project_root)
    pcap_path = manifest_row["pcap_path"]
    abs_path = root / pcap_path
    disk_size = abs_path.stat().st_size if abs_path.is_file() else None
    collect_train = manifest_row.get("split") == "train"
    issues = IssueCollector(issue_cap_per_code=policy.issue_cap_per_code)
    base = _manifest_base(manifest_row, disk_size)

    if not abs_path.is_file():
        if policy.progress_dir:
            LiveProgressStore(Path(policy.progress_dir)).clear(pcap_path)
        issues.add_file(
            ISSUE_OPEN_FAILURE,
            f"PCAP missing on disk: {pcap_path}",
            pcap_path=pcap_path,
        )
        integrity_row = {
            **base,
            "parse_success": False,
            "open_error": "file not found",
            **_empty_integrity_packet_fields(),
            "warnings": "",
            "hard_failures": ISSUE_OPEN_FAILURE,
        }
        return PcapAuditResult(
            pcap_path=pcap_path,
            integrity_row=integrity_row,
            training_row=None,
            issue_rows=issues.rows(),
            hard_failures=[ISSUE_OPEN_FAILURE],
            warnings=[],
            by_protocol={},
            scan_elapsed_seconds=time.perf_counter() - started,
            packet_count=None,
            file_size=disk_size,
        )

    integrity = IntegrityStats()
    train: TrainCharacterizationStats | None = None
    if collect_train:
        train = TrainCharacterizationStats(
            integrity=integrity,
            ip_cardinality_cap=policy.ip_cardinality_cap,
        )

    live: LiveProgressStore | None = None
    every = max(0, int(policy.progress_every_packets))
    if policy.progress_dir:
        live = LiveProgressStore(Path(policy.progress_dir))
        live.write(
            pcap_path=pcap_path,
            packets=0,
            elapsed_seconds=0.0,
            status="starting",
            file_size=disk_size,
        )

    try:
        for record in iter_packets(abs_path):
            integrity.observe(record)
            issues.maybe_add_packet(pcap_path, record)
            if train is not None:
                train.observe(record)
            if live is not None and every > 0 and integrity.packet_count % every == 0:
                live.write(
                    pcap_path=pcap_path,
                    packets=integrity.packet_count,
                    elapsed_seconds=time.perf_counter() - started,
                    status="running",
                    file_size=disk_size,
                )
    except (OSError, ValueError) as exc:
        if live is not None:
            live.clear(pcap_path)
        issues.add_file(
            ISSUE_OPEN_FAILURE,
            f"failed to open/read PCAP: {exc}",
            pcap_path=pcap_path,
        )
        integrity_row = {
            **base,
            "parse_success": False,
            "open_error": str(exc),
            **_empty_integrity_packet_fields(),
            "warnings": "",
            "hard_failures": ISSUE_OPEN_FAILURE,
        }
        return PcapAuditResult(
            pcap_path=pcap_path,
            integrity_row=integrity_row,
            training_row=None,
            issue_rows=issues.rows(),
            hard_failures=[ISSUE_OPEN_FAILURE],
            warnings=[],
            by_protocol={},
            scan_elapsed_seconds=time.perf_counter() - started,
            packet_count=None,
            file_size=disk_size,
        )

    if live is not None:
        live.clear(pcap_path)

    warnings, hard_failures = _apply_file_level_observations(
        issues=issues,
        pcap_path=pcap_path,
        stats=integrity,
        train=train,
        malformed_high_rate=policy.malformed_high_rate,
        malformed_catastrophic_rate=policy.malformed_catastrophic_rate,
    )

    integrity_row = {
        **base,
        "parse_success": True,
        "open_error": None,
        **integrity.to_integrity_fields(),
        "warnings": ";".join(warnings),
        "hard_failures": ";".join(hard_failures),
    }

    training_row = None
    if train is not None:
        training_row = {
            "pcap_path": pcap_path,
            "filename": manifest_row.get("filename"),
            "source": manifest_row.get("source"),
            "split": manifest_row.get("split"),
            "binary_label": manifest_row.get("binary_label"),
            "attack_family": manifest_row.get("attack_family"),
            "attack_type": manifest_row.get("attack_type"),
            "profiling_type": manifest_row.get("profiling_type"),
            "profiling_variant": manifest_row.get("profiling_variant"),
            "device": manifest_row.get("device"),
            "capture_session": manifest_row.get("capture_session"),
            **train.to_characterization_fields(),
            "audit_strategy_version": AUDIT_STRATEGY_VERSION,
        }

    return PcapAuditResult(
        pcap_path=pcap_path,
        integrity_row=integrity_row,
        training_row=training_row,
        issue_rows=issues.rows(),
        hard_failures=hard_failures,
        warnings=warnings,
        by_protocol=dict(integrity.by_protocol),
        scan_elapsed_seconds=time.perf_counter() - started,
        packet_count=integrity.packet_count,
        file_size=disk_size,
    )


def worker_crash_result(
    manifest_row: dict[str, Any],
    *,
    detail: str,
) -> PcapAuditResult:
    """Build a hard-failure result when a worker process crashes unexpectedly."""
    pcap_path = manifest_row["pcap_path"]
    issues = IssueCollector()
    issues.add_file(
        ISSUE_WORKER_CRASH,
        detail,
        pcap_path=pcap_path,
        severity=SEVERITY_HARD_FAILURE,
    )
    base = _manifest_base(manifest_row, None)
    integrity_row = {
        **base,
        "parse_success": False,
        "open_error": detail,
        **_empty_integrity_packet_fields(),
        "warnings": "",
        "hard_failures": ISSUE_WORKER_CRASH,
    }
    return PcapAuditResult(
        pcap_path=pcap_path,
        integrity_row=integrity_row,
        training_row=None,
        issue_rows=issues.rows(),
        hard_failures=[ISSUE_WORKER_CRASH],
        warnings=[],
        by_protocol={},
        scan_elapsed_seconds=None,
        packet_count=None,
        file_size=None,
    )
