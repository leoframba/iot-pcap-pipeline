"""Unified structured audit issue collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iot_pcap_pipeline.audit.policy import (
    DEFAULT_ISSUE_CAP_PER_CODE,
    HARD_FAILURE_CODES,
    SCOPE_CORPUS,
    SCOPE_FILE,
    SCOPE_PACKET,
    SEVERITY_HARD_FAILURE,
    SEVERITY_WARNING,
)
from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus


def normalize_packet_issue_code(record: PacketRecord) -> str | None:
    """Map a PacketRecord to a fixed issue code, or None if not an issue example."""
    status = record.parse_status
    detail = (record.parse_detail or "").lower()
    proto = (record.protocol_name or "unknown").lower()

    if status == ParseStatus.OK:
        return None

    if status == ParseStatus.UNSUPPORTED:
        if proto == "lldp" or "0x88cc" in detail:
            return "unsupported:lldp"
        if proto.startswith("ethertype_"):
            return f"unsupported:{proto}"
        if proto.startswith("ip_proto_"):
            return f"unsupported:{proto}"
        if proto.startswith("linktype_"):
            return f"unsupported:{proto}"
        return f"unsupported:{proto}"

    if status == ParseStatus.PARTIAL:
        if "tcp" in detail:
            return "partial:tcp_truncated"
        if "udp" in detail:
            return "partial:udp_truncated"
        if "ipv4" in detail:
            return "partial:ipv4"
        if "ipv6" in detail:
            return "partial:ipv6"
        if "arp" in detail:
            return "partial:arp"
        if "llc" in detail:
            return "partial:llc"
        return "partial:other"

    if status == ParseStatus.MALFORMED:
        if "ethernet" in detail:
            return "malformed:ethernet"
        if "ipv4" in detail:
            return "malformed:ipv4"
        if "ipv6" in detail:
            return "malformed:ipv6"
        return "malformed:other"

    if status == ParseStatus.ERROR:
        return "error:unexpected"

    return None


@dataclass
class AuditIssue:
    scope: str
    severity: str
    issue_code: str
    detail: str
    pcap_path: str | None = None
    packet_index: int | None = None
    timestamp: float | None = None
    parse_status: str | None = None
    protocol_name: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "pcap_path": self.pcap_path,
            "packet_index": self.packet_index,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "issue_code": self.issue_code,
            "detail": self.detail,
            "parse_status": self.parse_status,
            "protocol_name": self.protocol_name,
        }


@dataclass
class IssueCollector:
    """Collect corpus/file issues unbounded and packet examples with a per-code cap."""

    issue_cap_per_code: int = DEFAULT_ISSUE_CAP_PER_CODE
    issues: list[AuditIssue] = field(default_factory=list)
    _packet_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(
        self,
        *,
        scope: str,
        issue_code: str,
        detail: str,
        severity: str | None = None,
        pcap_path: str | None = None,
        packet_index: int | None = None,
        timestamp: float | None = None,
        parse_status: str | None = None,
        protocol_name: str | None = None,
    ) -> None:
        resolved_severity = severity or (
            SEVERITY_HARD_FAILURE
            if issue_code in HARD_FAILURE_CODES
            else SEVERITY_WARNING
        )
        self.issues.append(
            AuditIssue(
                scope=scope,
                severity=resolved_severity,
                issue_code=issue_code,
                detail=detail,
                pcap_path=pcap_path,
                packet_index=packet_index,
                timestamp=timestamp,
                parse_status=parse_status,
                protocol_name=protocol_name,
            )
        )

    def add_corpus(self, issue_code: str, detail: str, *, pcap_path: str | None = None) -> None:
        self.add(
            scope=SCOPE_CORPUS,
            issue_code=issue_code,
            detail=detail,
            pcap_path=pcap_path,
        )

    def add_file(
        self,
        issue_code: str,
        detail: str,
        *,
        pcap_path: str,
        severity: str | None = None,
    ) -> None:
        self.add(
            scope=SCOPE_FILE,
            issue_code=issue_code,
            detail=detail,
            pcap_path=pcap_path,
            severity=severity,
        )

    def maybe_add_packet(self, pcap_path: str, record: PacketRecord) -> None:
        code = normalize_packet_issue_code(record)
        if code is None:
            return
        key = (pcap_path, code)
        count = self._packet_counts.get(key, 0)
        if count >= self.issue_cap_per_code:
            self._packet_counts[key] = count + 1
            return
        self._packet_counts[key] = count + 1
        self.add(
            scope=SCOPE_PACKET,
            issue_code=code,
            detail=record.parse_detail or "",
            pcap_path=pcap_path,
            packet_index=record.packet_index,
            timestamp=record.timestamp,
            parse_status=record.parse_status.value,
            protocol_name=record.protocol_name,
            severity=(
                SEVERITY_HARD_FAILURE
                if record.parse_status == ParseStatus.ERROR
                else SEVERITY_WARNING
            ),
        )

    def rows(self) -> list[dict[str, Any]]:
        return [issue.to_row() for issue in self.issues]

    def hard_failure_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == SEVERITY_HARD_FAILURE)

    def has_hard_failures(self) -> bool:
        return self.hard_failure_count() > 0
