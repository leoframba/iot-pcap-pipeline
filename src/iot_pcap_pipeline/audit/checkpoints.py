"""Atomic per-PCAP audit checkpoints for resumable corpus scans."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from iot_pcap_pipeline.audit.policy import AUDIT_STRATEGY_VERSION
from iot_pcap_pipeline.audit.worker import AuditPolicy, PcapAuditResult


def checkpoint_id(pcap_path: str) -> str:
    return hashlib.sha256(pcap_path.encode("utf-8")).hexdigest()


class CheckpointStore:
    """Store and validate completed per-PCAP audit results."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, pcap_path: str) -> Path:
        return self.root / f"{checkpoint_id(pcap_path)}.json"

    def clear(self) -> int:
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        for path in self.root.glob("*.json.tmp"):
            path.unlink(missing_ok=True)
        return removed

    def load_valid(
        self,
        *,
        pcap_path: str,
        policy: AuditPolicy,
        manifest_file_size: int | None,
        disk_file_size: int | None,
        split: str | None,
    ) -> PcapAuditResult | None:
        path = self.path_for(pcap_path)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        meta = payload.get("meta") or {}
        result_payload = payload.get("result")
        if not isinstance(result_payload, dict):
            return None

        expected = {
            "audit_strategy_version": AUDIT_STRATEGY_VERSION,
            "pcap_path": pcap_path,
            "manifest_file_size": manifest_file_size,
            "disk_file_size": disk_file_size,
            "split": split,
            "ip_cardinality_cap": policy.ip_cardinality_cap,
            "issue_cap_per_code": policy.issue_cap_per_code,
            "malformed_high_rate": policy.malformed_high_rate,
            "malformed_catastrophic_rate": policy.malformed_catastrophic_rate,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                return None

        try:
            return PcapAuditResult(
                pcap_path=result_payload["pcap_path"],
                integrity_row=result_payload["integrity_row"],
                training_row=result_payload.get("training_row"),
                issue_rows=list(result_payload.get("issue_rows") or []),
                hard_failures=list(result_payload.get("hard_failures") or []),
                warnings=list(result_payload.get("warnings") or []),
                by_protocol=dict(result_payload.get("by_protocol") or {}),
                scan_elapsed_seconds=result_payload.get("scan_elapsed_seconds"),
                packet_count=result_payload.get("packet_count"),
                file_size=result_payload.get("file_size"),
                from_checkpoint=True,
            )
        except (KeyError, TypeError):
            return None

    def save(self, result: PcapAuditResult, *, policy: AuditPolicy) -> Path:
        """Atomically write a completed PCAP checkpoint."""
        integrity = result.integrity_row
        payload = {
            "meta": {
                "audit_strategy_version": AUDIT_STRATEGY_VERSION,
                "pcap_path": result.pcap_path,
                "manifest_file_size": integrity.get("file_size_manifest"),
                "disk_file_size": integrity.get("file_size_bytes"),
                "split": integrity.get("split"),
                "ip_cardinality_cap": policy.ip_cardinality_cap,
                "issue_cap_per_code": policy.issue_cap_per_code,
                "malformed_high_rate": policy.malformed_high_rate,
                "malformed_catastrophic_rate": policy.malformed_catastrophic_rate,
            },
            "result": asdict(result),
        }
        final_path = self.path_for(result.pcap_path)
        tmp_path = final_path.with_suffix(".json.tmp")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
        return final_path
