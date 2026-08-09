"""Manifest ↔ disk reconciliation for Phase 1B.2."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.audit.issues import IssueCollector
from iot_pcap_pipeline.audit.policy import (
    ISSUE_MANIFEST_DUPLICATE,
    ISSUE_MANIFEST_EXTRA,
    ISSUE_MANIFEST_MISSING,
    ISSUE_MANIFEST_SIZE_MISMATCH,
    ISSUE_MANIFEST_SPLIT_MISMATCH,
)
from iot_pcap_pipeline.dataset.inventory import discover_pcaps
from iot_pcap_pipeline.paths import PROJECT_ROOT, to_repo_relative


@dataclass
class ReconciliationResult:
    inventory_rows: list[dict[str, Any]]
    split_by_path: dict[str, str]
    discovered_paths: set[str]
    issues: IssueCollector
    hard_fail: bool = False
    warnings: list[str] = field(default_factory=list)


def load_inventory_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        # Normalize empty strings to None for optional fields.
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
        if row.get("file_size") is not None:
            row["file_size"] = int(row["file_size"])
    return rows


def load_split_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, str] = {}
    for row in rows:
        pcap_path = row["pcap_path"]
        mapping[pcap_path] = row["split"]
    return mapping


def reconcile_manifests(
    *,
    inventory_path: Path,
    split_path: Path,
    raw_root: Path,
    project_root: Path | None = None,
    issues: IssueCollector | None = None,
) -> ReconciliationResult:
    root = (project_root or PROJECT_ROOT).resolve()
    collector = issues or IssueCollector()
    inventory_rows = load_inventory_csv(inventory_path)
    split_by_path = load_split_csv(split_path)

    seen: set[str] = set()
    hard_fail = False

    for row in inventory_rows:
        pcap_path = row["pcap_path"]
        if pcap_path in seen:
            collector.add_corpus(
                ISSUE_MANIFEST_DUPLICATE,
                f"duplicate pcap_path in inventory: {pcap_path}",
                pcap_path=pcap_path,
            )
            hard_fail = True
        seen.add(pcap_path)

        split_from_inventory = row.get("split")
        split_from_split = split_by_path.get(pcap_path)
        if split_from_split is None:
            collector.add_corpus(
                ISSUE_MANIFEST_SPLIT_MISMATCH,
                f"pcap missing from dataset_split.csv: {pcap_path}",
                pcap_path=pcap_path,
            )
            hard_fail = True
        elif split_from_split != split_from_inventory:
            collector.add_corpus(
                ISSUE_MANIFEST_SPLIT_MISMATCH,
                (
                    f"split mismatch for {pcap_path}: "
                    f"inventory={split_from_inventory} split_csv={split_from_split}"
                ),
                pcap_path=pcap_path,
            )
            hard_fail = True

        abs_path = root / pcap_path
        if not abs_path.is_file():
            collector.add_corpus(
                ISSUE_MANIFEST_MISSING,
                f"manifest PCAP missing on disk: {pcap_path}",
                pcap_path=pcap_path,
            )
            hard_fail = True
        else:
            disk_size = abs_path.stat().st_size
            manifest_size = int(row.get("file_size") or -1)
            if disk_size != manifest_size:
                collector.add_corpus(
                    ISSUE_MANIFEST_SIZE_MISMATCH,
                    (
                        f"file size mismatch for {pcap_path}: "
                        f"manifest={manifest_size} disk={disk_size}"
                    ),
                    pcap_path=pcap_path,
                )
                hard_fail = True

    for split_path_key in split_by_path:
        if split_path_key not in seen:
            collector.add_corpus(
                ISSUE_MANIFEST_SPLIT_MISMATCH,
                f"dataset_split.csv path absent from inventory: {split_path_key}",
                pcap_path=split_path_key,
            )
            hard_fail = True

    discovered = {
        to_repo_relative(path, root) for path in discover_pcaps(raw_root)
    }
    for missing in sorted(seen - discovered):
        # Already reported as missing-on-disk when abs path check failed;
        # still ensure discovery set agreement.
        if (root / missing).is_file():
            collector.add_corpus(
                ISSUE_MANIFEST_MISSING,
                f"inventory path not found by discovery: {missing}",
                pcap_path=missing,
            )
            hard_fail = True

    for extra in sorted(discovered - seen):
        collector.add_corpus(
            ISSUE_MANIFEST_EXTRA,
            f"PCAP on disk not represented in manifest: {extra}",
            pcap_path=extra,
        )
        hard_fail = True

    return ReconciliationResult(
        inventory_rows=inventory_rows,
        split_by_path=split_by_path,
        discovered_paths=discovered,
        issues=collector,
        hard_fail=hard_fail,
    )
