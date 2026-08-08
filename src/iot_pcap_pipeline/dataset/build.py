"""Orchestrate Phase 1A manifest generation."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.dataset.inventory import build_inventory
from iot_pcap_pipeline.dataset.schema import INVENTORY_COLUMNS, SPLIT_COLUMNS
from iot_pcap_pipeline.dataset.split import assign_profiling_splits
from iot_pcap_pipeline.paths import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_SEED,
    PROJECT_ROOT,
)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                col: ("" if row.get(col) is None else row.get(col)) for col in columns
            }
            writer.writerow(serialized)


def _publisher_area(row: dict[str, Any]) -> str | None:
    if row.get("source") != "attacks":
        return None
    path = str(row.get("pcap_path", "")).replace("\\", "/")
    parts = path.lower().split("/")
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return None


def format_summary(rows: list[dict[str, Any]]) -> str:
    """Format the Phase 1A terminal summary."""
    total = len(rows)
    publisher_train = sum(1 for r in rows if _publisher_area(r) == "train")
    publisher_test = sum(1 for r in rows if _publisher_area(r) == "test")
    profiling = sum(1 for r in rows if r.get("source") == "profiling")

    attack_train = sum(
        1
        for r in rows
        if r.get("binary_label") == "ATTACK" and r.get("split") == "train"
    )
    attack_test = sum(
        1
        for r in rows
        if r.get("binary_label") == "ATTACK" and r.get("split") == "test"
    )

    benign_pub_train = sum(
        1
        for r in rows
        if r.get("source") == "attacks"
        and r.get("binary_label") == "BENIGN"
        and r.get("split") == "train"
    )
    benign_pub_test = sum(
        1
        for r in rows
        if r.get("source") == "attacks"
        and r.get("binary_label") == "BENIGN"
        and r.get("split") == "test"
    )
    benign_prof_train = sum(
        1
        for r in rows
        if r.get("source") == "profiling"
        and r.get("binary_label") == "BENIGN"
        and r.get("split") == "train"
    )
    benign_prof_test = sum(
        1
        for r in rows
        if r.get("source") == "profiling"
        and r.get("binary_label") == "BENIGN"
        and r.get("split") == "test"
    )

    profiling_type_counts = Counter(
        r.get("profiling_type")
        for r in rows
        if r.get("source") == "profiling" and r.get("binary_label") == "BENIGN"
    )
    attack_family_counts = Counter(
        r.get("attack_family")
        for r in rows
        if r.get("binary_label") == "ATTACK"
    )
    attack_type_counts = Counter(
        r.get("attack_type") for r in rows if r.get("binary_label") == "ATTACK"
    )
    unresolved = [r for r in rows if r.get("binary_label") == "UNKNOWN"]

    lines = [
        "Corpus",
        "------",
        f"publisher train-area PCAPs: {publisher_train}",
        f"publisher test-area PCAPs:  {publisher_test}",
        f"profiling PCAPs:            {profiling}",
        f"total:                      {total}",
        "",
        "Binary labels",
        "-------------",
        f"ATTACK / train: {attack_train}",
        f"ATTACK / test:  {attack_test}",
        f"BENIGN / publisher train: {benign_pub_train}",
        f"BENIGN / publisher test:  {benign_pub_test}",
        f"BENIGN / profiling train: {benign_prof_train}",
        f"BENIGN / profiling test:  {benign_prof_test}",
        "",
        "Profiling types (BENIGN profiling)",
        "----------------------------------",
    ]
    for key in sorted(k for k in profiling_type_counts if k is not None):
        lines.append(f"{key}: {profiling_type_counts[key]}")

    lines.extend(
        [
            "",
            "Attack families",
            "---------------",
        ]
    )
    for key in sorted(k for k in attack_family_counts if k is not None):
        lines.append(f"{key}: {attack_family_counts[key]}")

    lines.extend(
        [
            "",
            "Attack types",
            "------------",
        ]
    )
    for key in sorted(k for k in attack_type_counts if k is not None):
        lines.append(f"{key}: {attack_type_counts[key]}")

    lines.extend(
        [
            "",
            "Unresolved",
            "----------",
            f"count: {len(unresolved)}",
        ]
    )
    for row in unresolved:
        reason = row.get("unresolved_reason") or "unspecified"
        lines.append(f"- {row.get('pcap_path')}: {reason}")

    return "\n".join(lines) + "\n"


def build_manifests(
    raw_root: Path | None = None,
    output_dir: Path | None = None,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    project_root: Path | None = None,
    summary_file: TextIO | None = None,
) -> dict[str, Any]:
    """Build pcap_inventory.csv and dataset_split.csv."""
    root = (project_root or PROJECT_ROOT).resolve()
    raw = (raw_root or DEFAULT_RAW_ROOT).resolve()
    out = (output_dir or DEFAULT_MANIFEST_DIR).resolve()

    inventory = build_inventory(raw, project_root=root, validate=False)
    inventory, split_records = assign_profiling_splits(inventory, seed=seed)

    inventory_path = out / "pcap_inventory.csv"
    split_path = out / "dataset_split.csv"
    _write_csv(inventory_path, inventory, INVENTORY_COLUMNS)
    _write_csv(split_path, split_records, SPLIT_COLUMNS)

    summary = format_summary(inventory)
    if summary_file is not None:
        summary_file.write(summary)
    else:
        print(summary, end="")

    return {
        "inventory_path": inventory_path,
        "split_path": split_path,
        "inventory": inventory,
        "split_records": split_records,
        "summary": summary,
    }
