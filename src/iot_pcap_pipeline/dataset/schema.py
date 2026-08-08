"""Conditional schema validation for Phase 1A manifests."""

from __future__ import annotations

from typing import Any

INVENTORY_COLUMNS = [
    "pcap_path",
    "filename",
    "dataset_scope",
    "source",
    "split",
    "binary_label",
    "attack_family",
    "attack_type",
    "profiling_type",
    "profiling_variant",
    "device",
    "capture_session",
    "file_size",
    "grouping_key",
    "unresolved_reason",
]

SPLIT_COLUMNS = [
    "pcap_path",
    "source",
    "grouping_key",
    "split",
    "split_origin",
    "strategy_version",
    "seed",
]

ALWAYS_REQUIRED = (
    "pcap_path",
    "filename",
    "dataset_scope",
    "source",
    "split",
    "binary_label",
    "file_size",
)

VALID_SPLITS = {"train", "test"}
VALID_LABELS = {"ATTACK", "BENIGN", "UNKNOWN"}
VALID_SOURCES = {"attacks", "profiling", "unknown"}


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def validate_inventory_row(row: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages for one inventory row."""
    errors: list[str] = []

    for field in ALWAYS_REQUIRED:
        if _is_empty(row.get(field)):
            errors.append(f"missing required field: {field}")

    label = row.get("binary_label")
    source = row.get("source")
    split = row.get("split")

    if label not in VALID_LABELS:
        errors.append(f"invalid binary_label: {label!r}")
    if source not in VALID_SOURCES:
        errors.append(f"invalid source: {source!r}")
    if split not in VALID_SPLITS and label != "UNKNOWN":
        errors.append(f"invalid split: {split!r}")

    if label == "ATTACK":
        if _is_empty(row.get("attack_family")):
            errors.append("ATTACK rows require attack_family")
        if _is_empty(row.get("attack_type")):
            errors.append("ATTACK rows require attack_type")

    if (
        source == "profiling"
        and label != "UNKNOWN"
        and _is_empty(row.get("profiling_type"))
    ):
        errors.append("profiling rows require profiling_type")

    if label == "UNKNOWN" and _is_empty(row.get("unresolved_reason")):
        errors.append("UNKNOWN rows require unresolved_reason")

    return errors


def validate_inventory(rows: list[dict[str, Any]]) -> None:
    """Raise ValueError if any row fails conditional schema validation."""
    all_errors: list[str] = []
    seen_paths: set[str] = set()

    for idx, row in enumerate(rows):
        path = row.get("pcap_path")
        if path in seen_paths:
            all_errors.append(f"row {idx}: duplicate pcap_path {path!r}")
        if path:
            seen_paths.add(str(path))

        for err in validate_inventory_row(row):
            all_errors.append(f"row {idx} ({path}): {err}")

    if all_errors:
        preview = "\n".join(all_errors[:20])
        extra = len(all_errors) - 20
        suffix = f"\n... and {extra} more" if extra > 0 else ""
        raise ValueError(f"Inventory validation failed:\n{preview}{suffix}")
