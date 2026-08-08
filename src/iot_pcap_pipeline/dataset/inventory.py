"""PCAP discovery and conservative metadata classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iot_pcap_pipeline.dataset.schema import INVENTORY_COLUMNS, validate_inventory
from iot_pcap_pipeline.dataset.taxonomy import (
    classify_attack_stem,
    is_publisher_benign_stem,
    resolve_device_alias,
    strip_split_suffix,
)
from iot_pcap_pipeline.paths import DATASET_SCOPE, PROJECT_ROOT, to_repo_relative


def discover_pcaps(raw_root: Path) -> list[Path]:
    """Recursively discover *.pcap files under raw_root, sorted."""
    root = raw_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw root does not exist: {root}")
    return sorted(p for p in root.rglob("*.pcap") if p.is_file())


def _empty_row() -> dict[str, Any]:
    return {col: None for col in INVENTORY_COLUMNS}


def _parts_lower(path: Path) -> list[str]:
    return [part.lower() for part in path.parts]


def _classify_attack(path: Path, project_root: Path) -> dict[str, Any]:
    row = _empty_row()
    row["pcap_path"] = to_repo_relative(path, project_root)
    row["filename"] = path.name
    row["dataset_scope"] = DATASET_SCOPE
    row["source"] = "attacks"
    row["file_size"] = path.stat().st_size

    parts = _parts_lower(path)
    stem = path.stem
    base, suffix_split = strip_split_suffix(stem)

    dir_split: str | None = None
    if "train" in parts:
        dir_split = "train"
    if "test" in parts:
        if dir_split is not None:
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"  # placeholder; marked unresolved
            row["unresolved_reason"] = "path contains both train and test"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        dir_split = "test"

    if dir_split is None:
        row["binary_label"] = "UNKNOWN"
        row["split"] = "train"
        row["unresolved_reason"] = "attack PCAP missing train/test directory"
        row["grouping_key"] = f"unresolved:{row['pcap_path']}"
        return row

    if suffix_split is not None and suffix_split != dir_split:
        row["binary_label"] = "UNKNOWN"
        row["split"] = dir_split
        row["unresolved_reason"] = (
            f"contradictory split: directory={dir_split}, filename={suffix_split}"
        )
        row["grouping_key"] = f"unresolved:{row['pcap_path']}"
        return row

    row["split"] = dir_split

    if is_publisher_benign_stem(stem):
        row["binary_label"] = "BENIGN"
        row["capture_session"] = "Benign"
        row["grouping_key"] = f"attacks:Benign:{dir_split}"
        return row

    taxonomy = classify_attack_stem(stem)
    if taxonomy is None:
        row["binary_label"] = "UNKNOWN"
        row["unresolved_reason"] = f"unrecognized attack label: {base}"
        row["grouping_key"] = f"unresolved:{row['pcap_path']}"
        return row

    row["binary_label"] = "ATTACK"
    row["attack_family"] = taxonomy.family
    row["attack_type"] = taxonomy.attack_type
    row["capture_session"] = taxonomy.capture_session
    row["grouping_key"] = f"attacks:{taxonomy.attack_type}:{dir_split}:{taxonomy.capture_session}"
    return row


def _classify_profiling(path: Path, project_root: Path) -> dict[str, Any]:
    row = _empty_row()
    row["pcap_path"] = to_repo_relative(path, project_root)
    row["filename"] = path.name
    row["dataset_scope"] = DATASET_SCOPE
    row["source"] = "profiling"
    row["binary_label"] = "BENIGN"
    row["file_size"] = path.stat().st_size
    # Split assigned later; leave temporarily unset for validation after split.
    row["split"] = None

    parts = path.parts
    # Find PCAP/ segment and use following folder as experiment type.
    try:
        pcap_idx = next(i for i, part in enumerate(parts) if part.lower() == "pcap")
    except StopIteration:
        row["binary_label"] = "UNKNOWN"
        row["split"] = "train"
        row["unresolved_reason"] = "profiling path missing PCAP directory"
        row["grouping_key"] = f"unresolved:{row['pcap_path']}"
        return row

    if pcap_idx + 1 >= len(parts) - 1:
        row["binary_label"] = "UNKNOWN"
        row["split"] = "train"
        row["unresolved_reason"] = "profiling path missing experiment subdirectory"
        row["grouping_key"] = f"unresolved:{row['pcap_path']}"
        return row

    experiment = parts[pcap_idx + 1]
    experiment_key = experiment.lower()
    stem = path.stem

    if experiment_key == "idle":
        if stem != "Idle":
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = f"unexpected idle filename: {path.name}"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        row["profiling_type"] = "idle"
        row["profiling_variant"] = None
        row["capture_session"] = "Idle"
        row["grouping_key"] = "profiling:singleton:Idle"
        return row

    if experiment_key == "active":
        if stem != "Active":
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = f"unexpected active filename: {path.name}"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        row["profiling_type"] = "active"
        row["profiling_variant"] = "standard"
        row["capture_session"] = "Active"
        row["grouping_key"] = "profiling:singleton:Active"
        return row

    if experiment_key == "broker":
        if stem != "ActiveBroker":
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = f"unexpected broker filename: {path.name}"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        row["profiling_type"] = "active"
        row["profiling_variant"] = "active_broker"
        row["capture_session"] = "ActiveBroker"
        row["grouping_key"] = "profiling:singleton:ActiveBroker"
        return row

    if experiment_key == "power":
        if not stem.endswith("_Power"):
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = f"unexpected power filename: {path.name}"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        raw_device = stem[: -len("_Power")]
        device = resolve_device_alias(raw_device)
        if device is None:
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = f"unknown power device alias: {raw_device}"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        row["profiling_type"] = "power"
        row["profiling_variant"] = None
        row["device"] = device
        row["capture_session"] = f"{device}_Power"
        row["grouping_key"] = f"profiling:device:{device}"
        return row

    if experiment_key == "interactions":
        if pcap_idx + 2 >= len(parts) - 1:
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = "interaction path missing device directory"
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        raw_device_dir = parts[pcap_idx + 2]
        device = resolve_device_alias(raw_device_dir)
        if device is None:
            row["binary_label"] = "UNKNOWN"
            row["split"] = "train"
            row["unresolved_reason"] = (
                f"unknown interaction device alias: {raw_device_dir}"
            )
            row["grouping_key"] = f"unresolved:{row['pcap_path']}"
            return row
        # Preserve raw stem tokens (including PRECORDING) in capture_session.
        row["profiling_type"] = "interaction"
        row["profiling_variant"] = None
        row["device"] = device
        row["capture_session"] = stem
        row["grouping_key"] = f"profiling:device:{device}"
        return row

    row["binary_label"] = "UNKNOWN"
    row["split"] = "train"
    row["unresolved_reason"] = f"unrecognized profiling experiment folder: {experiment}"
    row["grouping_key"] = f"unresolved:{row['pcap_path']}"
    return row


def classify_pcap(path: Path, project_root: Path | None = None) -> dict[str, Any]:
    """Classify a single PCAP path into an inventory row."""
    root = (project_root or PROJECT_ROOT).resolve()
    resolved = path.resolve()
    parts_lower = _parts_lower(resolved)

    if "attacks" in parts_lower:
        return _classify_attack(resolved, root)
    if "profiling" in parts_lower:
        return _classify_profiling(resolved, root)

    row = _empty_row()
    row["pcap_path"] = to_repo_relative(resolved, root)
    row["filename"] = resolved.name
    row["dataset_scope"] = DATASET_SCOPE
    row["source"] = "unknown"
    row["binary_label"] = "UNKNOWN"
    row["split"] = "train"
    row["file_size"] = resolved.stat().st_size
    row["unresolved_reason"] = "PCAP outside attacks/ or profiling/ trees"
    row["grouping_key"] = f"unresolved:{row['pcap_path']}"
    return row


def build_inventory(
    raw_root: Path,
    project_root: Path | None = None,
    *,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Discover and classify all PCAPs under raw_root.

    Profiling rows may have split=None until assign_profiling_splits runs.
    Set validate=True only after splits are assigned.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    rows = [classify_pcap(path, root) for path in discover_pcaps(raw_root)]
    rows.sort(key=lambda r: r["pcap_path"])
    if validate:
        validate_inventory(rows)
    return rows
