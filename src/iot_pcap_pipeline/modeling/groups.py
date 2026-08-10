"""Derive atomic modeling_group_key values for Phase 2A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelingGroupSpec:
    modeling_group_key: str
    kind: str  # attack_lineage | spoofing | publisher_benign | profiling_device | profiling_singleton


def modeling_group_key_for_row(row: dict[str, Any]) -> ModelingGroupSpec:
    """Return the atomic modeling group for one TRAIN inventory row."""
    binary = (row.get("binary_label") or "").strip()
    source = (row.get("source") or "").strip()
    if binary == "ATTACK":
        family = (row.get("attack_family") or "").strip()
        attack_type = (row.get("attack_type") or "").strip()
        if not family or not attack_type:
            raise ValueError(
                f"ATTACK row missing family/type: {row.get('pcap_path')}"
            )
        key = f"{family}|{attack_type}"
        kind = "spoofing" if family == "Spoofing" else "attack_lineage"
        return ModelingGroupSpec(modeling_group_key=key, kind=kind)

    if binary == "BENIGN":
        if source == "attacks" or Path(row.get("pcap_path") or "").name.startswith(
            "Benign_"
        ):
            # Publisher benign capture (Benign_train.pcap).
            stem = Path(row.get("pcap_path") or "Benign_train").stem
            return ModelingGroupSpec(
                modeling_group_key=f"benign|publisher|{stem}",
                kind="publisher_benign",
            )
        device = (row.get("device") or "").strip()
        if device:
            return ModelingGroupSpec(
                modeling_group_key=f"benign|device|{device}",
                kind="profiling_device",
            )
        stem = Path(row.get("pcap_path") or "unknown").stem
        return ModelingGroupSpec(
            modeling_group_key=f"benign|singleton|{stem}",
            kind="profiling_singleton",
        )

    raise ValueError(
        f"unsupported binary_label for modeling group: {binary!r} "
        f"({row.get('pcap_path')})"
    )


def benign_category_for_row(row: dict[str, Any]) -> str:
    """Sampling-summary category for a BENIGN row."""
    source = (row.get("source") or "").strip()
    profiling_type = (row.get("profiling_type") or "").strip().lower()
    if source == "attacks" or not profiling_type:
        return "publisher_benign"
    if profiling_type in {"idle", "active", "interaction", "power"}:
        return f"profiling_{profiling_type}"
    return f"profiling_{profiling_type or 'other'}"
