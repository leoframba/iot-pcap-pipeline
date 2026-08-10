"""Gate 2A freeze: lock TRAIN modeling split + chosen sampling plan for Phase 2B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.modeling.sampling import CANDIDATE_PLANS
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    MODELING_SPLIT_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

GATE_2A_STATUS = "passed"
GATE_2A_DECISION = (
    "Freeze the Phase 2A TRAIN modeling split (modeling_group_key lineages; "
    "Owltron_Camera + Idle benign validation; Spoofing and publisher benign "
    "fit-only with documented limitations) and select group_balanced as the "
    "V1 FIT sampling policy: per-modeling_group budgets "
    "DDoS/DoS=60k, MQTT=30k, Recon/Spoofing/BENIGN uncapped; "
    "TRAIN-validation never sampled; TEST sealed until model freeze."
)

FROZEN_SAMPLING_PLAN_ID = "group_balanced"
DEFAULT_GATE_2A_COMPLETE_PATH = (
    DEFAULT_MODELING_DIR / "v1" / "gate_2a_complete.json"
)
DEFAULT_SAMPLING_PLAN_PATH = DEFAULT_MODELING_DIR / "v1" / "sampling_plan.json"
DEFAULT_SPLIT_MANIFEST_PATH = (
    DEFAULT_MODELING_DIR / "v1" / "modeling_split_manifest.csv"
)


def _plan_by_id(plan_id: str) -> dict[str, Any]:
    for plan in CANDIDATE_PLANS:
        if plan["plan_id"] == plan_id:
            return plan
    raise FeatureExtractionError(f"unknown sampling plan_id: {plan_id!r}")


def apply_gate_2a_freeze_to_plan(
    plan_payload: dict[str, Any],
    *,
    plan_id: str = FROZEN_SAMPLING_PLAN_ID,
) -> dict[str, Any]:
    """Return a copy of sampling_plan payload with Gate 2A freeze fields."""
    chosen = _plan_by_id(plan_id)
    out = dict(plan_payload)
    out["status"] = "frozen"
    out["gate_2a_status"] = GATE_2A_STATUS
    out["gate_2a_decision"] = GATE_2A_DECISION
    out["frozen_sampling_plan_id"] = plan_id
    out["frozen_sampling_plan"] = {
        "plan_id": chosen["plan_id"],
        "cap_mode": chosen.get("cap_mode", "per_pcap"),
        "caps": chosen["caps"],
        "description": chosen["description"],
    }
    out["next"] = (
        "Phase 2B: materialize FIT training view under frozen group_balanced "
        "reservoir sampling; train baselines on TRAIN-fit; evaluate/threshold "
        "on unsampled TRAIN-validation only. Do not consult TEST."
    )
    return out


def freeze_gate_2a(
    *,
    plan_id: str = FROZEN_SAMPLING_PLAN_ID,
    sampling_plan_path: Path | str | None = None,
    complete_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze Gate 2A into sampling_plan.json + gate_2a_complete.json."""
    root = (project_root or PROJECT_ROOT).resolve()
    plan_path = Path(sampling_plan_path or DEFAULT_SAMPLING_PLAN_PATH)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    marker_path = Path(complete_path or DEFAULT_GATE_2A_COMPLETE_PATH)
    if not marker_path.is_absolute():
        marker_path = root / marker_path
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    if not plan_path.is_file():
        raise FeatureExtractionError(
            f"sampling_plan.json missing: {plan_path}. "
            "Run characterize-modeling-split first."
        )
    if not split_path.is_file():
        raise FeatureExtractionError(
            f"modeling_split_manifest.csv missing: {split_path}"
        )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    frozen_plan = apply_gate_2a_freeze_to_plan(payload, plan_id=plan_id)

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan_path.with_suffix(plan_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(frozen_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(plan_path)

    complete = {
        "gate_2a_status": GATE_2A_STATUS,
        "gate_2a_decision": GATE_2A_DECISION,
        "modeling_split_strategy_version": MODELING_SPLIT_STRATEGY_VERSION,
        "frozen_sampling_plan_id": plan_id,
        "frozen_sampling_plan": frozen_plan["frozen_sampling_plan"],
        "validation_sampling": "never",
        "split": {
            "fit_pcaps": frozen_plan.get("split_summary", {})
            .get("fit", {})
            .get("pcap_count"),
            "validation_pcaps": frozen_plan.get("split_summary", {})
            .get("validation", {})
            .get("pcap_count"),
        },
        "artifacts": {
            "modeling_split_manifest": to_repo_relative(
                split_path, project_root=root
            ),
            "sampling_plan": to_repo_relative(plan_path, project_root=root),
            "sampling_summary": frozen_plan.get("artifacts", {}).get(
                "sampling_summary", "data/modeling/v1/sampling_summary.csv"
            ),
            "gate_2a_complete": to_repo_relative(marker_path, project_root=root),
        },
        "next": frozen_plan["next"],
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    mtmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    mtmp.write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mtmp.replace(marker_path)
    return complete


def format_gate_2a_freeze_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Gate 2A — modeling split + sampling policy FROZEN",
        f"gate_2a_status: {payload.get('gate_2a_status')}",
        f"frozen_sampling_plan_id: {payload.get('frozen_sampling_plan_id')}",
        f"modeling_split_strategy_version: "
        f"{payload.get('modeling_split_strategy_version')}",
        f"validation_sampling: {payload.get('validation_sampling')}",
    ]
    arts = payload.get("artifacts") or {}
    if arts:
        lines.append("artifacts:")
        for key, value in sorted(arts.items()):
            lines.append(f"  {key}: {value}")
    lines.append(f"next: {payload.get('next')}")
    return "\n".join(lines) + "\n"
