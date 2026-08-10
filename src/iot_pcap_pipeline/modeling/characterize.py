"""Orchestrate Phase 2A modeling-split characterization artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.features.dataset import require_train_build_complete
from iot_pcap_pipeline.modeling.sampling import (
    CANDIDATE_PLANS,
    ReservoirContract,
    build_split_summary,
    simulate_plan,
    write_sampling_summary,
)
from iot_pcap_pipeline.modeling.split import (
    assign_modeling_split,
    validate_split_acceptance,
    write_modeling_split_manifest,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    DEFAULT_MODELING_SEED,
    MODELING_SPLIT_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

DEFAULT_MODELING_V1_DIR = DEFAULT_MODELING_DIR / "v1"
DEFAULT_SPLIT_MANIFEST_PATH = DEFAULT_MODELING_V1_DIR / "modeling_split_manifest.csv"
DEFAULT_SAMPLING_PLAN_PATH = DEFAULT_MODELING_V1_DIR / "sampling_plan.json"
DEFAULT_SAMPLING_SUMMARY_PATH = DEFAULT_MODELING_V1_DIR / "sampling_summary.csv"


@dataclass
class ModelingCharacterizationResult:
    split_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    plan_payload: dict[str, Any]
    split_manifest_path: Path
    sampling_plan_path: Path
    sampling_summary_path: Path
    limitations: list[str] = field(default_factory=list)
    acceptance_issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.acceptance_issues


def characterize_modeling_split(
    *,
    inventory_path: Path | str | None = None,
    build_manifest_path: Path | str | None = None,
    train_complete_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    sampling_plan_path: Path | str | None = None,
    sampling_summary_path: Path | str | None = None,
    project_root: Path | None = None,
    base_seed: int = DEFAULT_MODELING_SEED,
    progress_file: TextIO | None = None,
) -> ModelingCharacterizationResult:
    """Assign TRAIN modeling split and simulate FIT sampling candidates."""
    root = (project_root or PROJECT_ROOT).resolve()
    require_train_build_complete(train_complete_path, project_root=root)

    out_dir = Path(output_dir or DEFAULT_MODELING_V1_DIR)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    split_path = Path(split_manifest_path or (out_dir / "modeling_split_manifest.csv"))
    if not split_path.is_absolute():
        split_path = root / split_path
    plan_path = Path(sampling_plan_path or (out_dir / "sampling_plan.json"))
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    summary_path = Path(sampling_summary_path or (out_dir / "sampling_summary.csv"))
    if not summary_path.is_absolute():
        summary_path = root / summary_path

    if progress_file is not None:
        progress_file.write(
            "Phase 2A: assigning modeling groups (fit vs validation)...\n"
        )
        progress_file.flush()

    assigned = assign_modeling_split(
        inventory_path=inventory_path,
        build_manifest_path=build_manifest_path,
        train_complete_path=train_complete_path,
        project_root=root,
        base_seed=base_seed,
    )
    issues = validate_split_acceptance(assigned.rows, project_root=root)
    if issues:
        raise FeatureExtractionError(
            "modeling split acceptance failed:\n- " + "\n- ".join(issues)
        )

    write_modeling_split_manifest(split_path, assigned.rows)

    fit_rows = [r for r in assigned.rows if r["modeling_split"] == "fit"]
    val_rows = [r for r in assigned.rows if r["modeling_split"] == "validation"]
    if progress_file is not None:
        progress_file.write(
            f"Assigned fit={len(fit_rows)} validation={len(val_rows)} PCAPs\n"
        )
        progress_file.write("Simulating FIT-only sampling candidate plans...\n")
        progress_file.flush()

    summary_rows: list[dict[str, Any]] = []
    for plan in CANDIDATE_PLANS:
        summary_rows.extend(
            simulate_plan(fit_rows, plan, base_seed=base_seed)
        )
    write_sampling_summary(summary_path, summary_rows)

    contract = ReservoirContract()
    split_summary = build_split_summary(assigned.rows)
    plan_payload: dict[str, Any] = {
        "status": "characterization_only",
        "modeling_split_strategy_version": MODELING_SPLIT_STRATEGY_VERSION,
        "base_seed": base_seed,
        "seed_formula": (
            f"int(sha256('{MODELING_SPLIT_STRATEGY_VERSION}|{{base_seed}}|{{salt}}')"
            ".hexdigest()[:16], 16)"
        ),
        "reservoir_contract": {
            "unit": contract.unit,
            "scope": contract.scope,
            "algorithm": contract.algorithm,
            "seed_formula": contract.seed_formula,
            "note": contract.note,
        },
        "validation_sampling": "never",
        "limitations": assigned.limitations,
        "selection_notes": assigned.selection_notes,
        "split_summary": split_summary,
        "candidate_plans": [
            {
                "plan_id": p["plan_id"],
                "description": p["description"],
                "caps": p["caps"],
            }
            for p in CANDIDATE_PLANS
        ],
        "artifacts": {
            "modeling_split_manifest": to_repo_relative(split_path, project_root=root),
            "sampling_summary": to_repo_relative(summary_path, project_root=root),
            "sampling_plan": to_repo_relative(plan_path, project_root=root),
        },
        "next": (
            "Review split + sampling_summary ratios, then freeze chosen plan_id "
            "/ caps before Phase 2B. Do not train models or consult TEST."
        ),
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan_path.with_suffix(plan_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(plan_path)

    return ModelingCharacterizationResult(
        split_rows=assigned.rows,
        summary_rows=summary_rows,
        plan_payload=plan_payload,
        split_manifest_path=split_path,
        sampling_plan_path=plan_path,
        sampling_summary_path=summary_path,
        limitations=assigned.limitations,
        acceptance_issues=issues,
    )


def format_modeling_characterization_summary(
    result: ModelingCharacterizationResult,
    *,
    project_root: Path | None = None,
) -> str:
    root = (project_root or PROJECT_ROOT).resolve()

    def _rel(path: Path) -> str:
        return to_repo_relative(path, project_root=root)

    fit_n = sum(1 for r in result.split_rows if r["modeling_split"] == "fit")
    val_n = sum(1 for r in result.split_rows if r["modeling_split"] == "validation")
    fit_g = len(
        {
            r["modeling_group_key"]
            for r in result.split_rows
            if r["modeling_split"] == "fit"
        }
    )
    val_g = len(
        {
            r["modeling_group_key"]
            for r in result.split_rows
            if r["modeling_split"] == "validation"
        }
    )
    lines = [
        "Phase 2A — modeling split + sampling characterization",
        f"status: {'passed' if result.passed else 'failed'}",
        f"modeling_split_strategy_version: {MODELING_SPLIT_STRATEGY_VERSION}",
        f"pcaps: fit={fit_n} validation={val_n} total={len(result.split_rows)}",
        f"modeling_groups: fit={fit_g} validation={val_g}",
        f"manifest: {_rel(result.split_manifest_path)}",
        f"sampling_plan: {_rel(result.sampling_plan_path)}",
        f"sampling_summary: {_rel(result.sampling_summary_path)}",
        "validation_sampling: never (all val windows retained)",
        "sampling status: characterization_only (caps not frozen)",
    ]
    if result.limitations:
        lines.append("limitations:")
        for lim in result.limitations:
            lines.append(f"  - {lim}")
    notes = result.plan_payload.get("selection_notes") or []
    if notes:
        lines.append("holdouts:")
        for note in notes:
            if note.get("family") == "BENIGN_profiling":
                lines.append(
                    f"  - BENIGN device={note.get('held_out_device_group')} "
                    f"singleton={note.get('held_out_singleton')} "
                    f"share={note.get('device_fraction_of_profiling')}"
                )
            else:
                lines.append(
                    f"  - {note.get('family')}: held_out={note.get('held_out_group')} "
                    f"frac={note.get('actual_fraction')}"
                )
    lines.append(
        "Next: review artifacts and freeze sampling plan before Phase 2B. "
        "Do not train models or use TEST."
    )
    return "\n".join(lines) + "\n"
