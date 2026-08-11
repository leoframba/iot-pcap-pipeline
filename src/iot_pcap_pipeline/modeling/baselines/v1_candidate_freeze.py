"""Phase 2B.5: close model exploration and freeze the V1 candidate definition.

Locks model family + model-input feature selection. Threshold remains unfrozen.
TEST stays sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.ablations import (
    ABLATION_VERSION,
    DEFAULT_ABLATION_ROOT,
)
from iot_pcap_pipeline.modeling.baselines.feature22_boost import (
    DEFAULT_FEATURE22_BOOST_ROOT,
    FEATURE22_BOOST_VERSION,
)
from iot_pcap_pipeline.modeling.baselines.model_input import (
    DROPPED_TEMPORAL_FEATURES,
    V1_MODEL_INPUT_CONTRACT_PATH,
    V1_MODEL_INPUT_FEATURES,
    V1_MODEL_INPUT_VERSION,
    build_v1_model_input_contract,
)
from iot_pcap_pipeline.modeling.baselines.models import HGB_PARAMS, RANDOM_SEED
from iot_pcap_pipeline.modeling.freeze import FROZEN_SAMPLING_PLAN_ID
from iot_pcap_pipeline.modeling.view import file_sha256
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

GATE_2B5_STATUS = "passed"
GATE_2B5_VERSION = "phase2b5_v1"
DEFAULT_V1_CANDIDATE_FREEZE_PATH = (
    DEFAULT_MODELING_DIR / "v1" / "v1_candidate_freeze.json"
)

GATE_2B5_DECISION = (
    "Close Phase 2B model-family exploration. Select HistGradientBoostingClassifier "
    "with the 22 nontemporal model-input features (unweighted, group_balanced FIT) "
    "as the V1 candidate. Keep all 27 features in the extractor/Parquet schema. "
    "XGBoost-22 is recorded as runner-up only. Reject CatBoost-22 and all other "
    "families/configs tested in 2B.4/2B.4B/2B.4C/2B.4D. Threshold remains "
    "unfrozen; TEST remains sealed."
)

MODEL_EXPLORATION_RANKING: list[dict[str, str]] = [
    {
        "rank": "1",
        "model_id": "hgb_22",
        "status": "final_v1_candidate",
        "notes": "HGB, 22 nontemporal features, unweighted, group_balanced FIT",
    },
    {
        "rank": "2",
        "model_id": "xgboost_22",
        "status": "runner_up",
        "notes": "Very strong; not selected (prefer characterized HGB on near-ties)",
    },
    {
        "rank": "reject",
        "model_id": "catboost_22",
        "status": "rejected",
        "notes": "Material Recon/min-family deficit vs HGB-22",
    },
    {
        "rank": "reject",
        "model_id": "other_families_and_27_feature_variants",
        "status": "rejected_or_baseline",
        "notes": (
            "Includes AdaBoost, RandomForest, ExtraTrees, HGB-27, XGBoost-27, "
            "CatBoost-27, balanced-weight variants"
        ),
    },
]

HGB22_ARTIFACT = DEFAULT_ABLATION_ROOT / "models" / "C_22_unweighted.joblib"
HGB22_FROM_REMATCH = (
    DEFAULT_FEATURE22_BOOST_ROOT / "models" / "hgb_22.joblib"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _require_parent_complete(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FeatureExtractionError(f"{label} complete marker missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise FeatureExtractionError(f"{label} not passed: {payload.get('status')!r}")
    return payload


def build_v1_candidate_freeze_payload(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()

    ablation = _require_parent_complete(
        root / DEFAULT_ABLATION_ROOT / "ablation_complete.json",
        label="Phase 2B.3B",
    )
    rematch = _require_parent_complete(
        root / DEFAULT_FEATURE22_BOOST_ROOT / "feature22_boost_complete.json",
        label="Phase 2B.4D",
    )

    hgb_src = root / HGB22_ARTIFACT
    if not hgb_src.is_file():
        raise FeatureExtractionError(f"missing HGB-22 artifact: {hgb_src}")
    hgb_sha = file_sha256(hgb_src)

    rematch_copy = root / HGB22_FROM_REMATCH
    if rematch_copy.is_file():
        rematch_sha = file_sha256(rematch_copy)
        if rematch_sha != hgb_sha:
            raise FeatureExtractionError(
                f"HGB-22 SHA mismatch between 2B.3B ({hgb_sha}) and "
                f"2B.4D rematch copy ({rematch_sha})"
            )

    return {
        "status": "frozen",
        "strategy_version": GATE_2B5_VERSION,
        "gate_2b5_status": GATE_2B5_STATUS,
        "gate_2b5_decision": GATE_2B5_DECISION,
        "model_exploration_status": "closed",
        "feature_selection_status": "resolved",
        "final_feature_count": 22,
        "extractor_feature_count": 27,
        "parquet_feature_count": 27,
        "model_input": {
            "model_input_version": V1_MODEL_INPUT_VERSION,
            "feature_count": 22,
            "feature_names": list(V1_MODEL_INPUT_FEATURES),
            "excluded_from_model_input": list(DROPPED_TEMPORAL_FEATURES),
            "parent_feature_strategy_version": "phase1c2_v1",
            "parent_feature_count": len(V1_FEATURE_NAMES),
            "selection_policy": (
                "Exclude five temporal/span features from X only; keep them in "
                "extractor and Parquet for future experiments."
            ),
        },
        "v1_candidate": {
            "model_id": "hgb_22",
            "model_family": "HistGradientBoostingClassifier",
            "class_weights": "none",
            "training_view": FROZEN_SAMPLING_PLAN_ID,
            "feature_count": 22,
            "feature_names": list(V1_MODEL_INPUT_FEATURES),
            "hyperparameters": dict(HGB_PARAMS),
            "random_state": RANDOM_SEED,
            "source_phase": ABLATION_VERSION,
            "source_variant": "C_22_unweighted",
            "model_artifact": to_repo_relative(hgb_src, project_root=root),
            "model_artifact_sha256": hgb_sha,
        },
        "runner_up": {
            "model_id": "xgboost_22",
            "status": "runner_up_not_selected",
            "source_phase": FEATURE22_BOOST_VERSION,
        },
        "ranking": list(MODEL_EXPLORATION_RANKING),
        "threshold": {
            "status": "unfrozen",
            "note": (
                "Operating threshold not frozen by this gate. Use prior C / "
                "2B.3C refine points for provisional selection in a later step."
            ),
        },
        "test": {"access": False, "pcaps_read": 0},
        "parents": {
            "phase2b3b_ablation_status": ablation.get("status"),
            "phase2b4d_feature22_boost_status": rematch.get("status"),
        },
        "next": (
            "Freeze a provisional operating threshold for HGB-22 on "
            "TRAIN-validation only, then package the V1 candidate. Do not "
            "consult TEST until model + threshold + model-input are frozen."
        ),
    }


def write_frozen_model_input_contract(
    *,
    project_root: Path | None = None,
) -> Path:
    """Rewrite model-input contract as frozen (feature selection resolved)."""
    root = (project_root or PROJECT_ROOT).resolve()
    payload = build_v1_model_input_contract(project_root=root)
    payload["status"] = "frozen"
    payload["feature_selection_status"] = "resolved"
    payload["final_feature_count"] = 22
    payload["provisional_model_family"] = "HistGradientBoostingClassifier"
    payload["provisional_class_weight"] = None
    payload["notes"] = [
        "Does not invalidate Phase 1C or require feature-dataset rebuilds.",
        "Feature selection resolved in Phase 2B.5: model input is 22 nontemporal features.",
        "V1 candidate is unweighted HGB on group_balanced FIT.",
        "Threshold remains unfrozen; TEST remains sealed.",
    ]
    out = root / V1_MODEL_INPUT_CONTRACT_PATH
    _atomic_json(out, payload)
    return out


def freeze_v1_candidate(
    *,
    project_root: Path | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Freeze V1 candidate + model-input selection; close model exploration."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_path or DEFAULT_V1_CANDIDATE_FREEZE_PATH)
    if not out.is_absolute():
        out = root / out

    payload = build_v1_candidate_freeze_payload(project_root=root)
    model_input_path = write_frozen_model_input_contract(project_root=root)
    payload["artifacts"] = {
        "v1_candidate_freeze": to_repo_relative(out, project_root=root),
        "model_input_contract": to_repo_relative(model_input_path, project_root=root),
        "hgb22_model_artifact": payload["v1_candidate"]["model_artifact"],
    }
    _atomic_json(out, payload)
    return payload


def format_v1_candidate_freeze_summary(payload: dict[str, Any]) -> str:
    cand = payload.get("v1_candidate") or {}
    thr = payload.get("threshold") or {}
    mi = payload.get("model_input") or {}
    lines = [
        "Phase 2B.5 — V1 candidate freeze (model exploration closed)",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"model_exploration_status: {payload.get('model_exploration_status')}",
        f"feature_selection_status: {payload.get('feature_selection_status')}",
        f"extractor/parquet: {payload.get('extractor_feature_count')} features",
        f"model_input: {mi.get('feature_count')} features "
        f"({mi.get('model_input_version')})",
        f"v1_candidate: {cand.get('model_family')} / {cand.get('model_id')} / "
        f"weights={cand.get('class_weights')} / view={cand.get('training_view')}",
        f"model_sha: {str(cand.get('model_artifact_sha256') or '')[:16]}…",
        f"runner_up: {(payload.get('runner_up') or {}).get('model_id')}",
        f"threshold: {thr.get('status')}",
        f"test_access: {(payload.get('test') or {}).get('access')}",
    ]
    arts = payload.get("artifacts") or {}
    if arts.get("v1_candidate_freeze"):
        lines.append(f"freeze: {arts['v1_candidate_freeze']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
