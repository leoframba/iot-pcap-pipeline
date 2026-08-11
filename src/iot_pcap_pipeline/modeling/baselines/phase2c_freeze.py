"""Phase 2C close: freeze V1 model package (H0 + threshold).

Hyperparameter exploration and threshold tuning are closed. TEST stays sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.modeling.baselines.hgb_sensitivity import (
    DEFAULT_SENSITIVITY_ROOT,
    SENSITIVITY_VERSION,
)
from iot_pcap_pipeline.modeling.baselines.model_input import (
    DROPPED_TEMPORAL_FEATURES,
    V1_MODEL_INPUT_CONTRACT_PATH,
    V1_MODEL_INPUT_FEATURES,
    V1_MODEL_INPUT_VERSION,
)
from iot_pcap_pipeline.modeling.baselines.v1_candidate_freeze import (
    DEFAULT_V1_CANDIDATE_FREEZE_PATH,
)
from iot_pcap_pipeline.modeling.freeze import FROZEN_SAMPLING_PLAN_ID
from iot_pcap_pipeline.modeling.view import file_sha256
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

PHASE2C_VERSION = "phase2c_v1"
DEFAULT_PHASE2C_FREEZE_PATH = DEFAULT_MODELING_DIR / "v1" / "v1_model_package.json"

# Frozen operating threshold (ATTACK iff score >= threshold).
# Selected on TRAIN-validation for H0 at the ≤0.1% benign-FPR operating point
# after Phase 2C.1 (see final_validation_comparison.csv H0_baseline/secondary).
FROZEN_V1_THRESHOLD = 0.9490790963172913

FROZEN_H0_PARAMS: dict[str, Any] = {
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_features": 1.0,
    "early_stopping": False,
    "random_state": 42,
    "class_weight": None,
}

GATE_2C_DECISION = (
    "Close Phase 2C. Freeze HistGradientBoostingClassifier H0 on 22 nontemporal "
    "features (unweighted, group_balanced FIT) with decision threshold "
    f"{FROZEN_V1_THRESHOLD}. Hyperparameter exploration closed (2C.1 kept H0 after "
    "VAL failed to confirm H8). Threshold tuning closed. TEST remains sealed."
)

# Validation operating point pinned at freeze time (TRAIN-validation only).
FROZEN_VAL_OPERATING_POINT: dict[str, Any] = {
    "split": "TRAIN-validation",
    "rows": 4_944_060,
    "pcaps": 20,
    "fpr_target": 0.001,
    "operating_point_name": "secondary_leq_0.1pct_benign_fpr",
    "benign_fpr": 0.0008887308922858159,
    "benign_fpr_pct": 0.088873,
    "benign_fp": 20,
    "benign_support": 22_504,
    "ddos_recall": 0.9985615232495709,
    "dos_recall": 0.9973727119713864,
    "mqtt_recall": 0.9928806349886892,
    "recon_recall": 0.8689133016627079,
    "ddos_recall_pct": 99.8562,
    "dos_recall_pct": 99.7373,
    "mqtt_recall_pct": 99.2881,
    "recon_recall_pct": 86.8913,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _require_passed(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FeatureExtractionError(f"{label} missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if status not in ("passed", "frozen"):
        raise FeatureExtractionError(f"{label} not ready: status={status!r}")
    return payload


def build_phase2c_freeze_payload(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()

    cand = _require_passed(
        root / DEFAULT_V1_CANDIDATE_FREEZE_PATH,
        label="Phase 2B.5 v1_candidate_freeze",
    )
    sens = _require_passed(
        root / DEFAULT_SENSITIVITY_ROOT / "sensitivity_complete.json",
        label="Phase 2C.1 sensitivity_complete",
    )
    if sens.get("frozen_config_id") != "H0":
        raise FeatureExtractionError(
            f"Phase 2C.1 frozen_config_id is {sens.get('frozen_config_id')!r}, "
            "expected H0"
        )
    if sens.get("hyperparameter_exploration_status") != "closed":
        raise FeatureExtractionError(
            "Phase 2C.1 hyperparameter exploration is not closed"
        )

    model_path = root / DEFAULT_SENSITIVITY_ROOT / "models" / "H0_full_fit.joblib"
    if not model_path.is_file():
        # Fall back to 2B.5 pinned artifact (same H0 hyperparams).
        model_path = root / Path(cand["v1_candidate"]["model_artifact"])
    if not model_path.is_file():
        raise FeatureExtractionError(f"missing H0 model artifact: {model_path}")
    model_sha = file_sha256(model_path)

    val_compare = root / DEFAULT_SENSITIVITY_ROOT / "final_validation_comparison.csv"
    model_input = root / V1_MODEL_INPUT_CONTRACT_PATH

    return {
        "status": "frozen",
        "strategy_version": PHASE2C_VERSION,
        "gate_2c_status": "passed",
        "gate_2c_decision": GATE_2C_DECISION,
        "phase": "2C",
        "hyperparameter_exploration_status": "closed",
        "threshold_tuning_status": "closed",
        "model_family": "HistGradientBoostingClassifier",
        "feature_count": 22,
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "feature_names": list(V1_MODEL_INPUT_FEATURES),
        "excluded_from_model_input": list(DROPPED_TEMPORAL_FEATURES),
        "class_weight": None,
        "class_weighting": "none",
        "training_view": FROZEN_SAMPLING_PLAN_ID,
        "hyperparameter_config_id": "H0",
        "hyperparameters": dict(FROZEN_H0_PARAMS),
        "decision_rule": "ATTACK if score >= threshold",
        "threshold": {
            "status": "frozen",
            "value": FROZEN_V1_THRESHOLD,
            "selection_split": "TRAIN-validation",
            "selection_basis": (
                "H0 full-FIT model; highest-recall operating point with "
                "empirical benign FPR <= 0.1% (secondary / fpr_0.1pct)"
            ),
            "source_artifact": to_repo_relative(val_compare, project_root=root),
            "source_row": "H0_baseline / secondary",
        },
        "validation_operating_point": dict(FROZEN_VAL_OPERATING_POINT),
        "model_artifact": to_repo_relative(model_path, project_root=root),
        "model_artifact_sha256": model_sha,
        "parents": {
            "phase2b5_v1_candidate_freeze_status": cand.get("status"),
            "phase2c1_sensitivity_status": sens.get("status"),
            "phase2c1_frozen_config_id": sens.get("frozen_config_id"),
            "phase2c1_strategy_version": SENSITIVITY_VERSION,
        },
        "test": {"access": False, "pcaps_read": 0, "status": "sealed"},
        "scope_closed": [
            "model_family",
            "model_input_features",
            "hyperparameters",
            "class_weighting",
            "training_view",
            "decision_threshold",
        ],
        "next": (
            "Phase 2C closed. Package/export the V1 candidate for inference. "
            "Do not reopen hyperparameter or threshold search. "
            "TEST remains sealed until an explicit TEST evaluation gate."
        ),
        "artifacts": {
            "v1_model_package": to_repo_relative(
                root / DEFAULT_PHASE2C_FREEZE_PATH, project_root=root
            ),
            "v1_candidate_freeze": to_repo_relative(
                root / DEFAULT_V1_CANDIDATE_FREEZE_PATH, project_root=root
            ),
            "model_input_contract": to_repo_relative(model_input, project_root=root),
            "sensitivity_complete": to_repo_relative(
                root / DEFAULT_SENSITIVITY_ROOT / "sensitivity_complete.json",
                project_root=root,
            ),
            "final_validation_comparison": to_repo_relative(
                val_compare, project_root=root
            ),
        },
    }


def freeze_phase2c(
    *,
    project_root: Path | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write the frozen V1 model package and close Phase 2C."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_path or DEFAULT_PHASE2C_FREEZE_PATH)
    if not out.is_absolute():
        out = root / out

    payload = build_phase2c_freeze_payload(project_root=root)
    payload["artifacts"]["v1_model_package"] = to_repo_relative(out, project_root=root)
    _atomic_json(out, payload)
    return payload


def format_phase2c_freeze_summary(payload: dict[str, Any]) -> str:
    thr = payload.get("threshold") or {}
    val = payload.get("validation_operating_point") or {}
    lines = [
        "Phase 2C — V1 model package freeze",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"model_family: {payload.get('model_family')}",
        f"features: {payload.get('feature_count')} ({payload.get('model_input_version')})",
        f"hyperparameters: {payload.get('hyperparameter_config_id')} frozen",
        f"class_weighting: {payload.get('class_weighting')}",
        f"training_view: {payload.get('training_view')}",
        f"threshold: {thr.get('value')} ({thr.get('status')})",
        f"decision_rule: {payload.get('decision_rule')}",
        (
            f"val_op: benign FPR={val.get('benign_fpr_pct')}% "
            f"({val.get('benign_fp')}/{val.get('benign_support')}); "
            f"Recon={val.get('recon_recall_pct')}%"
        ),
        f"hyperparameter_exploration: {payload.get('hyperparameter_exploration_status')}",
        f"threshold_tuning: {payload.get('threshold_tuning_status')}",
        f"test: {(payload.get('test') or {}).get('status')}",
    ]
    arts = payload.get("artifacts") or {}
    if arts.get("v1_model_package"):
        lines.append(f"package: {arts['v1_model_package']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
