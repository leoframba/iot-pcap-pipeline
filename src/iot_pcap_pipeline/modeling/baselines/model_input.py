"""Separately versioned V1 model-input selector (subset of the 27 feature schema).

Feature extraction and Parquet retain all 27 V1 columns (Phase 1C unchanged).
This contract selects which columns enter X for the provisional V1 HGB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.paths import DEFAULT_MODELING_DIR, PROJECT_ROOT, to_repo_relative

# Model-input version is independent of FEATURE_STRATEGY_VERSION (phase1c2_v1).
V1_MODEL_INPUT_VERSION = "v1_hgb22_nontemporal"
V1_MODEL_INPUT_CONTRACT_PATH = (
    DEFAULT_MODELING_DIR / "v1" / "model_input" / f"{V1_MODEL_INPUT_VERSION}.json"
)

DROPPED_TEMPORAL_FEATURES: tuple[str, ...] = (
    "window_span_seconds",
    "iat_mean_seconds",
    "iat_std_seconds",
    "iat_p50_seconds",
    "iat_p95_seconds",
)

V1_MODEL_INPUT_FEATURES: tuple[str, ...] = tuple(
    n for n in V1_FEATURE_NAMES if n not in DROPPED_TEMPORAL_FEATURES
)
assert len(V1_MODEL_INPUT_FEATURES) == 22
assert set(DROPPED_TEMPORAL_FEATURES).isdisjoint(V1_MODEL_INPUT_FEATURES)
assert len(V1_FEATURE_NAMES) - len(DROPPED_TEMPORAL_FEATURES) == 22

# Back-compat aliases used by 2B.3B ablations.
FEATURES_22 = V1_MODEL_INPUT_FEATURES


def build_v1_model_input_contract(*, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    return {
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "parent_feature_strategy_version": "phase1c2_v1",
        "parent_feature_count": len(V1_FEATURE_NAMES),
        "model_input_feature_count": len(V1_MODEL_INPUT_FEATURES),
        "selection_policy": (
            "Exclude the five temporal / span features from model input only; "
            "keep them in the extractor and Parquet schema for future experiments."
        ),
        "excluded_from_model_input": list(DROPPED_TEMPORAL_FEATURES),
        "feature_names": list(V1_MODEL_INPUT_FEATURES),
        "provisional_model_family": "HistGradientBoostingClassifier",
        "provisional_class_weight": None,
        "notes": [
            "Does not invalidate Phase 1C or require feature-dataset rebuilds.",
            "Threshold remains unfrozen until a focused C operating-point review.",
        ],
        "contract_path": to_repo_relative(
            V1_MODEL_INPUT_CONTRACT_PATH, project_root=root
        ),
    }


def write_v1_model_input_contract(
    *,
    path: Path | str | None = None,
    project_root: Path | None = None,
) -> Path:
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(path or V1_MODEL_INPUT_CONTRACT_PATH)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_v1_model_input_contract(project_root=root)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out
