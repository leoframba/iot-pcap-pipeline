"""Phase 2D — one-shot sealed TEST evaluation (measurement only).

2D.0 freezes the pre-TEST contract without reading any TEST feature shards.
Later steps may score TEST exactly once under that contract; no decisions may
change based on TEST results.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO

import joblib
import numpy as np
import pyarrow.parquet as pq

from iot_pcap_pipeline.dataset.taxonomy import (
    classify_attack_stem,
    is_publisher_benign_stem,
)
from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    V1_FEATURE_NAMES,
)
from iot_pcap_pipeline.features.validate_dataset import DEFAULT_TEST_BUILD_COMPLETE_JSON
from iot_pcap_pipeline.features.dataset import (
    DEFAULT_TEST_BUILD_MANIFEST_PATH,
    EXPECTED_TEST_PCAP_COUNT,
)
from iot_pcap_pipeline.modeling.baselines.constants import (
    FORBIDDEN_MODEL_COLUMNS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.data import (
    assert_feature_columns,
    encode_labels,
)
from iot_pcap_pipeline.modeling.baselines.metrics import (
    ConfusionCounts,
    GroupAccumulator,
    RunningScoreStats,
    global_ranking_metrics,
    macro_mean,
    metrics_from_confusion,
)
from iot_pcap_pipeline.modeling.baselines.model_input import (
    V1_MODEL_INPUT_CONTRACT_PATH,
    V1_MODEL_INPUT_FEATURES,
    V1_MODEL_INPUT_VERSION,
)
from iot_pcap_pipeline.modeling.baselines.models import attack_score_from_estimator
from iot_pcap_pipeline.modeling.baselines.phase2c_freeze import (
    DEFAULT_PHASE2C_FREEZE_PATH,
    FROZEN_H0_PARAMS,
    FROZEN_V1_THRESHOLD,
    PHASE2C_VERSION,
)
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_TRAINING_VIEW_CONTRACT_PATH,
    file_sha256,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

PHASE2D_VERSION = "phase2d_v1"
DEFAULT_FINAL_TEST_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "final_test" / PHASE2D_VERSION
)
DEFAULT_FINAL_TEST_CONTRACT_PATH = DEFAULT_FINAL_TEST_ROOT / "final_test_contract.json"

EXPECTED_TEST_ROWS = 6_206_674
EXPECTED_TEST_ATTACK_PCAPS = 20
EXPECTED_TEST_BENIGN_PCAPS = 9

EXPECTED_MODEL_FAMILY = "HistGradientBoostingClassifier"
EXPECTED_MODEL_FEATURE_COUNT = 22
EXPECTED_EXTRACTOR_FEATURE_COUNT = 27

ATTACK_FAMILIES_REPORT_ORDER: tuple[str, ...] = (
    "DDoS",
    "DoS",
    "MQTT",
    "Recon",
    "Spoofing",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _require_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FeatureExtractionError(f"{label} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FeatureExtractionError(f"{label} missing: {path}")
    return path


def _assert_eq(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise FeatureExtractionError(
            f"Phase 2D integrity check failed: {label}={actual!r}, expected {expected!r}"
        )


def _inventory_from_test_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read TEST build manifest CSV only (no feature Parquet shards)."""
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise FeatureExtractionError(f"empty TEST build manifest: {manifest_path}")

    n_attack = sum(1 for r in rows if (r.get("binary_label") or "").upper() == "ATTACK")
    n_benign = sum(1 for r in rows if (r.get("binary_label") or "").upper() == "BENIGN")
    total_rows = 0
    for r in rows:
        raw = r.get("output_row_count") or "0"
        try:
            total_rows += int(raw)
        except ValueError as exc:
            raise FeatureExtractionError(
                f"invalid output_row_count in TEST manifest: {raw!r}"
            ) from exc
        status = (r.get("status") or "").strip().lower()
        if status != "ok":
            raise FeatureExtractionError(
                f"TEST manifest row not ok: pcap_id={r.get('pcap_id')!r} status={status!r}"
            )

    return {
        "pcap_count": len(rows),
        "attack_pcaps": n_attack,
        "benign_pcaps": n_benign,
        "total_feature_rows": total_rows,
        "feature_shards_opened": 0,
    }


def build_final_test_contract(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build Gate 2D.0 contract. Does not open TEST feature Parquet shards."""
    root = (project_root or PROJECT_ROOT).resolve()

    package_path = _require_file(
        root / DEFAULT_PHASE2C_FREEZE_PATH, label="Phase 2C freeze (v1_model_package)"
    )
    package = _require_json(package_path, label="Phase 2C freeze")
    _assert_eq(package.get("status"), "frozen", label="phase2c.status")
    _assert_eq(
        package.get("strategy_version"), PHASE2C_VERSION, label="phase2c.strategy_version"
    )
    _assert_eq(
        package.get("model_family"),
        EXPECTED_MODEL_FAMILY,
        label="phase2c.model_family",
    )
    _assert_eq(
        package.get("feature_count"),
        EXPECTED_MODEL_FEATURE_COUNT,
        label="phase2c.feature_count",
    )
    _assert_eq(
        package.get("hyperparameter_config_id"),
        "H0",
        label="phase2c.hyperparameter_config_id",
    )
    thr = package.get("threshold") or {}
    _assert_eq(thr.get("status"), "frozen", label="phase2c.threshold.status")
    _assert_eq(thr.get("value"), FROZEN_V1_THRESHOLD, label="phase2c.threshold.value")
    _assert_eq(
        package.get("hyperparameters"),
        FROZEN_H0_PARAMS,
        label="phase2c.hyperparameters",
    )
    _assert_eq(
        list(package.get("feature_names") or []),
        list(V1_MODEL_INPUT_FEATURES),
        label="phase2c.feature_names",
    )

    model_rel = package.get("model_artifact")
    if not model_rel:
        raise FeatureExtractionError("Phase 2C package missing model_artifact")
    model_path = _require_file(root / Path(model_rel), label="frozen HGB model artifact")
    model_sha = file_sha256(model_path)
    _assert_eq(
        model_sha,
        package.get("model_artifact_sha256"),
        label="model_artifact_sha256 vs phase2c pin",
    )

    model_input_path = _require_file(
        root / V1_MODEL_INPUT_CONTRACT_PATH, label="22-feature model-input contract"
    )
    model_input = _require_json(model_input_path, label="model-input contract")
    _assert_eq(
        model_input.get("model_input_version"),
        V1_MODEL_INPUT_VERSION,
        label="model_input_version",
    )
    _assert_eq(
        model_input.get("feature_selection_status"),
        "resolved",
        label="model_input.feature_selection_status",
    )
    _assert_eq(
        int(model_input.get("final_feature_count") or model_input.get("feature_count") or 0),
        EXPECTED_MODEL_FEATURE_COUNT,
        label="model_input.feature_count",
    )

    schema_path = _require_file(
        root / DEFAULT_FEATURE_SCHEMA_PATH, label="27-feature schema"
    )
    schema_sha = feature_schema_sha256(schema_path)

    train_contract_path = _require_file(
        root / DEFAULT_TRAINING_VIEW_CONTRACT_PATH, label="training-view contract"
    )
    split_path = _require_file(
        root / DEFAULT_SPLIT_MANIFEST_PATH, label="modeling split manifest"
    )

    test_complete_path = _require_file(
        root / DEFAULT_TEST_BUILD_COMPLETE_JSON, label="TEST validation marker"
    )
    test_complete = _require_json(test_complete_path, label="TEST validation marker")
    _assert_eq(
        test_complete.get("validation_status"),
        "passed",
        label="test_build_complete.validation_status",
    )
    _assert_eq(
        int(test_complete.get("pcap_count") or 0),
        EXPECTED_TEST_PCAP_COUNT,
        label="test_build_complete.pcap_count",
    )
    _assert_eq(
        int(test_complete.get("total_feature_rows") or 0),
        EXPECTED_TEST_ROWS,
        label="test_build_complete.total_feature_rows",
    )
    _assert_eq(
        test_complete.get("feature_schema_sha256"),
        schema_sha,
        label="test_build_complete.feature_schema_sha256",
    )

    test_manifest_path = _require_file(
        root / DEFAULT_TEST_BUILD_MANIFEST_PATH, label="TEST structural build marker"
    )
    inventory = _inventory_from_test_manifest(test_manifest_path)
    _assert_eq(
        inventory["pcap_count"],
        EXPECTED_TEST_PCAP_COUNT,
        label="test_build_manifest.pcap_count",
    )
    _assert_eq(
        inventory["total_feature_rows"],
        EXPECTED_TEST_ROWS,
        label="test_build_manifest.total_feature_rows",
    )
    _assert_eq(
        inventory["attack_pcaps"],
        EXPECTED_TEST_ATTACK_PCAPS,
        label="test_build_manifest.attack_pcaps",
    )
    _assert_eq(
        inventory["benign_pcaps"],
        EXPECTED_TEST_BENIGN_PCAPS,
        label="test_build_manifest.benign_pcaps",
    )
    _assert_eq(
        inventory["total_feature_rows"],
        int(test_complete.get("total_feature_rows") or 0),
        label="manifest vs complete row count",
    )

    pins = {
        "model_artifact_sha256": model_sha,
        "phase2c_freeze_sha256": file_sha256(package_path),
        "model_input_contract_sha256": file_sha256(model_input_path),
        "feature_schema_sha256": schema_sha,
        "training_view_contract_sha256": file_sha256(train_contract_path),
        "modeling_split_manifest_sha256": file_sha256(split_path),
        "test_build_manifest_sha256": file_sha256(test_manifest_path),
        "test_build_complete_sha256": file_sha256(test_complete_path),
    }

    return {
        "status": "prepared",
        "gate_2d0_status": "passed",
        "strategy_version": PHASE2D_VERSION,
        "evaluation_mode": "one_shot_sealed_test",
        "model_family": EXPECTED_MODEL_FAMILY,
        "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
        "extractor_feature_count": EXPECTED_EXTRACTOR_FEATURE_COUNT,
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "feature_names": list(V1_MODEL_INPUT_FEATURES),
        "hyperparameter_config_id": "H0",
        "hyperparameters": dict(FROZEN_H0_PARAMS),
        "threshold": FROZEN_V1_THRESHOLD,
        "decision_rule": "score >= threshold",
        "model_family_status": "frozen",
        "feature_selection_status": "frozen",
        "hyperparameter_status": "frozen",
        "threshold_status": "frozen",
        "retraining_allowed": False,
        "threshold_sweep_allowed": False,
        "alternate_models_allowed": False,
        "test_sampling_allowed": False,
        "expected_test_pcaps": EXPECTED_TEST_PCAP_COUNT,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "expected_test_attack_pcaps": EXPECTED_TEST_ATTACK_PCAPS,
        "expected_test_benign_pcaps": EXPECTED_TEST_BENIGN_PCAPS,
        "test_inventory_check": {
            **inventory,
            "source": "test_build_manifest.csv + test_build_complete.json",
            "feature_parquet_rows_read": 0,
        },
        "model_artifact": to_repo_relative(model_path, project_root=root),
        "pins": pins,
        "artifacts": {
            "phase2c_freeze": to_repo_relative(package_path, project_root=root),
            "model_artifact": to_repo_relative(model_path, project_root=root),
            "model_input_contract": to_repo_relative(
                model_input_path, project_root=root
            ),
            "feature_schema": to_repo_relative(schema_path, project_root=root),
            "training_view_contract": to_repo_relative(
                train_contract_path, project_root=root
            ),
            "modeling_split_manifest": to_repo_relative(split_path, project_root=root),
            "test_build_manifest": to_repo_relative(
                test_manifest_path, project_root=root
            ),
            "test_build_complete": to_repo_relative(
                test_complete_path, project_root=root
            ),
        },
        "test_feature_access": {
            "parquet_shards_opened": False,
            "rows_read": 0,
            "note": (
                "Gate 2D.0 pins markers and hashes only; TEST feature shards "
                "remain unread until the one-shot evaluation command."
            ),
        },
        "scope_limits": [
            "Measurement only: no model/feature/hyperparameter/sampling/threshold "
            "decisions may change based on TEST.",
            "Exactly one sealed TEST evaluation after this contract is frozen.",
            "No retraining, threshold sweep, alternate models, or TEST sampling.",
        ],
        "next": (
            "Review final_test_contract.json. When ready, run the one-shot "
            "sealed TEST evaluation command (Gate 2D.1+). Do not alter the "
            "frozen candidate based on TEST."
        ),
    }


def prepare_final_test(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Freeze final_test_contract.json (Gate 2D.0). No TEST feature rows read."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_FINAL_TEST_ROOT)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    # Clear any prior evaluation / preflight markers so a re-prepare is clean.
    (out / "final_test_complete.json").unlink(missing_ok=True)
    (out / "preflight_complete.json").unlink(missing_ok=True)

    contract = build_final_test_contract(project_root=root)
    contract_path = out / "final_test_contract.json"
    contract["artifacts"]["final_test_contract"] = to_repo_relative(
        contract_path, project_root=root
    )
    _atomic_json(contract_path, contract)

    return {
        "status": "prepared",
        "gate_2d0_status": contract["gate_2d0_status"],
        "strategy_version": PHASE2D_VERSION,
        "contract_path": to_repo_relative(contract_path, project_root=root),
        "expected_test_pcaps": EXPECTED_TEST_PCAP_COUNT,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "feature_parquet_rows_read": 0,
        "next": contract["next"],
    }


def format_prepare_final_test_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2D.0 — Pre-TEST contract freeze",
        f"status: {payload.get('status')}",
        f"gate_2d0_status: {payload.get('gate_2d0_status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"expected_test_pcaps: {payload.get('expected_test_pcaps')}",
        f"expected_test_rows: {payload.get('expected_test_rows')}",
        f"feature_parquet_rows_read: {payload.get('feature_parquet_rows_read')}",
    ]
    if payload.get("contract_path"):
        lines.append(f"contract: {payload['contract_path']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"


def preflight_final_test(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    **forbidden_overrides: Any,
) -> dict[str, Any]:
    """Gate 2D.2 — verify contracts + TEST inventory; no predictions/metrics.

    Final stopping point before one-shot sealed TEST evaluation.
    """
    reject_final_test_overrides(**forbidden_overrides)
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_FINAL_TEST_ROOT)
    if not out.is_absolute():
        out = root / out

    contract_path = out / "final_test_contract.json"
    if not contract_path.is_file():
        raise FeatureExtractionError(
            f"final_test_contract.json missing at {contract_path}. "
            "Run prepare-final-test first."
        )

    contract = load_final_test_contract(contract_path, project_root=root)
    live_pins = verify_contract_integrity(contract, project_root=root)

    arts = contract.get("artifacts") or {}
    specs = load_test_pcap_specs(
        arts.get("test_build_manifest") or DEFAULT_TEST_BUILD_MANIFEST_PATH,
        project_root=root,
    )
    split_ids = load_modeling_split_pcap_ids(
        arts.get("modeling_split_manifest") or DEFAULT_SPLIT_MANIFEST_PATH,
        project_root=root,
    )
    assert_test_inventory_integrity(
        specs,
        expected_pcaps=EXPECTED_TEST_PCAP_COUNT,
        expected_rows=EXPECTED_TEST_ROWS,
        modeling_split_pcap_ids=split_ids,
    )

    # Confirm pinned model file is loadable without scoring any TEST rows.
    _ = load_frozen_hgb_estimator(contract, project_root=root)

    # Re-prepare clears eval markers; preflight also refuses a stale complete.
    (out / "final_test_complete.json").unlink(missing_ok=True)

    payload = {
        "status": "passed",
        "gate_2d2_status": "passed",
        "strategy_version": PHASE2D_VERSION,
        "evaluation_mode": "one_shot_sealed_test",
        "model_hash_verified": True,
        "model_input_hash_verified": True,
        "feature_schema_hash_verified": True,
        "phase2c_freeze_verified": True,
        "training_view_contract_verified": True,
        "modeling_split_manifest_verified": True,
        "test_build_manifest_verified": True,
        "test_build_complete_verified": True,
        "threshold_verified": True,
        "threshold": FROZEN_V1_THRESHOLD,
        "hyperparameter_config_id": "H0",
        "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
        "extractor_feature_count": EXPECTED_EXTRACTOR_FEATURE_COUNT,
        "expected_test_pcaps": EXPECTED_TEST_PCAP_COUNT,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "expected_test_attack_pcaps": EXPECTED_TEST_ATTACK_PCAPS,
        "expected_test_benign_pcaps": EXPECTED_TEST_BENIGN_PCAPS,
        "inventory_pcap_count": len(specs),
        "inventory_row_count": sum(s.expected_row_count for s in specs),
        "train_fit_validation_overlap_pcaps": 0,
        "predictions_generated": False,
        "metrics_generated": False,
        "feature_parquet_rows_read": 0,
        "test_feature_shards_opened": False,
        "ready_for_one_shot_test": True,
        "verified_pins": live_pins,
        "artifacts": {
            "final_test_contract": to_repo_relative(contract_path, project_root=root),
            "preflight_complete": to_repo_relative(
                out / "preflight_complete.json", project_root=root
            ),
        },
        "next": (
            "Preflight passed. Open one-shot sealed TEST evaluation with "
            "`iot-pcap-pipeline run-final-test` when ready. Do not alter the "
            "frozen candidate based on TEST."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "preflight_complete.json", payload)
    return payload


def format_preflight_final_test_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2D.2 — Preflight / dry run",
        f"status: {payload.get('status')}",
        f"gate_2d2_status: {payload.get('gate_2d2_status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"model_hash_verified: {payload.get('model_hash_verified')}",
        f"model_input_hash_verified: {payload.get('model_input_hash_verified')}",
        f"feature_schema_hash_verified: {payload.get('feature_schema_hash_verified')}",
        f"phase2c_freeze_verified: {payload.get('phase2c_freeze_verified')}",
        f"expected_test_pcaps: {payload.get('expected_test_pcaps')}",
        f"expected_test_rows: {payload.get('expected_test_rows')}",
        f"predictions_generated: {payload.get('predictions_generated')}",
        f"metrics_generated: {payload.get('metrics_generated')}",
        f"ready_for_one_shot_test: {payload.get('ready_for_one_shot_test')}",
    ]
    arts = payload.get("artifacts") or {}
    if arts.get("preflight_complete"):
        lines.append(f"preflight: {arts['preflight_complete']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Gate 2D.1 — evaluator + integrity (measurement only; no decision changes)
# ---------------------------------------------------------------------------

FEATURES_27 = list(V1_FEATURE_NAMES)
FEATURES_22 = list(V1_MODEL_INPUT_FEATURES)
assert len(FEATURES_27) == EXPECTED_EXTRACTOR_FEATURE_COUNT
assert len(FEATURES_22) == EXPECTED_MODEL_FEATURE_COUNT
assert set(FEATURES_22).issubset(set(FEATURES_27))

_MODEL_FEATURE_INDICES: tuple[int, ...] = tuple(
    FEATURES_27.index(name) for name in FEATURES_22
)

# Reject accidental post-hoc comparison knobs on the sealed TEST entrypoint.
FORBIDDEN_FINAL_TEST_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "model_path",
        "model_artifact",
        "threshold",
        "features",
        "feature_names",
        "feature_selector",
        "hyperparameters",
        "hyperparameter",
        "params",
        "sampling",
        "sample",
        "sample_fraction",
        "class_weight",
        "estimator",
    }
)

DEFAULT_TEST_BATCH_ROWS = 65_536


@dataclass(frozen=True)
class TestPcapSpec:
    pcap_id: str
    binary_label: str
    feature_parquet_path: str
    expected_row_count: int
    pcap_path: str = ""
    attack_family: str = ""
    attack_type: str = ""
    benign_category: str = ""


@dataclass(frozen=True)
class TestScoreBatch:
    spec: TestPcapSpec
    y_true: np.ndarray
    scores: np.ndarray
    y_pred: np.ndarray
    n_rows: int


def derive_test_pcap_metadata(
    *,
    pcap_path: str,
    binary_label: str,
) -> dict[str, str]:
    """Derive attack/benign grouping metadata from the TEST PCAP path."""
    path = Path(pcap_path)
    stem = path.stem
    label = binary_label.strip().upper()
    if label == "ATTACK":
        tax = classify_attack_stem(stem)
        if tax is None:
            raise FeatureExtractionError(
                f"unrecognized ATTACK stem for TEST metadata: {stem!r} ({pcap_path})"
            )
        return {
            "attack_family": tax.family,
            "attack_type": tax.attack_type,
            "benign_category": "",
        }
    if label != "BENIGN":
        raise FeatureExtractionError(f"unknown binary_label for metadata: {label!r}")

    parts_lower = [p.lower() for p in path.parts]
    if is_publisher_benign_stem(stem) or stem.lower().startswith("benign"):
        return {
            "attack_family": "",
            "attack_type": "",
            "benign_category": "publisher",
        }
    if "power" in parts_lower:
        return {
            "attack_family": "",
            "attack_type": "",
            "benign_category": "profiling_power",
        }
    if "idle" in parts_lower:
        return {
            "attack_family": "",
            "attack_type": "",
            "benign_category": "profiling_idle",
        }
    if "interactions" in parts_lower or "interaction" in parts_lower:
        return {
            "attack_family": "",
            "attack_type": "",
            "benign_category": "profiling_interaction",
        }
    raise FeatureExtractionError(
        f"unable to derive benign_category for TEST PCAP: {pcap_path}"
    )


def reject_final_test_overrides(**kwargs: Any) -> None:
    """Refuse any attempt to override frozen model/threshold/features/sampling."""
    if not kwargs:
        return
    forbidden = sorted(FORBIDDEN_FINAL_TEST_OVERRIDE_KEYS & set(kwargs))
    extra = sorted(set(kwargs) - FORBIDDEN_FINAL_TEST_OVERRIDE_KEYS)
    parts: list[str] = []
    if forbidden:
        parts.append(f"forbidden overrides: {forbidden}")
    if extra:
        parts.append(f"unsupported arguments: {extra}")
    raise FeatureExtractionError(
        "final TEST evaluation rejects overrides ("
        + "; ".join(parts)
        + "). Use the frozen Phase 2D contract only."
    )


def model_feature_column_indices() -> tuple[int, ...]:
    """Indices of the exact ordered 22 model features within the 27 V1 schema."""
    return _MODEL_FEATURE_INDICES


def select_model_features_x22(X27: np.ndarray) -> np.ndarray:
    """Select exact ordered 22 model features from a verified 27-feature matrix."""
    if X27.ndim != 2:
        raise FeatureExtractionError(f"expected 2D feature matrix, got ndim={X27.ndim}")
    if X27.shape[1] != EXPECTED_EXTRACTOR_FEATURE_COUNT:
        raise FeatureExtractionError(
            f"extractor feature count {X27.shape[1]} != {EXPECTED_EXTRACTOR_FEATURE_COUNT}"
        )
    X22 = np.ascontiguousarray(X27[:, list(_MODEL_FEATURE_INDICES)], dtype=np.float32)
    if X22.shape[1] != EXPECTED_MODEL_FEATURE_COUNT:
        raise FeatureExtractionError(
            f"model feature count {X22.shape[1]} != {EXPECTED_MODEL_FEATURE_COUNT}"
        )
    if not np.isfinite(X22).all():
        raise FeatureExtractionError("non-finite values in model input X (22 features)")
    return X22


def decide_attack(scores: np.ndarray, *, threshold: float) -> np.ndarray:
    """ATTACK iff score >= threshold (uint8 1/0)."""
    if threshold != FROZEN_V1_THRESHOLD:
        raise FeatureExtractionError(
            f"threshold {threshold!r} != frozen {FROZEN_V1_THRESHOLD!r}"
        )
    return (np.asarray(scores) >= threshold).astype(np.uint8, copy=False)


def load_final_test_contract(
    path: Path | str | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    p = Path(path or DEFAULT_FINAL_TEST_CONTRACT_PATH)
    if not p.is_absolute():
        p = root / p
    payload = _require_json(p, label="final_test_contract")
    if payload.get("strategy_version") != PHASE2D_VERSION:
        raise FeatureExtractionError(
            f"final_test_contract strategy_version={payload.get('strategy_version')!r}, "
            f"expected {PHASE2D_VERSION!r}"
        )
    if payload.get("gate_2d0_status") != "passed":
        raise FeatureExtractionError(
            f"Gate 2D.0 not passed: {payload.get('gate_2d0_status')!r}"
        )
    return payload


def verify_contract_integrity(
    contract: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, str]:
    """Recompute pinned hashes and assert frozen scalars still match."""
    root = (project_root or PROJECT_ROOT).resolve()
    pins = contract.get("pins") or {}
    arts = contract.get("artifacts") or {}

    _assert_eq(
        contract.get("threshold"),
        FROZEN_V1_THRESHOLD,
        label="contract.threshold",
    )
    _assert_eq(
        contract.get("hyperparameters"),
        FROZEN_H0_PARAMS,
        label="contract.hyperparameters",
    )
    _assert_eq(
        list(contract.get("feature_names") or []),
        FEATURES_22,
        label="contract.feature_names",
    )
    _assert_eq(
        int(contract.get("model_feature_count") or 0),
        EXPECTED_MODEL_FEATURE_COUNT,
        label="contract.model_feature_count",
    )
    _assert_eq(
        int(contract.get("expected_test_pcaps") or 0),
        EXPECTED_TEST_PCAP_COUNT,
        label="contract.expected_test_pcaps",
    )
    _assert_eq(
        int(contract.get("expected_test_rows") or 0),
        EXPECTED_TEST_ROWS,
        label="contract.expected_test_rows",
    )

    checks = {
        "model_artifact": (arts.get("model_artifact"), "model_artifact_sha256"),
        "phase2c_freeze": (arts.get("phase2c_freeze"), "phase2c_freeze_sha256"),
        "model_input_contract": (
            arts.get("model_input_contract"),
            "model_input_contract_sha256",
        ),
        "training_view_contract": (
            arts.get("training_view_contract"),
            "training_view_contract_sha256",
        ),
        "modeling_split_manifest": (
            arts.get("modeling_split_manifest"),
            "modeling_split_manifest_sha256",
        ),
        "test_build_manifest": (
            arts.get("test_build_manifest"),
            "test_build_manifest_sha256",
        ),
        "test_build_complete": (
            arts.get("test_build_complete"),
            "test_build_complete_sha256",
        ),
    }
    live: dict[str, str] = {}
    for label, (rel, pin_key) in checks.items():
        if not rel:
            raise FeatureExtractionError(f"contract missing artifact: {label}")
        path = _require_file(root / Path(rel), label=label)
        digest = file_sha256(path)
        expected = pins.get(pin_key)
        _assert_eq(digest, expected, label=pin_key)
        live[pin_key] = digest

    schema_path = _require_file(
        root / Path(arts.get("feature_schema") or DEFAULT_FEATURE_SCHEMA_PATH),
        label="feature_schema",
    )
    schema_sha = feature_schema_sha256(schema_path)
    _assert_eq(
        schema_sha,
        pins.get("feature_schema_sha256"),
        label="feature_schema_sha256",
    )
    live["feature_schema_sha256"] = schema_sha
    return live


def load_frozen_hgb_estimator(
    contract: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> Any:
    """Load the pinned HGB artifact; refuse hash drift."""
    root = (project_root or PROJECT_ROOT).resolve()
    rel = contract.get("model_artifact") or (contract.get("artifacts") or {}).get(
        "model_artifact"
    )
    if not rel:
        raise FeatureExtractionError("contract missing model_artifact")
    path = _require_file(root / Path(rel), label="frozen HGB model artifact")
    digest = file_sha256(path)
    expected = (contract.get("pins") or {}).get("model_artifact_sha256")
    _assert_eq(digest, expected, label="model_artifact_sha256 at load")
    return joblib.load(path)


def load_test_pcap_specs(
    manifest_path: Path | str,
    *,
    project_root: Path | None = None,
) -> list[TestPcapSpec]:
    """Load TEST inventory from the structural build manifest (no Parquet reads)."""
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    path = _require_file(path, label="TEST build manifest")

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise FeatureExtractionError(f"empty TEST build manifest: {path}")

    specs: list[TestPcapSpec] = []
    seen: set[str] = set()
    for row in rows:
        pcap_id = str(row.get("pcap_id") or "").strip()
        if not pcap_id:
            raise FeatureExtractionError("TEST manifest row missing pcap_id")
        if pcap_id in seen:
            raise FeatureExtractionError(
                f"TEST PCAP appears more than once in inventory: {pcap_id}"
            )
        seen.add(pcap_id)
        label = str(row.get("binary_label") or "").strip().upper()
        if label not in LABEL_MAPPING:
            raise FeatureExtractionError(
                f"TEST manifest unknown binary_label={label!r} for {pcap_id}"
            )
        out_rel = str(row.get("output_path") or "").strip()
        if not out_rel:
            raise FeatureExtractionError(f"TEST manifest missing output_path for {pcap_id}")
        try:
            n_rows = int(row.get("output_row_count") or "0")
        except ValueError as exc:
            raise FeatureExtractionError(
                f"invalid output_row_count for {pcap_id}"
            ) from exc
        status = str(row.get("status") or "").strip().lower()
        if status != "ok":
            raise FeatureExtractionError(
                f"TEST manifest status not ok for {pcap_id}: {status!r}"
            )
        pcap_path = str(row.get("pcap_path") or "").strip()
        meta = derive_test_pcap_metadata(pcap_path=pcap_path, binary_label=label)
        specs.append(
            TestPcapSpec(
                pcap_id=pcap_id,
                binary_label=label,
                feature_parquet_path=out_rel,
                expected_row_count=n_rows,
                pcap_path=pcap_path,
                attack_family=meta["attack_family"],
                attack_type=meta["attack_type"],
                benign_category=meta["benign_category"],
            )
        )
    return specs


def load_modeling_split_pcap_ids(
    split_manifest_path: Path | str,
    *,
    project_root: Path | None = None,
) -> set[str]:
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(split_manifest_path)
    if not path.is_absolute():
        path = root / path
    path = _require_file(path, label="modeling split manifest")
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = str(row.get("pcap_id") or "").strip()
            if pid:
                ids.add(pid)
    if not ids:
        raise FeatureExtractionError(f"no pcap_ids in modeling split manifest: {path}")
    return ids


def assert_test_inventory_integrity(
    specs: list[TestPcapSpec],
    *,
    expected_pcaps: int,
    expected_rows: int,
    modeling_split_pcap_ids: set[str],
) -> None:
    """Hard inventory gates before any scoring."""
    if len(specs) != expected_pcaps:
        raise FeatureExtractionError(
            f"TEST PCAP count {len(specs)} != {expected_pcaps}"
        )
    pcap_ids = [s.pcap_id for s in specs]
    if len(set(pcap_ids)) != len(pcap_ids):
        raise FeatureExtractionError("duplicate TEST pcap_id in inventory")
    total_rows = sum(s.expected_row_count for s in specs)
    if total_rows != expected_rows:
        raise FeatureExtractionError(
            f"TEST row count {total_rows} != {expected_rows}"
        )
    overlap = sorted(set(pcap_ids) & modeling_split_pcap_ids)
    if overlap:
        raise FeatureExtractionError(
            "TRAIN/FIT/validation PCAP IDs found in TEST inventory: "
            f"{overlap[:5]}{'…' if len(overlap) > 5 else ''}"
        )


def assert_parquet_v1_schema(pf: pq.ParquetFile, *, path: Path) -> None:
    """Verify shard carries the frozen 27-feature schema (plus identity cols)."""
    names = list(pf.schema.names)
    missing = [n for n in FEATURES_27 if n not in names]
    if missing:
        raise FeatureExtractionError(
            f"TEST shard missing V1 features ({path}): {missing[:5]}"
        )
    leaked = sorted(FORBIDDEN_MODEL_COLUMNS & set(FEATURES_27))
    if leaked:
        # Defensive: V1 feature names must never intersect metadata forbids.
        raise FeatureExtractionError(
            f"internal schema error: V1 features overlap forbidden metadata: {leaked}"
        )


def iter_test_score_batches(
    specs: list[TestPcapSpec],
    estimator: Any,
    *,
    project_root: Path | None = None,
    threshold: float = FROZEN_V1_THRESHOLD,
    batch_rows: int = DEFAULT_TEST_BATCH_ROWS,
) -> Iterator[TestScoreBatch]:
    """Score each TEST shard once under the frozen decision path.

    Path: Parquet → verify 27 schema → select 22 → predict_proba → score ≥ thr.
    """
    if threshold != FROZEN_V1_THRESHOLD:
        raise FeatureExtractionError(
            f"threshold override rejected: {threshold!r} != {FROZEN_V1_THRESHOLD!r}"
        )
    root = (project_root or PROJECT_ROOT).resolve()
    read_cols = FEATURES_27 + ["binary_label"]

    for spec in specs:
        path = Path(spec.feature_parquet_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FeatureExtractionError(f"TEST feature shard missing: {path}")
        pf = pq.ParquetFile(path)
        assert_parquet_v1_schema(pf, path=path)

        scored = 0
        for batch in pf.iter_batches(batch_size=batch_rows, columns=read_cols):
            feat_cols = [c for c in batch.schema.names if c != "binary_label"]
            assert_feature_columns(feat_cols, expected=FEATURES_27)
            leaked = sorted(FORBIDDEN_MODEL_COLUMNS & set(feat_cols))
            if leaked:
                raise FeatureExtractionError(
                    f"metadata columns entered feature batch for {spec.pcap_id}: {leaked}"
                )
            arrays = [
                batch.column(name).to_numpy(zero_copy_only=False) for name in FEATURES_27
            ]
            X27 = np.column_stack(arrays).astype(np.float32, copy=False)
            if not np.isfinite(X27).all():
                raise FeatureExtractionError(
                    f"non-finite extractor features in TEST shard {path}"
                )
            X22 = select_model_features_x22(X27)
            if X22.shape[1] != EXPECTED_MODEL_FEATURE_COUNT:
                raise FeatureExtractionError(
                    f"model X width {X22.shape[1]} != 22 for {spec.pcap_id}"
                )
            # Ensure metadata never entered model X (columns are pure floats).
            y_true = encode_labels(batch.column("binary_label").to_pylist())
            scores = attack_score_from_estimator(estimator, X22)
            if scores.shape[0] != X22.shape[0]:
                raise FeatureExtractionError(
                    f"score length mismatch for {spec.pcap_id}: "
                    f"{scores.shape[0]} vs {X22.shape[0]}"
                )
            y_pred = decide_attack(scores, threshold=threshold)
            n = int(X22.shape[0])
            scored += n
            yield TestScoreBatch(
                spec=spec,
                y_true=y_true,
                scores=np.asarray(scores, dtype=np.float32),
                y_pred=y_pred,
                n_rows=n,
            )

        if scored != spec.expected_row_count:
            raise FeatureExtractionError(
                f"TEST rows scored for {spec.pcap_id}: {scored} != "
                f"manifest expected {spec.expected_row_count}"
            )


def evaluate_sealed_test_inventory(
    specs: list[TestPcapSpec],
    estimator: Any,
    *,
    project_root: Path | None = None,
    threshold: float = FROZEN_V1_THRESHOLD,
    expected_pcaps: int,
    expected_rows: int,
    modeling_split_pcap_ids: set[str],
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """Run the sealed scoring path over an inventory (production or synthetic).

    Accumulates everything needed for global / family / benign / per-PCAP
    reporting in a single pass. Scoring path is unchanged.
    """
    assert_test_inventory_integrity(
        specs,
        expected_pcaps=expected_pcaps,
        expected_rows=expected_rows,
        modeling_split_pcap_ids=modeling_split_pcap_ids,
    )
    if threshold != FROZEN_V1_THRESHOLD:
        raise FeatureExtractionError(
            f"threshold override rejected: {threshold!r} != {FROZEN_V1_THRESHOLD!r}"
        )

    global_counts = ConfusionCounts()
    global_scores = RunningScoreStats()
    y_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    family_acc: dict[str, GroupAccumulator] = {}
    benign_acc: dict[str, GroupAccumulator] = {}
    pcap_acc: dict[str, GroupAccumulator] = {}

    rows_scored = 0
    for batch in iter_test_score_batches(
        specs,
        estimator,
        project_root=project_root,
        threshold=threshold,
    ):
        spec = batch.spec
        y_t = batch.y_true
        y_p = batch.y_pred
        scores = batch.scores

        global_counts.update(y_t, y_p)
        global_scores.update(scores)
        y_chunks.append(np.asarray(y_t, dtype=np.uint8))
        score_chunks.append(np.asarray(scores, dtype=np.float32))

        pid = spec.pcap_id
        if pid not in pcap_acc:
            pcap_acc[pid] = GroupAccumulator(
                key=pid, kind="pcap", binary_label=spec.binary_label
            )
        pcap_acc[pid].update(
            pcap_id=pid, y_true=y_t, y_pred=y_p, scores=scores
        )

        if spec.binary_label == "ATTACK":
            fam = spec.attack_family
            if not fam:
                raise FeatureExtractionError(f"ATTACK PCAP missing family: {pid}")
            if fam not in family_acc:
                family_acc[fam] = GroupAccumulator(
                    key=fam, kind="attack_group", binary_label="ATTACK"
                )
            family_acc[fam].update(
                pcap_id=pid, y_true=y_t, y_pred=y_p, scores=scores
            )
        else:
            cat = spec.benign_category
            if not cat:
                raise FeatureExtractionError(
                    f"BENIGN PCAP missing benign_category: {pid}"
                )
            # Fine-grained category (publisher / profiling_*).
            if cat not in benign_acc:
                benign_acc[cat] = GroupAccumulator(
                    key=cat, kind="benign_group", binary_label="BENIGN"
                )
            benign_acc[cat].update(
                pcap_id=pid, y_true=y_t, y_pred=y_p, scores=scores
            )
            # Rollups.
            if cat == "publisher":
                rollup = "publisher_benign"
            else:
                rollup = "profiling_benign"
            if rollup not in benign_acc:
                benign_acc[rollup] = GroupAccumulator(
                    key=rollup, kind="benign_group", binary_label="BENIGN"
                )
            benign_acc[rollup].update(
                pcap_id=pid, y_true=y_t, y_pred=y_p, scores=scores
            )

        rows_scored += batch.n_rows
        if progress_file is not None and rows_scored % 500_000 < batch.n_rows:
            progress_file.write(f"  scored {rows_scored} TEST rows\n")
            progress_file.flush()

    if rows_scored != expected_rows:
        raise FeatureExtractionError(
            f"TEST rows scored {rows_scored} != expected {expected_rows}"
        )
    if len(pcap_acc) != expected_pcaps:
        raise FeatureExtractionError(
            f"TEST PCAPs scored {len(pcap_acc)} != expected {expected_pcaps}"
        )

    y_true_all = (
        np.concatenate(y_chunks) if y_chunks else np.empty(0, dtype=np.uint8)
    )
    scores_all = (
        np.concatenate(score_chunks) if score_chunks else np.empty(0, dtype=np.float32)
    )
    ranking = global_ranking_metrics(y_true_all, scores_all)
    conf_metrics = metrics_from_confusion(global_counts, threshold=threshold)
    score_summary = global_scores.summary()

    # Attack family rows in canonical order.
    attack_family_rows: list[dict[str, Any]] = []
    family_recalls: list[float | None] = []
    for fam in ATTACK_FAMILIES_REPORT_ORDER:
        acc = family_acc.get(fam)
        if acc is None:
            attack_family_rows.append(
                {
                    "attack_family": fam,
                    "pcap_count": 0,
                    "rows": 0,
                    "tp": 0,
                    "fn": 0,
                    "recall": None,
                    "score_mean": None,
                    "score_p05": None,
                    "score_p50": None,
                    "score_p95": None,
                    "present": False,
                }
            )
            continue
        base = acc.to_attack_row()
        attack_family_rows.append(
            {
                "attack_family": fam,
                "pcap_count": base["pcap_count"],
                "rows": base["row_count"],
                "tp": base["tp"],
                "fn": base["fn"],
                "recall": base["recall"],
                "score_mean": base["attack_score_mean"],
                "score_p05": base["attack_score_p05"],
                "score_p50": base["attack_score_p50"],
                "score_p95": base["attack_score_p95"],
                "present": True,
            }
        )
        family_recalls.append(base["recall"])

    present_recalls = [r for r in family_recalls if r is not None]
    macro_family_recall = macro_mean(present_recalls)
    min_family_recall = float(min(present_recalls)) if present_recalls else None

    # Benign groups: rollups first, then fine-grained categories present.
    benign_order = [
        "publisher_benign",
        "profiling_benign",
        "publisher",
        "profiling_interaction",
        "profiling_power",
        "profiling_idle",
    ]
    benign_group_rows: list[dict[str, Any]] = []
    seen_benign: set[str] = set()
    for key in benign_order:
        acc = benign_acc.get(key)
        if acc is None:
            continue
        seen_benign.add(key)
        base = acc.to_benign_row()
        benign_group_rows.append(
            {
                "benign_group": key,
                "pcap_count": base["pcap_count"],
                "rows": base["row_count"],
                "fp": base["fp"],
                "tn": base["tn"],
                "fpr": base["fpr"],
                "specificity": base["specificity"],
                "score_mean": base["attack_score_mean"],
                "score_p95": base["attack_score_p95"],
                "score_p99": base["attack_score_p99"],
                "score_max": base["max_attack_score"],
            }
        )
    for key, acc in sorted(benign_acc.items()):
        if key in seen_benign:
            continue
        base = acc.to_benign_row()
        benign_group_rows.append(
            {
                "benign_group": key,
                "pcap_count": base["pcap_count"],
                "rows": base["row_count"],
                "fp": base["fp"],
                "tn": base["tn"],
                "fpr": base["fpr"],
                "specificity": base["specificity"],
                "score_mean": base["attack_score_mean"],
                "score_p95": base["attack_score_p95"],
                "score_p99": base["attack_score_p99"],
                "score_max": base["max_attack_score"],
            }
        )

    pcap_rows: list[dict[str, Any]] = []
    for spec in specs:
        acc = pcap_acc[spec.pcap_id]
        base = acc.to_pcap_row(
            modeling_group_key=(
                f"{spec.attack_family}|{spec.attack_type}"
                if spec.binary_label == "ATTACK"
                else f"benign|{spec.benign_category}"
            ),
            binary_label=spec.binary_label,
            benign_category=spec.benign_category,
        )
        s = acc.score_stats.summary()
        pcap_rows.append(
            {
                "pcap_id": spec.pcap_id,
                "pcap_path": spec.pcap_path,
                "binary_label": spec.binary_label,
                "attack_family": spec.attack_family,
                "attack_type": spec.attack_type,
                "benign_category": spec.benign_category,
                "rows": base["row_count"],
                "tp": base["tp"],
                "fp": base["fp"],
                "tn": base["tn"],
                "fn": base["fn"],
                "recall": base["recall"],
                "fpr": base["fpr"],
                "specificity": base["specificity"],
                "score_mean": s["attack_score_mean"],
                "score_p05": s["attack_score_p05"],
                "score_p50": s["attack_score_p50"],
                "score_p95": s["attack_score_p95"],
                "score_p99": s["attack_score_p99"],
                "score_max": s["max_attack_score"],
            }
        )

    family_recall_map = {
        r["attack_family"]: r["recall"] for r in attack_family_rows if r["present"]
    }

    global_metrics = {
        "threshold": threshold,
        "decision_rule": "ATTACK if score >= threshold",
        "tp": global_counts.tp,
        "tn": global_counts.tn,
        "fp": global_counts.fp,
        "fn": global_counts.fn,
        "attack_recall": conf_metrics["attack_recall"],
        "specificity": conf_metrics["specificity"],
        "benign_fpr": conf_metrics["benign_fpr"],
        "precision": conf_metrics["precision"],
        "f1": conf_metrics["f1"],
        "balanced_accuracy": conf_metrics["balanced_accuracy"],
        "roc_auc": ranking["roc_auc"],
        "pr_auc": ranking["pr_auc"],
        "rows_scored": rows_scored,
        "pcaps_scored": len(pcap_acc),
        "score_mean": score_summary["attack_score_mean"],
        "score_p05": score_summary["attack_score_p05"],
        "score_p50": score_summary["attack_score_p50"],
        "score_p95": score_summary["attack_score_p95"],
        "score_p99": score_summary["attack_score_p99"],
        "score_max": score_summary["max_attack_score"],
        "macro_attack_family_recall": macro_family_recall,
        "min_attack_family_recall": min_family_recall,
        "ddos_recall": family_recall_map.get("DDoS"),
        "dos_recall": family_recall_map.get("DoS"),
        "mqtt_recall": family_recall_map.get("MQTT"),
        "recon_recall": family_recall_map.get("Recon"),
        "spoofing_recall": family_recall_map.get("Spoofing"),
        "each_row_scored_once": True,
        "each_pcap_scored_once": True,
        "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
        "extractor_feature_count": EXPECTED_EXTRACTOR_FEATURE_COUNT,
        "feature_names": list(FEATURES_22),
    }

    return {
        "global_metrics": global_metrics,
        "attack_family_rows": attack_family_rows,
        "benign_group_rows": benign_group_rows,
        "pcap_rows": pcap_rows,
        "rows_scored": rows_scored,
        "pcaps_scored": len(pcap_acc),
        # Back-compat aliases used by older summary formatters / tests.
        "threshold": threshold,
        "confusion": {
            "tp": global_counts.tp,
            "fp": global_counts.fp,
            "tn": global_counts.tn,
            "fn": global_counts.fn,
        },
        "attack_recall": conf_metrics["attack_recall"],
        "benign_fpr": conf_metrics["benign_fpr"],
        "per_pcap": pcap_rows,
        "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
        "extractor_feature_count": EXPECTED_EXTRACTOR_FEATURE_COUNT,
        "feature_names": list(FEATURES_22),
        "each_row_scored_once": True,
        "each_pcap_scored_once": True,
    }


def _build_validation_vs_test_rows(
    *,
    project_root: Path,
    global_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare frozen TRAIN-validation operating point to one-shot TEST."""
    package_path = project_root / DEFAULT_PHASE2C_FREEZE_PATH
    val: dict[str, Any] = {}
    if package_path.is_file():
        package = json.loads(package_path.read_text(encoding="utf-8"))
        val = dict(package.get("validation_operating_point") or {})

    def row(metric: str, validation: Any, test: Any) -> dict[str, Any]:
        try:
            v = float(validation) if validation is not None else None
        except (TypeError, ValueError):
            v = None
        try:
            t = float(test) if test is not None else None
        except (TypeError, ValueError):
            t = None
        delta = None if v is None or t is None else t - v
        return {
            "metric": metric,
            "validation": v,
            "test": t,
            "delta_test_minus_validation": delta,
        }

    return [
        row("benign_fpr", val.get("benign_fpr"), global_metrics.get("benign_fpr")),
        row("ddos_recall", val.get("ddos_recall"), global_metrics.get("ddos_recall")),
        row("dos_recall", val.get("dos_recall"), global_metrics.get("dos_recall")),
        row("mqtt_recall", val.get("mqtt_recall"), global_metrics.get("mqtt_recall")),
        row("recon_recall", val.get("recon_recall"), global_metrics.get("recon_recall")),
        row(
            "attack_recall",
            None,
            global_metrics.get("attack_recall"),
        ),
        row("specificity", None, global_metrics.get("specificity")),
        row("precision", None, global_metrics.get("precision")),
        row("f1", None, global_metrics.get("f1")),
        row("balanced_accuracy", None, global_metrics.get("balanced_accuracy")),
        row("roc_auc", None, global_metrics.get("roc_auc")),
        row("pr_auc", None, global_metrics.get("pr_auc")),
        row(
            "macro_attack_family_recall",
            None,
            global_metrics.get("macro_attack_family_recall"),
        ),
        row(
            "min_attack_family_recall",
            None,
            global_metrics.get("min_attack_family_recall"),
        ),
    ]


def _atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def run_final_test(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    progress_file: TextIO | None = None,
    **forbidden_overrides: Any,
) -> dict[str, Any]:
    """One-shot sealed TEST evaluation bound to the Phase 2D contract.

    Does not accept model/threshold/feature/hyperparameter/sampling overrides.
    Supporting metric artifacts are written before final_test_complete.json.
    """
    reject_final_test_overrides(**forbidden_overrides)
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_FINAL_TEST_ROOT)
    if not out.is_absolute():
        out = root / out

    contract = load_final_test_contract(project_root=root)
    verify_contract_integrity(contract, project_root=root)

    preflight_path = out / "preflight_complete.json"
    if not preflight_path.is_file():
        raise FeatureExtractionError(
            f"preflight_complete.json missing at {preflight_path}. "
            "Run preflight-final-test before one-shot TEST evaluation."
        )
    preflight = _require_json(preflight_path, label="preflight_complete")
    if preflight.get("status") != "passed" or not preflight.get(
        "ready_for_one_shot_test"
    ):
        raise FeatureExtractionError(
            "Phase 2D.2 preflight not ready: "
            f"status={preflight.get('status')!r} "
            f"ready_for_one_shot_test={preflight.get('ready_for_one_shot_test')!r}"
        )

    estimator = load_frozen_hgb_estimator(contract, project_root=root)

    arts = contract.get("artifacts") or {}
    specs = load_test_pcap_specs(
        arts.get("test_build_manifest") or DEFAULT_TEST_BUILD_MANIFEST_PATH,
        project_root=root,
    )
    split_ids = load_modeling_split_pcap_ids(
        arts.get("modeling_split_manifest") or DEFAULT_SPLIT_MANIFEST_PATH,
        project_root=root,
    )

    if progress_file is not None:
        progress_file.write(
            "Phase 2D — sealed TEST evaluation (frozen H0 + threshold; "
            "no overrides)\n"
        )
        progress_file.flush()

    metrics = evaluate_sealed_test_inventory(
        specs,
        estimator,
        project_root=root,
        threshold=FROZEN_V1_THRESHOLD,
        expected_pcaps=EXPECTED_TEST_PCAP_COUNT,
        expected_rows=EXPECTED_TEST_ROWS,
        modeling_split_pcap_ids=split_ids,
        progress_file=progress_file,
    )
    global_metrics = metrics["global_metrics"]
    val_vs_test = _build_validation_vs_test_rows(
        project_root=root, global_metrics=global_metrics
    )

    out.mkdir(parents=True, exist_ok=True)

    # Supporting artifacts first — completion marker is written last.
    global_path = out / "test_global_metrics.json"
    attack_path = out / "test_attack_family_metrics.csv"
    benign_path = out / "test_benign_group_metrics.csv"
    pcap_path = out / "test_pcap_metrics.csv"
    compare_path = out / "validation_vs_test.csv"

    _atomic_json(global_path, global_metrics)
    _atomic_csv(
        attack_path,
        metrics["attack_family_rows"],
        [
            "attack_family",
            "pcap_count",
            "rows",
            "tp",
            "fn",
            "recall",
            "score_mean",
            "score_p05",
            "score_p50",
            "score_p95",
            "present",
        ],
    )
    _atomic_csv(
        benign_path,
        metrics["benign_group_rows"],
        [
            "benign_group",
            "pcap_count",
            "rows",
            "fp",
            "tn",
            "fpr",
            "specificity",
            "score_mean",
            "score_p95",
            "score_p99",
            "score_max",
        ],
    )
    _atomic_csv(
        pcap_path,
        metrics["pcap_rows"],
        [
            "pcap_id",
            "pcap_path",
            "binary_label",
            "attack_family",
            "attack_type",
            "benign_category",
            "rows",
            "tp",
            "fp",
            "tn",
            "fn",
            "recall",
            "fpr",
            "specificity",
            "score_mean",
            "score_p05",
            "score_p50",
            "score_p95",
            "score_p99",
            "score_max",
        ],
    )
    _atomic_csv(
        compare_path,
        val_vs_test,
        ["metric", "validation", "test", "delta_test_minus_validation"],
    )

    artifact_rels = {
        "final_test_contract": to_repo_relative(
            out / "final_test_contract.json", project_root=root
        ),
        "preflight_complete": to_repo_relative(preflight_path, project_root=root),
        "test_global_metrics": to_repo_relative(global_path, project_root=root),
        "test_attack_family_metrics": to_repo_relative(attack_path, project_root=root),
        "test_benign_group_metrics": to_repo_relative(benign_path, project_root=root),
        "test_pcap_metrics": to_repo_relative(pcap_path, project_root=root),
        "validation_vs_test": to_repo_relative(compare_path, project_root=root),
        "final_test_complete": to_repo_relative(
            out / "final_test_complete.json", project_root=root
        ),
    }

    complete = {
        "status": "passed",
        "gate_2d_status": "passed",
        "strategy_version": PHASE2D_VERSION,
        "evaluation_mode": "one_shot_sealed_test",
        "measurement_only": True,
        "retraining_allowed": False,
        "threshold_sweep_allowed": False,
        "alternate_models_allowed": False,
        "test_sampling_allowed": False,
        "threshold": FROZEN_V1_THRESHOLD,
        "decision_rule": "ATTACK if score >= threshold",
        "hyperparameter_config_id": "H0",
        "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
        "extractor_feature_count": EXPECTED_EXTRACTOR_FEATURE_COUNT,
        "model_artifact": contract.get("model_artifact"),
        "model_artifact_sha256": (contract.get("pins") or {}).get(
            "model_artifact_sha256"
        ),
        "metrics": global_metrics,
        "pcap_count": metrics["pcaps_scored"],
        "row_count": metrics["rows_scored"],
        "predictions_generated": True,
        "metrics_generated": True,
        "test": {
            "access": True,
            "pcaps_read": metrics["pcaps_scored"],
            "rows_read": metrics["rows_scored"],
            "one_shot": True,
        },
        "artifacts": artifact_rels,
        "next": (
            "TEST evaluation complete. Do not change the model, features, "
            "hyperparameters, sampling, or threshold based on these results."
        ),
    }
    # Final successful operation: completion marker.
    _atomic_json(out / "final_test_complete.json", complete)
    return complete


def format_final_test_summary(payload: dict[str, Any]) -> str:
    m = payload.get("metrics") or {}
    lines = [
        "Phase 2D — One-shot sealed TEST evaluation",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"threshold: {payload.get('threshold')}",
        f"pcaps: {payload.get('pcap_count')}",
        f"rows: {payload.get('row_count')}",
        f"confusion: tp={m.get('tp')} fp={m.get('fp')} "
        f"tn={m.get('tn')} fn={m.get('fn')}",
        f"attack_recall: {m.get('attack_recall')}",
        f"benign_fpr: {m.get('benign_fpr')}",
        f"macro_family_recall: {m.get('macro_attack_family_recall')}",
        f"min_family_recall: {m.get('min_attack_family_recall')}",
        f"roc_auc: {m.get('roc_auc')} pr_auc: {m.get('pr_auc')}",
        f"measurement_only: {payload.get('measurement_only')}",
    ]
    arts = payload.get("artifacts") or {}
    if arts.get("final_test_complete"):
        lines.append(f"complete: {arts['final_test_complete']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
