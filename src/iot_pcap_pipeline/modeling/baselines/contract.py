"""Baseline experiment contract + Gate 2B.1 precondition checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    V1_FEATURE_NAMES,
)
from iot_pcap_pipeline.modeling.baselines.constants import (
    BASELINE_STRATEGY_VERSION,
    DECISION_THRESHOLD,
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_PCAPS,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_PCAPS,
    EXPECTED_VAL_ROWS,
    LABEL_MAPPING,
    POSITIVE_CLASS,
)
from iot_pcap_pipeline.modeling.freeze import FROZEN_SAMPLING_PLAN_ID
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_FIT_VIEW_COMPLETE_PATH,
    DEFAULT_FIT_VIEW_MANIFEST_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_TRAINING_VIEW_CONTRACT_PATH,
    file_sha256,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    MODELING_SPLIT_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

DEFAULT_BASELINES_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / BASELINE_STRATEGY_VERSION
)
DEFAULT_BASELINE_CONTRACT_PATH = DEFAULT_BASELINES_ROOT / "baseline_contract.json"


def build_baseline_contract_payload(
    *,
    project_root: Path | None = None,
    fit_view_manifest_path: Path | None = None,
    training_view_contract_path: Path | None = None,
    split_manifest_path: Path | None = None,
    feature_schema_path: Path | None = None,
    smoke_only: bool = False,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    fit_man = Path(fit_view_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    train_contract = Path(
        training_view_contract_path or DEFAULT_TRAINING_VIEW_CONTRACT_PATH
    )
    if not train_contract.is_absolute():
        train_contract = root / train_contract
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path
    schema_path = Path(feature_schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not schema_path.is_absolute():
        schema_path = root / schema_path

    return {
        "strategy_version": BASELINE_STRATEGY_VERSION,
        "task": "binary_classification",
        "positive_class": POSITIVE_CLASS,
        "label_mapping": dict(LABEL_MAPPING),
        "feature_count": len(V1_FEATURE_NAMES),
        "feature_names": list(V1_FEATURE_NAMES),
        "training_view": FROZEN_SAMPLING_PLAN_ID,
        "modeling_split_strategy_version": MODELING_SPLIT_STRATEGY_VERSION,
        "fit_rows": EXPECTED_FIT_ROWS,
        "fit_attack_rows": EXPECTED_FIT_ATTACK,
        "fit_benign_rows": EXPECTED_FIT_BENIGN,
        "fit_pcaps": EXPECTED_FIT_PCAPS,
        "validation_rows": EXPECTED_VAL_ROWS,
        "validation_pcaps": EXPECTED_VAL_PCAPS,
        "validation_sampling": "never",
        "decision_threshold": DECISION_THRESHOLD,
        "class_weights": "none",
        "threshold_tuning": False,
        "test_access": False,
        "score_calibration": False,
        "score_note": (
            "predict_proba attack-class score is not a calibrated real-world "
            "attack probability; FIT was intentionally resampled under "
            "group_balanced."
        ),
        "smoke_only": bool(smoke_only),
        "pins": {
            "feature_schema_sha256": feature_schema_sha256(schema_path),
            "training_view_contract_sha256": file_sha256(train_contract),
            "fit_view_manifest_sha256": file_sha256(fit_man),
            "modeling_split_manifest_sha256": file_sha256(split_path),
        },
        "artifacts": {
            "feature_schema": to_repo_relative(schema_path, project_root=root),
            "training_view_contract": to_repo_relative(
                train_contract, project_root=root
            ),
            "fit_view_manifest": to_repo_relative(fit_man, project_root=root),
            "modeling_split_manifest": to_repo_relative(
                split_path, project_root=root
            ),
        },
    }


def write_baseline_contract(
    path: Path | str | None = None,
    *,
    project_root: Path | None = None,
    smoke_only: bool = False,
    **kwargs: Any,
) -> Path:
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(path or DEFAULT_BASELINE_CONTRACT_PATH)
    if not out.is_absolute():
        out = root / out
    payload = build_baseline_contract_payload(
        project_root=root, smoke_only=smoke_only, **kwargs
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def require_fit_view_ready(
    *,
    fit_complete_path: Path | str | None = None,
    project_root: Path | None = None,
    smoke_only: bool = False,
) -> dict[str, Any]:
    """Refuse unless Phase 2B.1 fit view completion marker is passed."""
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(fit_complete_path or DEFAULT_FIT_VIEW_COMPLETE_PATH)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FeatureExtractionError(
            f"fit_view_complete.json missing: {path}. "
            "Run build-modeling-fit-view first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise FeatureExtractionError(
            f"fit view not passed: status={payload.get('status')!r} in {path}"
        )
    if payload.get("sampling_plan_id") != FROZEN_SAMPLING_PLAN_ID:
        raise FeatureExtractionError(
            f"fit view sampling_plan_id={payload.get('sampling_plan_id')!r}; "
            f"expected {FROZEN_SAMPLING_PLAN_ID!r}"
        )
    if payload.get("modeling_split_strategy_version") != MODELING_SPLIT_STRATEGY_VERSION:
        raise FeatureExtractionError(
            "fit view modeling_split_strategy_version mismatch: "
            f"{payload.get('modeling_split_strategy_version')!r}"
        )
    if payload.get("validation_sampling") != "never":
        raise FeatureExtractionError(
            "fit view must record validation_sampling=never"
        )
    totals = payload.get("totals") or {}
    if int(totals.get("validation_pcaps_touched", -1)) != 0:
        raise FeatureExtractionError("fit view touched validation PCAPs")
    if int(totals.get("test_pcaps_touched", -1)) != 0:
        raise FeatureExtractionError("fit view touched TEST PCAPs")

    if not smoke_only:
        if int(totals.get("total_rows", -1)) != EXPECTED_FIT_ROWS:
            raise FeatureExtractionError(
                f"fit rows {totals.get('total_rows')} != {EXPECTED_FIT_ROWS}"
            )
        if int(totals.get("attack_rows", -1)) != EXPECTED_FIT_ATTACK:
            raise FeatureExtractionError(
                f"fit attack {totals.get('attack_rows')} != {EXPECTED_FIT_ATTACK}"
            )
        if int(totals.get("benign_rows", -1)) != EXPECTED_FIT_BENIGN:
            raise FeatureExtractionError(
                f"fit benign {totals.get('benign_rows')} != {EXPECTED_FIT_BENIGN}"
            )
        if int(totals.get("fit_pcaps", -1)) != EXPECTED_FIT_PCAPS:
            raise FeatureExtractionError(
                f"fit pcaps {totals.get('fit_pcaps')} != {EXPECTED_FIT_PCAPS}"
            )
    return payload


def verify_pinned_hashes(
    contract: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> None:
    """Ensure on-disk artifact hashes still match the baseline contract pins."""
    root = (project_root or PROJECT_ROOT).resolve()
    pins = contract.get("pins") or {}
    arts = contract.get("artifacts") or {}

    checks = (
        ("feature_schema_sha256", arts.get("feature_schema"), "schema"),
        (
            "training_view_contract_sha256",
            arts.get("training_view_contract"),
            "file",
        ),
        ("fit_view_manifest_sha256", arts.get("fit_view_manifest"), "file"),
        (
            "modeling_split_manifest_sha256",
            arts.get("modeling_split_manifest"),
            "file",
        ),
    )
    for pin_key, rel, kind in checks:
        if not rel:
            raise FeatureExtractionError(f"baseline contract missing artifact for {pin_key}")
        path = root / str(rel)
        if not path.is_file():
            raise FeatureExtractionError(f"pinned artifact missing: {path}")
        if kind == "schema":
            actual = feature_schema_sha256(path)
        else:
            actual = file_sha256(path)
        expected = str(pins.get(pin_key) or "")
        if actual != expected:
            raise FeatureExtractionError(
                f"{pin_key} mismatch: actual={actual} pinned={expected}"
            )
