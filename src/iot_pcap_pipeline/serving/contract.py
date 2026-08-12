"""Load and verify the frozen V1 serving contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.modeling.baselines.model_input import V1_MODEL_INPUT_FEATURES
from iot_pcap_pipeline.modeling.baselines.phase2c_freeze import FROZEN_V1_THRESHOLD
from iot_pcap_pipeline.modeling.view import file_sha256
from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

DEFAULT_SERVING_CONTRACT_PATH = (
    PROJECT_ROOT / "artifacts" / "v1" / "serving_contract.json"
)
EXPECTED_MODEL_SHA256 = (
    "c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb"
)
EXPECTED_FEATURE_SCHEMA_SHA256 = (
    "d3ee4f40f9e2a3da8f2821ea41d5115a8117b1cd921e7a9fb8558026aa02e69b"
)
FROZEN_MIN_COMPLETE_WINDOWS = 3
FROZEN_MIN_ATTACK_WINDOWS = 3
FROZEN_ATTACK_RATE_THRESHOLD = 0.005
FROZEN_POLICY_ID = "K3_R0.005"


def load_serving_contract(
    path: Path | str | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    p = Path(path or DEFAULT_SERVING_CONTRACT_PATH)
    if not p.is_absolute():
        p = root / p
    if not p.is_file():
        raise FeatureExtractionError(f"serving_contract.json missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_serving_contract(
    contract: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Refuse drift vs frozen V1 model / feature / aggregation pins."""
    root = (project_root or PROJECT_ROOT).resolve()
    doc = contract if contract is not None else load_serving_contract(path, project_root=root)

    if doc.get("status") != "frozen":
        raise FeatureExtractionError(
            f"serving contract status must be frozen, got {doc.get('status')!r}"
        )
    if doc.get("serving_contract_version") != "v1":
        raise FeatureExtractionError(
            f"unexpected serving_contract_version: {doc.get('serving_contract_version')!r}"
        )
    if doc.get("frozen_policy_id") != FROZEN_POLICY_ID:
        raise FeatureExtractionError(
            f"frozen_policy_id {doc.get('frozen_policy_id')!r} != {FROZEN_POLICY_ID!r}"
        )

    window = doc.get("window_decision") or {}
    thr = float(window.get("window_attack_threshold"))
    if thr != FROZEN_V1_THRESHOLD:
        raise FeatureExtractionError(
            f"window_attack_threshold {thr!r} != {FROZEN_V1_THRESHOLD!r}"
        )

    pcap = doc.get("pcap_decision") or {}
    if int(pcap.get("minimum_complete_windows")) != FROZEN_MIN_COMPLETE_WINDOWS:
        raise FeatureExtractionError("minimum_complete_windows drift")
    if int(pcap.get("pcap_min_attack_windows")) != FROZEN_MIN_ATTACK_WINDOWS:
        raise FeatureExtractionError("pcap_min_attack_windows drift")
    if float(pcap.get("pcap_attack_rate_threshold")) != FROZEN_ATTACK_RATE_THRESHOLD:
        raise FeatureExtractionError("pcap_attack_rate_threshold drift")

    model = doc.get("model") or {}
    names = list(model.get("feature_names") or [])
    if names != list(V1_MODEL_INPUT_FEATURES):
        raise FeatureExtractionError(
            "serving contract feature_names drift vs v1_hgb22_nontemporal"
        )
    if int(model.get("feature_count") or 0) != 22:
        raise FeatureExtractionError("serving contract feature_count != 22")

    model_rel = model.get("model_artifact")
    if not model_rel:
        raise FeatureExtractionError("serving contract missing model_artifact")
    model_path = root / Path(model_rel)
    if not model_path.is_file():
        raise FeatureExtractionError(f"model artifact missing: {model_path}")
    model_sha = file_sha256(model_path)
    pinned = str(model.get("model_artifact_sha256") or "")
    if model_sha != pinned or pinned != EXPECTED_MODEL_SHA256:
        raise FeatureExtractionError(
            f"model SHA mismatch: actual={model_sha} pinned={pinned} "
            f"expected={EXPECTED_MODEL_SHA256}"
        )

    schema_rel = model.get("feature_schema")
    if not schema_rel:
        raise FeatureExtractionError("serving contract missing feature_schema")
    schema_path = root / Path(schema_rel)
    schema_sha = feature_schema_sha256(schema_path)
    pinned_schema = str(model.get("feature_schema_sha256") or "")
    if schema_sha != pinned_schema or pinned_schema != EXPECTED_FEATURE_SCHEMA_SHA256:
        raise FeatureExtractionError(
            f"feature schema SHA mismatch: actual={schema_sha} pinned={pinned_schema}"
        )

    return doc
