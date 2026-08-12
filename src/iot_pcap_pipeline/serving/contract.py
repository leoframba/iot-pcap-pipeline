"""Load and verify the frozen V1 serving contract (stdlib + artifacts only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.serving.errors import ServingError

DEFAULT_SERVING_CONTRACT_PATH = (
    PROJECT_ROOT / "artifacts" / "v1" / "serving_contract.json"
)
DEFAULT_MODEL_INPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "v1" / "v1_hgb22_nontemporal.json"
)
DEFAULT_FEATURE_SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "v1" / "feature_schema.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "v1" / "H0_full_fit.joblib"

EXPECTED_MODEL_SHA256 = (
    "c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb"
)
EXPECTED_FEATURE_SCHEMA_SHA256 = (
    "d3ee4f40f9e2a3da8f2821ea41d5115a8117b1cd921e7a9fb8558026aa02e69b"
)

# Frozen window / PCAP aggregation pins (must match serving_contract.json).
WINDOW_ATTACK_THRESHOLD = 0.9490790963172913
FROZEN_MIN_COMPLETE_WINDOWS = 3
FROZEN_MIN_ATTACK_WINDOWS = 3
FROZEN_ATTACK_RATE_THRESHOLD = 0.005
FROZEN_POLICY_ID = "K3_R0.005"

# V1 serving accepts classic libpcap Ethernet only (corpus distribution).
ACCEPTED_LINKTYPE = 1  # DLT_EN10MB
ACCEPTED_LINKTYPE_NAME = "DLT_EN10MB"


def sha256_file(path: Path | str) -> str:
    """SHA-256 of file bytes (empty files refused)."""
    p = Path(path)
    data = p.read_bytes()
    if not data:
        raise ServingError(f"refusing empty file hash: {p}")
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ServingError(f"JSON missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


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
        raise ServingError(f"serving_contract.json missing: {p}")
    return load_json(p)


def load_model_input_feature_names(
    path: Path | str | None = None,
    *,
    project_root: Path | None = None,
) -> list[str]:
    """Load ordered 22 feature names from checkout-ready model-input JSON."""
    root = (project_root or PROJECT_ROOT).resolve()
    p = Path(path or DEFAULT_MODEL_INPUT_PATH)
    if not p.is_absolute():
        p = root / p
    doc = load_json(p)
    names = list(doc.get("feature_names") or [])
    if len(names) != 22:
        raise ServingError(
            f"model-input feature_names length {len(names)} != 22 ({p})"
        )
    return names


def verify_serving_contract(
    contract: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Refuse drift vs frozen V1 artifacts under artifacts/v1/."""
    root = (project_root or PROJECT_ROOT).resolve()
    doc = (
        contract
        if contract is not None
        else load_serving_contract(path, project_root=root)
    )

    if doc.get("status") != "frozen":
        raise ServingError(
            f"serving contract status must be frozen, got {doc.get('status')!r}"
        )
    if doc.get("serving_contract_version") != "v1":
        raise ServingError(
            f"unexpected serving_contract_version: {doc.get('serving_contract_version')!r}"
        )
    if doc.get("frozen_policy_id") != FROZEN_POLICY_ID:
        raise ServingError(
            f"frozen_policy_id {doc.get('frozen_policy_id')!r} != {FROZEN_POLICY_ID!r}"
        )

    window = doc.get("window_decision") or {}
    thr = float(window.get("window_attack_threshold"))
    if thr != WINDOW_ATTACK_THRESHOLD:
        raise ServingError(
            f"window_attack_threshold {thr!r} != {WINDOW_ATTACK_THRESHOLD!r}"
        )

    pcap = doc.get("pcap_decision") or {}
    if int(pcap.get("minimum_complete_windows")) != FROZEN_MIN_COMPLETE_WINDOWS:
        raise ServingError("minimum_complete_windows drift")
    if int(pcap.get("pcap_min_attack_windows")) != FROZEN_MIN_ATTACK_WINDOWS:
        raise ServingError("pcap_min_attack_windows drift")
    if float(pcap.get("pcap_attack_rate_threshold")) != FROZEN_ATTACK_RATE_THRESHOLD:
        raise ServingError("pcap_attack_rate_threshold drift")

    model = doc.get("model") or {}
    names = list(model.get("feature_names") or [])
    artifact_names = load_model_input_feature_names(project_root=root)
    if names != artifact_names:
        raise ServingError(
            "serving_contract feature_names drift vs artifacts/v1/v1_hgb22_nontemporal.json"
        )
    if int(model.get("feature_count") or 0) != 22:
        raise ServingError("serving contract feature_count != 22")

    model_rel = model.get("model_artifact")
    if not model_rel:
        raise ServingError("serving contract missing model_artifact")
    model_path = root / Path(model_rel)
    if not model_path.is_file():
        raise ServingError(f"model artifact missing: {model_path}")
    model_sha = sha256_file(model_path)
    pinned = str(model.get("model_artifact_sha256") or "")
    if model_sha != pinned or pinned != EXPECTED_MODEL_SHA256:
        raise ServingError(
            f"model SHA mismatch: actual={model_sha} pinned={pinned} "
            f"expected={EXPECTED_MODEL_SHA256}"
        )

    schema_rel = model.get("feature_schema")
    if not schema_rel:
        raise ServingError("serving contract missing feature_schema")
    schema_path = root / Path(schema_rel)
    if not schema_path.is_file():
        raise ServingError(f"feature schema missing: {schema_path}")
    schema_sha = sha256_file(schema_path)
    pinned_schema = str(model.get("feature_schema_sha256") or "")
    if schema_sha != pinned_schema or pinned_schema != EXPECTED_FEATURE_SCHEMA_SHA256:
        raise ServingError(
            f"feature schema SHA mismatch: actual={schema_sha} pinned={pinned_schema}"
        )

    return doc
