"""Contract integrity tests for frozen V1 serving_contract.json."""

from __future__ import annotations

import json

from iot_pcap_pipeline.modeling.baselines.model_input import V1_MODEL_INPUT_FEATURES
from iot_pcap_pipeline.modeling.baselines.phase2c_freeze import FROZEN_V1_THRESHOLD
from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.serving.contract import (
    EXPECTED_MODEL_SHA256,
    FROZEN_ATTACK_RATE_THRESHOLD,
    FROZEN_MIN_ATTACK_WINDOWS,
    FROZEN_MIN_COMPLETE_WINDOWS,
    FROZEN_POLICY_ID,
    verify_serving_contract,
)


def test_verify_serving_contract_passes() -> None:
    doc = verify_serving_contract()
    assert doc["status"] == "frozen"
    assert doc["frozen_policy_id"] == FROZEN_POLICY_ID


def test_serving_model_sha_matches_frozen_artifact() -> None:
    doc = verify_serving_contract()
    assert doc["model"]["model_artifact_sha256"] == EXPECTED_MODEL_SHA256
    assert EXPECTED_MODEL_SHA256.startswith("c07ef408")


def test_serving_feature_order_matches_model_input_contract() -> None:
    doc = verify_serving_contract()
    assert list(doc["model"]["feature_names"]) == list(V1_MODEL_INPUT_FEATURES)
    artifact_input = json.loads(
        (PROJECT_ROOT / "artifacts/v1/v1_hgb22_nontemporal.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(doc["model"]["feature_names"]) == list(artifact_input["feature_names"])


def test_serving_window_threshold_frozen() -> None:
    doc = verify_serving_contract()
    assert doc["window_decision"]["window_attack_threshold"] == FROZEN_V1_THRESHOLD
    assert FROZEN_V1_THRESHOLD == 0.9490790963172913


def test_serving_pcap_aggregation_pins() -> None:
    doc = verify_serving_contract()
    pcap = doc["pcap_decision"]
    assert pcap["minimum_complete_windows"] == FROZEN_MIN_COMPLETE_WINDOWS
    assert pcap["pcap_min_attack_windows"] == FROZEN_MIN_ATTACK_WINDOWS
    assert pcap["pcap_attack_rate_threshold"] == FROZEN_ATTACK_RATE_THRESHOLD


def test_selection_justification_is_engineering_not_val_superiority() -> None:
    doc = verify_serving_contract()
    just = doc["selection_justification"]
    assert just["type"] == "engineering_operating_policy"
    assert "not because validation proved it superior" in just["note"]


def test_d0_complete_artifact_consistent() -> None:
    path = PROJECT_ROOT / "data" / "serving" / "v1" / "d0_complete.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["frozen_policy"]["policy_id"] == FROZEN_POLICY_ID
    assert payload["frozen_policy"]["K"] == FROZEN_MIN_ATTACK_WINDOWS
    assert payload["frozen_policy"]["R"] == FROZEN_ATTACK_RATE_THRESHOLD
    assert (
        payload["frozen_policy"]["minimum_complete_windows"]
        == FROZEN_MIN_COMPLETE_WINDOWS
    )
    assert payload["selection_justification"]["type"] == "engineering_operating_policy"
