"""D2.3 / D2.4: FastAPI lifespan engine load + GET /health."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from iot_pcap_pipeline.api.app import create_app
from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import FakePcapFetcher
from iot_pcap_pipeline.serving.contract import EXPECTED_MODEL_SHA256
from iot_pcap_pipeline.serving.errors import ServingError


def _empty_fetcher() -> FakePcapFetcher:
    return FakePcapFetcher(
        {},
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=1024,
    )


def test_health_returns_ok_and_model_identity() -> None:
    app = create_app(settings=ServingSettings(
        input_bucket="iomt-input", input_prefix="pcaps/", max_pcap_bytes=1024
    ), pcap_fetcher=_empty_fetcher())
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "model_version": "v1_hgb22_nontemporal",
        "serving_contract_version": "v1",
        "model_sha256": EXPECTED_MODEL_SHA256,
    }


def test_startup_fails_when_model_artifacts_missing(tmp_path: Path) -> None:
    app = create_app(
        project_root=tmp_path,
        settings=ServingSettings(
            input_bucket="iomt-input", input_prefix="pcaps/", max_pcap_bytes=1024
        ),
        pcap_fetcher=_empty_fetcher(),
    )
    with pytest.raises(ServingError):
        with TestClient(app):
            pass


def test_health_does_not_include_prediction_fields() -> None:
    app = create_app(settings=ServingSettings(
        input_bucket="iomt-input", input_prefix="pcaps/", max_pcap_bytes=1024
    ), pcap_fetcher=_empty_fetcher())
    with TestClient(app) as client:
        payload = client.get("/health").json()
    assert "prediction" not in payload
    assert "pcap_attack_score" not in payload
    assert "window_summary" not in payload
