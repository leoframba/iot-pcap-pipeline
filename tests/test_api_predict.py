"""D2.5–D2.10: POST /predict with FakePcapFetcher (no GCP credentials)."""

from __future__ import annotations

from pathlib import Path

import dpkt
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from iot_pcap_pipeline.api.app import create_app
from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import FakePcapFetcher
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from pcap_synth import eth_ip_tcp, write_pcap


def _settings(**overrides) -> ServingSettings:
    base = dict(
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=50 * 1024 * 1024,
    )
    base.update(overrides)
    return ServingSettings(**base)


def _synthetic_pcap(tmp_path: Path, *, n_windows: int = 3) -> Path:
    path = tmp_path / "synth.pcap"
    packets = [
        (float(i) * 0.001, eth_ip_tcp(sport=3000 + i, flags=dpkt.tcp.TH_SYN))
        for i in range(WINDOW_SIZE * n_windows)
    ]
    write_pcap(path, packets, linktype=1)
    return path


def _client_for_pcap(
    tmp_path: Path, pcap_path: Path, **settings_kw
) -> tuple[TestClient, str]:
    settings = _settings(**settings_kw)
    uri = f"gs://{settings.input_bucket}/{settings.input_prefix}example.pcap"
    fetcher = FakePcapFetcher(
        {uri: pcap_path},
        input_bucket=settings.input_bucket,
        input_prefix=settings.input_prefix,
        max_pcap_bytes=settings.max_pcap_bytes,
    )
    app = create_app(settings=settings, pcap_fetcher=fetcher)
    return TestClient(app), uri


def test_predict_schema_rejects_empty_instances() -> None:
    app = create_app(
        settings=_settings(),
        pcap_fetcher=FakePcapFetcher(
            {},
            input_bucket="iomt-input",
            input_prefix="pcaps/",
            max_pcap_bytes=1024,
        ),
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": []})
    assert response.status_code == 422


def test_predict_schema_rejects_non_gs_uri(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path)
    client, _ = _client_for_pcap(tmp_path, pcap)
    with client:
        response = client.post(
            "/predict",
            json={"instances": [{"gcs_uri": "https://example.com/x.pcap"}]},
        )
    assert response.status_code == 422


def test_predict_rejects_disallowed_bucket(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path)
    settings = _settings()
    fetcher = FakePcapFetcher(
        {"gs://other-bucket/pcaps/example.pcap": pcap},
        input_bucket=settings.input_bucket,
        input_prefix=settings.input_prefix,
        max_pcap_bytes=settings.max_pcap_bytes,
    )
    app = create_app(settings=settings, pcap_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"instances": [{"gcs_uri": "gs://other-bucket/pcaps/example.pcap"}]},
        )
    assert response.status_code == 400
    assert "bucket not allowed" in response.json()["detail"]


def test_predict_rejects_disallowed_prefix(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path)
    settings = _settings()
    uri = "gs://iomt-input/private/secret.pcap"
    fetcher = FakePcapFetcher(
        {uri: pcap},
        input_bucket=settings.input_bucket,
        input_prefix=settings.input_prefix,
        max_pcap_bytes=settings.max_pcap_bytes,
    )
    app = create_app(settings=settings, pcap_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 400
    assert "prefix not allowed" in response.json()["detail"]


def test_predict_rejects_oversized_object(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path)
    client, uri = _client_for_pcap(tmp_path, pcap, max_pcap_bytes=10)
    with client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 400
    assert "too large" in response.json()["detail"]


def test_predict_runs_real_classify_via_fake_gcs(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_windows=3)
    client, uri = _client_for_pcap(tmp_path, pcap)
    with client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predictions"}
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert pred["status"] in {"OK", "INSUFFICIENT_DATA"}
    assert pred["model"] == {
        "model_version": "v1_hgb22_nontemporal",
        "serving_contract_version": "v1",
        "score_semantics": "uncalibrated_model_score",
    }
    assert "model_sha256" not in pred["model"]
    assert pred["decision"]["pcap_min_attack_windows"] == 3
    assert pred["decision"]["pcap_attack_rate_threshold"] == 0.005
    assert pred["decision"]["minimum_complete_windows"] == 3
    assert pred["window_summary"]["total_windows"] == 3
    if pred["status"] == "OK":
        assert pred["prediction"] in {"ATTACK", "BENIGN"}
        assert pred["pcap_attack_score"] is not None


def test_predict_invalid_pcap_returns_contract_status(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pcap"
    bad.write_bytes(b"not a pcap")
    client, uri = _client_for_pcap(tmp_path, bad)
    with client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 200
    pred = response.json()["predictions"][0]
    assert pred["status"] == "INVALID_INPUT"
    assert pred["prediction"] is None
