"""D2 FastAPI acceptance: /health + /predict with FakePcapFetcher (no GCP creds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dpkt
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from iot_pcap_pipeline.api.app import MAX_CONCURRENT_PREDICTIONS, create_app
from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import FakePcapFetcher, PcapFetchError
from iot_pcap_pipeline.serving.classify import classify_pcap
from iot_pcap_pipeline.serving.contract import EXPECTED_MODEL_SHA256
from iot_pcap_pipeline.serving.model import V1InferenceEngine
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from pcap_synth import eth_ip_tcp, write_pcap


def _settings(**overrides: Any) -> ServingSettings:
    base: dict[str, Any] = dict(
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=50 * 1024 * 1024,
    )
    base.update(overrides)
    return ServingSettings(**base)


def _synthetic_pcap(
    tmp_path: Path, *, n_packets: int, linktype: int = 1, name: str = "synth.pcap"
) -> Path:
    path = tmp_path / name
    packets = [
        (float(i) * 0.001, eth_ip_tcp(sport=3000 + i, flags=dpkt.tcp.TH_SYN))
        for i in range(n_packets)
    ]
    write_pcap(path, packets, linktype=linktype)
    return path


def _omit_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _omit_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_omit_nones(v) for v in value]
    return value


def _app_for_pcap(
    pcap_path: Path,
    *,
    settings: ServingSettings | None = None,
    denied: bool = False,
    engine: V1InferenceEngine | None = None,
    map_uri: str | None = None,
) -> tuple[Any, FakePcapFetcher, str]:
    resolved = settings or _settings()
    uri = map_uri or (
        f"gs://{resolved.input_bucket}/{resolved.input_prefix}example.pcap"
    )
    mapping = {} if denied else {uri: pcap_path}
    fetcher = FakePcapFetcher(
        mapping,
        input_bucket=resolved.input_bucket,
        input_prefix=resolved.input_prefix,
        max_pcap_bytes=resolved.max_pcap_bytes,
        denied_uris=[uri] if denied else None,
    )
    app = create_app(settings=resolved, pcap_fetcher=fetcher, engine=engine)
    return app, fetcher, uri


def test_health_ok_model_identity() -> None:
    app = create_app(settings=_settings(), pcap_fetcher=FakePcapFetcher(
        {},
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=1024,
    ))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v1_hgb22_nontemporal",
        "serving_contract_version": "v1",
        "model_sha256": EXPECTED_MODEL_SHA256,
    }


def test_predict_requires_exactly_one_instance() -> None:
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
        assert client.post("/predict", json={"instances": []}).status_code == 422
        assert client.post(
            "/predict",
            json={
                "instances": [
                    {"gcs_uri": "gs://iomt-input/pcaps/a.pcap"},
                    {"gcs_uri": "gs://iomt-input/pcaps/b.pcap"},
                ]
            },
        ).status_code == 422


def test_predict_http_equals_direct_classify_pcap(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE * 3)
    engine = V1InferenceEngine.load_default()
    direct = _omit_nones(classify_pcap(pcap, engine=engine).to_dict())

    app, _, uri = _app_for_pcap(pcap, engine=engine)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 200
    http_pred = response.json()["predictions"][0]
    assert http_pred == direct
    assert id(client.app.state.engine) == id(engine)


def test_predict_insufficient_data_is_http_200(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=10)  # < 3 complete windows
    app, _, uri = _app_for_pcap(pcap)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 200
    pred = response.json()["predictions"][0]
    assert pred["status"] == "INSUFFICIENT_DATA"
    assert "prediction" not in pred  # exclude_none
    assert pred["window_summary"]["total_windows"] < 3


def test_predict_corrupt_pcap_is_422(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pcap"
    bad.write_bytes(b"not a pcap")
    app, _, uri = _app_for_pcap(bad)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 422
    assert "predictions" not in response.json()


def test_predict_unsupported_linktype_is_422(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE * 3, linktype=101)
    app, _, uri = _app_for_pcap(pcap)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 422
    assert "linktype" in response.json()["detail"].lower()


def test_predict_malformed_gs_uri_is_422(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE)
    app, _, _ = _app_for_pcap(pcap)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"instances": [{"gcs_uri": "https://example.com/x.pcap"}]},
        )
    assert response.status_code == 422


def test_predict_wrong_bucket_is_403(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE)
    settings = _settings()
    uri = "gs://other-bucket/pcaps/example.pcap"
    fetcher = FakePcapFetcher(
        {uri: pcap},
        input_bucket=settings.input_bucket,
        input_prefix=settings.input_prefix,
        max_pcap_bytes=settings.max_pcap_bytes,
    )
    app = create_app(settings=settings, pcap_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 403
    assert "bucket not allowed" in response.json()["detail"]


def test_predict_wrong_prefix_is_403(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE)
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
    assert response.status_code == 403
    assert "prefix not allowed" in response.json()["detail"]


def test_predict_object_missing_is_404(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE)
    settings = _settings()
    uri = f"gs://{settings.input_bucket}/{settings.input_prefix}missing.pcap"
    fetcher = FakePcapFetcher(
        {},  # not mapped
        input_bucket=settings.input_bucket,
        input_prefix=settings.input_prefix,
        max_pcap_bytes=settings.max_pcap_bytes,
    )
    app = create_app(settings=settings, pcap_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 404
    assert pcap.exists()  # local source unused


def test_predict_permission_denied_is_403(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE)
    app, _, uri = _app_for_pcap(pcap, denied=True)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 403
    assert "permission denied" in response.json()["detail"].lower()


def test_predict_oversize_is_413(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE * 3)
    app, _, uri = _app_for_pcap(pcap, settings=_settings(max_pcap_bytes=10))
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


def test_temp_pcap_deleted_after_success(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE * 3)
    app, fetcher, uri = _app_for_pcap(pcap)
    with TestClient(app) as client:
        assert client.post("/predict", json={"instances": [{"gcs_uri": uri}]}).status_code == 200
    assert fetcher.last_destination is not None
    assert not fetcher.last_destination.exists()
    assert not fetcher.last_destination.parent.exists()


def test_temp_pcap_deleted_after_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pcap"
    bad.write_bytes(b"not a pcap")
    app, fetcher, uri = _app_for_pcap(bad)
    with TestClient(app) as client:
        assert client.post("/predict", json={"instances": [{"gcs_uri": uri}]}).status_code == 422
    assert fetcher.last_destination is not None
    assert not fetcher.last_destination.exists()
    assert not fetcher.last_destination.parent.exists()


def test_repeated_requests_reuse_same_engine(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE * 3)
    engine = V1InferenceEngine.load_default()
    app, _, uri = _app_for_pcap(pcap, engine=engine)
    with TestClient(app) as client:
        assert id(client.app.state.engine) == id(engine)
        for _ in range(3):
            assert (
                client.post("/predict", json={"instances": [{"gcs_uri": uri}]}).status_code
                == 200
            )
        assert id(client.app.state.engine) == id(engine)
        assert client.app.state.max_concurrent_predictions == MAX_CONCURRENT_PREDICTIONS


def test_prediction_has_no_model_sha_or_probability_language(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE * 3)
    app, _, uri = _app_for_pcap(pcap)
    with TestClient(app) as client:
        response = client.post("/predict", json={"instances": [{"gcs_uri": uri}]})
    assert response.status_code == 200
    pred = response.json()["predictions"][0]
    assert "model_sha256" not in pred
    assert "model_sha256" not in pred["model"]
    blob = json.dumps(pred).lower()
    assert "probability" not in blob
    assert "confidence" not in blob
    assert pred["model"]["score_semantics"] == "uncalibrated_model_score"


def test_predict_rejects_extra_fields(tmp_path: Path) -> None:
    pcap = _synthetic_pcap(tmp_path, n_packets=WINDOW_SIZE)
    app, _, uri = _app_for_pcap(pcap)
    with TestClient(app) as client:
        assert (
            client.post(
                "/predict",
                json={"instances": [{"gcs_uri": uri}], "extra": True},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/predict",
                json={"instances": [{"gcs_uri": uri, "local_path": "/tmp/x"}]},
            ).status_code
            == 422
        )


def test_predict_internal_fetch_500_is_sanitized(tmp_path: Path) -> None:
    """GCS/internal fetch failures must not leak exception text to clients."""

    class _BoomFetcher:
        def fetch(self, gcs_uri: str, destination: Path) -> Path:
            raise PcapFetchError(
                "GCS request failed: secret-bucket /path/to/artifact sha=abc",
                status_code=500,
            )

    settings = _settings()
    app = create_app(settings=settings, pcap_fetcher=_BoomFetcher())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"instances": [{"gcs_uri": "gs://iomt-input/pcaps/x.pcap"}]},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "internal serving error"
    assert "secret-bucket" not in response.text
    assert "artifact" not in response.text


def test_d2_complete_artifact_present() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "serving" / "v1" / "d2_complete.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "D2"
    assert payload["status"] == "complete"
    assert payload["max_concurrent_predictions"] == 1
    assert payload["http_direct_inference_parity"] == "passed"
    assert payload["temporary_cleanup"] == "passed"
    assert payload["ci"] == "passed"


def test_d3_complete_artifact_present() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "serving" / "v1" / "d3_complete.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "D3"
    assert payload["status"] == "complete"
    assert payload["python_version"] == "3.11"
    assert payload["scikit_learn_version"] == "1.9.0"
    assert payload["docker_base_image"] == "python:3.11-slim-bookworm"
    assert payload["docker_platform"] == "linux/amd64"
    assert payload["container_port"] == 8080
    assert payload["uvicorn_workers"] == 1
    assert payload["non_root_uid"] == 10001
    assert payload["model_sha256"] == EXPECTED_MODEL_SHA256
    assert payload["serving_contract_version"] == "v1"
    assert payload["docker_smoke"] == "passed"
    assert payload["http_direct_inference_parity"] == "passed"
    assert payload["startup_determinism"] == "passed"
    assert payload["image_tag"].startswith("iomt-ids:v1-")
    assert "Artifact Registry" in payload["image_digest_note"]
    assert "Vertex endpoint IDs" in payload["out_of_scope"]
    assert payload["record_status"] == "historical"
    assert payload["superseded_by"].endswith("d4_vertex_complete.json")
    assert payload["release_image"].startswith(
        "us-west1-docker.pkg.dev/iot-pcap-pipeline/iomt-serving/iomt-ids@sha256:"
    )


def test_d4_vertex_complete_artifact_present() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "serving"
        / "v1"
        / "d4_vertex_complete.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "D4"
    assert payload["status"] == "complete"
    assert payload["region"] == "us-west1"
    assert payload["container_platform"] == "linux/amd64"
    assert payload["scikit_learn_version"] == "1.9.0"
    assert payload["model_sha256"] == EXPECTED_MODEL_SHA256
    assert payload["serving_contract_version"] == "v1"
    digest = payload["artifact_registry_digest"]
    assert digest.startswith("sha256:")
    assert payload["release_image"].endswith(digest)
    assert payload["vertex_model_id"]
    assert str(payload["vertex_model_version"]) == "2"
    assert payload["vertex_endpoint_id"]
    assert payload["endpoint_smoke"] == "passed"
    assert payload["cloud_vs_local_parity"] == "passed"
    blob = json.dumps(payload)
    assert "BEGIN PRIVATE KEY" not in blob
    assert "private_key" not in blob.lower()
    secrets = payload.get("secrets_excluded", [])
    assert "service_account_key" in secrets
    assert "access_token" in secrets
