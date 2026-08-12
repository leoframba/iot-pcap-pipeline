"""FastAPI application: load frozen V1 engine once at startup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from iot_pcap_pipeline.api.schemas import PredictRequest, PredictResponse
from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import GcsPcapFetcher, PcapFetchError, PcapFetcher
from iot_pcap_pipeline.serving.classify import classify_pcap
from iot_pcap_pipeline.serving.model import V1InferenceEngine


def create_app(
    *,
    project_root: Path | str | None = None,
    settings: ServingSettings | None = None,
    pcap_fetcher: PcapFetcher | None = None,
) -> FastAPI:
    """Build the serving app.

    The inference engine is loaded during lifespan startup (contract + SHA +
    class/feature checks). Startup fails hard if verification fails so a
    broken model never looks healthy.

    Inject ``settings`` / ``pcap_fetcher`` for tests (e.g. FakePcapFetcher).
    Production defaults: ``ServingSettings.from_env()`` + ``GcsPcapFetcher``.
    """
    root = Path(project_root).resolve() if project_root is not None else None
    resolved_settings = settings if settings is not None else ServingSettings.from_env()
    resolved_fetcher: PcapFetcher
    if pcap_fetcher is not None:
        resolved_fetcher = pcap_fetcher
    else:
        resolved_fetcher = GcsPcapFetcher(
            input_bucket=resolved_settings.input_bucket,
            input_prefix=resolved_settings.input_prefix,
            max_pcap_bytes=resolved_settings.max_pcap_bytes,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        app.state.pcap_fetcher = resolved_fetcher
        app.state.engine = V1InferenceEngine.load_default(project_root=root)
        yield

    app = FastAPI(
        title="iot-pcap-pipeline",
        version="v1",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        engine: V1InferenceEngine = request.app.state.engine
        model = engine.contract.get("model") or {}
        return {
            "status": "ok",
            "model_version": model.get("model_version"),
            "serving_contract_version": engine.contract.get(
                "serving_contract_version"
            ),
            "model_sha256": engine.model_sha256,
        }

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        engine: V1InferenceEngine = request.app.state.engine
        fetcher: PcapFetcher = request.app.state.pcap_fetcher
        gcs_uri = body.instances[0].gcs_uri

        with TemporaryDirectory(prefix="iomt-pcap-") as tmp:
            destination = Path(tmp) / "input.pcap"
            try:
                fetcher.fetch(gcs_uri, destination)
            except PcapFetchError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            result = classify_pcap(destination, engine=engine)
            return PredictResponse.from_classify_dict(result.to_dict())

    return app


# Default ASGI entrypoint: ``uvicorn iot_pcap_pipeline.api.app:app``.
app = create_app()
