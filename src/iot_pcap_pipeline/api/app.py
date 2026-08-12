"""FastAPI application: load frozen V1 engine once at startup."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from iot_pcap_pipeline.api.schemas import PredictRequest, PredictResponse
from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import (
    GcsPcapFetcher,
    LocalDirectoryPcapFetcher,
    PcapFetchError,
    PcapFetcher,
)
from iot_pcap_pipeline.serving.classify import classify_pcap
from iot_pcap_pipeline.serving.errors import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_INVALID_INPUT,
    STATUS_OK,
    STATUS_UNSUPPORTED_INPUT,
    ServingError,
)
from iot_pcap_pipeline.serving.model import V1InferenceEngine

logger = logging.getLogger("iot_pcap_pipeline.api")

# D2.12 / D2.13: single in-process prediction at a time until D4 benchmarking.
MAX_CONCURRENT_PREDICTIONS = 1
INTERNAL_SERVING_ERROR_DETAIL = "internal serving error"

_HTTP_INFERENCE_ERROR_STATUSES = frozenset(
    {STATUS_INVALID_INPUT, STATUS_UNSUPPORTED_INPUT}
)
_HTTP_SUCCESS_STATUSES = frozenset({STATUS_OK, STATUS_INSUFFICIENT_DATA})


def create_app(
    *,
    project_root: Path | str | None = None,
    settings: ServingSettings | None = None,
    pcap_fetcher: PcapFetcher | None = None,
    engine: V1InferenceEngine | None = None,
    max_concurrent_predictions: int = MAX_CONCURRENT_PREDICTIONS,
) -> FastAPI:
    """Build the serving app.

    The inference engine is loaded during lifespan startup (contract + SHA +
    class/feature checks). Startup fails hard if verification fails so a
    broken model never looks healthy.

    Inject ``settings`` / ``pcap_fetcher`` / ``engine`` for tests
    (e.g. FakePcapFetcher). Production defaults: ``ServingSettings.from_env()``
    + ``GcsPcapFetcher`` + ``V1InferenceEngine.load_default()``.
    """
    if max_concurrent_predictions < 1:
        raise ValueError("max_concurrent_predictions must be >= 1")

    root = Path(project_root).resolve() if project_root is not None else None
    resolved_settings = settings if settings is not None else ServingSettings.from_env()
    resolved_fetcher: PcapFetcher
    if pcap_fetcher is not None:
        resolved_fetcher = pcap_fetcher
    elif resolved_settings.pcap_fetcher == "local":
        resolved_fetcher = LocalDirectoryPcapFetcher(
            resolved_settings.local_pcap_root or "",
            input_bucket=resolved_settings.input_bucket,
            input_prefix=resolved_settings.input_prefix,
            max_pcap_bytes=resolved_settings.max_pcap_bytes,
        )
    else:
        resolved_fetcher = GcsPcapFetcher(
            input_bucket=resolved_settings.input_bucket,
            input_prefix=resolved_settings.input_prefix,
            max_pcap_bytes=resolved_settings.max_pcap_bytes,
        )
    injected_engine = engine
    predict_semaphore = threading.Semaphore(int(max_concurrent_predictions))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        app.state.pcap_fetcher = resolved_fetcher
        app.state.predict_semaphore = predict_semaphore
        app.state.max_concurrent_predictions = int(max_concurrent_predictions)
        if injected_engine is not None:
            app.state.engine = injected_engine
        else:
            app.state.engine = V1InferenceEngine.load_default(project_root=root)
        yield

    app = FastAPI(
        title="iot-pcap-pipeline",
        version="v1",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        eng: V1InferenceEngine = request.app.state.engine
        model = eng.contract.get("model") or {}
        return {
            "status": "ok",
            "model_version": model.get("model_version"),
            "serving_contract_version": eng.contract.get("serving_contract_version"),
            "model_sha256": eng.model_sha256,
        }

    @app.post(
        "/predict",
        response_model=PredictResponse,
        response_model_exclude_none=True,
    )
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        """Blocking predict (runs in FastAPI threadpool). Serializes via semaphore."""
        eng: V1InferenceEngine = request.app.state.engine
        fetcher: PcapFetcher = request.app.state.pcap_fetcher
        sem: threading.Semaphore = request.app.state.predict_semaphore
        gcs_uri = body.instances[0].gcs_uri
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        download_seconds: float | None = None
        inference_seconds: float | None = None
        pcap_bytes: int | None = None
        total_windows: int | None = None
        status: str | None = None
        prediction: str | None = None

        sem.acquire()
        try:
            with TemporaryDirectory(prefix="iomt-pcap-") as tmp:
                destination = Path(tmp) / "input.pcap"
                t_dl0 = time.perf_counter()
                try:
                    fetcher.fetch(gcs_uri, destination)
                except PcapFetchError as exc:
                    http_status = int(exc.status_code)
                    detail = (
                        INTERNAL_SERVING_ERROR_DETAIL
                        if http_status >= 500
                        else str(exc)
                    )
                    if http_status >= 500:
                        logger.error(
                            "predict fetch internal error request_id=%s: %s",
                            request_id,
                            exc,
                            exc_info=exc,
                        )
                    _log_predict(
                        request_id=request_id,
                        status="FETCH_ERROR",
                        prediction=None,
                        pcap_bytes=None,
                        download_seconds=time.perf_counter() - t_dl0,
                        inference_seconds=None,
                        total_seconds=time.perf_counter() - t0,
                        total_windows=None,
                        http_status=http_status,
                    )
                    raise HTTPException(status_code=http_status, detail=detail) from exc
                download_seconds = time.perf_counter() - t_dl0
                pcap_bytes = destination.stat().st_size

                t_inf0 = time.perf_counter()
                try:
                    result = classify_pcap(destination, engine=eng)
                except ServingError as exc:
                    logger.error(
                        "predict serving error request_id=%s: %s",
                        request_id,
                        exc,
                        exc_info=exc,
                    )
                    _log_predict(
                        request_id=request_id,
                        status="SERVING_ERROR",
                        prediction=None,
                        pcap_bytes=pcap_bytes,
                        download_seconds=download_seconds,
                        inference_seconds=time.perf_counter() - t_inf0,
                        total_seconds=time.perf_counter() - t0,
                        total_windows=None,
                        http_status=500,
                    )
                    raise HTTPException(
                        status_code=500, detail=INTERNAL_SERVING_ERROR_DETAIL
                    ) from exc
                except Exception as exc:  # noqa: BLE001 - map unexpected to 500
                    logger.error(
                        "predict unexpected error request_id=%s: %s",
                        request_id,
                        exc,
                        exc_info=exc,
                    )
                    _log_predict(
                        request_id=request_id,
                        status="UNEXPECTED_ERROR",
                        prediction=None,
                        pcap_bytes=pcap_bytes,
                        download_seconds=download_seconds,
                        inference_seconds=time.perf_counter() - t_inf0,
                        total_seconds=time.perf_counter() - t0,
                        total_windows=None,
                        http_status=500,
                    )
                    raise HTTPException(
                        status_code=500, detail=INTERNAL_SERVING_ERROR_DETAIL
                    ) from exc
                inference_seconds = time.perf_counter() - t_inf0

                status = result.status
                prediction = result.prediction
                total_windows = int(result.window_summary.get("total_windows") or 0)

                if status in _HTTP_INFERENCE_ERROR_STATUSES:
                    _log_predict(
                        request_id=request_id,
                        status=status,
                        prediction=prediction,
                        pcap_bytes=pcap_bytes,
                        download_seconds=download_seconds,
                        inference_seconds=inference_seconds,
                        total_seconds=time.perf_counter() - t0,
                        total_windows=total_windows,
                        http_status=422,
                    )
                    raise HTTPException(
                        status_code=422,
                        detail=result.detail or status,
                    )

                if status not in _HTTP_SUCCESS_STATUSES:
                    logger.error(
                        "predict unexpected classify status request_id=%s status=%s",
                        request_id,
                        status,
                    )
                    _log_predict(
                        request_id=request_id,
                        status=status,
                        prediction=prediction,
                        pcap_bytes=pcap_bytes,
                        download_seconds=download_seconds,
                        inference_seconds=inference_seconds,
                        total_seconds=time.perf_counter() - t0,
                        total_windows=total_windows,
                        http_status=500,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=INTERNAL_SERVING_ERROR_DETAIL,
                    )

                _log_predict(
                    request_id=request_id,
                    status=status,
                    prediction=prediction,
                    pcap_bytes=pcap_bytes,
                    download_seconds=download_seconds,
                    inference_seconds=inference_seconds,
                    total_seconds=time.perf_counter() - t0,
                    total_windows=total_windows,
                    http_status=200,
                )
                return PredictResponse.from_classify_dict(result.to_dict())
        finally:
            sem.release()

    return app


def _log_predict(
    *,
    request_id: str,
    status: str | None,
    prediction: str | None,
    pcap_bytes: int | None,
    download_seconds: float | None,
    inference_seconds: float | None,
    total_seconds: float,
    total_windows: int | None,
    http_status: int,
) -> None:
    # Intentionally omit GCS URIs, payloads, IPs, MACs, and feature vectors.
    logger.info(
        "predict request_id=%s http_status=%s status=%s prediction=%s "
        "pcap_bytes=%s download_seconds=%.6f inference_seconds=%.6f "
        "total_seconds=%.6f total_windows=%s",
        request_id,
        http_status,
        status,
        prediction,
        pcap_bytes,
        download_seconds if download_seconds is not None else -1.0,
        inference_seconds if inference_seconds is not None else -1.0,
        total_seconds,
        total_windows,
    )


# Default ASGI entrypoint: ``uvicorn iot_pcap_pipeline.api.app:app --workers 1``.
app = create_app()
