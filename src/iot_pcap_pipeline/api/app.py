"""FastAPI application: load frozen V1 engine once at startup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request

from iot_pcap_pipeline.serving.model import V1InferenceEngine


def create_app(*, project_root: Path | str | None = None) -> FastAPI:
    """Build the serving app.

    The inference engine is loaded during lifespan startup (contract + SHA +
    class/feature checks). Startup fails hard if verification fails so a
    broken model never looks healthy.
    """
    root = Path(project_root).resolve() if project_root is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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

    return app


# Default ASGI entrypoint: ``uvicorn iot_pcap_pipeline.api.app:app``.
app = create_app()
