# V1 serving image for FastAPI / Vertex (port 8080, single Uvicorn worker).
# Always build for linux/amd64 so images match Vertex / GCE and do not depend
# on the developer machine architecture (e.g. Apple Silicon).
#
# Build:
#   docker build --platform=linux/amd64 -t iomt-ids:v1-<gitsha> .
# Run (local smoke without GCP):
#   docker run --rm -p 8080:8080 \
#     -e IOMT_PCAP_FETCHER=local \
#     -e IOMT_LOCAL_PCAP_ROOT=/fixtures \
#     -e IOMT_INPUT_BUCKET=iomt-input \
#     -e IOMT_INPUT_PREFIX=pcaps/ \
#     -v /path/to/fixtures:/fixtures:ro \
#     iomt-ids:v1-<gitsha>
#
# No project IDs, bucket secrets, or service-account JSON are baked into this image.
# Google ADC is discovered at runtime when deployed.

# Platform is set by `docker build --platform=linux/amd64` (CI and release),
# not a constant FROM --platform (BuildKit FromPlatformFlagConstDisallowed).
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

WORKDIR /app

# Non-root runtime user (no elevated privileges required).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Locked serving deps only (no research / no dev groups).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY artifacts/v1 ./artifacts/v1

RUN chown -R appuser:appuser /app
USER appuser

# Editable install keeps PROJECT_ROOT=/app so artifacts/v1 resolves correctly.
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_NO_DEV=1 \
  PATH="/app/.venv/bin:$PATH" \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

RUN uv sync --frozen --extra serving --no-dev

# D3.7 — fail the image build if frozen contract / model SHA are wrong.
RUN python -c "from iot_pcap_pipeline.serving.contract import EXPECTED_MODEL_SHA256, DEFAULT_MODEL_PATH, sha256_file, verify_serving_contract; from iot_pcap_pipeline.serving.model import V1InferenceEngine; doc = verify_serving_contract(); digest = sha256_file(DEFAULT_MODEL_PATH); assert digest == EXPECTED_MODEL_SHA256, (digest, EXPECTED_MODEL_SHA256); assert digest == doc['model']['model_artifact_sha256']; engine = V1InferenceEngine.load_default(); assert engine.model_sha256 == EXPECTED_MODEL_SHA256; assert len(engine.feature_names) == 22; print('ok build-time serving artifacts + engine load', digest[:16] + '…')"

# Runtime config is environment-only (examples; unset is fine — app has provisional defaults):
#   IOMT_INPUT_BUCKET
#   IOMT_INPUT_PREFIX
#   IOMT_MAX_PCAP_BYTES
# Docker/CI smoke without GCP:
#   IOMT_PCAP_FETCHER=local
#   IOMT_LOCAL_PCAP_ROOT=/fixtures
# Do not set GOOGLE_APPLICATION_CREDENTIALS to a baked-in key path.

EXPOSE 8080

# Application readiness is GET /health (also Vertex healthRoute). No curl in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).read()"

# D2 concurrency contract: one process, one loaded engine, one in-process semaphore.
CMD ["uvicorn", "iot_pcap_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
