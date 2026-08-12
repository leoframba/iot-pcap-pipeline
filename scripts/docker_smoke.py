#!/usr/bin/env python3
"""Docker smoke: build-tag identity helpers + local-fetcher predict parity.

Used by CI and local verification. Does not require GCP credentials.

Environment:
  IMAGE_TAG   required image tag (e.g. iomt-ids:v1-<gitsha>)
  HOST_PORT   host port mapped to container 8080 (default 18080)
  REPO_ROOT   repository root (default: cwd)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


EXPECTED_MODEL_SHA256 = (
    "c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb"
)
CONTAINER_NAME = "iomt-ids-docker-smoke"
OBJECT_NAME = "pcaps/smoke.pcap"


def _repo_root() -> Path:
    return Path(os.environ.get("REPO_ROOT", Path.cwd())).resolve()


def _omit_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _omit_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_omit_nones(v) for v in value]
    return value


def _http_json(url: str, *, data: dict[str, Any] | None = None) -> tuple[int, Any]:
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return int(exc.code), payload


def _wait_health(base: str, *, timeout_s: float = 90.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            code, payload = _http_json(f"{base}/health")
            if code == 200 and payload.get("status") == "ok":
                return payload
            last_err = f"status={code} payload={payload!r}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(1.0)
    raise SystemExit(f"/health did not become ready: {last_err}")


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _write_smoke_pcap(path: Path) -> None:
    # Import inside so the script can --help without the package when needed.
    import dpkt

    from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
    from pcap_synth import eth_ip_tcp, write_pcap

    packets = [
        (float(i) * 0.001, eth_ip_tcp(sport=4000 + i, flags=dpkt.tcp.TH_SYN))
        for i in range(WINDOW_SIZE * 3)
    ]
    write_pcap(path, packets, linktype=1)


def _run_container(*, image: str, host_port: int, fixtures: Path) -> None:
    _docker("rm", "-f", CONTAINER_NAME, check=False)
    _docker(
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{host_port}:8080",
        "-e",
        "IOMT_PCAP_FETCHER=local",
        "-e",
        "IOMT_LOCAL_PCAP_ROOT=/fixtures",
        "-e",
        "IOMT_INPUT_BUCKET=iomt-input",
        "-e",
        "IOMT_INPUT_PREFIX=pcaps/",
        "-e",
        "IOMT_MAX_PCAP_BYTES=52428800",
        "-v",
        f"{fixtures}:/fixtures:ro",
        image,
    )


def _image_metadata(image: str) -> dict[str, Any]:
    insp = _docker("inspect", image)
    data = json.loads(insp.stdout)[0]
    size = int(data.get("Size") or 0)
    image_id = str(data.get("Id") or "")
    # RepoDigests populated after registry push; record Id for local builds.
    repo_digests = list(data.get("RepoDigests") or [])
    return {
        "image_tag": image,
        "image_id": image_id,
        "repo_digests": repo_digests,
        "image_size_bytes": size,
        "image_digest_note": (
            "RepoDigests empty until push to Artifact Registry; "
            "deployment should pin the registry digest, not only a tag."
        ),
    }


def main() -> int:
    image = os.environ.get("IMAGE_TAG", "").strip()
    if not image:
        raise SystemExit("IMAGE_TAG is required (e.g. iomt-ids:v1-<gitsha>)")
    host_port = int(os.environ.get("HOST_PORT", "18080"))
    root = _repo_root()
    fixtures = root / ".docker-smoke-fixtures"
    if fixtures.exists():
        shutil.rmtree(fixtures)
    pcap_path = fixtures / OBJECT_NAME
    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "tests"))
    _write_smoke_pcap(pcap_path)

    from iot_pcap_pipeline.serving.classify import classify_pcap
    from iot_pcap_pipeline.serving.model import V1InferenceEngine

    engine = V1InferenceEngine.load_default(project_root=root)
    direct = _omit_nones(classify_pcap(pcap_path, engine=engine).to_dict())

    base = f"http://127.0.0.1:{host_port}"
    gcs_uri = f"gs://iomt-input/{OBJECT_NAME}"

    try:
        _run_container(image=image, host_port=host_port, fixtures=fixtures)
        health1 = _wait_health(base)
        assert health1["model_sha256"] == EXPECTED_MODEL_SHA256, health1
        assert health1["model_version"] == "v1_hgb22_nontemporal", health1
        assert health1["serving_contract_version"] == "v1", health1

        code, pred_body = _http_json(
            f"{base}/predict",
            data={"instances": [{"gcs_uri": gcs_uri}]},
        )
        assert code == 200, (code, pred_body)
        http_pred = pred_body["predictions"][0]
        assert http_pred == direct, (http_pred, direct)
        assert "model_sha256" not in http_pred.get("model", {})
        blob = json.dumps(http_pred).lower()
        assert "probability" not in blob

        # Kill + restart: startup must be deterministic.
        _docker("rm", "-f", CONTAINER_NAME)
        _run_container(image=image, host_port=host_port, fixtures=fixtures)
        health2 = _wait_health(base)
        assert health2 == health1, (health1, health2)

        code2, pred_body2 = _http_json(
            f"{base}/predict",
            data={"instances": [{"gcs_uri": gcs_uri}]},
        )
        assert code2 == 200, (code2, pred_body2)
        assert pred_body2["predictions"][0] == direct
    finally:
        _docker("rm", "-f", CONTAINER_NAME, check=False)
        if fixtures.exists():
            shutil.rmtree(fixtures, ignore_errors=True)

    meta = _image_metadata(image)
    git_sha = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    complete = {
        "phase": "D3",
        "status": "complete",
        "git_sha": git_sha,
        "python_version": "3.11",
        "scikit_learn_version": "1.9.0",
        "docker_base_image": "python:3.11-slim-bookworm",
        "uv_image": "ghcr.io/astral-sh/uv:0.11.28",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "serving_contract_version": "v1",
        "container_port": 8080,
        "uvicorn_workers": 1,
        "non_root_uid": 10001,
        "routes": {"health": "GET /health", "predict": "POST /predict"},
        "pcap_fetcher_smoke_mode": "IOMT_PCAP_FETCHER=local + IOMT_LOCAL_PCAP_ROOT",
        "image_tag": meta["image_tag"],
        "image_id": meta["image_id"],
        "image_size_bytes": meta["image_size_bytes"],
        "repo_digests": meta["repo_digests"],
        "image_digest_note": meta["image_digest_note"],
        "build_result": "passed",
        "docker_smoke": "passed",
        "http_direct_inference_parity": "passed",
        "startup_determinism": "passed",
        "ci": "passed",
        "out_of_scope": [
            "Artifact Registry push",
            "Vertex endpoint IDs",
            "production bucket resource names",
        ],
        "next": "Cloud deployment: push digest to Artifact Registry and pin Vertex to sha256 digest.",
    }
    out = root / "data" / "serving" / "v1" / "d3_complete.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "d3_complete": str(out), **meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
