#!/usr/bin/env python3
"""Vertex AI rawPredict smoke for the V1 IoMT IDS endpoint.

Requires Application Default Credentials. Does not read or write keys.

Environment:
  ENDPOINT_ID   required Vertex endpoint id
  GCS_URI       required gs://bucket/object.pcap (must be on the allowlisted prefix)
  PROJECT_ID    default iot-pcap-pipeline
  REGION        default us-west1
  LOCAL_PCAP    optional path; when set, compare Vertex JSON to local classify_pcap()
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EXPECTED_MODEL_SHA256 = (
    "c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb"
)
DEFAULT_PROJECT = "iot-pcap-pipeline"
DEFAULT_REGION = "us-west1"


def _omit_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _omit_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_omit_nones(v) for v in value]
    return value


def _access_token() -> str:
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        from google.auth.transport.requests import Request

        try:
            credentials, _project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except DefaultCredentialsError:
            credentials = None
        if credentials is not None:
            credentials.refresh(Request())
            token = getattr(credentials, "token", None)
            if token:
                return str(token)
    except ImportError:
        pass

    import subprocess

    proc = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=False,
        text=True,
        capture_output=True,
    )
    token = (proc.stdout or "").strip()
    if proc.returncode == 0 and token:
        return token
    raise SystemExit(
        "No Application Default Credentials. Run gcloud auth application-default "
        "login, or ensure gcloud auth print-access-token works. Do not pass keys."
    )


def _raw_predict(
    *,
    project: str,
    region: str,
    endpoint_id: str,
    gcs_uri: str,
    token: str,
) -> tuple[int, Any]:
    url = (
        f"https://{region}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{region}/endpoints/{endpoint_id}:rawPredict"
    )
    body = json.dumps({"instances": [{"gcs_uri": gcs_uri}]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return int(exc.code), payload


def main() -> int:
    endpoint_id = os.environ.get("ENDPOINT_ID", "").strip()
    gcs_uri = os.environ.get("GCS_URI", "").strip()
    if not endpoint_id:
        raise SystemExit("ENDPOINT_ID is required")
    if not gcs_uri.startswith("gs://"):
        raise SystemExit("GCS_URI must be a gs:// URI")

    project = os.environ.get("PROJECT_ID", DEFAULT_PROJECT).strip() or DEFAULT_PROJECT
    region = os.environ.get("REGION", DEFAULT_REGION).strip() or DEFAULT_REGION
    local_pcap = os.environ.get("LOCAL_PCAP", "").strip()

    token = _access_token()
    code, payload = _raw_predict(
        project=project,
        region=region,
        endpoint_id=endpoint_id,
        gcs_uri=gcs_uri,
        token=token,
    )
    if code != 200:
        print(json.dumps({"ok": False, "http_status": code, "body": payload}, indent=2))
        return 1

    blob = json.dumps(payload).lower()
    if "probability" in blob:
        raise SystemExit("response contained forbidden alias 'probability'")

    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise SystemExit(f"expected one prediction, got {payload!r}")
    pred = predictions[0]
    model = pred.get("model") or {}
    if model.get("serving_contract_version") != "v1":
        raise SystemExit(f"serving_contract_version drift: {model}")
    if model.get("score_semantics") != "uncalibrated_model_score":
        raise SystemExit(f"score_semantics drift: {model}")
    if "model_sha256" in model:
        raise SystemExit("predict payload must not include model_sha256")

    result: dict[str, Any] = {
        "ok": True,
        "http_status": code,
        "endpoint_id": endpoint_id,
        "region": region,
        "gcs_uri": gcs_uri,
        "prediction": pred.get("prediction"),
        "status": pred.get("status"),
        "pcap_attack_score": pred.get("pcap_attack_score"),
        "window_summary": pred.get("window_summary"),
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
    }

    if local_pcap:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        from iot_pcap_pipeline.serving.classify import classify_pcap
        from iot_pcap_pipeline.serving.model import V1InferenceEngine

        engine = V1InferenceEngine.load_default(project_root=root)
        if engine.model_sha256 != EXPECTED_MODEL_SHA256:
            raise SystemExit(f"local model SHA drift: {engine.model_sha256}")
        direct = _omit_nones(classify_pcap(Path(local_pcap), engine=engine).to_dict())
        cloud = _omit_nones(pred)
        if cloud != direct:
            print(
                json.dumps(
                    {"ok": False, "cloud_vs_local_parity": "failed", "cloud": cloud, "local": direct},
                    indent=2,
                )
            )
            return 1
        result["cloud_vs_local_parity"] = "passed"
        result["local_pcap"] = local_pcap
    else:
        result["cloud_vs_local_parity"] = "skipped"
        result["local_pcap_note"] = "set LOCAL_PCAP to compare against classify_pcap()"

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
