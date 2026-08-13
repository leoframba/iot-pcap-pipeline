# Cloud deployment (Vertex AI)

V1 serving path:

```text
Artifact Registry → Vertex AI Model Registry → Vertex Endpoint → GCS input bucket
                 ↳ runtime service account (objectViewer on the bucket)
```

This document records the **successful** `us-west1` deployment. Resource IDs are not secrets. Do **not** commit service-account keys, tokens, or `.env` files.

Immutable identities: [`data/serving/v1/d4_vertex_complete.json`](../data/serving/v1/d4_vertex_complete.json).

The endpoint is **not** a public demo. Undeploy idle replicas when you are not testing (`gcloud ai endpoints undeploy-model`). Redeploy from the pinned digest below.

## Identities

| Resource | Value |
| --- | --- |
| GCP project | `iot-pcap-pipeline` (`123709655981`) |
| Region | `us-west1` |
| Artifact Registry repo | `iomt-serving` |
| Release image | `us-west1-docker.pkg.dev/iot-pcap-pipeline/iomt-serving/iomt-ids@sha256:68ea9a9a251cba9bdad1862ff1a90ef6947495fd1c02ce3fed3bcdb085564752` |
| Image tag (mutable) | `v1-amd64` |
| Container platform | `linux/amd64` |
| Vertex model | `2282511372472287232` **version 2** |
| Endpoint | `5077544697069568` (`iomt-ids-endpoint`) |
| Deployed model id | `6280157530182123520` |
| Machine | `n1-standard-2`, 1 replica |
| Runtime SA | `iomt-predictor@iot-pcap-pipeline.iam.gserviceaccount.com` |
| Input bucket | `iomt-pcap-input-123709655981` |
| Object prefix | `pcaps/` |
| Health / predict | `GET /health`, `POST /predict` on container port `8080` |

Older tag `v1` digest `sha256:a6e590b9…` is **not** the AMD64 release (first local/arm-era push). Vertex model **version 1** used that digest. Pin version **2** / the `68ea9a9a…` digest only.

Do not use tag `:latest`.

## 1. Build and push (linux/amd64)

From the repo root, authenticated to the project:

```bash
PROJECT=iot-pcap-pipeline
REGION=us-west1
AR_REPO=iomt-serving
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/iomt-ids"

gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT}" || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

docker build --platform linux/amd64 -t "${IMAGE}:v1-amd64" .
docker push "${IMAGE}:v1-amd64"
docker buildx imagetools inspect "${IMAGE}:v1-amd64"
```

Record the **digest** from the inspect output. Vertex must pin `@sha256:…`, not the tag.

## 2. GCS input bucket

```bash
PROJECT_NUMBER="$(gcloud projects describe iot-pcap-pipeline --format='value(projectNumber)')"
BUCKET="iomt-pcap-input-${PROJECT_NUMBER}"

gcloud storage buckets create "gs://${BUCKET}" --location=us-west1 --project=iot-pcap-pipeline
# Upload objects only under pcaps/
gcloud storage cp path/to/example.pcap "gs://${BUCKET}/pcaps/example.pcap"
```

The API allowlists this bucket + prefix. Arbitrary `gs://` URIs are rejected.

## 3. Runtime service account

The Vertex prediction container runs as a **dedicated** SA. Grant it `roles/storage.objectViewer` on the input bucket only. Do not download a JSON key; Vertex uses the attached SA.

```bash
gcloud iam service-accounts create iomt-predictor \
  --display-name="IoMT Vertex Predict runtime" \
  --project=iot-pcap-pipeline

gcloud storage buckets add-iam-policy-binding "gs://iomt-pcap-input-123709655981" \
  --member="serviceAccount:iomt-predictor@iot-pcap-pipeline.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

Also grant the Vertex service agent permission to act as this SA (`iam.serviceAccountUser`) when deploying.

## 4. Upload model to Vertex Model Registry

Custom container (no prebuilt sklearn runtime). Environment must match the bucket:

```bash
DIGEST="us-west1-docker.pkg.dev/iot-pcap-pipeline/iomt-serving/iomt-ids@sha256:68ea9a9a251cba9bdad1862ff1a90ef6947495fd1c02ce3fed3bcdb085564752"

gcloud ai models upload \
  --region=us-west1 \
  --display-name=iomt-ids \
  --container-image-uri="${DIGEST}" \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-ports=8080 \
  --container-env-vars="IOMT_INPUT_BUCKET=iomt-pcap-input-123709655981,IOMT_INPUT_PREFIX=pcaps/,IOMT_MAX_PCAP_BYTES=536870912"
```

Subsequent digest pins create a new **model version**. V1 cloud serving is version `2`.

## 5. Endpoint + deploy

```bash
gcloud ai endpoints create \
  --region=us-west1 \
  --display-name=iomt-ids-endpoint

gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-west1 \
  --model=2282511372472287232@2 \
  --display-name=iomt-ids \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --service-account=iomt-predictor@iot-pcap-pipeline.iam.gserviceaccount.com \
  --traffic-split=0=100
```

Replace `ENDPOINT_ID` with the created endpoint (recorded value: `5077544697069568`).

## 6. Smoke (same request as production)

Requires Application Default Credentials (user or SA), **not** a checked-in key:

```bash
export ENDPOINT_ID=5077544697069568
export GCS_URI=gs://iomt-pcap-input-123709655981/pcaps/example.pcap
# optional local file for cloud-vs-local parity:
# export LOCAL_PCAP=/path/to/the/same.pcap
uv sync --extra serving
uv run python scripts/vertex_smoke.py
```

`rawPredict` body:

```json
{"instances":[{"gcs_uri":"gs://iomt-pcap-input-123709655981/pcaps/example.pcap"}]}
```

## 7. Tear down (cost)

```bash
gcloud ai endpoints undeploy-model 5077544697069568 \
  --region=us-west1 \
  --deployed-model-id=6280157530182123520
```

The Artifact Registry image, Model Registry version, bucket, and this document remain the proof of deployment. Redeploy from the pinned digest when you need the endpoint again.

## What not to commit

- Service-account JSON keys
- `gcloud auth print-access-token` output
- `.env` with project tokens
- Live endpoint URLs advertised as a permanent public demo
