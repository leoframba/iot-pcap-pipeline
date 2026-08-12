# V1 serving artifacts

Frozen binary-IDS package for reproducible Docker / serving builds.

| File | Role |
|------|------|
| `H0_full_fit.joblib` | Frozen HistGradientBoostingClassifier (H0, 22 nontemporal features) |
| `v1_model_package.json` | Threshold, feature list, SHA-256, decision rule |
| `v1_hgb22_nontemporal.json` | Model-input feature contract |
| `feature_schema.json` | Full V1 27-feature schema (serving selects the 22) |
| `serving_contract.json` | Frozen PCAP aggregation (K=3, R=0.005, min windows=3) |

## Immutable model identity

```text
SHA-256(H0_full_fit.joblib) =
  c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb
```

Decision threshold (frozen): `0.9490790963172913`  
Rule: ATTACK if `score >= threshold`

PCAP aggregation (frozen in `serving_contract.json`):

- `pcap_attack_score = attack_windows / total_complete_windows`
- `INSUFFICIENT_DATA` if `total_complete_windows < 3`
- ATTACK iff `attack_windows >= 3` AND `pcap_attack_score >= 0.005`

Research training path (gitignored under `data/modeling/`):

`data/modeling/v1/hgb_sensitivity/phase2c1_v1/models/H0_full_fit.joblib`

This directory is the **canonical checkout-ready copy** for deployment. Research
JSON under `data/modeling/v1/` remains the evaluation provenance; serving and
Docker should load from `artifacts/v1/`.

## Fresh clone → verify → build

```bash
# After clone (artifacts/v1 is tracked in Git; ~720KB joblib)
shasum -a 256 artifacts/v1/H0_full_fit.joblib
# must equal c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb

python -c "
import json, hashlib, pathlib
p = pathlib.Path('artifacts/v1/H0_full_fit.joblib')
digest = hashlib.sha256(p.read_bytes()).hexdigest()
pkg = json.loads(pathlib.Path('artifacts/v1/v1_model_package.json').read_text())
assert digest == pkg['model_artifact_sha256'], (digest, pkg['model_artifact_sha256'])
print('ok', digest)
"
```

If the binary is ever moved to object storage, keep the same SHA in
`v1_model_package.json` and document the URI here; the Docker build must still
resolve to this exact digest before packaging.

## Serving import guidance

Import decoder / windowing / extractor / serving modules directly. Do **not**
import `iot_pcap_pipeline.cli` or research Parquet builders in the serving image:

```python
from iot_pcap_pipeline.serving import V1InferenceEngine, classify_pcap

engine = V1InferenceEngine.load_default()  # once per process
result = classify_pcap("capture.pcap", engine=engine)
```

`serving/contract.py` depends only on stdlib + `artifacts/v1/*.json` (no PyArrow).

The CLI entrypoint is `iot_pcap_pipeline.cli:main` (research only).
