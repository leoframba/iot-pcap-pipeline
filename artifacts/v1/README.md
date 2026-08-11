# V1 serving artifacts

Frozen binary-IDS package for reproducible Docker / serving builds.

| File | Role |
|------|------|
| `H0_full_fit.joblib` | Frozen HistGradientBoostingClassifier (H0, 22 nontemporal features) |
| `v1_model_package.json` | Threshold, feature list, SHA-256, decision rule |
| `v1_hgb22_nontemporal.json` | Model-input feature contract |
| `feature_schema.json` | Full V1 27-feature schema (serving selects the 22) |

## Immutable model identity

```text
SHA-256(H0_full_fit.joblib) =
  c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb
```

Decision threshold (frozen): `0.9490790963172913`  
Rule: ATTACK if `score >= threshold`

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

Import decoder / windowing / extractor modules directly. Do **not** import
`iot_pcap_pipeline.cli` or research Parquet builders in the serving image:

```python
from iot_pcap_pipeline.pcap.decode import ...
from iot_pcap_pipeline.windowing.stream import ...
from iot_pcap_pipeline.features.extractor import extract_features
```

The CLI entrypoint is `iot_pcap_pipeline.cli:main` (research only).
