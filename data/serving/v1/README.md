# D0 — V1 serving contract (research / selection)

Selection evidence for PCAP-level aggregation lives here.
Frozen checkout-ready contract: `artifacts/v1/serving_contract.json`
(**K=3**, **R=0.005**, **minimum_complete_windows=3**).

| File | Role |
|------|------|
| `serving_contract_draft.json` | Draft semantics (K/R pending) |
| `pcap_aggregation_candidates.json` | Predeclared 12-policy grid + selection priority |
| `pcap_aggregation_by_pcap.csv` | Per-VAL-PCAP window counts + policy predictions |
| `pcap_aggregation_summary.csv` | Policy-level summary metrics (ranked) |
| `pcap_aggregation_review.json` | Recommended policy + freeze gate status |
| `d0_complete.json` | D0 closure marker (engineering justification recorded) |
| `d1_complete.json` | D1 local inference closure + parity status |

```bash
uv run iot-pcap-pipeline evaluate-pcap-aggregation
```

**Frozen policy:** K=3, R=0.005, `minimum_complete_windows=3`  
(see `artifacts/v1/serving_contract.json`)

**Local inference:** `classify_pcap()` / `V1InferenceEngine.load_default()` under
`src/iot_pcap_pipeline/serving/`. Prediction `model` block matches the frozen
schema (`model_version`, `serving_contract_version`, `score_semantics` only).

**Rules**

- Window model, 22 inputs, and window threshold `0.9490790963172913` are frozen.
- Scores are uncalibrated decision scores — never call them probabilities.
- Choose K/R on TRAIN-validation only. Do not use TEST to select.
- After freeze, do not adjust K/R if TEST characterization disappoints.
