# D0 — V1 serving contract (research / selection)

Selection evidence for PCAP-level aggregation lives here.
The checkout-ready freeze will be `artifacts/v1/serving_contract.json`
(Commit 3, after TRAIN-validation review of K/R).

| File | Role |
|------|------|
| `serving_contract_draft.json` | Draft semantics (K/R pending) |
| `pcap_aggregation_candidates.json` | Predeclared 12-policy grid + selection priority |
| `pcap_aggregation_by_pcap.csv` | Per-VAL-PCAP window counts + policy predictions |
| `pcap_aggregation_summary.csv` | Policy-level summary metrics (ranked) |
| `pcap_aggregation_review.json` | Recommended policy + freeze gate status |
| `d0_complete.json` | Closure marker after freeze + tests (Commit 4) |

```bash
uv run iot-pcap-pipeline evaluate-pcap-aggregation
```

**Rules**

- Window model, 22 inputs, and window threshold `0.9490790963172913` are frozen.
- Scores are uncalibrated decision scores — never call them probabilities.
- Choose K/R on TRAIN-validation only. Do not use TEST to select.
- After freeze, do not adjust K/R if TEST characterization disappoints.
