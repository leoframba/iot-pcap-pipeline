# CICIoMT Binary IDS — PCAP preprocessing pipeline

Phase 1 targets a deterministic, reusable conversion path:

```text
PCAP → packet parsing → fixed windows → clean network features → training-ready dataset
```

## Dataset scope

`dataset_scope = wifi_mqtt`

This repository currently inventories and will convert only the **Wi-Fi / MQTT**
slice of CICIoMT2024 located under `data/raw/WiFI_and_MQTT/`.

**Bluetooth PCAPs are intentionally out of scope for the V1 binary classifier.**
They are not an accidental omission; the paper’s CSV feature-conversion work also
focuses on Wi-Fi and MQTT because of the nature of those features.

## Phase 1A

Phase 1A builds metadata manifests only:

- `data/manifests/pcap_inventory.csv`
- `data/manifests/dataset_split.csv`

```bash
uv run iot-pcap-pipeline build-manifests
```

## Phase 1B.1

Streaming DPKT reader + normalized `PacketRecord` decoder.

```bash
uv run iot-pcap-pipeline inspect-pcaps path/to/file.pcap --max-packets 20000
```

## Phase 1B.2

Full-corpus parser/integrity audit and TRAIN-only behavioral characterization:

```bash
uv run iot-pcap-pipeline audit-corpus --workers 4 --resume
```

Artifacts:

- `data/audit/pcap_integrity.csv` — all manifested PCAPs (TRAIN and TEST)
- `data/audit/training_characterization.csv` — TRAIN PCAPs only
- `data/audit/audit_issues.csv` — corpus/file/packet observations
- `data/audit/.work/` — atomic per-PCAP checkpoints for resume

Execution notes:

- PCAPs are scanned with a process pool (default 4 workers; `--workers 1` is sequential)
- Files are scheduled largest-first for runtime balance; final CSVs remain canonically sorted
- Valid checkpoints are reused; changing file size / policy / strategy forces a rescan
- Live per-file packet progress goes to stderr every `--progress-every` packets (default 250k)

To watch progress while redirecting stderr:

```bash
mkdir -p data/audit
uv run iot-pcap-pipeline audit-corpus --workers 4 --resume \
  2> data/audit/audit_progress.log \
  | tee data/audit/audit_summary.log
# elsewhere:
tail -f data/audit/audit_progress.log
```

**Methodological guardrail:** `pcap_integrity.csv` may include TEST protocol
counts solely for parser/integrity validation. Feature selection and behavioral
comparisons for Phase 1C must use `training_characterization.csv` only. Do not
choose features by comparing held-out TEST distributions.

## Phase 1B.3

Timestamp-only ordering probe. Reads PCAPs in original capture order, ignores
frame buffers, and measures adjacent timestamp deltas (positive / duplicate /
negative) so Phase 1C can choose an IAT / rate policy with evidence.

```bash
uv run iot-pcap-pipeline probe-timestamps \
  data/raw/WiFI_and_MQTT/attacks/pcap/train/Benign_train.pcap \
  data/raw/WiFI_and_MQTT/attacks/pcap/test/Benign_test.pcap \
  data/raw/WiFI_and_MQTT/profiling/PCAP/Broker/ActiveBroker.pcap \
  data/raw/WiFI_and_MQTT/profiling/PCAP/Interactions/M1T_Camera/M1T_Camera_WAN_RECORDING.pcap \
  data/raw/WiFI_and_MQTT/profiling/PCAP/Idle/Idle.pcap \
  data/raw/WiFI_and_MQTT/attacks/pcap/train/MQTT-DDoS-Connect_Flood_train.pcap \
  data/raw/WiFI_and_MQTT/attacks/pcap/test/Recon-VulScan_test.pcap \
  data/raw/WiFI_and_MQTT/attacks/pcap/test/TCP_IP-DoS-SYN_test.pcap \
  data/raw/WiFI_and_MQTT/attacks/pcap/train/TCP_IP-DDoS-SYN2_train.pcap \
  data/raw/WiFI_and_MQTT/attacks/pcap/train/TCP_IP-DDoS-ICMP2_train.pcap
```

Artifacts:

- `data/audit/timestamp_probe.csv` — one row per probed PCAP
- `data/audit/timestamp_reversal_examples.csv` — bounded first-N and
  largest-N (by magnitude) negative events (`example_kind` column)

Notes:

- No Ethernet/IP/TCP decoding; timestamps only
- Negative-delta percentiles are exact; positive-delta percentiles use a
  deterministic reservoir sample when the positive count exceeds
  `--positive-sample-cap` (default 100,000) — method is recorded in the CSV
- Optional run-length stats: `negative_run_count` / `max` / `mean`
- Diagnostic only — not an ML training input
- Prefer PCAPs with high `negative_delta_count` from Phase 1B.2; any path works

Interpretation focus columns: `negative_delta_count`, `negative_delta_ratio`,
`negative_delta_p50_magnitude`, `negative_delta_p95_magnitude`,
`negative_delta_p99_magnitude`, `negative_delta_max_magnitude`.

## Phase 1C.1 — Gate A (passed)

TRAIN-only windowing-policy characterization. Timestamp-only (no frame decode).
Each TRAIN PCAP was scanned **once** while all six candidate configs updated
concurrently. Artifact:

- `data/features/windowing_characterization_train.csv` — 85 TRAIN PCAPs × 6
  policies (`phase1c1_v2`)

**Frozen V1 windowing policy (Gate A):**

```text
WINDOW_SIZE = 25
INACTIVITY_TIMEOUT_SECONDS = 5.0
BACKWARD_RESET_SECONDS = 1.0
```

Semantics:

- `delta > 5s` → close segment; drop incomplete window
- `delta < -1s` → close segment; drop incomplete window
- `-1s <= delta < 0` → keep packet (sanitized IAT → 0 in later phases)
- `delta == 0` → keep packet; IAT = 0
- full window → exactly 25 non-overlapping packets
- window span → `max(timestamp) - min(timestamp)`

Rationale (summary): 25/5 maximizes detection responsiveness vs 50/100, keeps
benign profiling retention strong except sparse power (accepted), and avoids
30s windows that can span minutes. ~2.95% zero-span windows at size 25 are
tolerated; V1 will not rely on packets/sec or bytes/sec.

```bash
uv run iot-pcap-pipeline characterize-windowing --workers 4
```

## Phase 1C.2 — Gate B (passed)

V1 production window engine + 27-feature extractor using the frozen Gate-A
policy. Same core path for offline and future inference:

```text
iter_packets → iter_windows → extract_features
```

**Frozen feature contract:** `FEATURE_STRATEGY_VERSION = phase1c2_v1`
(exactly 27 ordered numeric features; see `data/features/v1/feature_schema.json`).

```bash
# Write schema only
uv run iot-pcap-pipeline extract-features --write-schema-only

# Representative TRAIN smoke (capped windows per PCAP)
uv run iot-pcap-pipeline extract-features --smoke --max-windows-per-pcap 10000

# Arbitrary PCAP (no labels required)
uv run iot-pcap-pipeline extract-features path/to/capture.pcap --output /tmp/feats.csv
```

Artifacts:

- `data/features/v1/feature_schema.json` — frozen V1 contract
- `data/features/v1/smoke/train_features_smoke.csv` — diagnostic TRAIN smoke
- `data/features/v1/smoke/training_feature_characterization.csv`

Parse-status policy:

- `OK` / `UNSUPPORTED` / `PARTIAL` / `MALFORMED` → included in windows
- `ERROR` → abort extraction for that PCAP

Notes carried into Phase 1C.3:

- `tcp_urg_ratio` was constant-zero in the smoke set but remains in V1.
  After the full TRAIN feature build, report globally constant features.
  Dropping a constant from the **model input** requires an explicit
  pre-training schema/model-contract decision; TEST must not decide it.
- Full-corpus extraction must stream windows to Parquet. Do not reuse the
  smoke helper that accumulates all rows in a Python list.

## Phase 1C.3a — Streaming Parquet smoke

Storage layer for the frozen `phase1c2_v1` extractor. Feature values and
windowing are unchanged; only the write path is new.

```text
PCAP → iter_packets → iter_windows → extract_features → buffer → Parquet
```

- `FEATURE_STRATEGY_VERSION = phase1c2_v1` (unchanged)
- `FEATURE_BUILD_STRATEGY_VERSION = phase1c3_v1` (build/storage contract)
- Atomic shards: write `<pcap-id>.parquet.tmp`, then `os.replace` to final
- Resume checkpoints under `data/features/v1/.work/train/`
- `pcap_id` is path-stable: `<stem>-<sha256(repo-relative-path)[:16]>`
- Checkpoint identity includes `pcap_path`, `binary_label`, `output_path`,
  and `output_file_size` (label corrections force rebuild)

```bash
# Unit tests (storage contract)
uv run pytest tests/test_features_parquet.py -q

# TRAIN-only real smoke (Benign_train, Idle, Recon-VulScan_train) + resume check
uv run iot-pcap-pipeline build-feature-parquet --smoke
```

Artifacts:

- `data/features/v1/smoke/parquet/<pcap-id>.parquet`
- `data/features/v1/.work/train/<pcap-id>.json`

Not in 1C.3a: full 85-PCAP TRAIN, TEST extraction, multi-worker scheduler,
global manifests, constant-feature report, or model training.

## Phase 1C.3b step 1 — TRAIN corpus orchestration

Multiprocess wrapper around the frozen single-PCAP Parquet builder.

```text
inventory → 85 TRAIN PCAPs → size check → largest-first
  → ProcessPoolExecutor → build_pcap_parquet → build_manifest.csv
```

```bash
# Orchestration smoke (6 modest TRAIN PCAPs, workers=4)
uv run iot-pcap-pipeline build-feature-dataset --split train --workers 4 --resume --smoke

# Full TRAIN corpus (do not start until smoke is green)
uv run iot-pcap-pipeline build-feature-dataset --split train --workers 4 --resume
```

Artifacts (full TRAIN):

- `data/features/v1/train/<pcap-id>.parquet` (gitignored)
- `data/features/v1/.work/train/<pcap-id>.json`
- `data/features/v1/build_manifest.csv` (deterministic, sorted by `pcap_path`)

Smoke artifacts:

- `data/features/v1/smoke/dataset/<pcap-id>.parquet`
- `data/features/v1/.work/smoke/<pcap-id>.json`
- `data/features/v1/smoke/build_manifest_smoke.csv`
Not yet: TRAIN-wide feature statistics, constant-feature report, TEST
extraction, class balancing, or model training.

## Phase 1C.3b — validate TRAIN build (read-only)

After the full 85-PCAP TRAIN Parquet build, validate shards against the
manifest, audit integrity packet counts, and frozen 25/5s/1s windowing
characterization. Streams feature columns only (no PCAP decode).

```bash
uv run iot-pcap-pipeline validate-feature-dataset --split train
```

Artifacts:

- `data/features/v1/train_feature_summary.csv`
- `data/features/v1/train_constant_features.csv`
- `data/features/v1/train_build_complete.json` — written **only** if all checks pass

Constant features (e.g. `tcp_urg_ratio`) are reported only; do not drop from
V1 without an explicit pre-training contract. TEST is not consulted.

## Phase 1C.3b — TRAIN per-group characterization (read-only)

After validation, characterize feature distributions over existing Parquet
shards (no PCAP decode). Groups: all TRAIN, publisher benign, profiling
types, attack families, and attack types.

```bash
uv run iot-pcap-pipeline characterize-feature-dataset --split train
```

Artifacts:

- `data/features/v1/train_feature_group_summary.csv`
- `data/features/v1/train_feature_pcap_diagnostics.csv`
- `data/features/v1/train_feature_group_characterization.json`

Do not drop features from this report alone.

## Phase 1C.3b step 2 (next)

TEST extraction after TRAIN validation / characterization review.

Raw PCAPs under `data/raw/` are immutable source data and must not be modified.
