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

## Phase 1B.3 (current)

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

Raw PCAPs under `data/raw/` are immutable source data and must not be modified.
