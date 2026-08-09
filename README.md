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

## Phase 1B.2 (current)

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

Raw PCAPs under `data/raw/` are immutable source data and must not be modified.
