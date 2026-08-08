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

## Phase 1A (current)

Phase 1A builds metadata manifests only:

- `data/manifests/pcap_inventory.csv`
- `data/manifests/dataset_split.csv`

No packet parsing, windowing, feature extraction, or model training happens yet.

```bash
uv run iot-pcap-pipeline build-manifests
```

Raw PCAPs under `data/raw/` are immutable source data and must not be modified.
