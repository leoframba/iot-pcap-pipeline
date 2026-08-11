# MQTT structural features — closed after port-gated M4

**Status:** closed (`stop_mqtt_structural_features`)  
**Corrected probe:** `phase_v2m1b` (plaintext MQTT ports = `{1883}` only)  
**Superseded:** `phase_v2m1/attempt1` (ungated TCP false positives)

## What was wrong in attempt1

The extractor parsed **every** TCP payload as potential MQTT whenever the first-byte type nibble was 1–14. Ordinary HTTP/TLS-like traffic was misclassified as MQTT and often marked `INVALID`, which inflated benign “violation” rates (especially `Active.pcap`).

## What v2m1b found (FIT only, `:1883` gated)

Conditional on windows with MQTT frames:

| Group | mean `mqtt_invalid_ratio` | windows with any invalid |
|-------|---------------------------|--------------------------|
| mqtt_malformed | ~0.0000 | ~0.11% |
| publisher benign | ~0.0001 | ~0.04% |
| profiling benign | ~0.0001 | ~0.05% |

- `ActiveBroker.pcap`: still ~0 invalid (real MQTT).
- `Active.pcap`: invalid rate collapses once non-1883 TCP is ignored.
- `mqtt_publish_wildcard_topic_count`: **0** everywhere, including malformed.
- INCOMPLETE rates also collapse after the port gate.

## Decision

MQTT_Malformed_Data on this FIT corpus does **not** expose protocol-structural violations that are uncommon in benign plaintext MQTT under the defended checks. Do **not** add these features to V1 or train a V2 structural-MQTT model from this signal.

Keep V1’s known limitation: MQTT malformed-data recall ~23% on TEST remains a documented blind spot, not a feature-engineering target on this capture evidence.
