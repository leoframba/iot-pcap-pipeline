# V1 assessment — Phase 2D evaluation complete

**Status:** evaluation complete  
**Model:** frozen HGB-22 (H0)  
**Threshold:** `0.9490790963172913` (ATTACK iff `score >= threshold`)  
**TEST:** one-shot sealed evaluation on 29 PCAPs / 6,206,674 windows  
**Constraint:** do **not** adjust threshold, features, hyperparameters, or retrain against TEST

Source artifacts: `data/modeling/v1/final_test/phase2d_v1/`.

## Headline TEST result

At the frozen threshold the model scored every TEST window once and produced:

| Quantity | Value |
|----------|-------|
| TP / TN / FP / FN | 6,180,714 / 15,149 / 12 / 10,799 |
| Aggregate attack recall | 99.826% |
| Specificity | 99.921% |
| Benign FPR | 0.07915% |
| Macro attack-family recall (5 families) | 79.73% |
| Minimum attack-family recall | 8.02% (Spoofing) |

## Validation vs TEST (families with meaningful VAL coverage)

| Metric | Validation | TEST |
|--------|------------|------|
| Benign FPR | 0.0889% | 0.0792% |
| DDoS recall | 99.856% | 99.860% |
| DoS recall | 99.737% | 99.882% |
| MQTT recall | 99.288% | 99.484% |
| Recon recall | 86.891% | 91.392% |

On the four families with meaningful validation coverage, TEST generalizes at least as well as validation. Recon improves by ~4.5 percentage points. Benign FPR stays below the frozen ≤0.1% operating target.

## Blind spots

### Spoofing

ARP Spoofing recall is **56 / 698 = 8.02%**. That single family drives minimum-family recall to 8.02% and five-family macro recall to 79.73%.

Aggregate 99.826% attack recall is misleading: ~5.9M TEST attack windows are DDoS/DoS, where performance is nearly perfect. A binary-IDS claim based only on aggregate recall would hide the spoofing failure.

#### V2A follow-up (FIT only) — closed as a limitation

After V1 freeze, experiment `v2a1` tested whether ARP IP↔MAC *relationship* features could recover spoofing without touching TEST or altering the V1 package.

- **Stateless (25-packet windows):** conflict features fire on only **7 / 6,423** spoofing FIT windows. Benign has equal/higher `arp_ratio`, so this is not rediscovering “more ARP.”
- **Stateful (whole-PCAP feasibility):** only **1** spoofing sender IP ever claims two SHAs (**~4.9%** of spoof ARP observations). Publisher/profiling benign show **higher** multi-MAC rates (**~43% / ~26%**).

**Decision:** do not add ARP-identity features or train a V2 ARP-identity model on this corpus. Document spoofing as a detection limitation. Write-up: `data/experiments/v2_arp/phase_v2a1/ARP_SPOOFING_LIMITATION.md` (closure marker: `v2a_experiment_closed.json`).

### Attack-type heterogeneity

MQTT family average is excellent (**99.48%**), but **MQTT_Malformed_Data** is only **160 / 698 = 22.92%**.

#### V2M follow-up (FIT only) — closed after port-gated probe

Experiment `v2m1` / corrected `v2m1b` tested whether raw-TCP MQTT *structural* violations separate malformed from benign MQTT without touching TEST.

- **attempt1** parsed all TCP payloads and produced false INVALID MQTT on mixed benign traffic (especially non-MQTT TCP in `Active.pcap`).
- **v2m1b** gates parsing to FIT-pinned plaintext port **1883** only. Conditional `mqtt_invalid_ratio` is ~0 for malformed and benign; PUBLISH wildcards never appear on the wire.

**Decision:** do not add MQTT structural features or train a V2 structural-MQTT model on this corpus. Write-up: `data/experiments/v2_mqtt/phase_v2m1b/MQTT_STRUCTURAL_LIMITATION.md`.

Recon varies substantially:

| Type | Recall |
|------|--------|
| Port Scan | 95.17% |
| OS Scan | 83.77% |
| Ping Sweep | 72.97% |
| VulScan | 40.34% |

### Likely cause: V1 feature information, not HGB tuning

This pattern is consistent with a feature-information limit rather than an untuned HGB. ARP spoofing is fundamentally about inconsistent IP↔MAC identity/mapping; V1 deliberately does not model those relationships. Malformed MQTT may require protocol/payload validity signals the frozen extractor does not parse. The model can only infer those attacks indirectly from packet-size / protocol / flag / cardinality statistics.

## Benign performance (with caveat)

All **12** false positives came from the publisher benign TEST PCAP:
**12 / 15,101 = 0.0795% FPR**. The eight profiling PCAPs produced **zero** false positives.

Those eight profiling PCAPs generated only **60** total windows, and one profiling PCAP generated **zero** windows. Do **not** treat “0% profiling FPR” as evidence of strong device generalization — zero observed FPs in 60 windows is weak evidence.

## Fair deployment statement

V1 is **not** a comprehensive IoMT attack detector.

A fair statement is:

> V1 provides very strong detection of DDoS, DoS, MQTT flooding, and most reconnaissance traffic at approximately 0.08% observed benign FPR, but does not reliably detect ARP spoofing, MQTT malformed-data attacks, or some low-volume reconnaissance types.

## Closure rules

- Phase 2D evaluation: **complete**
- Frozen model / threshold / 22-feature selector / hashes: **unchanged**
- No threshold adjustment, feature experiment, or retraining against this TEST set
- V2A ARP-identity follow-up on FIT: **closed** — spoofing remains a documented limitation (`data/experiments/v2_arp/phase_v2a1/`)
- Further work (if any) belongs to a later version (V2+), designed without using TEST for selection; prefer other blind spots on this corpus rather than ARP IP↔MAC identity features
