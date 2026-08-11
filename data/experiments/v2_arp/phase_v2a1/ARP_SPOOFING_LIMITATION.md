# ARP Spoofing — documented detection limitation

**Status:** closed (do not pursue ARP IP↔MAC identity features on this corpus)  
**Experiment:** V2A phase `v2a1` (`data/experiments/v2_arp/phase_v2a1/`)  
**V1 package:** unchanged (extractor, 27-feature schema, HGB-22, threshold, TEST artifacts)

## Why this is a limitation

V1 TEST spoofing recall is **~8%** (`56 / 698`). Spoofing is about inconsistent IP↔MAC bindings. V1 never models those relationships; it only sees aggregate protocol ratios (including `arp_ratio`) and related volume/flag/cardinality statistics.

V2A asked whether *relationship-derived* ARP features could fix that **without** opening TEST for selection.

## What V2A checked (FIT only)

### A5 — Stateless features inside frozen 25-packet windows

Artifact: `arp_probe_complete.json` / `arp_feature_summary.csv`

- Spoofing FIT: sender-IP conflict features nonzero on **7 / 6,423** windows (~0.11%).
- Given ARP is present: conflict rate is still only **~1.1%** of spoofing ARP windows.
- Benign FIT has **equal or higher** `arp_ratio` than spoofing — failure is not “spoof PCAPs have more ARP.”

### A6 — Whole-PCAP / stateful feasibility (no production extractor)

Artifact: `arp_stateful_feasibility_complete.json`

| Group | Conflict IPs | Conflict obs ratio | Novel MAC claims |
|-------|--------------|--------------------|------------------|
| Spoofing FIT | **1** | **4.9%** | 1 |
| Publisher benign FIT | 15 | **42.9%** | 15 |
| Profiling benign FIT | 10 | **26.1%** | 10 |

On spoofing, the sole dual-MAC sender IP first appears **~5,464 packets / ~109 s** after the original claim. Later flip-flops exist on that IP, but most spoof ARP traffic never involves it.

Benign multi-MAC rates are **higher** than spoofing (ActiveBroker dominates profiling). Whole-PCAP multi-MAC history is therefore **not spoof-specific** on this FIT corpus.

## Decision

1. **Do not** add ARP identity features to the frozen V1 package.
2. **Do not** train a V2 model from A2–A6 ARP-identity candidates on this dataset.
3. Treat ARP Spoofing as a **documented detection limitation** of the current pipeline + CICIoMT Wi-Fi/MQTT captures used here.
4. Keep the existing fair deployment statement: V1 is strong on DDoS/DoS/MQTT floods/most recon at ~0.08% benign FPR, and **does not reliably detect ARP spoofing**.

## If spoofing detection is required later

That work needs a different evidence regime (captures where legitimate vs spoofed IP↔MAC bindings are actually separable), not more engineering around a signal this corpus does not expose cleanly. Prefer other V1 blind spots (e.g. MQTT malformed-data) for the next feature effort on this dataset.
