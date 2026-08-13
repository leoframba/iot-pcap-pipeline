# Future work (V2)

V1 is a **deployment-oriented generalist**: raw PCAP → frozen 25-packet windows → 22 nontemporal features → HistGradientBoosting → PCAP-level BENIGN/ATTACK, served on Vertex AI.

That is a complete serving system for the attacks V1 can see. It is not a comprehensive IoMT detector. Aggregate TEST attack recall of **99.826%** is dominated by DDoS/DoS (~5.9M of the attack windows). The remaining failures are low-volume and protocol-semantic:

| Blind spot | TEST recall | What V1 cannot see |
| --- | --- | --- |
| ARP spoofing | 8.02% (56 / 698) | IP↔MAC identity / mapping |
| MQTT malformed | ~22.92% (160 / 698) | protocol / payload validity |
| Recon VulScan | 40.34% | sparse, low-volume recon |

V2 should treat those as a **research program**, not as “retrain HGB with better sampling.”

## Why resampling this corpus is not V2

Spoofing FIT has **6,423 windows from one spoofing lineage/capture**. Spoofing has **no independent TRAIN-validation lineage**. Reweighting or oversampling that capture changes its training mass; it does not:

- add new spoofing implementations, devices, or topologies
- create a validation lineage that is independent of FIT
- tell you whether a specialist generalizes

Flood traffic will still dominate any global loss unless sampling is designed **after** there is independent rare-class data to sample. The first V2 investment is additional independent captures for rare attack families **and** more realistic benign environments (V1’s eight profiling TEST PCAPs produced only 60 windows; “0% profiling FPR” is weak evidence).

Then evaluate alternative sampling so DDoS/DoS windows cannot drown the rare classes. Sampling is step two, not the whole plan.

## Architecture: generalist + specialists + gated fusion

Retraining the same 22-feature HGB is the least interesting V2. A more honest architecture keeps V1 (or its successor) as a flood/recon generalist and adds specialists only where the 25-packet nontemporal view is the wrong inductive bias:

```text
                         ┌─ generalist HGB ─────────┐
PCAP → shared parsing ───┼─ low-volume specialist ─┼→ gated/meta fusion → verdict
                         ├─ ARP/context specialist ─┤
                         └─ MQTT specialist ────────┘
```

- **Shared parsing** stays deterministic (DPKT, Ethernet-only unless the contract is explicitly reopened).
- **Generalist** keeps short windows and cheap nontemporal features for high-volume floods.
- **Low-volume specialist** may use longer windows, session/flow state, or rate-of-change features that 25-packet snapshots wash out (VulScan, ping sweep).
- **ARP/context specialist** is for identity/mapping attacks — **not** “add `arp_ratio`.”
- **MQTT specialist** is for protocol-semantic abuse — **not** “parse every TCP payload as MQTT.”

Specialist scores combine through a **gated or learned fusion** layer. A naïve OR of independent detectors compounds false positives (each specialist’s benign FPR adds). Fusion should see *when* a specialist is in-distribution (e.g. ARP present, MQTT on a known port) and otherwise defer to the generalist.

## Closed probes on *this* corpus (do not relitigate as V1 patches)

Post-freeze FIT-only probes already asked whether cheap relationship/structural features recover the blind spots **without** touching TEST. They did not.

**V2A (ARP IP↔MAC identity)** — [`data/experiments/v2_arp/phase_v2a1/ARP_SPOOFING_LIMITATION.md`](../data/experiments/v2_arp/phase_v2a1/ARP_SPOOFING_LIMITATION.md)

- Stateless 25-packet conflict features fire on **7 / 6,423** spoofing FIT windows.
- Benign has equal or higher `arp_ratio` than spoofing.
- Whole-PCAP multi-MAC rates are **higher** in publisher/profiling benign (~43% / ~26%) than in spoofing (~4.9%).

**Decision:** do not add ARP-identity features to V1, and do not train an ARP-identity specialist from these candidates on CICIoMT2024 Wi-Fi/MQTT FIT.

**V2M (MQTT structural violations)** — [`data/experiments/v2_mqtt/phase_v2m1b/MQTT_STRUCTURAL_LIMITATION.md`](../data/experiments/v2_mqtt/phase_v2m1b/MQTT_STRUCTURAL_LIMITATION.md)

- Ungated TCP parsing produced false INVALID MQTT on mixed benign traffic (`Active.pcap`).
- Port-gated to FIT-pinned plaintext **1883**, conditional `mqtt_invalid_ratio` is ~0 for malformed and benign; PUBLISH wildcards never appear on the wire.

**Decision:** do not add those structural features to V1, and do not train a structural-MQTT specialist from this signal on this capture evidence.

V2 specialists for ARP/MQTT therefore require **new independent data** (or a genuinely different state representation that these probes did not test). They are not “turn the closed FIT features back on.”

## Selection and evaluation rules

V1 already pays for lineage-aware TRAIN/VAL/TEST. V2 must not spend that rigor:

- Keep **lineage-grouped** splits. Do not leak a rare-family capture into both specialist training and fusion validation.
- If fusion is learned, train it on **out-of-fold specialist predictions** (the fusion model must not see in-sample specialist scores from the same windows).
- Do **not** reuse V1 TEST for another final-performance claim. Freeze V2 against a **new untouched evaluation set** (new PCAPs, or a predeclared holdout that was never used for V1 threshold, K/R, or specialist design).
- Do not adjust V1’s frozen threshold, 22-feature set, or serving contract in order to “look better” on V1 TEST.

## One-line version

**V1:** deployment-oriented generalist (floods + most recon, ~0.08% benign FPR).  
**V2:** rare-class independent data + multiscale/stateful specialists + principled fusion.
