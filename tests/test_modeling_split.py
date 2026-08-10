"""Phase 2A modeling split + sampling characterization tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from iot_pcap_pipeline.features.dataset import EXPECTED_TRAIN_PCAP_COUNT
from iot_pcap_pipeline.modeling.groups import modeling_group_key_for_row
from iot_pcap_pipeline.modeling.sampling import (
    windows_after_cap,
)
from iot_pcap_pipeline.modeling.seeds import stable_seed_u64
from iot_pcap_pipeline.modeling.split import (
    _select_closest_group,
    validate_split_acceptance,
)
from iot_pcap_pipeline.paths import (
    FEATURE_BUILD_STRATEGY_VERSION,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    WINDOWING_STRATEGY_VERSION,
)
from iot_pcap_pipeline.windowing.policy import (
    BACKWARD_RESET_SECONDS,
    INACTIVITY_TIMEOUT_SECONDS,
    WINDOW_SIZE,
)


def test_stable_seed_reproducible_and_not_builtin_hash() -> None:
    a = stable_seed_u64("DDoS", base_seed=42)
    b = stable_seed_u64("DDoS", base_seed=42)
    c = stable_seed_u64("DoS", base_seed=42)
    assert a == b
    assert a != c
    # Must not depend on PYTHONHASHSEED / built-in hash.
    assert a == int(
        __import__("hashlib")
        .sha256(b"phase2a_v1|42|DDoS")
        .hexdigest()[:16],
        16,
    )


def test_modeling_group_chunks_share_key() -> None:
    a = modeling_group_key_for_row(
        {
            "binary_label": "ATTACK",
            "attack_family": "DDoS",
            "attack_type": "DDoS_ICMP",
            "pcap_path": "data/raw/x/TCP_IP-DDoS-ICMP1_train.pcap",
        }
    )
    b = modeling_group_key_for_row(
        {
            "binary_label": "ATTACK",
            "attack_family": "DDoS",
            "attack_type": "DDoS_ICMP",
            "pcap_path": "data/raw/x/TCP_IP-DDoS-ICMP2_train.pcap",
        }
    )
    assert a.modeling_group_key == b.modeling_group_key == "DDoS|DDoS_ICMP"


def test_select_closest_group_prefers_near_20pct() -> None:
    candidates = [
        ("DDoS|DDoS_UDP", 342),
        ("DDoS|DDoS_ICMP", 322),
        ("DDoS|DDoS_TCP", 168),
        ("DDoS|DDoS_SYN", 168),
    ]
    key, frac, reason = _select_closest_group(
        candidates,
        family_total=1000,
        target_fraction=0.20,
        seed_salt="attack_family|DDoS",
        base_seed=42,
    )
    assert key in {"DDoS|DDoS_TCP", "DDoS|DDoS_SYN"}
    assert abs(frac - 0.168) < 1e-9
    assert "closest_group_to_target_validation_fraction" in reason


def test_windows_after_cap() -> None:
    assert windows_after_cap(100, None) == 100
    assert windows_after_cap(100, 25) == 25
    assert windows_after_cap(10, 25) == 10


def test_validate_split_acceptance_catches_group_overlap() -> None:
    rows = [
        {
            "pcap_path": "a.pcap",
            "pcap_id": "a",
            "modeling_group_key": "DDoS|DDoS_ICMP",
            "modeling_split": "fit",
            "binary_label": "ATTACK",
            "attack_family": "DDoS",
            "feature_parquet_path": "data/features/v1/train/a.parquet",
        },
        {
            "pcap_path": "b.pcap",
            "pcap_id": "b",
            "modeling_group_key": "DDoS|DDoS_ICMP",
            "modeling_split": "validation",
            "binary_label": "ATTACK",
            "attack_family": "DDoS",
            "feature_parquet_path": "data/features/v1/train/b.parquet",
        },
    ]
    # Pad to avoid only the count check — still should flag atomicity/overlap.
    issues = validate_split_acceptance(rows)
    assert any("group not atomic" in i or "overlap" in i for i in issues)


def _write_train_complete(path: Path, schema_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation_status": "passed",
                "pcap_count": EXPECTED_TRAIN_PCAP_COUNT,
                "feature_strategy_version": FEATURE_STRATEGY_VERSION,
                "feature_build_strategy_version": FEATURE_BUILD_STRATEGY_VERSION,
                "feature_schema_sha256": schema_hash,
                "windowing_strategy_version": WINDOWING_STRATEGY_VERSION,
                "windowing": {
                    "window_size": WINDOW_SIZE,
                    "inactivity_timeout_seconds": INACTIVITY_TIMEOUT_SECONDS,
                    "backward_reset_seconds": BACKWARD_RESET_SECONDS,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_allocate_group_budget_proportional() -> None:
    from iot_pcap_pipeline.modeling.sampling import allocate_group_budget

    members = [
        {"pcap_path": "b.pcap", "window_count": 100},
        {"pcap_path": "a.pcap", "window_count": 300},
    ]
    out = allocate_group_budget(members, 200)
    assert sum(out.values()) == 200
    assert out["a.pcap"] == 150
    assert out["b.pcap"] == 50
    # Uncapped / under budget keeps all.
    assert allocate_group_budget(members, None) == {"a.pcap": 300, "b.pcap": 100}
    assert allocate_group_budget(members, 10_000) == {"a.pcap": 300, "b.pcap": 100}


def test_full_corpus_characterization_smoke() -> None:
    """Run against the real frozen TRAIN corpus when present."""
    inv = PROJECT_ROOT / "data" / "manifests" / "pcap_inventory.csv"
    man = PROJECT_ROOT / "data" / "features" / "v1" / "build_manifest.csv"
    complete = PROJECT_ROOT / "data" / "features" / "v1" / "train_build_complete.json"
    if not (inv.is_file() and man.is_file() and complete.is_file()):
        pytest.skip("frozen TRAIN artifacts not present")

    from iot_pcap_pipeline.modeling.characterize import characterize_modeling_split

    out = PROJECT_ROOT / "data" / "modeling" / "v1"
    result = characterize_modeling_split(
        inventory_path=inv,
        build_manifest_path=man,
        train_complete_path=complete,
        output_dir=out,
    )
    assert result.passed
    assert len(result.split_rows) == EXPECTED_TRAIN_PCAP_COUNT

    fit = [r for r in result.split_rows if r["modeling_split"] == "fit"]
    val = [r for r in result.split_rows if r["modeling_split"] == "validation"]
    assert fit and val
    assert not (
        {r["modeling_group_key"] for r in fit}
        & {r["modeling_group_key"] for r in val}
    )

    # ICMP chunks must share a side.
    icmp = [
        r
        for r in result.split_rows
        if r["modeling_group_key"] == "DDoS|DDoS_ICMP"
    ]
    assert len(icmp) == 8
    assert len({r["modeling_split"] for r in icmp}) == 1

    spoof = [r for r in result.split_rows if r["attack_family"] == "Spoofing"]
    assert len(spoof) == 1 and spoof[0]["modeling_split"] == "fit"

    pub = [
        r
        for r in result.split_rows
        if str(r["modeling_group_key"]).startswith("benign|publisher|")
    ]
    assert pub and all(r["modeling_split"] == "fit" for r in pub)

    # Undersized device holdout should pull Idle into validation.
    idle = [
        r
        for r in result.split_rows
        if r["modeling_group_key"] == "benign|singleton|Idle"
    ]
    assert len(idle) == 1 and idle[0]["modeling_split"] == "validation"
    owl = [
        r
        for r in result.split_rows
        if r["modeling_group_key"] == "benign|device|Owltron_Camera"
    ]
    assert owl and all(r["modeling_split"] == "validation" for r in owl)

    plan = json.loads(result.sampling_plan_path.read_text(encoding="utf-8"))
    assert plan["status"] == "characterization_only"
    assert plan["validation_sampling"] == "never"
    assert "reservoir" in plan["reservoir_contract"]["algorithm"]
    plan_ids = {p["plan_id"] for p in plan["candidate_plans"]}
    assert "group_balanced" in plan_ids

    with result.sampling_summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert summary
    assert {r["plan_id"] for r in summary} >= {
        "fullish",
        "family_balanced",
        "aggressive",
        "group_balanced",
    }
    gb = [r for r in summary if r["plan_id"] == "group_balanced"]
    assert gb and all(r["cap_mode"] == "per_modeling_group" for r in gb)
