"""PCAP → TRAIN-fit / TRAIN-validation assignment by modeling_group_key."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from iot_pcap_pipeline.features.dataset import (
    DEFAULT_BUILD_MANIFEST_PATH,
    EXPECTED_TRAIN_PCAP_COUNT,
    require_train_build_complete,
)
from iot_pcap_pipeline.features.parquet import pcap_id_from_path
from iot_pcap_pipeline.modeling.groups import (
    benign_category_for_row,
    modeling_group_key_for_row,
)
from iot_pcap_pipeline.modeling.seeds import stable_seed_u64
from iot_pcap_pipeline.paths import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MODELING_SEED,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

ModelingSplit = Literal["fit", "validation"]

TARGET_ATTACK_VAL_FRACTION = 0.20
TARGET_PROFILING_VAL_FRACTION = 0.20
# Floor for "device alone is large enough" — below this, add a singleton.
MIN_PROFILING_VAL_FRACTION = 0.15
# Reject tiny device-only holdouts (e.g. Singcall ~44 windows).
MIN_DEVICE_VAL_WINDOWS = 1_000

SPLIT_MANIFEST_COLUMNS: tuple[str, ...] = (
    "pcap_path",
    "pcap_id",
    "modeling_group_key",
    "modeling_split",
    "binary_label",
    "attack_family",
    "attack_type",
    "profiling_type",
    "device",
    "window_count",
    "feature_parquet_path",
    "selection_reason",
    "benign_category",
    "group_kind",
)


@dataclass
class SplitAssignmentResult:
    rows: list[dict[str, Any]]
    limitations: list[str] = field(default_factory=list)
    selection_notes: list[dict[str, Any]] = field(default_factory=list)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _inventory_train_rows(inventory_path: Path) -> list[dict[str, str]]:
    rows = _load_csv(inventory_path)
    train = [
        r
        for r in rows
        if r.get("split") == "train" and r.get("binary_label") in {"BENIGN", "ATTACK"}
    ]
    train.sort(key=lambda r: r["pcap_path"])
    return train


def _select_closest_group(
    candidates: list[tuple[str, int]],
    *,
    family_total: int,
    target_fraction: float,
    seed_salt: str,
    base_seed: int,
) -> tuple[str, float, str]:
    """Pick group whose window fraction is closest to target; seeded tie-break."""
    if not candidates:
        raise FeatureExtractionError(f"no candidates for holdout: {seed_salt}")
    if family_total <= 0:
        raise FeatureExtractionError(f"empty family total for {seed_salt}")

    seed = stable_seed_u64(seed_salt, base_seed=base_seed)
    # Stable order for ties: sort by (abs_distance, seeded_rank, key).
    ranked: list[tuple[float, int, str, int, float]] = []
    for key, windows in candidates:
        frac = windows / family_total
        dist = abs(frac - target_fraction)
        # Derive a deterministic rank from seed + key without hash().
        rank = stable_seed_u64(f"{seed_salt}|tie|{key}", base_seed=base_seed) ^ seed
        ranked.append((dist, rank, key, windows, frac))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]))
    dist, _rank, key, windows, frac = ranked[0]
    reason = (
        "closest_group_to_target_validation_fraction"
        f"|target={target_fraction:.2f}|actual={frac:.6f}|windows={windows}"
    )
    return key, frac, reason


def assign_modeling_split(
    *,
    inventory_path: Path | str | None = None,
    build_manifest_path: Path | str | None = None,
    train_complete_path: Path | str | None = None,
    project_root: Path | None = None,
    base_seed: int = DEFAULT_MODELING_SEED,
    target_attack_val_fraction: float = TARGET_ATTACK_VAL_FRACTION,
    target_profiling_val_fraction: float = TARGET_PROFILING_VAL_FRACTION,
    min_profiling_val_fraction: float = MIN_PROFILING_VAL_FRACTION,
    min_device_val_windows: int = MIN_DEVICE_VAL_WINDOWS,
) -> SplitAssignmentResult:
    """Assign each TRAIN PCAP to fit or validation via atomic modeling groups."""
    root = (project_root or PROJECT_ROOT).resolve()
    require_train_build_complete(
        train_complete_path,
        project_root=root,
    )

    inv_path = Path(inventory_path or (DEFAULT_MANIFEST_DIR / "pcap_inventory.csv"))
    if not inv_path.is_absolute():
        inv_path = root / inv_path
    man_path = Path(build_manifest_path or DEFAULT_BUILD_MANIFEST_PATH)
    if not man_path.is_absolute():
        man_path = root / man_path

    train_rows = _inventory_train_rows(inv_path)
    if len(train_rows) != EXPECTED_TRAIN_PCAP_COUNT:
        raise FeatureExtractionError(
            f"Expected {EXPECTED_TRAIN_PCAP_COUNT} TRAIN PCAPs in inventory, "
            f"found {len(train_rows)}"
        )
    if any(r.get("split") == "test" for r in train_rows):
        raise FeatureExtractionError("TEST PCAPs leaked into TRAIN inventory selection")

    manifest_rows = _load_csv(man_path)
    by_path = {r["pcap_path"]: r for r in manifest_rows}
    if len(manifest_rows) != EXPECTED_TRAIN_PCAP_COUNT:
        raise FeatureExtractionError(
            f"Expected {EXPECTED_TRAIN_PCAP_COUNT} build_manifest rows, "
            f"found {len(manifest_rows)}"
        )

    # Enrich rows with group + window counts.
    enriched: list[dict[str, Any]] = []
    for row in train_rows:
        pcap_path = row["pcap_path"]
        man = by_path.get(pcap_path)
        if man is None:
            raise FeatureExtractionError(
                f"TRAIN PCAP missing from build_manifest: {pcap_path}"
            )
        if (man.get("status") or "").strip() != "ok":
            raise FeatureExtractionError(
                f"build_manifest status not ok for {pcap_path}: {man.get('status')!r}"
            )
        try:
            window_count = int(man["output_row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FeatureExtractionError(
                f"invalid output_row_count for {pcap_path}: {exc}"
            ) from exc
        abs_pcap = root / pcap_path
        pcap_id = pcap_id_from_path(abs_pcap, project_root=root)
        out_rel = (man.get("output_path") or "").strip()
        if not out_rel:
            out_rel = f"data/features/v1/train/{pcap_id}.parquet"
        spec = modeling_group_key_for_row(row)
        enriched.append(
            {
                "pcap_path": pcap_path,
                "pcap_id": pcap_id,
                "modeling_group_key": spec.modeling_group_key,
                "group_kind": spec.kind,
                "binary_label": row.get("binary_label") or "",
                "attack_family": row.get("attack_family") or "",
                "attack_type": row.get("attack_type") or "",
                "profiling_type": row.get("profiling_type") or "",
                "device": row.get("device") or "",
                "source": row.get("source") or "",
                "window_count": window_count,
                "feature_parquet_path": out_rel,
                "benign_category": (
                    benign_category_for_row(row)
                    if row.get("binary_label") == "BENIGN"
                    else ""
                ),
                "modeling_split": "",
                "selection_reason": "",
            }
        )

    # Index by modeling group.
    group_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_windows: dict[str, int] = defaultdict(int)
    group_kind: dict[str, str] = {}
    for row in enriched:
        key = row["modeling_group_key"]
        group_members[key].append(row)
        group_windows[key] += int(row["window_count"])
        group_kind[key] = str(row["group_kind"])

    val_groups: set[str] = set()
    group_reasons: dict[str, str] = {}
    limitations: list[str] = []
    selection_notes: list[dict[str, Any]] = []

    # --- ATTACK families: hold out one lineage closest to target fraction ---
    attack_by_family: dict[str, list[str]] = defaultdict(list)
    for key, kind in group_kind.items():
        if kind in {"attack_lineage", "spoofing"}:
            family = key.split("|", 1)[0]
            attack_by_family[family].append(key)

    for family, keys in sorted(attack_by_family.items()):
        if family == "Spoofing":
            for key in keys:
                group_reasons[key] = (
                    "spoofing_fit_only|no_independent_train_validation_group"
                )
            limitations.append(
                "Spoofing: single TRAIN capture assigned entirely to fit; "
                "TRAIN-validation has no Spoofing family coverage."
            )
            selection_notes.append(
                {
                    "family": family,
                    "held_out_group": None,
                    "reason": "spoofing_fit_only",
                }
            )
            continue

        if len(keys) < 2:
            for key in keys:
                group_reasons[key] = "single_group_family_fit_only"
            continue

        family_total = sum(group_windows[k] for k in keys)
        candidates = [(k, group_windows[k]) for k in keys]
        held_key, frac, reason = _select_closest_group(
            candidates,
            family_total=family_total,
            target_fraction=target_attack_val_fraction,
            seed_salt=f"attack_family|{family}",
            base_seed=base_seed,
        )
        val_groups.add(held_key)
        group_reasons[held_key] = reason
        for key in keys:
            if key != held_key:
                group_reasons[key] = "fit_remainder_after_lineage_holdout"
        selection_notes.append(
            {
                "family": family,
                "held_out_group": held_key,
                "actual_fraction": frac,
                "target_fraction": target_attack_val_fraction,
                "reason": reason,
                "family_total_windows": family_total,
            }
        )

    # --- BENIGN: publisher fit-only; device (+ optional singleton) ---
    limitations.append(
        "Publisher benign (Benign_train): fit only; TRAIN-validation has no "
        "independent publisher-benign coverage. Validation FPR primarily "
        "measures held-out profiling benign."
    )
    for key, kind in group_kind.items():
        if kind == "publisher_benign":
            group_reasons[key] = "publisher_benign_fit_only"

    profiling_device_keys = [
        k for k, kind in group_kind.items() if kind == "profiling_device"
    ]
    profiling_singleton_keys = [
        k for k, kind in group_kind.items() if kind == "profiling_singleton"
    ]
    profiling_total = sum(
        group_windows[k] for k in profiling_device_keys + profiling_singleton_keys
    )

    # Capable devices: must include power + interaction among members.
    capable_devices: list[tuple[str, int]] = []
    for key in profiling_device_keys:
        types = {
            (m.get("profiling_type") or "").strip().lower()
            for m in group_members[key]
        }
        windows = group_windows[key]
        if "power" in types and "interaction" in types and windows >= min_device_val_windows:
            capable_devices.append((key, windows))

    if not capable_devices:
        raise FeatureExtractionError(
            "No profiling device group meets power+interaction and "
            f"min_device_val_windows={min_device_val_windows}"
        )

    # Prefer device whose fraction of profiling_total is closest to target.
    # If all are below the 15% band, still pick closest (honestly document).
    held_device, device_frac, device_reason = _select_closest_group(
        capable_devices,
        family_total=profiling_total,
        target_fraction=target_profiling_val_fraction,
        seed_salt="benign_profiling_device",
        base_seed=base_seed,
    )
    val_groups.add(held_device)
    group_reasons[held_device] = (
        f"profiling_device_holdout|{device_reason}"
        f"|profiling_share={device_frac:.6f}"
    )

    # States covered by held device.
    val_states = {
        (m.get("profiling_type") or "").strip().lower()
        for m in group_members[held_device]
    }
    # Add a singleton if state coverage is thin OR device alone is undersized.
    need_singleton_for_states = len(val_states) < 2
    need_singleton_for_size = device_frac < min_profiling_val_fraction
    held_singleton: str | None = None
    singleton_reason = ""
    if (need_singleton_for_states or need_singleton_for_size) and profiling_singleton_keys:
        preferred = ["Idle", "Active", "ActiveBroker"]
        present = {
            Path(group_members[k][0]["pcap_path"]).stem: k
            for k in profiling_singleton_keys
        }
        if need_singleton_for_size:
            # Prefer Idle for slow/idle benign behavior when size is short.
            for stem in preferred:
                if stem in present:
                    held_singleton = present[stem]
                    singleton_reason = (
                        "profiling_singleton_added_for_min_validation_fraction"
                        f"|device_share={device_frac:.6f}"
                        f"|min={min_profiling_val_fraction:.2f}"
                        f"|singleton={stem}"
                    )
                    break
        elif need_singleton_for_states:
            for stem in preferred:
                if stem in present:
                    key = present[stem]
                    ptype = (group_members[key][0].get("profiling_type") or "").lower()
                    if ptype and ptype not in val_states:
                        held_singleton = key
                        singleton_reason = (
                            "profiling_singleton_added_for_state_coverage"
                            f"|singleton={stem}"
                        )
                        break
        if held_singleton is None:
            cands = [(k, group_windows[k]) for k in profiling_singleton_keys]
            held_singleton, _, _ = _select_closest_group(
                cands,
                family_total=profiling_total,
                target_fraction=target_profiling_val_fraction,
                seed_salt="benign_profiling_singleton",
                base_seed=base_seed,
            )
            singleton_reason = (
                "profiling_singleton_added_fallback_closest"
                f"|need_states={need_singleton_for_states}"
                f"|need_size={need_singleton_for_size}"
            )

    if held_singleton is not None:
        val_groups.add(held_singleton)
        group_reasons[held_singleton] = singleton_reason
        val_states |= {
            (m.get("profiling_type") or "").strip().lower()
            for m in group_members[held_singleton]
        }

    combined_val_windows = group_windows[held_device] + (
        group_windows[held_singleton] if held_singleton else 0
    )
    combined_frac = (
        combined_val_windows / profiling_total if profiling_total else 0.0
    )

    for key in profiling_device_keys + profiling_singleton_keys:
        if key not in group_reasons:
            group_reasons[key] = "profiling_fit_remainder"

    selection_notes.append(
        {
            "family": "BENIGN_profiling",
            "held_out_device_group": held_device,
            "held_out_singleton": held_singleton,
            "device_fraction_of_profiling": device_frac,
            "combined_validation_fraction_of_profiling": combined_frac,
            "combined_validation_windows": combined_val_windows,
            "target_fraction": target_profiling_val_fraction,
            "min_profiling_val_fraction": min_profiling_val_fraction,
            "singleton_added_for_states": need_singleton_for_states,
            "singleton_added_for_size": need_singleton_for_size
            and held_singleton is not None,
            "validation_profiling_states": sorted(s for s in val_states if s),
            "profiling_total_windows": profiling_total,
            "note": (
                "Device groups are small vs 15–25% target; add Idle/Active/"
                "ActiveBroker when device alone is below "
                f"{min_profiling_val_fraction:.0%} or has <2 states."
            ),
        }
    )

    # Assign every PCAP.
    for row in enriched:
        key = row["modeling_group_key"]
        if key in val_groups:
            row["modeling_split"] = "validation"
        else:
            row["modeling_split"] = "fit"
        row["selection_reason"] = group_reasons.get(key, "")

    # Acceptance: group atomicity already by construction; verify no split groups.
    for key, members in group_members.items():
        splits = {m["modeling_split"] for m in members}
        if len(splits) != 1:
            raise FeatureExtractionError(
                f"modeling_group_key split across fit/validation: {key}"
            )

    return SplitAssignmentResult(
        rows=sorted(enriched, key=lambda r: r["pcap_path"]),
        limitations=limitations,
        selection_notes=selection_notes,
    )


def write_modeling_split_manifest(
    path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPLIT_MANIFEST_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["pcap_path"]):
            writer.writerow({c: row.get(c, "") for c in SPLIT_MANIFEST_COLUMNS})
    tmp.replace(path)
    return path


def validate_split_acceptance(
    rows: list[dict[str, Any]],
    *,
    project_root: Path | None = None,
) -> list[str]:
    """Return list of acceptance violations (empty = pass)."""
    root = (project_root or PROJECT_ROOT).resolve()
    issues: list[str] = []
    if len(rows) != EXPECTED_TRAIN_PCAP_COUNT:
        issues.append(
            f"expected {EXPECTED_TRAIN_PCAP_COUNT} rows, got {len(rows)}"
        )
    paths = [r["pcap_path"] for r in rows]
    if len(paths) != len(set(paths)):
        issues.append("duplicate pcap_path in modeling split")
    if any("/test/" in p.replace("\\", "/") or p.endswith("_test.pcap") for p in paths):
        # Soft path heuristic; inventory split is authoritative.
        pass
    fit = {r["pcap_path"] for r in rows if r["modeling_split"] == "fit"}
    val = {r["pcap_path"] for r in rows if r["modeling_split"] == "validation"}
    if fit & val:
        issues.append(f"PCAP overlap fit/validation: {sorted(fit & val)}")
    if fit | val != set(paths):
        issues.append("fit∪validation does not cover all assigned PCAPs")

    # Group overlap
    fit_groups = {
        r["modeling_group_key"] for r in rows if r["modeling_split"] == "fit"
    }
    val_groups = {
        r["modeling_group_key"]
        for r in rows
        if r["modeling_split"] == "validation"
    }
    if fit_groups & val_groups:
        issues.append(
            "modeling_group_key overlap fit/validation: "
            + ", ".join(sorted(fit_groups & val_groups))
        )

    # Lineage atomicity
    by_group: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_group[r["modeling_group_key"]].add(r["modeling_split"])
    for key, splits in by_group.items():
        if len(splits) != 1:
            issues.append(f"group not atomic: {key} -> {sorted(splits)}")

    # Spoofing
    spoof = [r for r in rows if r.get("attack_family") == "Spoofing"]
    if len(spoof) != 1 or spoof[0]["modeling_split"] != "fit":
        issues.append("Spoofing must be exactly 1 PCAP in fit")
    if any(r["modeling_split"] == "validation" and r.get("attack_family") == "Spoofing" for r in rows):
        issues.append("Spoofing must not appear in validation")

    # Publisher benign fit only
    pub = [
        r
        for r in rows
        if str(r.get("modeling_group_key", "")).startswith("benign|publisher|")
    ]
    if not pub or any(r["modeling_split"] != "fit" for r in pub):
        issues.append("publisher benign must be fit only")

    # Device groups atomic (already covered) — ensure relative paths stay under project.
    for r in rows:
        rel = r.get("feature_parquet_path") or ""
        if rel.startswith("/"):
            issues.append(f"absolute feature_parquet_path: {rel}")
        try:
            to_repo_relative(root / r["pcap_path"], project_root=root)
        except Exception:  # noqa: BLE001
            issues.append(f"bad pcap_path: {r['pcap_path']}")

    return issues
