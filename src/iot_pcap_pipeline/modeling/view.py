"""Phase 2B.1: materialize frozen TRAIN-fit view under group_balanced."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from iot_pcap_pipeline.features.parquet import (
    DEFAULT_BUFFER_ROWS,
    PARQUET_COLUMN_NAMES,
    StreamingFeatureParquetWriter,
    feature_parquet_arrow_schema,
    feature_schema_sha256,
    load_build_checkpoint,
    parquet_row_count,
    parquet_schema_matches,
    write_build_checkpoint,
)
from iot_pcap_pipeline.features.schema import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    V1_FEATURE_NAMES,
)
from iot_pcap_pipeline.modeling.freeze import (
    DEFAULT_GATE_2A_COMPLETE_PATH,
    FROZEN_SAMPLING_PLAN_ID,
)
from iot_pcap_pipeline.modeling.sampling import (
    CANDIDATE_PLANS,
    allocate_fit_sample_sizes,
    family_cap,
    reservoir_indices,
)
from iot_pcap_pipeline.modeling.seeds import reservoir_seed_for_pcap
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    DEFAULT_MODELING_SEED,
    MODELING_SPLIT_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

DEFAULT_SPLIT_MANIFEST_PATH = (
    DEFAULT_MODELING_DIR / "v1" / "modeling_split_manifest.csv"
)
DEFAULT_TRAINING_VIEW_CONTRACT_PATH = (
    DEFAULT_MODELING_DIR / "v1" / "training_view_contract.json"
)
DEFAULT_FIT_VIEW_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "views" / FROZEN_SAMPLING_PLAN_ID
)
DEFAULT_FIT_VIEW_DIR = DEFAULT_FIT_VIEW_ROOT / "fit"
DEFAULT_FIT_VIEW_MANIFEST_PATH = DEFAULT_FIT_VIEW_ROOT / "fit_view_manifest.csv"
DEFAULT_FIT_VIEW_COMPLETE_PATH = DEFAULT_FIT_VIEW_ROOT / "fit_view_complete.json"
DEFAULT_FIT_VIEW_CHECKPOINT_DIR = (
    DEFAULT_MODELING_DIR / "v1" / ".work" / "views" / FROZEN_SAMPLING_PLAN_ID
)

FIT_VIEW_MANIFEST_COLUMNS: tuple[str, ...] = (
    "pcap_id",
    "modeling_group_key",
    "binary_label",
    "attack_family",
    "attack_type",
    "benign_category",
    "source_parquet_path",
    "source_row_count",
    "sampling_mode",
    "group_budget",
    "allocated_sample_rows",
    "reservoir_seed",
    "output_parquet_path",
    "output_row_count",
    "output_file_size",
    "selection_sha256",
    "status",
    "resumed",
)

FORBIDDEN_OUTPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "attack_family",
        "attack_type",
        "modeling_group_key",
        "device",
        "pcap_path",
    }
)


def file_sha256(path: Path | str) -> str:
    data = Path(path).read_bytes()
    if not data:
        raise FeatureExtractionError(f"refusing empty file hash: {path}")
    return hashlib.sha256(data).hexdigest()


def _plan_by_id(plan_id: str) -> dict[str, Any]:
    for plan in CANDIDATE_PLANS:
        if plan["plan_id"] == plan_id:
            return plan
    raise FeatureExtractionError(f"unknown sampling plan_id: {plan_id!r}")


def build_training_view_contract_payload(
    *,
    split_manifest_path: Path,
    feature_schema_path: Path | None = None,
    base_seed: int = DEFAULT_MODELING_SEED,
    plan_id: str = FROZEN_SAMPLING_PLAN_ID,
) -> dict[str, Any]:
    """Construct the immutable Phase 2B.1 training-view contract payload."""
    plan = _plan_by_id(plan_id)
    schema_path = Path(feature_schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    return {
        "modeling_split_strategy_version": MODELING_SPLIT_STRATEGY_VERSION,
        "sampling_plan_id": plan_id,
        "cap_mode": plan.get("cap_mode", "per_modeling_group"),
        "caps": plan["caps"],
        "modeling_split_manifest_sha256": file_sha256(split_manifest_path),
        "feature_schema_sha256": feature_schema_sha256(schema_path),
        "base_seed": int(base_seed),
        "expected_fit_pcaps": 65,
        "expected_validation_pcaps": 20,
        "expected_fit_rows": 704_305,
        "expected_attack_rows": 493_235,
        "expected_benign_rows": 211_070,
        "expected_attack_by_family": {
            "DDoS": 180_000,
            "DoS": 180_000,
            "MQTT": 92_053,
            "Recon": 34_759,
            "Spoofing": 6_423,
        },
        "reservoir_algorithm": "deterministic_reservoir_sample_without_replacement",
        "seed_formula": (
            f"int(sha256('{MODELING_SPLIT_STRATEGY_VERSION}|{{base_seed}}|{{pcap_id}}')"
            ".hexdigest()[:16], 16)"
        ),
        "validation_sampling": "never",
        "output_columns": list(PARQUET_COLUMN_NAMES),
        "forbidden_output_columns": sorted(FORBIDDEN_OUTPUT_COLUMNS),
    }


def write_training_view_contract(
    path: Path | str | None = None,
    *,
    split_manifest_path: Path | str | None = None,
    feature_schema_path: Path | str | None = None,
    base_seed: int = DEFAULT_MODELING_SEED,
    plan_id: str = FROZEN_SAMPLING_PLAN_ID,
    project_root: Path | None = None,
) -> Path:
    """Write ``training_view_contract.json`` (atomic)."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(path or DEFAULT_TRAINING_VIEW_CONTRACT_PATH)
    if not out.is_absolute():
        out = root / out
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path
    schema_path = Path(feature_schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not schema_path.is_absolute():
        schema_path = root / schema_path
    payload = build_training_view_contract_payload(
        split_manifest_path=split_path,
        feature_schema_path=schema_path,
        base_seed=base_seed,
        plan_id=plan_id,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def require_gate_2a_passed(
    *,
    gate_path: Path | str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(gate_path or DEFAULT_GATE_2A_COMPLETE_PATH)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FeatureExtractionError(
            f"Gate 2A marker missing: {path}. Freeze modeling split first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate_2a_status") != "passed":
        raise FeatureExtractionError(
            f"Gate 2A not passed in {path}: status={payload.get('gate_2a_status')!r}"
        )
    if payload.get("frozen_sampling_plan_id") != FROZEN_SAMPLING_PLAN_ID:
        raise FeatureExtractionError(
            "Gate 2A frozen plan is "
            f"{payload.get('frozen_sampling_plan_id')!r}; "
            f"expected {FROZEN_SAMPLING_PLAN_ID!r}"
        )
    return payload


def load_and_verify_training_view_contract(
    *,
    contract_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    feature_schema_path: Path | str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Load the frozen contract and refuse if pinned hashes drifted."""
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(contract_path or DEFAULT_TRAINING_VIEW_CONTRACT_PATH)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FeatureExtractionError(
            f"training_view_contract.json missing: {path}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))

    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path
    schema_path = Path(feature_schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not schema_path.is_absolute():
        schema_path = root / schema_path

    if contract.get("modeling_split_strategy_version") != MODELING_SPLIT_STRATEGY_VERSION:
        raise FeatureExtractionError(
            "contract modeling_split_strategy_version mismatch: "
            f"{contract.get('modeling_split_strategy_version')!r}"
        )
    if contract.get("sampling_plan_id") != FROZEN_SAMPLING_PLAN_ID:
        raise FeatureExtractionError(
            f"contract sampling_plan_id must be {FROZEN_SAMPLING_PLAN_ID!r}"
        )

    actual_manifest = file_sha256(split_path)
    pinned_manifest = str(contract.get("modeling_split_manifest_sha256") or "")
    if actual_manifest != pinned_manifest:
        raise FeatureExtractionError(
            "modeling_split_manifest.csv hash mismatch vs training_view_contract: "
            f"actual={actual_manifest} pinned={pinned_manifest}. "
            "Refuse to materialize; re-freeze / rewrite contract only deliberately."
        )

    actual_schema = feature_schema_sha256(schema_path)
    pinned_schema = str(contract.get("feature_schema_sha256") or "")
    if actual_schema != pinned_schema:
        raise FeatureExtractionError(
            "feature schema hash mismatch vs training_view_contract: "
            f"actual={actual_schema} pinned={pinned_schema}"
        )
    return contract


def _load_split_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection_sha256(window_indices: list[int]) -> str:
    digest = hashlib.sha256()
    for wi in window_indices:
        digest.update(f"{int(wi)}\n".encode("utf-8"))
    return digest.hexdigest()


def _check_batch_finite(batch: pa.RecordBatch) -> None:
    import numpy as np

    for name in V1_FEATURE_NAMES:
        arr = batch.column(batch.schema.get_field_index(name)).to_numpy(
            zero_copy_only=False
        )
        if not bool(np.isfinite(np.asarray(arr, dtype=np.float64)).all()):
            raise FeatureExtractionError(
                f"non-finite values in feature column {name!r}"
            )


@dataclass
class FitViewShardResult:
    pcap_id: str
    modeling_group_key: str
    binary_label: str
    attack_family: str
    attack_type: str
    benign_category: str
    source_parquet_path: str
    source_row_count: int
    sampling_mode: str
    group_budget: str
    allocated_sample_rows: int
    reservoir_seed: int
    output_parquet_path: str
    output_row_count: int
    output_file_size: int
    selection_sha256: str
    status: str
    resumed: bool

    def as_manifest_row(self) -> dict[str, Any]:
        return {
            "pcap_id": self.pcap_id,
            "modeling_group_key": self.modeling_group_key,
            "binary_label": self.binary_label,
            "attack_family": self.attack_family,
            "attack_type": self.attack_type,
            "benign_category": self.benign_category,
            "source_parquet_path": self.source_parquet_path,
            "source_row_count": self.source_row_count,
            "sampling_mode": self.sampling_mode,
            "group_budget": self.group_budget,
            "allocated_sample_rows": self.allocated_sample_rows,
            "reservoir_seed": self.reservoir_seed,
            "output_parquet_path": self.output_parquet_path,
            "output_row_count": self.output_row_count,
            "output_file_size": self.output_file_size,
            "selection_sha256": self.selection_sha256,
            "status": self.status,
            "resumed": str(bool(self.resumed)).lower(),
        }


@dataclass
class FitViewBuildResult:
    passed: bool
    contract: dict[str, Any]
    shard_results: list[FitViewShardResult]
    totals: dict[str, Any]
    issues: list[str] = field(default_factory=list)
    view_manifest_path: Path | None = None
    view_complete_path: Path | None = None

    @property
    def successful_pcaps(self) -> int:
        return sum(1 for r in self.shard_results if r.status == "ok")


def _checkpoint_reusable(
    *,
    checkpoint_path: Path,
    output_path: Path,
    pcap_id: str,
    source_path: Path,
    source_row_count: int,
    allocated_k: int,
    reservoir_seed: int,
    schema_sha256: str,
    contract_sha256: str,
    project_root: Path,
) -> dict[str, Any] | None:
    payload = load_build_checkpoint(checkpoint_path)
    if payload is None:
        return None
    if payload.get("pcap_id") != pcap_id:
        return None
    if payload.get("source_parquet_path") != to_repo_relative(
        source_path, project_root=project_root
    ):
        return None
    if int(payload.get("source_row_count", -1)) != int(source_row_count):
        return None
    if int(payload.get("source_file_size", -1)) != int(source_path.stat().st_size):
        return None
    if int(payload.get("allocated_sample_rows", -1)) != int(allocated_k):
        return None
    if int(payload.get("reservoir_seed", -1)) != int(reservoir_seed):
        return None
    if payload.get("feature_schema_sha256") != schema_sha256:
        return None
    if payload.get("training_view_contract_sha256") != contract_sha256:
        return None
    if payload.get("status") != "ok":
        return None
    if not output_path.is_file():
        return None
    if int(payload.get("output_file_size", -1)) != int(output_path.stat().st_size):
        return None
    if parquet_row_count(output_path) != int(payload.get("output_row_count", -1)):
        return None
    if int(payload.get("output_row_count", -1)) != int(allocated_k):
        return None
    if not parquet_schema_matches(output_path):
        return None
    return payload


def materialize_fit_shard(
    *,
    row: dict[str, Any],
    allocated_k: int,
    group_budget: int | None,
    output_path: Path,
    checkpoint_path: Path,
    base_seed: int,
    schema_sha256: str,
    contract_sha256: str,
    project_root: Path,
    resume: bool,
    buffer_rows: int = DEFAULT_BUFFER_ROWS,
) -> FitViewShardResult:
    """Sample one FIT source Parquet into an output shard (atomic + resume)."""
    root = project_root
    pcap_id = str(row["pcap_id"])
    source_rel = str(row["feature_parquet_path"])
    source = Path(source_rel)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_file():
        raise FeatureExtractionError(f"source feature Parquet missing: {source}")

    source_n = parquet_row_count(source)
    if source_n is None:
        raise FeatureExtractionError(f"cannot read row count: {source}")
    manifest_n = int(row["window_count"])
    if source_n != manifest_n:
        raise FeatureExtractionError(
            f"{pcap_id}: source rows {source_n} != manifest window_count {manifest_n}"
        )
    if allocated_k > source_n:
        raise FeatureExtractionError(
            f"{pcap_id}: allocated k={allocated_k} > source rows {source_n}"
        )

    if not parquet_schema_matches(source):
        raise FeatureExtractionError(
            f"{pcap_id}: source schema does not match frozen V1 feature schema"
        )

    seed = reservoir_seed_for_pcap(pcap_id, base_seed=base_seed)
    sampling_mode = "full" if allocated_k == source_n else "reservoir"
    budget_str = "" if group_budget is None else str(int(group_budget))
    out = Path(output_path)
    if not out.is_absolute():
        out = root / out
    out_rel = to_repo_relative(out, project_root=root)
    source_rel_norm = to_repo_relative(source, project_root=root)

    if resume:
        reused = _checkpoint_reusable(
            checkpoint_path=checkpoint_path,
            output_path=out,
            pcap_id=pcap_id,
            source_path=source,
            source_row_count=source_n,
            allocated_k=allocated_k,
            reservoir_seed=seed,
            schema_sha256=schema_sha256,
            contract_sha256=contract_sha256,
            project_root=root,
        )
        if reused is not None:
            return FitViewShardResult(
                pcap_id=pcap_id,
                modeling_group_key=str(row.get("modeling_group_key") or ""),
                binary_label=str(row.get("binary_label") or ""),
                attack_family=str(row.get("attack_family") or ""),
                attack_type=str(row.get("attack_type") or ""),
                benign_category=str(row.get("benign_category") or ""),
                source_parquet_path=source_rel_norm,
                source_row_count=source_n,
                sampling_mode=str(reused.get("sampling_mode") or sampling_mode),
                group_budget=budget_str,
                allocated_sample_rows=allocated_k,
                reservoir_seed=seed,
                output_parquet_path=out_rel,
                output_row_count=int(reused["output_row_count"]),
                output_file_size=int(reused["output_file_size"]),
                selection_sha256=str(reused["selection_sha256"]),
                status="ok",
                resumed=True,
            )

    selected: set[int] | None
    if sampling_mode == "full":
        selected = None
    else:
        selected = set(reservoir_indices(source_n, allocated_k, seed))

    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)

    arrow_schema = feature_parquet_arrow_schema()
    writer = StreamingFeatureParquetWriter(
        tmp, buffer_rows=buffer_rows, schema=arrow_schema
    )
    selected_window_indices: list[int] = []
    started = time.perf_counter()
    try:
        pf = pq.ParquetFile(source)
        physical_i = 0
        for batch in pf.iter_batches(batch_size=buffer_rows, columns=list(PARQUET_COLUMN_NAMES)):
            names = batch.schema.names
            if set(FORBIDDEN_OUTPUT_COLUMNS) & set(names):
                raise FeatureExtractionError(
                    f"{pcap_id}: forbidden modeling columns present in source batch"
                )
            _check_batch_finite(batch)
            rows = batch.to_pylist()
            for row_dict in rows:
                take = selected is None or physical_i in selected
                if take:
                    writer.append(row_dict)
                    selected_window_indices.append(int(row_dict["window_index"]))
                physical_i += 1
        if physical_i != source_n:
            raise FeatureExtractionError(
                f"{pcap_id}: streamed {physical_i} rows but metadata has {source_n}"
            )
        written = writer.close()
        if written != allocated_k:
            raise FeatureExtractionError(
                f"{pcap_id}: wrote {written} rows but allocated k={allocated_k}"
            )
        if len(selected_window_indices) != allocated_k:
            raise FeatureExtractionError(
                f"{pcap_id}: selection length mismatch vs allocated k"
            )
        os.replace(tmp, out)
    except Exception:
        writer.abort()
        tmp.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    selection_hash = _selection_sha256(selected_window_indices)
    output_size = out.stat().st_size
    payload = {
        "pcap_id": pcap_id,
        "source_parquet_path": source_rel_norm,
        "source_row_count": source_n,
        "source_file_size": int(source.stat().st_size),
        "sampling_mode": sampling_mode,
        "group_budget": budget_str,
        "allocated_sample_rows": allocated_k,
        "reservoir_seed": seed,
        "output_parquet_path": out_rel,
        "output_row_count": written,
        "output_file_size": output_size,
        "selection_sha256": selection_hash,
        "feature_schema_sha256": schema_sha256,
        "training_view_contract_sha256": contract_sha256,
        "modeling_split_strategy_version": MODELING_SPLIT_STRATEGY_VERSION,
        "elapsed_seconds": elapsed,
        "status": "ok",
    }
    write_build_checkpoint(payload, checkpoint_path)

    return FitViewShardResult(
        pcap_id=pcap_id,
        modeling_group_key=str(row.get("modeling_group_key") or ""),
        binary_label=str(row.get("binary_label") or ""),
        attack_family=str(row.get("attack_family") or ""),
        attack_type=str(row.get("attack_type") or ""),
        benign_category=str(row.get("benign_category") or ""),
        source_parquet_path=source_rel_norm,
        source_row_count=source_n,
        sampling_mode=sampling_mode,
        group_budget=budget_str,
        allocated_sample_rows=allocated_k,
        reservoir_seed=seed,
        output_parquet_path=out_rel,
        output_row_count=written,
        output_file_size=output_size,
        selection_sha256=selection_hash,
        status="ok",
        resumed=False,
    )


def write_fit_view_manifest(path: Path, rows: list[FitViewShardResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIT_VIEW_MANIFEST_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.pcap_id):
            writer.writerow(row.as_manifest_row())
    tmp.replace(path)
    return path


def validate_fit_view_totals(
    shard_results: list[FitViewShardResult],
    contract: dict[str, Any],
    *,
    validation_pcap_count: int,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    ok = [r for r in shard_results if r.status == "ok"]
    expected_fit = int(contract["expected_fit_pcaps"])
    if len(ok) != expected_fit:
        issues.append(f"successful FIT PCAPs {len(ok)} != expected {expected_fit}")
    if validation_pcap_count != 0:
        # Builder must never copy validation; this is a sanity param from caller.
        issues.append(
            f"validation PCAPs touched unexpectedly: {validation_pcap_count}"
        )

    total = sum(r.output_row_count for r in ok)
    attack = sum(r.output_row_count for r in ok if r.binary_label == "ATTACK")
    benign = sum(r.output_row_count for r in ok if r.binary_label == "BENIGN")
    by_family: dict[str, int] = defaultdict(int)
    for r in ok:
        if r.binary_label == "ATTACK":
            by_family[r.attack_family] += r.output_row_count

    if total != int(contract["expected_fit_rows"]):
        issues.append(
            f"total rows {total} != expected {contract['expected_fit_rows']}"
        )
    if attack != int(contract["expected_attack_rows"]):
        issues.append(
            f"attack rows {attack} != expected {contract['expected_attack_rows']}"
        )
    if benign != int(contract["expected_benign_rows"]):
        issues.append(
            f"benign rows {benign} != expected {contract['expected_benign_rows']}"
        )

    expected_fam = contract.get("expected_attack_by_family") or {}
    for fam, exp in expected_fam.items():
        got = int(by_family.get(fam, 0))
        if got != int(exp):
            issues.append(f"family {fam}: rows {got} != expected {exp}")

    for r in ok:
        if r.output_row_count != r.allocated_sample_rows:
            issues.append(
                f"{r.pcap_id}: output_row_count {r.output_row_count} "
                f"!= allocated {r.allocated_sample_rows}"
            )

    totals = {
        "fit_pcaps": len(ok),
        "validation_pcaps_touched": validation_pcap_count,
        "test_pcaps_touched": 0,
        "total_rows": total,
        "attack_rows": attack,
        "benign_rows": benign,
        "attack_by_family": dict(sorted(by_family.items())),
    }
    return totals, issues


def build_modeling_fit_view(
    *,
    resume: bool = True,
    project_root: Path | None = None,
    split_manifest_path: Path | str | None = None,
    contract_path: Path | str | None = None,
    gate_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    feature_schema_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    buffer_rows: int = DEFAULT_BUFFER_ROWS,
) -> FitViewBuildResult:
    """Materialize Gate 2A frozen FIT view (no validation / TEST / training)."""
    root = (project_root or PROJECT_ROOT).resolve()
    require_gate_2a_passed(gate_path=gate_path, project_root=root)
    contract = load_and_verify_training_view_contract(
        contract_path=contract_path,
        split_manifest_path=split_manifest_path,
        feature_schema_path=feature_schema_path,
        project_root=root,
    )
    contract_file = Path(contract_path or DEFAULT_TRAINING_VIEW_CONTRACT_PATH)
    if not contract_file.is_absolute():
        contract_file = root / contract_file
    contract_sha = file_sha256(contract_file)

    schema_path = Path(feature_schema_path or DEFAULT_FEATURE_SCHEMA_PATH)
    if not schema_path.is_absolute():
        schema_path = root / schema_path
    schema_sha = feature_schema_sha256(schema_path)

    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path
    all_rows = _load_split_manifest(split_path)
    fit_rows = [r for r in all_rows if r.get("modeling_split") == "fit"]
    val_rows = [r for r in all_rows if r.get("modeling_split") == "validation"]
    if len(fit_rows) != int(contract["expected_fit_pcaps"]):
        raise FeatureExtractionError(
            f"FIT PCAP count {len(fit_rows)} != contract "
            f"{contract['expected_fit_pcaps']}"
        )
    if len(val_rows) != int(contract["expected_validation_pcaps"]):
        raise FeatureExtractionError(
            f"validation PCAP count {len(val_rows)} != contract "
            f"{contract['expected_validation_pcaps']}"
        )

    plan_id = str(contract["sampling_plan_id"])
    plan = _plan_by_id(plan_id)
    base_seed = int(contract.get("base_seed", DEFAULT_MODELING_SEED))
    allocations = allocate_fit_sample_sizes(fit_rows, plan, base_seed=base_seed)
    preflight_total = sum(allocations.values())
    if preflight_total != int(contract["expected_fit_rows"]):
        raise FeatureExtractionError(
            f"preflight allocated rows {preflight_total} != "
            f"contract expected_fit_rows {contract['expected_fit_rows']}"
        )

    view_root = Path(output_dir or DEFAULT_FIT_VIEW_ROOT)
    if not view_root.is_absolute():
        view_root = root / view_root
    fit_dir = view_root / "fit"
    manifest_path = view_root / "fit_view_manifest.csv"
    complete_path = view_root / "fit_view_complete.json"
    ckpt_dir = Path(checkpoint_dir or DEFAULT_FIT_VIEW_CHECKPOINT_DIR)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    fit_dir.mkdir(parents=True, exist_ok=True)

    # Drop stale complete marker until acceptance passes.
    complete_path.unlink(missing_ok=True)

    caps = plan["caps"]
    shard_results: list[FitViewShardResult] = []
    for idx, row in enumerate(sorted(fit_rows, key=lambda r: r["pcap_id"]), start=1):
        pcap_id = str(row["pcap_id"])
        path_key = str(row["pcap_path"])
        k = int(allocations[path_key])
        budget = family_cap(row, caps)
        if progress_file is not None:
            progress_file.write(
                f"[{idx}/{len(fit_rows)}] {pcap_id} k={k} mode="
                f"{'full' if k == int(row['window_count']) else 'reservoir'}\n"
            )
            progress_file.flush()
        result = materialize_fit_shard(
            row=row,
            allocated_k=k,
            group_budget=budget,
            output_path=fit_dir / f"{pcap_id}.parquet",
            checkpoint_path=ckpt_dir / f"{pcap_id}.json",
            base_seed=base_seed,
            schema_sha256=schema_sha,
            contract_sha256=contract_sha,
            project_root=root,
            resume=resume,
            buffer_rows=buffer_rows,
        )
        shard_results.append(result)

    write_fit_view_manifest(manifest_path, shard_results)
    totals, issues = validate_fit_view_totals(
        shard_results,
        contract,
        validation_pcap_count=0,
    )

    # Ensure no validation pcap_ids appear in outputs.
    val_ids = {r["pcap_id"] for r in val_rows}
    for r in shard_results:
        if r.pcap_id in val_ids:
            issues.append(f"validation pcap materialized: {r.pcap_id}")

    passed = not issues
    result = FitViewBuildResult(
        passed=passed,
        contract=contract,
        shard_results=shard_results,
        totals=totals,
        issues=issues,
        view_manifest_path=manifest_path,
        view_complete_path=complete_path if passed else None,
    )
    if not passed:
        return result

    complete = {
        "status": "passed",
        "phase": "2b1_fit_view",
        "modeling_split_strategy_version": MODELING_SPLIT_STRATEGY_VERSION,
        "sampling_plan_id": plan_id,
        "base_seed": base_seed,
        "training_view_contract": to_repo_relative(contract_file, project_root=root),
        "training_view_contract_sha256": contract_sha,
        "feature_schema_sha256": schema_sha,
        "modeling_split_manifest_sha256": contract["modeling_split_manifest_sha256"],
        "totals": totals,
        "expected": {
            "fit_pcaps": contract["expected_fit_pcaps"],
            "fit_rows": contract["expected_fit_rows"],
            "attack_rows": contract["expected_attack_rows"],
            "benign_rows": contract["expected_benign_rows"],
            "attack_by_family": contract["expected_attack_by_family"],
        },
        "artifacts": {
            "fit_dir": to_repo_relative(fit_dir, project_root=root),
            "fit_view_manifest": to_repo_relative(manifest_path, project_root=root),
            "fit_view_complete": to_repo_relative(complete_path, project_root=root),
        },
        "validation_sampling": "never",
        "next": (
            "Inspect fit_view_manifest.csv + totals, then Phase 2B.2 model "
            "training on this view only. Do not consult TEST."
        ),
    }
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = complete_path.with_suffix(complete_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(complete_path)
    result.view_complete_path = complete_path
    return result


def format_fit_view_summary(result: FitViewBuildResult) -> str:
    lines = [
        "Phase 2B.1 — TRAIN-fit view materialization",
        f"status: {'passed' if result.passed else 'FAILED'}",
        f"fit_pcaps_ok: {result.successful_pcaps}",
        f"total_rows: {result.totals.get('total_rows')}",
        f"attack_rows: {result.totals.get('attack_rows')}",
        f"benign_rows: {result.totals.get('benign_rows')}",
        f"attack_by_family: {result.totals.get('attack_by_family')}",
    ]
    if result.view_manifest_path is not None:
        lines.append(f"fit_view_manifest: {result.view_manifest_path}")
    if result.view_complete_path is not None:
        lines.append(f"fit_view_complete: {result.view_complete_path}")
    if result.issues:
        lines.append("issues:")
        for issue in result.issues:
            lines.append(f"  - {issue}")
    else:
        lines.append(
            "next: inspect the view before Phase 2B.2 training; do not consult TEST."
        )
    return "\n".join(lines) + "\n"
