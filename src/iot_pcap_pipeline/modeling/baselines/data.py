"""Load FIT NumPy arrays and stream TRAIN-validation feature shards."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.constants import (
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_PCAPS,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_ATTACK,
    EXPECTED_VAL_BENIGN,
    EXPECTED_VAL_PCAPS,
    EXPECTED_VAL_ROWS,
    FORBIDDEN_MODEL_COLUMNS,
    LABEL_MAPPING,
    SMOKE_FIT_BUCKETS,
    SMOKE_ROWS_PER_GROUP,
    SMOKE_VAL_ATTACK_GROUPS,
    SMOKE_VAL_BENIGN_GROUPS,
)
from iot_pcap_pipeline.modeling.baselines.metrics import benign_group_key
from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

_FEATURE_LIST = list(V1_FEATURE_NAMES)
_READ_COLUMNS = _FEATURE_LIST + ["binary_label"]


def encode_labels(labels: list[str] | np.ndarray) -> np.ndarray:
    """Map BENIGN/ATTACK strings to uint8 using the frozen label mapping."""
    out = np.empty(len(labels), dtype=np.uint8)
    for i, lab in enumerate(labels):
        key = str(lab)
        if key not in LABEL_MAPPING:
            raise FeatureExtractionError(f"unknown binary_label: {key!r}")
        out[i] = LABEL_MAPPING[key]
    return out


def assert_feature_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Refuse if feature order drifts or metadata leaks into X."""
    names = list(columns)
    if names != _FEATURE_LIST:
        raise FeatureExtractionError(
            "feature column order mismatch: "
            f"got {names[:5]}... (n={len(names)}); "
            f"expected {list(V1_FEATURE_NAMES)[:5]}... (n={len(V1_FEATURE_NAMES)})"
        )
    leaked = sorted(FORBIDDEN_MODEL_COLUMNS & set(names))
    if leaked:
        raise FeatureExtractionError(
            f"forbidden metadata columns present in model input: {leaked}"
        )


def reject_test_path(path: Path | str) -> None:
    text = Path(path).as_posix().lower()
    if "/test/" in f"/{text}/" or text.endswith("/test") or "/features/v1/test/" in text:
        raise FeatureExtractionError(f"TEST path rejected: {path}")


@dataclass(frozen=True)
class FitArrays:
    X: np.ndarray  # float32 [n, 27]
    y: np.ndarray  # uint8 [n]
    pcap_ids: list[str]
    n_attack: int
    n_benign: int
    smoke_groups_loaded: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class ValidationPcapSpec:
    pcap_id: str
    pcap_path: str
    feature_parquet_path: str
    modeling_group_key: str
    binary_label: str
    attack_family: str
    attack_type: str
    benign_category: str
    window_count: int


def load_fit_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: r["pcap_id"])
    return rows


def load_validation_specs(
    split_manifest_path: Path,
    *,
    project_root: Path | None = None,
) -> list[ValidationPcapSpec]:
    root = (project_root or PROJECT_ROOT).resolve()
    with split_manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    specs: list[ValidationPcapSpec] = []
    for row in rows:
        if row.get("modeling_split") != "validation":
            continue
        feat = str(row["feature_parquet_path"])
        reject_test_path(feat)
        if "views/" in feat.replace("\\", "/"):
            raise FeatureExtractionError(
                f"validation must use unsampled TRAIN features, not a view: {feat}"
            )
        specs.append(
            ValidationPcapSpec(
                pcap_id=str(row["pcap_id"]),
                pcap_path=str(row["pcap_path"]),
                feature_parquet_path=feat,
                modeling_group_key=str(row.get("modeling_group_key") or ""),
                binary_label=str(row.get("binary_label") or ""),
                attack_family=str(row.get("attack_family") or ""),
                attack_type=str(row.get("attack_type") or ""),
                benign_category=str(row.get("benign_category") or ""),
                window_count=int(row["window_count"]),
            )
        )
    specs.sort(key=lambda s: s.pcap_id)
    return specs


def _fit_smoke_bucket(row: dict[str, str]) -> str | None:
    if row.get("binary_label") == "BENIGN":
        return "benign"
    family = (row.get("attack_family") or "").strip()
    if family in SMOKE_FIT_BUCKETS:
        return family
    return None


def _read_feature_prefix(
    path: Path,
    *,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read at most ``max_rows`` leading rows (features + labels only)."""
    if max_rows <= 0:
        return (
            np.empty((0, len(V1_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )
    pf = pq.ParquetFile(path)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    taken = 0
    for batch in pf.iter_batches(
        batch_size=min(8_192, max_rows), columns=_READ_COLUMNS
    ):
        cols = [c for c in batch.schema.names if c != "binary_label"]
        assert_feature_columns(cols)
        arrays = [
            batch.column(name).to_numpy(zero_copy_only=False)
            for name in _FEATURE_LIST
        ]
        X = np.column_stack(arrays).astype(np.float32, copy=False)
        y = encode_labels(batch.column("binary_label").to_pylist())
        if not np.isfinite(X).all():
            raise FeatureExtractionError(f"non-finite features in {path}")
        remain = max_rows - taken
        if X.shape[0] > remain:
            X = X[:remain]
            y = y[:remain]
        xs.append(X)
        ys.append(y)
        taken += int(X.shape[0])
        if taken >= max_rows:
            break
    if not xs:
        return (
            np.empty((0, len(V1_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def build_smoke_validation_budgets(
    specs: list[ValidationPcapSpec],
    *,
    rows_per_group: int = SMOKE_ROWS_PER_GROUP,
) -> dict[str, int]:
    """Per-PCAP row budgets so every required VAL group contributes.

    Owltron power keeps all available rows (typically 40). Other groups take
    up to ``rows_per_group`` from member PCAPs (deterministic pcap_id order).
    """
    budgets: dict[str, int] = {}

    for gkey in SMOKE_VAL_ATTACK_GROUPS:
        members = sorted(
            [s for s in specs if s.modeling_group_key == gkey],
            key=lambda s: s.pcap_id,
        )
        remain = int(rows_per_group)
        for spec in members:
            if remain <= 0:
                break
            take = min(int(spec.window_count), remain)
            if take > 0:
                budgets[spec.pcap_id] = take
                remain -= take

    benign_targets: dict[str, int | None] = {
        "profiling_idle": rows_per_group,
        "owltron_interaction": rows_per_group,
        "owltron_power": None,
    }
    for bkey in SMOKE_VAL_BENIGN_GROUPS:
        members = sorted(
            [
                s
                for s in specs
                if s.binary_label == "BENIGN"
                and benign_group_key(s.benign_category, s.modeling_group_key) == bkey
            ],
            key=lambda s: s.pcap_id,
        )
        cap = benign_targets[bkey]
        if cap is None:
            for spec in members:
                budgets[spec.pcap_id] = int(spec.window_count)
            continue
        remain = int(cap)
        for spec in members:
            if remain <= 0:
                break
            take = min(int(spec.window_count), remain)
            if take > 0:
                budgets[spec.pcap_id] = take
                remain -= take
    return budgets


def load_fit_arrays(
    fit_manifest_path: Path,
    *,
    project_root: Path | None = None,
    expected_rows: int | None = EXPECTED_FIT_ROWS,
    max_rows: int | None = None,
    smoke_only: bool = False,
    smoke_rows_per_group: int = SMOKE_ROWS_PER_GROUP,
    require_all_smoke_groups: bool = False,
) -> FitArrays:
    """Load FIT view features+labels into preallocated NumPy arrays.

    Smoke mode takes a fixed per-family slice (benign + DDoS + DoS + MQTT +
    Recon + Spoofing), not a single global row cap.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    rows = load_fit_manifest_rows(fit_manifest_path)
    if not smoke_only and len(rows) != EXPECTED_FIT_PCAPS:
        raise FeatureExtractionError(
            f"fit manifest PCAPs {len(rows)} != {EXPECTED_FIT_PCAPS}"
        )

    if smoke_only:
        return _load_fit_arrays_smoke(
            rows,
            project_root=root,
            rows_per_group=smoke_rows_per_group,
            require_all_groups=require_all_smoke_groups,
        )

    if max_rows is not None:
        n_alloc = int(max_rows)
    elif expected_rows is not None:
        n_alloc = int(expected_rows)
    else:
        n_alloc = sum(int(r["output_row_count"]) for r in rows)

    X = np.empty((n_alloc, len(V1_FEATURE_NAMES)), dtype=np.float32)
    y = np.empty((n_alloc,), dtype=np.uint8)
    pcap_ids: list[str] = []
    cursor = 0

    for row in rows:
        if cursor >= n_alloc:
            break
        rel = str(row["output_parquet_path"])
        reject_test_path(rel)
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FeatureExtractionError(f"FIT shard missing: {path}")

        table = pq.read_table(path, columns=_READ_COLUMNS)
        cols = [c for c in table.column_names if c != "binary_label"]
        assert_feature_columns(cols)

        features = np.column_stack(
            [
                table.column(name).to_numpy(zero_copy_only=False)
                for name in _FEATURE_LIST
            ]
        ).astype(np.float32, copy=False)
        labels = encode_labels(table.column("binary_label").to_pylist())
        if features.shape[0] != labels.shape[0]:
            raise FeatureExtractionError(f"X/y length mismatch in {path}")
        if not np.isfinite(features).all():
            raise FeatureExtractionError(f"non-finite features in FIT shard {path}")

        take = min(features.shape[0], n_alloc - cursor)
        X[cursor : cursor + take] = features[:take]
        y[cursor : cursor + take] = labels[:take]
        pcap_ids.append(str(row["pcap_id"]))
        cursor += take

    if cursor != n_alloc:
        if max_rows is not None:
            X = X[:cursor]
            y = y[:cursor]
        else:
            raise FeatureExtractionError(
                f"loaded FIT rows {cursor} != expected {n_alloc}"
            )

    n_attack = int((y == LABEL_MAPPING["ATTACK"]).sum())
    n_benign = int((y == LABEL_MAPPING["BENIGN"]).sum())
    if max_rows is None:
        if X.shape[0] != EXPECTED_FIT_ROWS:
            raise FeatureExtractionError(
                f"FIT rows {X.shape[0]} != {EXPECTED_FIT_ROWS}"
            )
        if n_attack != EXPECTED_FIT_ATTACK:
            raise FeatureExtractionError(
                f"FIT attack {n_attack} != {EXPECTED_FIT_ATTACK}"
            )
        if n_benign != EXPECTED_FIT_BENIGN:
            raise FeatureExtractionError(
                f"FIT benign {n_benign} != {EXPECTED_FIT_BENIGN}"
            )

    return FitArrays(
        X=X,
        y=y,
        pcap_ids=pcap_ids,
        n_attack=n_attack,
        n_benign=n_benign,
    )


def _load_fit_arrays_smoke(
    rows: list[dict[str, str]],
    *,
    project_root: Path,
    rows_per_group: int,
    require_all_groups: bool,
) -> FitArrays:
    by_bucket: dict[str, list[dict[str, str]]] = {b: [] for b in SMOKE_FIT_BUCKETS}
    for row in rows:
        bucket = _fit_smoke_bucket(row)
        if bucket is not None:
            by_bucket[bucket].append(row)

    missing = [b for b in SMOKE_FIT_BUCKETS if not by_bucket[b]]
    if require_all_groups and missing:
        raise FeatureExtractionError(
            f"smoke FIT missing required groups: {missing}"
        )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pcap_ids: list[str] = []
    loaded_groups: list[str] = []

    for bucket in SMOKE_FIT_BUCKETS:
        members = sorted(by_bucket[bucket], key=lambda r: r["pcap_id"])
        if not members:
            continue
        remain = int(rows_per_group)
        for row in members:
            if remain <= 0:
                break
            rel = str(row["output_parquet_path"])
            reject_test_path(rel)
            path = Path(rel)
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                raise FeatureExtractionError(f"FIT shard missing: {path}")
            features, labels = _read_feature_prefix(path, max_rows=remain)
            if features.shape[0] == 0:
                continue
            xs.append(features)
            ys.append(labels)
            pcap_ids.append(str(row["pcap_id"]))
            remain -= int(features.shape[0])
        if remain < rows_per_group:
            loaded_groups.append(bucket)

    if not xs:
        raise FeatureExtractionError("smoke FIT selection produced zero rows")
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return FitArrays(
        X=X,
        y=y,
        pcap_ids=pcap_ids,
        n_attack=int((y == LABEL_MAPPING["ATTACK"]).sum()),
        n_benign=int((y == LABEL_MAPPING["BENIGN"]).sum()),
        smoke_groups_loaded=tuple(loaded_groups),
    )


@dataclass
class ValidationBatch:
    spec: ValidationPcapSpec
    X: np.ndarray
    y: np.ndarray


def iter_validation_batches(
    specs: list[ValidationPcapSpec],
    *,
    project_root: Path | None = None,
    batch_rows: int = 65_536,
    max_rows: int | None = None,
    row_budget_by_pcap: dict[str, int] | None = None,
) -> Iterator[ValidationBatch]:
    """Yield feature/label batches from unsampled TRAIN-validation PCAPs.

    When ``row_budget_by_pcap`` is set (smoke), only listed PCAPs are read and
    each is capped independently — not a single global FIFO cap.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    emitted = 0
    for spec in specs:
        if row_budget_by_pcap is not None:
            if spec.pcap_id not in row_budget_by_pcap:
                continue
            pcap_budget = int(row_budget_by_pcap[spec.pcap_id])
            if pcap_budget <= 0:
                continue
        else:
            pcap_budget = None

        reject_test_path(spec.feature_parquet_path)
        path = Path(spec.feature_parquet_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FeatureExtractionError(f"validation shard missing: {path}")
        pf = pq.ParquetFile(path)
        pcap_emitted = 0
        for batch in pf.iter_batches(batch_size=batch_rows, columns=_READ_COLUMNS):
            cols = [c for c in batch.schema.names if c != "binary_label"]
            assert_feature_columns(cols)
            arrays = [
                batch.column(name).to_numpy(zero_copy_only=False)
                for name in _FEATURE_LIST
            ]
            X = np.column_stack(arrays).astype(np.float32, copy=False)
            y = encode_labels(batch.column("binary_label").to_pylist())
            if not np.isfinite(X).all():
                raise FeatureExtractionError(
                    f"non-finite features in validation shard {path}"
                )
            if pcap_budget is not None:
                remain = pcap_budget - pcap_emitted
                if remain <= 0:
                    break
                if X.shape[0] > remain:
                    X = X[:remain]
                    y = y[:remain]
            if max_rows is not None:
                remain = max_rows - emitted
                if remain <= 0:
                    return
                if X.shape[0] > remain:
                    X = X[:remain]
                    y = y[:remain]
            emitted += int(X.shape[0])
            pcap_emitted += int(X.shape[0])
            yield ValidationBatch(spec=spec, X=X, y=y)
            if pcap_budget is not None and pcap_emitted >= pcap_budget:
                break
            if max_rows is not None and emitted >= max_rows:
                return


def validate_validation_inventory(
    specs: list[ValidationPcapSpec],
    *,
    smoke_only: bool = False,
) -> dict[str, int]:
    if not smoke_only and len(specs) != EXPECTED_VAL_PCAPS:
        raise FeatureExtractionError(
            f"validation PCAPs {len(specs)} != {EXPECTED_VAL_PCAPS}"
        )
    total = sum(s.window_count for s in specs)
    attack = sum(
        s.window_count for s in specs if s.binary_label == "ATTACK"
    )
    benign = sum(
        s.window_count for s in specs if s.binary_label == "BENIGN"
    )
    if not smoke_only:
        if total != EXPECTED_VAL_ROWS:
            raise FeatureExtractionError(
                f"validation rows {total} != {EXPECTED_VAL_ROWS}"
            )
        if attack != EXPECTED_VAL_ATTACK:
            raise FeatureExtractionError(
                f"validation attack {attack} != {EXPECTED_VAL_ATTACK}"
            )
        if benign != EXPECTED_VAL_BENIGN:
            raise FeatureExtractionError(
                f"validation benign {benign} != {EXPECTED_VAL_BENIGN}"
            )
    return {
        "validation_pcaps": len(specs),
        "validation_rows": total,
        "validation_attack_rows": attack,
        "validation_benign_rows": benign,
    }


def assert_fit_val_disjoint(
    fit_pcap_ids: list[str],
    val_specs: list[ValidationPcapSpec],
) -> None:
    fit_set = set(fit_pcap_ids)
    overlap = sorted(fit_set & {s.pcap_id for s in val_specs})
    if overlap:
        raise FeatureExtractionError(
            f"validation PCAPs leaked into FIT: {overlap[:5]}"
        )
