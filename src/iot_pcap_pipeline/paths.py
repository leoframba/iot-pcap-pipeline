"""Project path helpers."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "WiFI_and_MQTT"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
DEFAULT_AUDIT_CHECKPOINT_DIR = DEFAULT_AUDIT_DIR / ".work"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"

DATASET_SCOPE = "wifi_mqtt"
SPLIT_STRATEGY_VERSION = "phase1a_v1"
DEFAULT_SPLIT_SEED = 42
AUDIT_STRATEGY_VERSION = "phase1b2_v1"
TIMESTAMP_PROBE_STRATEGY_VERSION = "phase1b3_v2"
WINDOWING_STRATEGY_VERSION = "phase1c1_v1"


def to_repo_relative(path: Path, project_root: Path | None = None) -> str:
    """Return a POSIX path relative to the project root when possible."""
    root = (project_root or PROJECT_ROOT).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()
