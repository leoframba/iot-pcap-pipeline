"""Project path helpers."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "WiFI_and_MQTT"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

DATASET_SCOPE = "wifi_mqtt"
SPLIT_STRATEGY_VERSION = "phase1a_v1"
DEFAULT_SPLIT_SEED = 42


def to_repo_relative(path: Path, project_root: Path | None = None) -> str:
    """Return a POSIX path relative to the project root when possible."""
    root = (project_root or PROJECT_ROOT).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()
