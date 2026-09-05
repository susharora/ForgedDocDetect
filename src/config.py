from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

# Machine-local configuration must not be able to silently change
# scientific/model-facing decisions.
ALLOWED_MACHINE_SECTIONS = {
    "paths",
    "runtime",
    "machine",
}


def _resolve_repo_path(path: str | Path) -> Path:
    """
    Resolve a path relative to the repository root unless it is already absolute.
    """
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load one YAML file and require a mapping at the top level.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(
            f"Top level of configuration must be a mapping: {path}"
        )

    return data


def load_experiment_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """
    Load the committed scientific experiment configuration.

    Examples of settings that belong here:
        - random seed
        - frozen split identity
        - preprocessing
        - augmentation
        - batch size
        - optimizer
        - learning rate
        - checkpoint-selection policy
        - Grad-CAM policy

    These settings must not vary silently between machines.
    """
    resolved_path = _resolve_repo_path(config_path)
    config = _load_yaml(resolved_path)

    return config, resolved_path


def load_machine_config(
    config_path: str | Path = "configs/local.yaml",
    *,
    required: bool = False,
) -> tuple[dict[str, Any], Path]:
    """
    Load machine-specific settings.

    Machine configuration is intentionally restricted to things such as:
        - dataset/output roots
        - worker count
        - device/runtime settings
        - machine identifier

    It must not contain training/model/preprocessing sections.
    """
    resolved_path = _resolve_repo_path(config_path)

    if not resolved_path.exists():
        if required:
            raise FileNotFoundError(
                f"Machine configuration file not found: {resolved_path}"
            )
        return {}, resolved_path

    config = _load_yaml(resolved_path)

    forbidden_sections = set(config) - ALLOWED_MACHINE_SECTIONS

    if forbidden_sections:
        raise ValueError(
            "Machine-local configuration contains forbidden top-level "
            f"sections: {sorted(forbidden_sections)}. "
            "Scientific settings must live in the committed experiment "
            "configuration."
        )

    return config, resolved_path