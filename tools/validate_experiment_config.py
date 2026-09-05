#!/usr/bin/env python3
"""
Validate a scientific experiment configuration before any run starts.

Checks:
- scientific experiment YAML loads correctly;
- machine-local YAML obeys the allowed machine-only contract;
- machine dataset root exists;
- experiment config SHA-256 is reported;
- frozen Tech-1 source workbook, split bundle and manifests exist;
- every frozen artifact matches its declared SHA-256;
- manifest row counts match the frozen configuration;
- experiment config and frozen split bundle agree with one another;
- held-out test policy remains explicitly locked.

This script is read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

# Allow this standalone tool to import src/config.py when executed as:
# python tools/validate_experiment_config.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config, load_machine_config


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_path(path_value: str | Path) -> Path:
    """Resolve repo-relative paths while preserving absolute machine paths."""
    path = Path(path_value).expanduser()

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Require a YAML value to be a mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping.")

    return value


def require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    """Return a required mapping key with a useful failure message."""
    if key not in mapping:
        raise KeyError(f"Missing required key: {label}.{key}")

    return mapping[key]


def check_sha256(label: str, path: Path, expected: str) -> None:
    """Verify that a file exists and matches its declared SHA-256."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    actual = sha256_file(path)

    if actual.lower() != str(expected).lower():
        raise ValueError(
            f"{label} SHA-256 mismatch:\n"
            f"  path:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    print(f"[PASS] {label} SHA-256")
    print(f"       {actual}")


def count_csv_rows(path: Path) -> int:
    """Count CSV data rows, excluding the header."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        try:
            next(reader)
        except StopIteration:
            return 0

        return sum(1 for _ in reader)


def check_csv_rows(label: str, path: Path, expected: int) -> None:
    """Verify the number of data rows in a CSV manifest."""
    actual = count_csv_rows(path)

    if actual != expected:
        raise ValueError(
            f"{label} row-count mismatch:\n"
            f"  path:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    print(f"[PASS] {label} rows = {actual}")


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return require_mapping(data, str(path))


def check_equal(label: str, actual: Any, expected: Any) -> None:
    """Fail if two provenance values disagree."""
    if actual != expected:
        raise ValueError(
            f"{label} mismatch:\n"
            f"  expected: {expected!r}\n"
            f"  actual:   {actual!r}"
        )

    print(f"[PASS] {label}")


def validate_machine_config(machine_cfg: dict[str, Any]) -> None:
    """Check the small amount of machine state required at this stage."""
    paths = require_mapping(
        require_key(machine_cfg, "paths", "machine_config"),
        "machine_config.paths",
    )

    dataset_root = resolve_path(
        require_key(paths, "dataset_root", "machine_config.paths")
    )

    runs_root = resolve_path(
        require_key(paths, "runs_root", "machine_config.paths")
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Configured dataset_root does not exist or is not a directory: "
            f"{dataset_root}"
        )

    print(f"[PASS] dataset_root exists")
    print(f"       {dataset_root}")

    # runs_root is allowed not to exist yet.
    # The future run initializer will create it safely.
    print(f"[INFO] runs_root")
    print(f"       {runs_root}")


def validate_seeds(config: dict[str, Any]) -> None:
    """Require the scientific seeds declared by the current contract."""
    reproducibility = require_mapping(
        require_key(config, "reproducibility", "config"),
        "reproducibility",
    )

    seeds = require_mapping(
        require_key(reproducibility, "seeds", "reproducibility"),
        "reproducibility.seeds",
    )

    required_seeds = ("python", "numpy", "torch", "dataloader")

    for name in required_seeds:
        value = require_key(seeds, name, "reproducibility.seeds")

        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"reproducibility.seeds.{name} must be an integer, "
                f"got {value!r}"
            )

    print("[PASS] required random seeds are explicitly defined")


def validate_frozen_data(config: dict[str, Any]) -> None:
    """Verify the Tech-1 frozen artifacts and cross-check the bundle."""

    data_cfg = require_mapping(
        require_key(config, "data", "config"),
        "data",
    )

    # ------------------------------------------------------------------
    # Frozen discovery workbook
    # ------------------------------------------------------------------
    source_cfg = require_mapping(
        require_key(data_cfg, "source_discovery", "data"),
        "data.source_discovery",
    )

    workbook_path = resolve_path(
        require_key(source_cfg, "workbook", "data.source_discovery")
    )

    workbook_sha = require_key(
        source_cfg,
        "sha256",
        "data.source_discovery",
    )

    check_sha256(
        "source discovery workbook",
        workbook_path,
        workbook_sha,
    )

    # ------------------------------------------------------------------
    # Frozen split bundle
    # ------------------------------------------------------------------
    split_cfg = require_mapping(
        require_key(data_cfg, "frozen_split", "data"),
        "data.frozen_split",
    )

    bundle_path = resolve_path(
        require_key(split_cfg, "bundle", "data.frozen_split")
    )

    bundle_sha = require_key(
        split_cfg,
        "bundle_sha256",
        "data.frozen_split",
    )

    check_sha256(
        "frozen split bundle",
        bundle_path,
        bundle_sha,
    )

    bundle = load_yaml_file(bundle_path)

    bundle_source = require_mapping(
        require_key(bundle, "source_discovery", "bundle"),
        "bundle.source_discovery",
    )

    bundle_counts = require_mapping(
        require_key(bundle, "counts", "bundle"),
        "bundle.counts",
    )

    bundle_artifacts = require_mapping(
        require_key(bundle, "artifacts", "bundle"),
        "bundle.artifacts",
    )

    # The experiment config and frozen bundle must describe the same
    # discovery source.
    check_equal(
        "bundle source workbook filename",
        workbook_path.name,
        require_key(
            bundle_source,
            "workbook",
            "bundle.source_discovery",
        ),
    )

    check_equal(
        "bundle source workbook SHA-256",
        workbook_sha,
        require_key(
            bundle_source,
            "workbook_sha256",
            "bundle.source_discovery",
        ),
    )

    # ------------------------------------------------------------------
    # Frozen manifests
    # ------------------------------------------------------------------
    manifest_specs = (
        ("cards", "rows"),
        ("project_train", "images"),
        ("dev_val", "images"),
    )

    for manifest_name, config_row_key in manifest_specs:
        manifest_cfg = require_mapping(
            require_key(
                split_cfg,
                manifest_name,
                "data.frozen_split",
            ),
            f"data.frozen_split.{manifest_name}",
        )

        bundle_manifest = require_mapping(
            require_key(
                bundle_artifacts,
                manifest_name,
                "bundle.artifacts",
            ),
            f"bundle.artifacts.{manifest_name}",
        )

        manifest_path = resolve_path(
            require_key(
                manifest_cfg,
                "path",
                f"data.frozen_split.{manifest_name}",
            )
        )

        manifest_sha = require_key(
            manifest_cfg,
            "sha256",
            f"data.frozen_split.{manifest_name}",
        )

        expected_rows = require_key(
            manifest_cfg,
            config_row_key,
            f"data.frozen_split.{manifest_name}",
        )

        # Experiment config must point to the exact file named by the bundle.
        check_equal(
            f"{manifest_name} filename agrees with bundle",
            manifest_path.name,
            require_key(
                bundle_manifest,
                "filename",
                f"bundle.artifacts.{manifest_name}",
            ),
        )

        check_equal(
            f"{manifest_name} SHA-256 agrees with bundle",
            manifest_sha,
            require_key(
                bundle_manifest,
                "sha256",
                f"bundle.artifacts.{manifest_name}",
            ),
        )

        check_equal(
            f"{manifest_name} expected rows agree with bundle",
            expected_rows,
            require_key(
                bundle_manifest,
                "rows",
                f"bundle.artifacts.{manifest_name}",
            ),
        )

        check_sha256(
            f"{manifest_name} manifest",
            manifest_path,
            manifest_sha,
        )

        check_csv_rows(
            f"{manifest_name} manifest",
            manifest_path,
            expected_rows,
        )

    # ------------------------------------------------------------------
    # Card/image counts
    # ------------------------------------------------------------------
    project_train_cfg = require_mapping(
        split_cfg["project_train"],
        "data.frozen_split.project_train",
    )

    dev_val_cfg = require_mapping(
        split_cfg["dev_val"],
        "data.frozen_split.dev_val",
    )

    check_equal(
        "project_train card count",
        require_key(
            project_train_cfg,
            "cards",
            "data.frozen_split.project_train",
        ),
        require_key(
            bundle_counts,
            "project_train_cards",
            "bundle.counts",
        ),
    )

    check_equal(
        "project_train image count",
        require_key(
            project_train_cfg,
            "images",
            "data.frozen_split.project_train",
        ),
        require_key(
            bundle_counts,
            "project_train_images",
            "bundle.counts",
        ),
    )

    check_equal(
        "dev_val card count",
        require_key(
            dev_val_cfg,
            "cards",
            "data.frozen_split.dev_val",
        ),
        require_key(
            bundle_counts,
            "dev_val_cards",
            "bundle.counts",
        ),
    )

    check_equal(
        "dev_val image count",
        require_key(
            dev_val_cfg,
            "images",
            "data.frozen_split.dev_val",
        ),
        require_key(
            bundle_counts,
            "dev_val_images",
            "bundle.counts",
        ),
    )


def validate_test_policy(config: dict[str, Any]) -> None:
    """Ensure the held-out test set is still explicitly protected."""
    test_cfg = require_mapping(
        require_key(config, "held_out_test", "config"),
        "held_out_test",
    )

    check_equal(
        "held-out test policy",
        require_key(test_cfg, "policy", "held_out_test"),
        "untouched_until_final_evaluation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Tech-2 experiment configuration and frozen provenance."
    )

    parser.add_argument(
        "--config",
        default="configs/experiments/resnet18_gradcam.yaml",
        help="Committed scientific experiment YAML.",
    )

    parser.add_argument(
        "--machine-config",
        default="configs/local.yaml",
        help="Machine-local YAML.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("TECH-2 EXPERIMENT CONFIGURATION VALIDATION")
    print("=" * 72)

    experiment_cfg, experiment_path = load_experiment_config(args.config)

    # Require the real machine config here rather than silently using {}.
    # Before an experiment can run, its machine paths must be explicit.
    machine_cfg, machine_path = load_machine_config(
        args.machine_config,
        required=True,
    )

    print(f"\nExperiment config:")
    print(f"  {experiment_path}")
    print(f"  SHA-256: {sha256_file(experiment_path)}")

    print(f"\nMachine config:")
    print(f"  {machine_path}")
    print("  SHA-256 intentionally not treated as scientific provenance")

    print("\n--- Machine contract ---")
    validate_machine_config(machine_cfg)

    print("\n--- Reproducibility contract ---")
    validate_seeds(experiment_cfg)

    print("\n--- Frozen Tech-1 provenance ---")
    validate_frozen_data(experiment_cfg)

    print("\n--- Held-out test contract ---")
    validate_test_policy(experiment_cfg)

    print("\n" + "=" * 72)
    print("EXPERIMENT CONFIGURATION VALIDATION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()