#!/usr/bin/env python3
"""
Create a new immutable experiment run directory and capture run provenance.

This tool:
- requires a clean Git working tree;
- re-runs the Tech-2 configuration/provenance validator;
- creates a unique runs/<run_id>/ directory;
- snapshots the scientific experiment config;
- records Git, machine, Python, PyTorch, CUDA and GPU provenance;
- records frozen Tech-1 artifact identities;
- never overwrites an existing run.

It does NOT train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config, load_machine_config


RUN_SUBDIRS = (
    "logs",
    "metrics",
    "predictions",
    "checkpoints",
    "localisation/maps",
    "localisation/rendered",
    "diagnostics",
    "manifests",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def run_command(args: list[str]) -> str:
    """Run a command in the repository root and return stripped stdout."""
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Require a clean repository and return the exact Git state."""
    commit_sha = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    status = run_command(
        ["git", "status", "--porcelain", "--untracked-files=all"]
    )

    if status:
        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit, stash, or remove outstanding changes before creating "
            "a scientific run.\n\n"
            f"git status --porcelain:\n{status}"
        )

    return {
        "commit_sha": commit_sha,
        "branch": branch,
        "working_tree_clean": True,
    }


def validate_config(config_path: Path, machine_path: Path) -> None:
    """
    Re-run the already validated Tech-2 provenance gate.

    Run creation must never bypass frozen-artifact validation.
    """
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "validate_experiment_config.py"),
        "--config",
        str(config_path),
        "--machine-config",
        str(machine_path),
    ]

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def safe_component(value: str) -> str:
    """Convert free text into a filesystem-safe run-ID component."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = value.strip("-._")

    if not value:
        raise ValueError("Run-ID component became empty after sanitization.")

    return value


def get_nvidia_driver_version() -> str | None:
    """Return the NVIDIA driver version when nvidia-smi is available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    versions = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }

    if not versions:
        return None

    return ",".join(sorted(versions))


def runtime_provenance(device_spec: str) -> dict[str, Any]:
    """Capture Python, PyTorch, CUDA and GPU details."""
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and torchvision must be installed before "
            "initializing a ResNet experiment run."
        ) from exc

    cuda_requested = device_spec.startswith("cuda")

    if cuda_requested and not torch.cuda.is_available():
        raise RuntimeError(
            f"Machine config requests device={device_spec!r}, "
            "but torch.cuda.is_available() is False."
        )

    gpu_devices: list[dict[str, Any]] = []
    selected_device: int | None = None

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)

            gpu_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_vram_bytes": props.total_memory,
                    "total_vram_gib": round(
                        props.total_memory / (1024**3),
                        3,
                    ),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

        if cuda_requested:
            parsed_device = torch.device(device_spec)

            selected_device = (
                parsed_device.index
                if parsed_device.index is not None
                else torch.cuda.current_device()
            )

    cudnn_version = None

    if torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "nvidia_driver_version": get_nvidia_driver_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": {
            "device_count": torch.cuda.device_count(),
            "selected_device": selected_device,
            "devices": gpu_devices,
        },
    }


def pretrained_checkpoint_provenance(
    experiment_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Record pretrained-weight identity if configured.

    Phase 0 has not selected ResNet weights yet, so the expected current
    result is explicitly 'not_configured'. Later Phase 2 will populate this.
    """
    model_cfg = experiment_cfg.get("model")

    if not isinstance(model_cfg, dict):
        return {
            "status": "not_configured",
            "identity": None,
            "sha256": None,
        }

    checkpoint_cfg = model_cfg.get("pretrained_checkpoint")

    if not isinstance(checkpoint_cfg, dict):
        return {
            "status": "not_configured",
            "identity": None,
            "sha256": None,
        }

    return {
        "status": "configured",
        "identity": checkpoint_cfg.get("identity"),
        "sha256": checkpoint_cfg.get("sha256"),
    }


def build_run_record(
    *,
    run_id: str,
    created_at: datetime,
    experiment_cfg: dict[str, Any],
    experiment_path: Path,
    experiment_sha256: str,
    machine_cfg: dict[str, Any],
    git_info: dict[str, Any],
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical run.yaml structure."""
    experiment = experiment_cfg["experiment"]
    seeds = experiment_cfg["reproducibility"]["seeds"]

    data_cfg = experiment_cfg["data"]
    source_cfg = data_cfg["source_discovery"]
    split_cfg = data_cfg["frozen_split"]

    machine = machine_cfg["machine"]
    machine_paths = machine_cfg["paths"]
    runtime_cfg = machine_cfg["runtime"]

    try:
        config_display_path = str(
            experiment_path.relative_to(REPO_ROOT)
        )
    except ValueError:
        config_display_path = str(experiment_path)

    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "created_at_utc": created_at.isoformat(),
            "stage": "phase0_infrastructure",
            "status": "initialized",
        },
        "experiment": {
            "name": experiment["name"],
            "pipeline": experiment["pipeline"],
            "task": experiment["task"],
        },
        "git": git_info,
        "scientific_config": {
            "path": config_display_path,
            "sha256": experiment_sha256,
            "snapshot": "experiment_config.yaml",
        },
        "machine": {
            "configured_id": machine["id"],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "dataset_root": str(
                Path(machine_paths["dataset_root"]).expanduser().resolve()
            ),
            "runs_root": str(
                Path(machine_paths["runs_root"]).expanduser().resolve()
            ),
            "runtime_device": runtime_cfg["device"],
            "num_workers": runtime_cfg["num_workers"],
        },
        "runtime": runtime_info,
        "reproducibility": {
            "seeds": seeds,
        },
        "frozen_data": {
            "dataset": data_cfg["dataset"],
            "source_discovery": {
                "workbook": Path(source_cfg["workbook"]).name,
                "sha256": source_cfg["sha256"],
            },
            "split_bundle": {
                "filename": Path(split_cfg["bundle"]).name,
                "sha256": split_cfg["bundle_sha256"],
            },
            "project_train": {
                "filename": Path(
                    split_cfg["project_train"]["path"]
                ).name,
                "sha256": split_cfg["project_train"]["sha256"],
                "cards": split_cfg["project_train"]["cards"],
                "images": split_cfg["project_train"]["images"],
            },
            "dev_val": {
                "filename": Path(
                    split_cfg["dev_val"]["path"]
                ).name,
                "sha256": split_cfg["dev_val"]["sha256"],
                "cards": split_cfg["dev_val"]["cards"],
                "images": split_cfg["dev_val"]["images"],
            },
            "held_out_test_policy": experiment_cfg[
                "held_out_test"
            ]["policy"],
        },
        "pretrained_checkpoint": pretrained_checkpoint_provenance(
            experiment_cfg
        ),
        "artifact_directories": list(RUN_SUBDIRS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a provenance-tracked Tech-2 run."
    )

    parser.add_argument(
        "--config",
        default="configs/experiments/resnet18_gradcam.yaml",
        help="Committed scientific experiment configuration.",
    )

    parser.add_argument(
        "--machine-config",
        default="configs/local.yaml",
        help="Machine-local configuration.",
    )

    args = parser.parse_args()

    experiment_cfg, experiment_path = load_experiment_config(
        args.config
    )

    machine_cfg, machine_path = load_machine_config(
        args.machine_config,
        required=True,
    )

    print("=" * 72)
    print("TECH-2 RUN INITIALIZATION")
    print("=" * 72)

    print("\n--- Git provenance ---")
    git_info = git_provenance()
    print(f"[PASS] clean Git commit: {git_info['commit_sha']}")

    print("\n--- Configuration/provenance gate ---")
    validate_config(experiment_path, machine_path)

    experiment_sha256 = sha256_file(experiment_path)

    runtime_cfg = machine_cfg["runtime"]
    runtime_info = runtime_provenance(runtime_cfg["device"])

    created_at = datetime.now(timezone.utc)

    experiment_name = safe_component(
        experiment_cfg["experiment"]["name"]
    )

    machine_id = safe_component(machine_cfg["machine"]["id"])

    timestamp = created_at.strftime("%Y%m%dT%H%M%S_%fZ")

    run_id = (
        f"{timestamp}_"
        f"{experiment_name}_"
        f"{machine_id}_"
        f"{git_info['commit_sha'][:8]}"
    )

    runs_root = Path(
        machine_cfg["paths"]["runs_root"]
    ).expanduser()

    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root

    runs_root = runs_root.resolve()
    run_dir = runs_root / run_id

    run_record = build_run_record(
        run_id=run_id,
        created_at=created_at,
        experiment_cfg=experiment_cfg,
        experiment_path=experiment_path,
        experiment_sha256=experiment_sha256,
        machine_cfg=machine_cfg,
        git_info=git_info,
        runtime_info=runtime_info,
    )

    # exist_ok=False is the central no-overwrite guarantee.
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        for subdir in RUN_SUBDIRS:
            (run_dir / subdir).mkdir(parents=True, exist_ok=False)

        # Byte-for-byte scientific-config snapshot.
        snapshot_path = run_dir / "experiment_config.yaml"
        shutil.copy2(experiment_path, snapshot_path)

        snapshot_sha256 = sha256_file(snapshot_path)

        if snapshot_sha256 != experiment_sha256:
            raise RuntimeError(
                "Experiment config snapshot SHA-256 differs from source."
            )

        run_yaml_path = run_dir / "run.yaml"

        with run_yaml_path.open("x", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(
                run_record,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

    except Exception:
        # Safe because this process created run_dir with exist_ok=False.
        shutil.rmtree(run_dir)
        raise

    print("\n" + "=" * 72)
    print("RUN INITIALIZATION: PASS")
    print("=" * 72)
    print(f"Run ID:   {run_id}")
    print(f"Run dir:  {run_dir}")
    print(f"run.yaml: {run_yaml_path}")
    print(f"Git SHA:  {git_info['commit_sha']}")
    print(f"Config:   {experiment_sha256}")


if __name__ == "__main__":
    main()