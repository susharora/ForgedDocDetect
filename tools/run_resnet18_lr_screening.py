#!/usr/bin/env python3
"""
Execute one REAL frozen Tech-2 ResNet-18 LR-screening experiment.

One invocation runs exactly one resolution:

    r256, seed 8

or:

    r512, seed 8

Scientific computation
-----------------------
The command invokes the already-validated production path:

    run_resolution_lr_screening()
        ->
    Stage A once
        ->
    Stage-B LR 3e-5
        ->
    Stage-B LR 1e-4
        ->
    Stage-B LR 3e-4
        ->
    frozen weighted-dev-loss LR selection
        ->
    persist_resolution_lr_screening_result()

Run provenance
--------------
Before training:

- requires clean Git;
- runs canonical experiment validation;
- snapshots the exact scientific YAML;
- records Git, machine and runtime provenance;
- hashes the production source modules;
- creates a unique immutable run directory.

After training:

- persists Stage-A raw-best checkpoint;
- persists all three Stage-B raw-best checkpoints;
- persists all four epoch histories;
- persists LR-screening summary;
- persists artifact manifest;
- updates run.yaml with result identity and timing.

Failure behavior
----------------
A failed run directory is deliberately retained with:

    status: failed
    failure type/message
    scientific log
    provenance

so failed scientific execution is auditable rather than silently removed.

Important boundaries
--------------------
This command has NO held-out-test input or route.

It does NOT:
- derive FPR10 thresholds;
- access held-out test;
- perform resolution selection;
- perform multi-seed stability;
- perform Grad-CAM.

This is the real training entry point. Do not run it until its separate
synthetic integration audit has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml


# ======================================================================
# Repository imports
# ======================================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    REPO_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            REPO_ROOT
        ),
    )

from src.reproducibility import (
    establish_pre_cuda_environment,
)

from src.config import (
    load_experiment_config,
    load_machine_config,
)

from src.lr_screening import (
    ResolutionLRScreeningResult,
    run_resolution_lr_screening,
)

from src.screening_artifacts import (
    ScreeningArtifactBundle,
    persist_resolution_lr_screening_result,
)


# ======================================================================
# Constants
# ======================================================================

RUN_SUBDIRECTORIES = (
    "logs",
    "metrics",
    "predictions",
    "checkpoints",
    "localisation/maps",
    "localisation/rendered",
    "diagnostics",
    "manifests",
)


CORE_SOURCE_FILES = (
    "src/data.py",
    "src/dataloading.py",
    "src/reproducibility.py",
    "src/modeling.py",
    "src/objective.py",
    "src/optimization.py",
    "src/engine.py",
    "src/training_control.py",
    "src/development_metrics.py",
    "src/stage_b_branching.py",
    "src/stage_runner.py",
    "src/lr_screening.py",
    "src/screening_artifacts.py",
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


LOGGER = logging.getLogger(
    "tech2.resnet18"
)


# ======================================================================
# File hashing
# ======================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ======================================================================
# Git provenance
# ======================================================================

def run_git_command(
    args: list[str],
) -> str:

    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return (
        result
        .stdout
        .strip()
    )


def require_clean_git() -> dict[str, Any]:

    commit_sha = run_git_command(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    branch = run_git_command(
        [
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ]
    )

    status = run_git_command(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    )

    if status:

        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit/remove outstanding changes before starting "
            "a scientific LR-screening run.\n\n"
            f"{status}"
        )

    if (
        len(
            commit_sha
        )
        != 40
    ):

        raise RuntimeError(
            "Git HEAD is not a full 40-character SHA."
        )

    return {
        "commit_sha":
            commit_sha,

        "branch":
            branch,

        "working_tree_clean_at_run_start":
            True,
    }


# ======================================================================
# Configuration validator
# ======================================================================

def run_canonical_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> dict[str, str]:

    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "validate_experiment_config.py"
            ),
            "--config",
            str(
                experiment_path
            ),
            "--machine-config",
            str(
                machine_path
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Canonical experiment configuration validation failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    handoff_lines = [
        line.strip()
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]

    if len(
        handoff_lines
    ) != 1:

        raise RuntimeError(
            "Expected exactly one validator handoff line:\n"
            f"{handoff_lines}"
        )

    match = VALIDATION_HANDOFF_PATTERN.match(
        handoff_lines[
            0
        ]
    )

    if match is None:

        raise RuntimeError(
            "Could not parse validator handoff:\n"
            f"{handoff_lines[0]}"
        )

    if (
        match.group(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Canonical validator did not return PASS."
        )

    artifact_path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    artifact_sha = match.group(
        "sha256"
    )

    if not artifact_path.is_file():

        raise FileNotFoundError(
            artifact_path
        )

    if (
        sha256_file(
            artifact_path
        )
        != artifact_sha
    ):

        raise RuntimeError(
            "Validator artifact SHA-256 mismatch."
        )

    return {
        "path":
            str(
                artifact_path
            ),

        "sha256":
            artifact_sha,
    }


# ======================================================================
# Small path helpers
# ======================================================================

def safe_component(
    value: str,
) -> str:

    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        value.strip(),
    )

    cleaned = cleaned.strip(
        "-._"
    )

    if not cleaned:

        raise ValueError(
            "Filesystem component became empty after sanitization."
        )

    return cleaned


def resolve_runs_root(
    machine_cfg: dict[str, Any],
) -> Path:

    value = (
        machine_cfg[
            "paths"
        ][
            "runs_root"
        ]
    )

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():

        path = (
            REPO_ROOT
            / path
        )

    return path.resolve()


# ======================================================================
# Atomic YAML updates
# ======================================================================

def write_yaml_atomic(
    *,
    path: Path,
    value: dict[str, Any],
) -> None:

    temporary_path = path.with_name(
        path.name
        + ".partial."
        + uuid.uuid4().hex
    )

    try:

        with temporary_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file:

            yaml.safe_dump(
                value,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        os.replace(
            temporary_path,
            path,
        )

    except Exception:

        if temporary_path.exists():

            temporary_path.unlink()

        raise


# ======================================================================
# Logging
# ======================================================================

def configure_run_logging(
    *,
    log_path: Path,
) -> None:

    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(
        logging.INFO
    )

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    formatter.converter = (
        time.gmtime
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="x",
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    stream_handler = logging.StreamHandler(
        sys.stdout
    )

    stream_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        file_handler
    )

    LOGGER.addHandler(
        stream_handler
    )


# ======================================================================
# Runtime provenance
# ======================================================================

def runtime_provenance(
    *,
    configured_device: str,
) -> dict[str, Any]:

    device = torch.device(
        configured_device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "Machine config requests CUDA but CUDA is unavailable."
        )

    selected_gpu: (
        dict[str, Any]
        | None
    ) = None

    if device.type == "cuda":

        index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )

        properties = (
            torch.cuda
            .get_device_properties(
                index
            )
        )

        selected_gpu = {
            "index":
                index,

            "name":
                properties.name,

            "total_memory_bytes":
                properties.total_memory,

            "total_memory_gib":
                (
                    properties.total_memory
                    / (
                        1024
                        ** 3
                    )
                ),

            "compute_capability":
                (
                    f"{properties.major}."
                    f"{properties.minor}"
                ),
        }

    cudnn_version = None

    if torch.backends.cudnn.is_available():

        cudnn_version = (
            torch.backends.cudnn.version()
        )

    return {
        "python":
            {
                "version":
                    platform.python_version(),

                "executable":
                    sys.executable,
            },

        "packages":
            {
                "torch":
                    str(
                        torch.__version__
                    ),

                "torchvision":
                    str(
                        torchvision.__version__
                    ),
            },

        "cuda":
            {
                "available":
                    torch.cuda.is_available(),

                "torch_cuda_version":
                    torch.version.cuda,

                "cudnn_version":
                    cudnn_version,
            },

        "configured_device":
            configured_device,

        "selected_gpu":
            selected_gpu,
    }


# ======================================================================
# Production source provenance
# ======================================================================

def production_source_hashes() -> dict[str, str]:

    result: dict[
        str,
        str,
    ] = {}

    for relative_path in (
        CORE_SOURCE_FILES
    ):

        path = (
            REPO_ROOT
            / relative_path
        )

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

        result[
            relative_path
        ] = sha256_file(
            path
        )

    return result


# ======================================================================
# Result logging
# ======================================================================

def log_screening_result(
    result: ResolutionLRScreeningResult,
) -> None:

    stage_a = (
        result
        .stage_a_result
    )

    LOGGER.info(
        "Stage A complete | epochs=%d | stop=%s | "
        "raw_best_epoch=%d | raw_best_dev_loss=%.12f | "
        "AUROC_at_raw_best=%.12f | best_AUROC=%.12f",
        stage_a.epochs_completed,
        stage_a.stop_reason,
        stage_a.raw_best_epoch,
        stage_a.raw_best_weighted_dev_loss,
        stage_a.dev_auroc_at_raw_best_checkpoint,
        stage_a.best_dev_auroc,
    )

    for candidate in (
        result
        .stage_b_candidates
    ):

        stage_b = (
            candidate
            .stage_b_result
        )

        disagreement = (
            stage_b
            .stage_b_auroc_disagreement
        )

        if disagreement is None:

            raise RuntimeError(
                "Completed Stage-B candidate missing AUROC disagreement."
            )

        LOGGER.info(
            "Stage B complete | backbone_lr=%.8g | "
            "epochs=%d | stop=%s | raw_best_epoch=%d | "
            "raw_best_dev_loss=%.12f | AUROC_at_raw_best=%.12f | "
            "best_AUROC=%.12f | AUROC_difference=%.12f | "
            "protocol_review=%s",
            candidate.backbone_lr,
            stage_b.epochs_completed,
            stage_b.stop_reason,
            stage_b.raw_best_epoch,
            stage_b.raw_best_weighted_dev_loss,
            stage_b.dev_auroc_at_raw_best_checkpoint,
            stage_b.best_dev_auroc,
            disagreement.difference,
            disagreement.protocol_review_flag,
        )

    LOGGER.info(
        "LR selection | selected_backbone_lr=%.8g | "
        "selected_raw_best_dev_loss=%.12f | exact_loss_tie=%s | "
        "protocol_review_required=%s",
        result.selected_backbone_lr,
        result.selected_raw_best_weighted_dev_loss,
        result.exact_loss_tie_encountered,
        result.protocol_review_required,
    )


# ======================================================================
# Bundle helpers
# ======================================================================

def artifact_lookup(
    *,
    bundle: ScreeningArtifactBundle,
) -> dict[
    str,
    Any,
]:

    result = {
        artifact.path:
            artifact

        for artifact
        in bundle.artifacts
    }

    if len(
        result
    ) != len(
        bundle.artifacts
    ):

        raise RuntimeError(
            "Persisted bundle contains duplicate artifact paths."
        )

    return result


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run one real frozen Tech-2 "
            "ResNet-18 resolution LR screen."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/experiments/"
            "resnet18_gradcam.yaml"
        ),
    )

    parser.add_argument(
        "--machine-config",
        default="configs/local.yaml",
    )

    parser.add_argument(
        "--resolution",
        required=True,
        choices=(
            "r256",
            "r512",
        ),
    )

    parser.add_argument(
        "--run-seed",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    # ==================================================================
    # Pre-run gates.
    #
    # All of these happen BEFORE run directory creation.
    # ==================================================================

    git_info = require_clean_git()

    if args.run_seed != 8:

        raise ValueError(
            "Initial LR screening requires frozen run_seed=8."
        )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            args.config
        )
    )

    # ==================================================================
    # PRE-CUDA deterministic environment.
    #
    # This MUST happen before runtime_provenance(), because that function
    # queries the CUDA device and may initialize CUDA.
    # ==================================================================

    cublas_workspace_config = (
        establish_pre_cuda_environment(
            experiment_cfg=experiment_cfg,
        )
    )

    machine_cfg, machine_path = (
        load_machine_config(
            args.machine_config,
            required=True,
        )
    )

    validator_artifact = (
        run_canonical_validator(
            experiment_path=experiment_path,
            machine_path=machine_path,
        )
    )

    experiment_sha = sha256_file(
        experiment_path
    )

    machine_id = str(
        machine_cfg[
            "machine"
        ][
            "id"
        ]
    )

    configured_device = str(
        machine_cfg[
            "runtime"
        ][
            "device"
        ]
    )

    runtime_info = runtime_provenance(
        configured_device=(
            configured_device
        )
    )

    if (
        os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        )
        != cublas_workspace_config
    ):

        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG changed after "
            "CUDA runtime provenance collection."
        )

    runtime_info[
        "cublas_workspace_config"
    ] = cublas_workspace_config

    source_hashes = (
        production_source_hashes()
    )

    # ==================================================================
    # Immutable run identity.
    # ==================================================================

    created_at = datetime.now(
        timezone.utc
    )

    timestamp = created_at.strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    run_id = (
        f"{timestamp}_"
        f"resnet18_lr_screen_"
        f"{safe_component(args.resolution)}_"
        f"seed{args.run_seed}_"
        f"{safe_component(machine_id)}_"
        f"{git_info['commit_sha'][:8]}"
    )

    runs_root = resolve_runs_root(
        machine_cfg
    )

    runs_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir = (
        runs_root
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for subdirectory in (
        RUN_SUBDIRECTORIES
    ):

        (
            run_dir
            / subdirectory
        ).mkdir(
            parents=True,
            exist_ok=False,
        )

    # ==================================================================
    # Exact scientific config snapshot.
    # ==================================================================

    config_snapshot_path = (
        run_dir
        / "experiment_config.yaml"
    )

    shutil.copy2(
        experiment_path,
        config_snapshot_path,
    )

    snapshot_sha = sha256_file(
        config_snapshot_path
    )

    if snapshot_sha != experiment_sha:

        raise RuntimeError(
            "Scientific config snapshot SHA does not match source."
        )

    # ==================================================================
    # Logging.
    # ==================================================================

    screening_log_path = (
        run_dir
        / "logs"
        / "lr_screening.log"
    )

    configure_run_logging(
        log_path=screening_log_path
    )

    run_yaml_path = (
        run_dir
        / "run.yaml"
    )

    # ==================================================================
    # Initial provenance record.
    # ==================================================================

    run_record: dict[
        str,
        Any,
    ] = {
        "schema_version":
            1,

        "run":
            {
                "run_id":
                    run_id,

                "created_at_utc":
                    created_at.isoformat(),

                "workflow":
                    "resolution_lr_screening",

                "status":
                    "initialized",

                "resolution_name":
                    args.resolution,

                "run_seed":
                    args.run_seed,
            },

        "git":
            git_info,

        "scientific_config":
            {
                "source_path":
                    str(
                        experiment_path
                    ),

                "snapshot":
                    "experiment_config.yaml",

                "sha256":
                    experiment_sha,

                "protocol_status":
                    experiment_cfg[
                        "experiment"
                    ][
                        "protocol_status"
                    ],
            },

        "validator":
            validator_artifact,

        "machine":
            {
                "id":
                    machine_id,

                "hostname":
                    platform.node(),

                "configured_device":
                    configured_device,

                "num_workers":
                    machine_cfg[
                        "runtime"
                    ][
                        "num_workers"
                    ],

                "dataset_root":
                    str(
                        machine_cfg[
                            "paths"
                        ][
                            "dataset_root"
                        ]
                    ),

                "runs_root":
                    str(
                        runs_root
                    ),
            },

        "runtime":
            runtime_info,

        "production_source_sha256":
            source_hashes,

        "frozen_data":
            {
                "project_train_manifest_sha256":
                    (
                        experiment_cfg[
                            "data"
                        ][
                            "frozen_split"
                        ][
                            "project_train"
                        ][
                            "sha256"
                        ]
                    ),

                "dev_val_manifest_sha256":
                    (
                        experiment_cfg[
                            "data"
                        ][
                            "frozen_split"
                        ][
                            "dev_val"
                        ][
                            "sha256"
                        ]
                    ),

                "held_out_test_accessed":
                    False,
            },

        "artifacts":
            {
                "screening_log":
                    "logs/lr_screening.log",

                "screening_summary":
                    None,

                "artifact_manifest":
                    None,

                "selected_checkpoint":
                    None,
            },

        "timing":
            {
                "screening_seconds":
                    None,

                "persistence_seconds":
                    None,

                "total_completed_training_epochs":
                    None,

                "mean_seconds_per_completed_epoch":
                    None,
            },
    }

    write_yaml_atomic(
        path=run_yaml_path,
        value=run_record,
    )

    LOGGER.info(
        "CUBLAS_WORKSPACE_CONFIG: %s",
        cublas_workspace_config,
    )

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "RESNET-18 REAL LR SCREENING"
    )

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "Run ID: %s",
        run_id,
    )

    LOGGER.info(
        "Git commit: %s",
        git_info[
            "commit_sha"
        ],
    )

    LOGGER.info(
        "Machine: %s",
        machine_id,
    )

    LOGGER.info(
        "Resolution: %s",
        args.resolution,
    )

    LOGGER.info(
        "Run seed: %d",
        args.run_seed,
    )

    LOGGER.info(
        "Device: %s",
        configured_device,
    )

    LOGGER.info(
        "Experiment config SHA-256: %s",
        experiment_sha,
    )

    LOGGER.info(
        "Held-out test: NOT ACCESSED"
    )

    try:

        # ==============================================================
        # Mark running before scientific computation begins.
        # ==============================================================

        started_at = datetime.now(
            timezone.utc
        )

        run_record[
            "run"
        ][
            "status"
        ] = "running"

        run_record[
            "run"
        ][
            "started_at_utc"
        ] = started_at.isoformat()

        write_yaml_atomic(
            path=run_yaml_path,
            value=run_record,
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "Starting scientific training:"
        )

        LOGGER.info(
            "  Stage A once, then Stage B for backbone LRs "
            "[3e-5, 1e-4, 3e-4]."
        )
        
        LOGGER.info(
           "  Live epoch timing/loss/AUROC progress will be logged below."
        )

        # ==============================================================
        # REAL scientific screening.
        # ==============================================================

        screening_start = time.perf_counter()

        screening_result = (
            run_resolution_lr_screening(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                repo_root=REPO_ROOT,
                resolution_name=(
                    args.resolution
                ),
                run_seed=args.run_seed,
            )
        )

        screening_seconds = (
            time.perf_counter()
            - screening_start
        )

        log_screening_result(
            screening_result
        )

        total_epochs = (
            screening_result
            .stage_a_result
            .epochs_completed
            +
            sum(
                candidate
                .stage_b_result
                .epochs_completed

                for candidate
                in (
                    screening_result
                    .stage_b_candidates
                )
            )
        )

        if total_epochs <= 0:

            raise RuntimeError(
                "Completed screening reports zero total training epochs."
            )

        mean_seconds_per_epoch = (
            screening_seconds
            / total_epochs
        )

        LOGGER.info(
            "Scientific screening complete | total_epochs=%d | "
            "duration_seconds=%.3f | mean_seconds_per_epoch=%.3f",
            total_epochs,
            screening_seconds,
            mean_seconds_per_epoch,
        )

        # ==============================================================
        # Durable persistence.
        # ==============================================================

        persistence_start = (
            time.perf_counter()
        )

        bundle = (
            persist_resolution_lr_screening_result(
                result=screening_result,
                run_dir=run_dir,
                run_id=run_id,
                git_commit=(
                    git_info[
                        "commit_sha"
                    ]
                ),
                experiment_config_sha256=(
                    experiment_sha
                ),
            )
        )

        persistence_seconds = (
            time.perf_counter()
            - persistence_start
        )

        artifacts = artifact_lookup(
            bundle=bundle
        )

        if (
            bundle.summary_path
            not in artifacts
        ):

            raise RuntimeError(
                "Persisted bundle is missing screening summary."
            )

        if (
            bundle.manifest_path
            not in artifacts
        ):

            raise RuntimeError(
                "Persisted bundle is missing artifact manifest."
            )

        selected_checkpoint_artifacts = [
            artifact

            for artifact
            in bundle.artifacts

            if (
                artifact.artifact_type
                == "raw_argmin_model_checkpoint"
                and artifact.stage
                == "stage_b"
                and artifact.selected
                is True
            )
        ]

        if len(
            selected_checkpoint_artifacts
        ) != 1:

            raise RuntimeError(
                "Persisted bundle must contain exactly one "
                "selected Stage-B checkpoint."
            )

        selected_checkpoint_artifact = (
            selected_checkpoint_artifacts[
                0
            ]
        )

        if (
            selected_checkpoint_artifact
            .model_state_sha256
            != screening_result
            .selected_stage_b_checkpoint_sha256
        ):

            raise RuntimeError(
                "Persisted selected checkpoint identity disagrees "
                "with screening result."
            )

        # ==============================================================
        # Successful run record.
        # ==============================================================

        completed_at = datetime.now(
            timezone.utc
        )

        run_record[
            "run"
        ][
            "status"
        ] = "completed"

        run_record[
            "run"
        ][
            "completed_at_utc"
        ] = completed_at.isoformat()

        run_record[
            "frozen_data"
        ][
            "held_out_test_accessed"
        ] = False

        run_record[
            "result"
        ] = {
            "selected_backbone_lr":
                (
                    screening_result
                    .selected_backbone_lr
                ),

            "selected_raw_best_weighted_dev_loss":
                (
                    screening_result
                    .selected_raw_best_weighted_dev_loss
                ),

            "selected_stage_b_checkpoint_model_state_sha256":
                (
                    screening_result
                    .selected_stage_b_checkpoint_sha256
                ),

            "exact_loss_tie_encountered":
                (
                    screening_result
                    .exact_loss_tie_encountered
                ),

            "protocol_review_required":
                (
                    screening_result
                    .protocol_review_required
                ),
        }

        run_record[
            "artifacts"
        ][
            "screening_summary"
        ] = {
            "path":
                bundle.summary_path,

            "sha256":
                artifacts[
                    bundle.summary_path
                ].sha256,
        }

        run_record[
            "artifacts"
        ][
            "artifact_manifest"
        ] = {
            "path":
                bundle.manifest_path,

            "sha256":
                artifacts[
                    bundle.manifest_path
                ].sha256,
        }

        run_record[
            "artifacts"
        ][
            "selected_checkpoint"
        ] = {
            "path":
                selected_checkpoint_artifact.path,

            "file_sha256":
                selected_checkpoint_artifact.sha256,

            "model_state_sha256":
                (
                    selected_checkpoint_artifact
                    .model_state_sha256
                ),

            "backbone_lr":
                selected_checkpoint_artifact.backbone_lr,

            "checkpoint_epoch":
                (
                    selected_checkpoint_artifact
                    .checkpoint_epoch
                ),

            "weighted_dev_loss":
                (
                    selected_checkpoint_artifact
                    .weighted_dev_loss
                ),
        }

        run_record[
            "timing"
        ] = {
            "screening_seconds":
                screening_seconds,

            "persistence_seconds":
                persistence_seconds,

            "total_completed_training_epochs":
                total_epochs,

            "mean_seconds_per_completed_epoch":
                mean_seconds_per_epoch,
        }

        write_yaml_atomic(
            path=run_yaml_path,
            value=run_record,
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "REAL LR SCREENING: PASS"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Run directory: %s",
            run_dir,
        )

        LOGGER.info(
            "Selected backbone LR: %.8g",
            screening_result
            .selected_backbone_lr,
        )

        LOGGER.info(
            "Selected raw-best weighted dev loss: %.12f",
            screening_result
            .selected_raw_best_weighted_dev_loss,
        )

        LOGGER.info(
            "Artifact manifest: %s",
            (
                run_dir
                / bundle.manifest_path
            ),
        )

        LOGGER.info(
            "Protocol review required: %s",
            screening_result
            .protocol_review_required,
        )

        LOGGER.info(
            "Held-out test: NOT ACCESSED"
        )

        return 0

    except Exception as exc:

        LOGGER.exception(
            "REAL LR SCREENING: FAIL"
        )

        failed_at = datetime.now(
            timezone.utc
        )

        run_record[
            "run"
        ][
            "status"
        ] = "failed"

        run_record[
            "run"
        ][
            "failed_at_utc"
        ] = failed_at.isoformat()

        run_record[
            "failure"
        ] = {
            "exception_type":
                type(
                    exc
                ).__name__,

            "message":
                str(
                    exc
                ),
        }

        run_record[
            "frozen_data"
        ][
            "held_out_test_accessed"
        ] = False

        try:

            write_yaml_atomic(
                path=run_yaml_path,
                value=run_record,
            )

        except Exception:

            LOGGER.exception(
                "Failed to update run.yaml with failure state."
            )

        return 1

    finally:

        for handler in list(
            LOGGER.handlers
        ):

            handler.flush()
            handler.close()

        LOGGER.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )