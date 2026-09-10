#!/usr/bin/env python3
"""
Execute one REAL selected-LR multi-seed stability run for Tech-2.

Purpose
-------
LR screening at seed 8 is already complete and frozen:

    r256 -> backbone LR 3e-4
    r512 -> backbone LR 1e-4

This tool performs confirmation/stability runs only:

    seed 9
    seed 10

For one resolution + seed:

    fresh Stage A
        ->
    Stage-A raw-best checkpoint
        ->
    ONE Stage-B branch using the already-selected LR
        ->
    raw-best weighted-dev-CE checkpoint
        ->
    durable artifacts

There is NO LR search here.

Scientific boundaries
---------------------
This tool:
- uses project_train for training;
- uses dev_val for checkpoint selection;
- uses the exact frozen Stage-A / Stage-B contracts;
- uses weighted dev CE as already frozen;
- computes AUROC only as the existing non-selection diagnostic;
- never accesses held-out test;
- never derives a threshold;
- never compares resolutions;
- never performs Grad-CAM.

Sequential-run Git policy
-------------------------
At the beginning of a run, tracked source/config modifications are
forbidden.

For unattended sequential runs on the same machine, the gate permits
ONLY untracked evidence produced by an earlier run beneath:

    runs/

and canonical validator logs matching:

    logs/validate_experiment_config_*.log

This means seed/resolution run #2 can start without an intermediate Git
commit while still executing from exactly the same scientific Git HEAD.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


# ======================================================================
# Repository path
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


# ======================================================================
# Reuse established production infrastructure
# ======================================================================

from src.config import (
    load_experiment_config,
    load_machine_config,
)

from src.lr_screening import (
    StageAInitializationEvidence,
    _build_stage_a_runtime,
)

from src.objective import (
    WeightedCrossEntropyObjective,
)

from src.reproducibility import (
    establish_pre_cuda_environment,
)

from src.screening_artifacts import (
    _stage_summary,
    _write_checkpoint,
    _write_history_csv,
    format_backbone_lr_tag,
)

from src.stage_b_branching import (
    StageBBranchInitializationEvidence,
    build_stage_b_branch,
)

from src.stage_runner import (
    StageRunResult,
    run_training_stage,
)

from tools.run_resnet18_lr_screening import (
    RUN_SUBDIRECTORIES,
    configure_run_logging,
    production_source_hashes,
    resolve_runs_root,
    run_canonical_validator,
    run_git_command,
    runtime_provenance,
    safe_component,
    sha256_file,
    write_yaml_atomic,
)


LOGGER = logging.getLogger(
    "tech2.resnet18"
)


# ======================================================================
# Frozen seed-8 screening evidence
# ======================================================================

SCREENING_CODE_COMMIT = (
    "745f47f3fccf43694e9cf3478a261f2fd902cac9"
)


LOCKED_SELECTIONS = {
    "r256": {
        "expected_backbone_lr":
            0.0003,

        "summary_path":
            (
                "runs/"
                "20260910T005651_789477Z_"
                "resnet18_lr_screen_r256_seed8_"
                "sush-AMD_745f47f3/"
                "metrics/lr_screening_summary.yaml"
            ),

        "summary_sha256":
            (
                "62b3b68b9c06de82ff54bbfdb90f5fc3"
                "12d24d821338997579304519994fe87c"
            ),
    },

    "r512": {
        "expected_backbone_lr":
            0.0001,

        "summary_path":
            (
                "runs/"
                "20260910T005713_188852Z_"
                "resnet18_lr_screen_r512_seed8_"
                "IMTA134_745f47f3/"
                "metrics/lr_screening_summary.yaml"
            ),

        "summary_sha256":
            (
                "abd8b8213401e641016c9c12af443d531"
                "97b8e55826b6e97916f965d411efb16"
            ),
    },
}


ALLOWED_STABILITY_SEEDS = {
    9,
    10,
}


# ======================================================================
# Result records
# ======================================================================

@dataclass(
    frozen=True
)
class LockedSelectionEvidence:

    resolution_name: str

    source_summary_path: str
    source_summary_sha256: str

    source_run_id: str
    source_git_commit: str

    screening_seed: int

    backbone_lr: float

    seed8_selected_weighted_dev_loss: float

    seed8_selected_model_state_sha256: str


@dataclass(
    frozen=True
)
class SelectedLRStabilityResult:

    resolution_name: str
    run_seed: int

    backbone_lr: float

    locked_selection: LockedSelectionEvidence

    stage_a_initialization: (
        StageAInitializationEvidence
    )

    stage_a_result: StageRunResult

    stage_b_initialization: (
        StageBBranchInitializationEvidence
    )

    stage_b_result: StageRunResult

    protocol_review_required: bool


@dataclass(
    frozen=True
)
class StabilityArtifactBundle:

    summary_path: str
    summary_sha256: str

    manifest_path: str
    manifest_sha256: str

    selected_checkpoint_path: str
    selected_checkpoint_file_sha256: str
    selected_checkpoint_model_state_sha256: str


# ======================================================================
# Path helpers
# ======================================================================

def resolve_repo_path(
    value: str | Path,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():

        path = (
            REPO_ROOT
            / path
        )

    return path.resolve()


def relative_to_run(
    *,
    run_dir: Path,
    path: Path,
) -> str:

    return (
        path
        .resolve()
        .relative_to(
            run_dir.resolve()
        )
        .as_posix()
    )


# ======================================================================
# Git gate
# ======================================================================

def require_scientifically_clean_git() -> dict[str, Any]:
    """
    Require immutable tracked scientific code/config.

    Untracked evidence from an earlier completed unattended run is
    permitted only beneath runs/ or as canonical validator logs.
    """

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

    permitted_untracked: list[
        str
    ] = []

    blockers: list[
        str
    ] = []

    for line in (
        status.splitlines()
        if status
        else []
    ):

        # Only ?? entries may ever be ignored.
        if line.startswith(
            "?? "
        ):

            path = line[
                3:
            ]

            if path.startswith(
                "runs/"
            ):

                permitted_untracked.append(
                    path
                )

                continue

            if (
                path.startswith(
                    "logs/validate_experiment_config_"
                )
                and path.endswith(
                    ".log"
                )
            ):

                permitted_untracked.append(
                    path
                )

                continue

        # Any modified/staged/deleted/renamed tracked file,
        # and any unrelated untracked file, remains a hard failure.
        blockers.append(
            line
        )

    if blockers:

        raise RuntimeError(
            "Scientific source tree is not clean.\n"
            "Only untracked prior run evidence under runs/ and "
            "canonical validator logs are permitted between "
            "unattended sequential stability runs.\n\n"
            + "\n".join(
                blockers
            )
        )

    if len(
        commit_sha
    ) != 40:

        raise RuntimeError(
            "Git HEAD is not a full 40-character SHA."
        )

    return {
        "commit_sha":
            commit_sha,

        "branch":
            branch,

        "tracked_scientific_tree_clean_at_run_start":
            True,

        "permitted_prior_untracked_evidence_count":
            len(
                permitted_untracked
            ),

        "permitted_prior_untracked_evidence":
            permitted_untracked,
    }


# ======================================================================
# Locked LR evidence
# ======================================================================

def load_locked_selection(
    *,
    resolution_name: str,
) -> LockedSelectionEvidence:

    if resolution_name not in LOCKED_SELECTIONS:

        raise ValueError(
            "Unsupported stability resolution:\n"
            f"  {resolution_name!r}"
        )

    source_cfg = (
        LOCKED_SELECTIONS[
            resolution_name
        ]
    )

    summary_relative = str(
        source_cfg[
            "summary_path"
        ]
    )

    summary_path = resolve_repo_path(
        summary_relative
    )

    if not summary_path.is_file():

        raise FileNotFoundError(
            "Locked seed-8 LR-screening summary is missing:\n"
            f"  {summary_path}"
        )

    expected_sha = str(
        source_cfg[
            "summary_sha256"
        ]
    )

    actual_sha = sha256_file(
        summary_path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Locked seed-8 LR-screening summary SHA mismatch:\n"
            f"  resolution={resolution_name}\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        summary = (
            yaml.safe_load(
                file
            )
            or {}
        )

    if not isinstance(
        summary,
        Mapping,
    ):

        raise TypeError(
            "Locked screening summary must be a mapping."
        )

    if summary.get(
        "schema_version"
    ) != 1:

        raise RuntimeError(
            "Unexpected locked screening-summary schema."
        )

    if summary.get(
        "artifact_type"
    ) != "resolution_lr_screening_summary":

        raise RuntimeError(
            "Unexpected locked screening artifact type."
        )

    run = summary.get(
        "run"
    )

    scope = summary.get(
        "scope"
    )

    selection = summary.get(
        "selection"
    )

    if not isinstance(
        run,
        Mapping,
    ):

        raise RuntimeError(
            "Locked screening summary missing run mapping."
        )

    if not isinstance(
        scope,
        Mapping,
    ):

        raise RuntimeError(
            "Locked screening summary missing scope mapping."
        )

    if not isinstance(
        selection,
        Mapping,
    ):

        raise RuntimeError(
            "Locked screening summary missing selection mapping."
        )

    if str(
        run.get(
            "git_commit"
        )
    ) != SCREENING_CODE_COMMIT:

        raise RuntimeError(
            "Locked screening summary came from unexpected "
            "scientific code commit."
        )

    if str(
        scope.get(
            "resolution_name"
        )
    ) != resolution_name:

        raise RuntimeError(
            "Locked screening summary resolution mismatch."
        )

    screening_seed = int(
        scope.get(
            "run_seed"
        )
    )

    if screening_seed != 8:

        raise RuntimeError(
            "Locked LR selection must originate from seed 8."
        )

    if scope.get(
        "held_out_test_accessed"
    ) is not False:

        raise RuntimeError(
            "Locked screening evidence does not preserve "
            "held-out-test boundary."
        )

    expected_candidates = [
        0.00003,
        0.0001,
        0.0003,
    ]

    observed_candidates = [
        float(
            value
        )
        for value
        in scope.get(
            "backbone_lr_candidates",
            [],
        )
    ]

    if observed_candidates != expected_candidates:

        raise RuntimeError(
            "Locked screening LR candidate set changed."
        )

    backbone_lr = float(
        selection.get(
            "selected_backbone_lr"
        )
    )

    expected_lr = float(
        source_cfg[
            "expected_backbone_lr"
        ]
    )

    if backbone_lr != expected_lr:

        raise RuntimeError(
            "Locked selected LR disagrees with frozen seed-8 "
            "screening result:\n"
            f"  resolution={resolution_name}\n"
            f"  expected={expected_lr}\n"
            f"  actual={backbone_lr}"
        )

    if selection.get(
        "exact_loss_tie_encountered"
    ) is not False:

        raise RuntimeError(
            "Expected seed-8 LR selection to have no exact tie."
        )

    if selection.get(
        "protocol_review_required"
    ) is not False:

        raise RuntimeError(
            "Seed-8 LR selection unexpectedly requires "
            "protocol review."
        )

    if selection.get(
        "auroc_changes_selection"
    ) is not False:

        raise RuntimeError(
            "AUROC must not have changed seed-8 LR selection."
        )

    selected_loss = float(
        selection[
            "selected_raw_best_weighted_dev_loss"
        ]
    )

    selected_model_sha = str(
        selection[
            "selected_stage_b_checkpoint_model_state_sha256"
        ]
    )

    if len(
        selected_model_sha
    ) != 64:

        raise RuntimeError(
            "Locked seed-8 model-state SHA is malformed."
        )

    return LockedSelectionEvidence(
        resolution_name=(
            resolution_name
        ),

        source_summary_path=(
            summary_relative
        ),

        source_summary_sha256=(
            actual_sha
        ),

        source_run_id=str(
            run[
                "run_id"
            ]
        ),

        source_git_commit=str(
            run[
                "git_commit"
            ]
        ),

        screening_seed=(
            screening_seed
        ),

        backbone_lr=(
            backbone_lr
        ),

        seed8_selected_weighted_dev_loss=(
            selected_loss
        ),

        seed8_selected_model_state_sha256=(
            selected_model_sha
        ),
    )


# ======================================================================
# Scientific computation
# ======================================================================

def run_selected_lr_stability(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    repo_root: Path,
    resolution_name: str,
    run_seed: int,
    locked_selection: LockedSelectionEvidence,
) -> SelectedLRStabilityResult:

    if run_seed not in ALLOWED_STABILITY_SEEDS:

        raise ValueError(
            "Selected-LR stability runner permits only "
            "confirmation seeds 9 and 10:\n"
            f"  actual={run_seed}"
        )

    if (
        locked_selection.resolution_name
        != resolution_name
    ):

        raise RuntimeError(
            "Locked LR evidence resolution mismatch."
        )

    backbone_lr = float(
        locked_selection.backbone_lr
    )

    root = Path(
        repo_root
    ).resolve()

    LOGGER.info(
        "Stability run start | resolution=%s | seed=%d | "
        "locked_backbone_lr=%.8g",
        resolution_name,
        run_seed,
        backbone_lr,
    )

    # ==================================================================
    # Stage A — fresh for this resolution + run seed.
    # ==================================================================

    LOGGER.info(
        "Stage A start | resolution=%s | seed=%d",
        resolution_name,
        run_seed,
    )

    (
        stage_a_model,
        stage_a_loaders,
        stage_a_optimizer,
        stage_a_objective,
        device,
        stage_a_initialization,
    ) = _build_stage_a_runtime(
        experiment_cfg=(
            experiment_cfg
        ),
        machine_cfg=(
            machine_cfg
        ),
        repo_root=(
            root
        ),
        resolution_name=(
            resolution_name
        ),
        run_seed=(
            run_seed
        ),
    )

    stage_a_result = run_training_stage(
        experiment_cfg=(
            experiment_cfg
        ),
        stage="stage_a",
        model=(
            stage_a_model
        ),
        project_train_loader=(
            stage_a_loaders
            .project_train
        ),
        dev_val_loader=(
            stage_a_loaders
            .dev_val
        ),
        optimizer=(
            stage_a_optimizer
        ),
        objective=(
            stage_a_objective
        ),
        device=(
            device
        ),
    )

    if stage_a_result.stage != "stage_a":

        raise RuntimeError(
            "Stage-A runner returned wrong stage identity."
        )

    if (
        stage_a_result
        .raw_best_checkpoint
        .stage
        != "stage_a"
    ):

        raise RuntimeError(
            "Stage A did not produce a raw-best Stage-A checkpoint."
        )

    stage_a_checkpoint = (
        stage_a_result
        .raw_best_checkpoint
    )

    stage_a_sha = (
        stage_a_checkpoint
        .model_state_sha256
    )

    LOGGER.info(
        "Stage A selected checkpoint | resolution=%s | seed=%d | "
        "epoch=%d | weighted_dev_loss=%.12f | model_sha256=%s",
        resolution_name,
        run_seed,
        stage_a_result.raw_best_epoch,
        stage_a_result.raw_best_weighted_dev_loss,
        stage_a_sha,
    )

    # Stage B must be reconstructed from the checkpoint, not continued
    # from the live Stage-A optimizer/model/DataLoader state.
    del stage_a_optimizer
    del stage_a_objective
    del stage_a_model
    del stage_a_loaders

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ==================================================================
    # Stage B — ONE already-selected LR only.
    # ==================================================================

    LOGGER.info(
        "Stage B selected-LR branch start | "
        "resolution=%s | seed=%d | backbone_lr=%.8g",
        resolution_name,
        run_seed,
        backbone_lr,
    )

    branch = build_stage_b_branch(
        experiment_cfg=(
            experiment_cfg
        ),
        machine_cfg=(
            machine_cfg
        ),
        repo_root=(
            root
        ),
        resolution_name=(
            resolution_name
        ),
        run_seed=(
            run_seed
        ),
        stage_a_checkpoint=(
            stage_a_checkpoint
        ),
        backbone_lr=(
            backbone_lr
        ),
    )

    initialization = (
        branch
        .initialization_evidence
    )

    if (
        initialization
        .stage_a_checkpoint_model_state_sha256
        != stage_a_sha
    ):

        raise RuntimeError(
            "Stage-B stability branch references wrong "
            "Stage-A checkpoint."
        )

    if (
        initialization
        .restored_model_state_sha256_after_device_move
        != stage_a_sha
    ):

        raise RuntimeError(
            "Stage-B stability branch did not restore exact "
            "Stage-A model state."
        )

    branch_objective = (
        WeightedCrossEntropyObjective(
            experiment_cfg=(
                experiment_cfg
            ),
            device=(
                branch.device
            ),
        )
    )

    stage_b_result = run_training_stage(
        experiment_cfg=(
            experiment_cfg
        ),
        stage="stage_b",
        model=(
            branch.model
        ),
        project_train_loader=(
            branch
            .dataloaders
            .project_train
        ),
        dev_val_loader=(
            branch
            .dataloaders
            .dev_val
        ),
        optimizer=(
            branch.optimizer
        ),
        objective=(
            branch_objective
        ),
        device=(
            branch.device
        ),
    )

    if stage_b_result.stage != "stage_b":

        raise RuntimeError(
            "Stage-B runner returned wrong stage identity."
        )

    if (
        stage_b_result
        .raw_best_checkpoint
        .stage
        != "stage_b"
    ):

        raise RuntimeError(
            "Stage B did not produce a raw-best Stage-B checkpoint."
        )

    disagreement = (
        stage_b_result
        .stage_b_auroc_disagreement
    )

    if disagreement is None:

        raise RuntimeError(
            "Stage-B stability result is missing AUROC "
            "disagreement evidence."
        )

    protocol_review_required = bool(
        disagreement
        .protocol_review_flag
    )

    LOGGER.info(
        "Stage B selected-LR branch complete | "
        "resolution=%s | seed=%d | backbone_lr=%.8g | "
        "epochs=%d | raw_best_epoch=%d | "
        "raw_best_dev_loss=%.12f | "
        "AUROC_at_raw_best=%.12f | best_AUROC=%.12f | "
        "AUROC_difference=%.12f | protocol_review=%s",
        resolution_name,
        run_seed,
        backbone_lr,
        stage_b_result.epochs_completed,
        stage_b_result.raw_best_epoch,
        stage_b_result.raw_best_weighted_dev_loss,
        stage_b_result.dev_auroc_at_raw_best_checkpoint,
        stage_b_result.best_dev_auroc,
        disagreement.difference,
        protocol_review_required,
    )

    del branch_objective
    del branch

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return SelectedLRStabilityResult(
        resolution_name=(
            resolution_name
        ),

        run_seed=(
            run_seed
        ),

        backbone_lr=(
            backbone_lr
        ),

        locked_selection=(
            locked_selection
        ),

        stage_a_initialization=(
            stage_a_initialization
        ),

        stage_a_result=(
            stage_a_result
        ),

        stage_b_initialization=(
            initialization
        ),

        stage_b_result=(
            stage_b_result
        ),

        protocol_review_required=(
            protocol_review_required
        ),
    )


# ======================================================================
# Artifact serialization helpers
# ======================================================================

def new_staging_path(
    final_path: Path,
) -> Path:

    return final_path.with_name(
        final_path.name
        + ".partial."
        + uuid.uuid4().hex
    )


def assert_absent(
    paths: list[
        Path
    ],
) -> None:

    existing = [
        path
        for path
        in paths
        if path.exists()
    ]

    if existing:

        raise FileExistsError(
            "Refusing to overwrite existing stability artifacts:\n"
            + "\n".join(
                str(
                    path
                )
                for path
                in existing
            )
        )


def stage_a_initialization_summary(
    evidence: StageAInitializationEvidence,
) -> dict[str, Any]:

    return {
        "fresh_model_state_sha256":
            evidence.fresh_model_state_sha256,

        "project_train_generator_initial_state_sha256":
            (
                evidence
                .project_train_generator_initial_state_sha256
            ),

        "dev_val_generator_initial_state_sha256":
            (
                evidence
                .dev_val_generator_initial_state_sha256
            ),

        "optimizer_state_entries_at_construction":
            (
                evidence
                .optimizer_state_entries_at_construction
            ),

        "pretrained_checkpoint_sha256":
            (
                evidence
                .model_provenance
                .pretrained_checkpoint_sha256
            ),

        "requested_device":
            evidence.requested_device,

        "actual_model_device":
            evidence.actual_model_device,
    }


def stage_b_initialization_summary(
    evidence: StageBBranchInitializationEvidence,
) -> dict[str, Any]:

    return {
        "stage_a_checkpoint_epoch":
            evidence.stage_a_checkpoint_epoch,

        "stage_a_checkpoint_weighted_dev_loss":
            (
                evidence
                .stage_a_checkpoint_weighted_dev_loss
            ),

        "stage_a_checkpoint_model_state_sha256":
            (
                evidence
                .stage_a_checkpoint_model_state_sha256
            ),

        "fresh_model_state_sha256_before_restore":
            (
                evidence
                .fresh_model_state_sha256_before_restore
            ),

        "restored_model_state_sha256_cpu":
            (
                evidence
                .restored_model_state_sha256_cpu
            ),

        "restored_model_state_sha256_after_device_move":
            (
                evidence
                .restored_model_state_sha256_after_device_move
            ),

        "project_train_generator_initial_state_sha256":
            (
                evidence
                .project_train_generator_initial_state_sha256
            ),

        "dev_val_generator_initial_state_sha256":
            (
                evidence
                .dev_val_generator_initial_state_sha256
            ),

        "optimizer_state_entries_at_construction":
            (
                evidence
                .optimizer_state_entries_at_construction
            ),
    }


# ======================================================================
# Stability persistence
# ======================================================================

def persist_stability_result(
    *,
    result: SelectedLRStabilityResult,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    experiment_config_sha256: str,
) -> StabilityArtifactBundle:

    root = Path(
        run_dir
    ).resolve()

    checkpoints_dir = (
        root
        / "checkpoints"
    )

    metrics_dir = (
        root
        / "metrics"
    )

    manifests_dir = (
        root
        / "manifests"
    )

    for directory in (
        checkpoints_dir,
        metrics_dir,
        manifests_dir,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    lr_tag = format_backbone_lr_tag(
        result.backbone_lr
    )

    stage_a_checkpoint_path = (
        checkpoints_dir
        / "stage_a_raw_best.pt"
    )

    stage_a_history_path = (
        metrics_dir
        / "stage_a_history.csv"
    )

    stage_b_checkpoint_path = (
        checkpoints_dir
        / (
            f"stage_b_lr_{lr_tag}"
            "_raw_best.pt"
        )
    )

    stage_b_history_path = (
        metrics_dir
        / (
            f"stage_b_lr_{lr_tag}"
            "_history.csv"
        )
    )

    summary_path = (
        metrics_dir
        / "stability_summary.yaml"
    )

    manifest_path = (
        manifests_dir
        / "stability_artifacts.yaml"
    )

    final_paths = [
        stage_a_checkpoint_path,
        stage_a_history_path,
        stage_b_checkpoint_path,
        stage_b_history_path,
        summary_path,
        manifest_path,
    ]

    assert_absent(
        final_paths
    )

    staged: list[
        tuple[
            Path,
            Path,
        ]
    ] = []

    artifact_records: list[
        dict[str, Any]
    ] = []

    moved: list[
        Path
    ] = []

    def stage_file(
        final_path: Path,
    ) -> Path:

        temporary = (
            new_staging_path(
                final_path
            )
        )

        staged.append(
            (
                temporary,
                final_path,
            )
        )

        return temporary

    try:

        # ==============================================================
        # Stage A checkpoint
        # ==============================================================

        temp = stage_file(
            stage_a_checkpoint_path
        )

        _write_checkpoint(
            path=(
                temp
            ),
            checkpoint=(
                result
                .stage_a_result
                .raw_best_checkpoint
            ),
            run_id=(
                run_id
            ),
            resolution_name=(
                result.resolution_name
            ),
            run_seed=(
                result.run_seed
            ),
            backbone_lr=None,
        )

        artifact_records.append(
            {
                "path":
                    relative_to_run(
                        run_dir=root,
                        path=(
                            stage_a_checkpoint_path
                        ),
                    ),

                "sha256":
                    sha256_file(
                        temp
                    ),

                "size_bytes":
                    temp.stat().st_size,

                "artifact_type":
                    "raw_argmin_model_checkpoint",

                "stage":
                    "stage_a",

                "backbone_lr":
                    None,

                "model_state_sha256":
                    (
                        result
                        .stage_a_result
                        .raw_best_checkpoint
                        .model_state_sha256
                    ),

                "checkpoint_epoch":
                    (
                        result
                        .stage_a_result
                        .raw_best_epoch
                    ),

                "weighted_dev_loss":
                    (
                        result
                        .stage_a_result
                        .raw_best_weighted_dev_loss
                    ),
            }
        )

        # ==============================================================
        # Stage A history
        # ==============================================================

        temp = stage_file(
            stage_a_history_path
        )

        _write_history_csv(
            path=(
                temp
            ),
            result=(
                result
                .stage_a_result
            ),
        )

        artifact_records.append(
            {
                "path":
                    relative_to_run(
                        run_dir=root,
                        path=(
                            stage_a_history_path
                        ),
                    ),

                "sha256":
                    sha256_file(
                        temp
                    ),

                "size_bytes":
                    temp.stat().st_size,

                "artifact_type":
                    "epoch_history_csv",

                "stage":
                    "stage_a",
            }
        )

        # ==============================================================
        # Stage B checkpoint
        # ==============================================================

        temp = stage_file(
            stage_b_checkpoint_path
        )

        _write_checkpoint(
            path=(
                temp
            ),
            checkpoint=(
                result
                .stage_b_result
                .raw_best_checkpoint
            ),
            run_id=(
                run_id
            ),
            resolution_name=(
                result.resolution_name
            ),
            run_seed=(
                result.run_seed
            ),
            backbone_lr=(
                result.backbone_lr
            ),
        )

        selected_checkpoint_file_sha = (
            sha256_file(
                temp
            )
        )

        selected_checkpoint_model_sha = (
            result
            .stage_b_result
            .raw_best_checkpoint
            .model_state_sha256
        )

        artifact_records.append(
            {
                "path":
                    relative_to_run(
                        run_dir=root,
                        path=(
                            stage_b_checkpoint_path
                        ),
                    ),

                "sha256":
                    selected_checkpoint_file_sha,

                "size_bytes":
                    temp.stat().st_size,

                "artifact_type":
                    "raw_argmin_model_checkpoint",

                "stage":
                    "stage_b",

                "backbone_lr":
                    result.backbone_lr,

                "model_state_sha256":
                    selected_checkpoint_model_sha,

                "checkpoint_epoch":
                    (
                        result
                        .stage_b_result
                        .raw_best_epoch
                    ),

                "weighted_dev_loss":
                    (
                        result
                        .stage_b_result
                        .raw_best_weighted_dev_loss
                    ),

                "selected":
                    True,
            }
        )

        # ==============================================================
        # Stage B history
        # ==============================================================

        temp = stage_file(
            stage_b_history_path
        )

        _write_history_csv(
            path=(
                temp
            ),
            result=(
                result
                .stage_b_result
            ),
        )

        artifact_records.append(
            {
                "path":
                    relative_to_run(
                        run_dir=root,
                        path=(
                            stage_b_history_path
                        ),
                    ),

                "sha256":
                    sha256_file(
                        temp
                    ),

                "size_bytes":
                    temp.stat().st_size,

                "artifact_type":
                    "epoch_history_csv",

                "stage":
                    "stage_b",

                "backbone_lr":
                    result.backbone_lr,

                "selected":
                    True,
            }
        )

        # ==============================================================
        # Summary
        # ==============================================================

        locked = (
            result
            .locked_selection
        )

        summary = {
            "schema_version":
                1,

            "artifact_type":
                "resolution_selected_lr_stability_summary",

            "run":
                {
                    "run_id":
                        run_id,

                    "git_commit":
                        git_commit,

                    "experiment_config_sha256":
                        experiment_config_sha256,
                },

            "scope":
                {
                    "resolution_name":
                        result.resolution_name,

                    "run_seed":
                        result.run_seed,

                    "backbone_lr":
                        result.backbone_lr,

                    "workflow":
                        "selected_lr_multi_seed_stability",

                    "held_out_test_accessed":
                        False,
                },

            "locked_lr_source":
                {
                    "policy":
                        "reuse_seed8_lr_screening_winner",

                    "summary_path":
                        locked.source_summary_path,

                    "summary_sha256":
                        locked.source_summary_sha256,

                    "source_run_id":
                        locked.source_run_id,

                    "source_git_commit":
                        locked.source_git_commit,

                    "screening_seed":
                        locked.screening_seed,

                    "selected_backbone_lr":
                        locked.backbone_lr,

                    "seed8_selected_weighted_dev_loss":
                        (
                            locked
                            .seed8_selected_weighted_dev_loss
                        ),

                    "seed8_selected_model_state_sha256":
                        (
                            locked
                            .seed8_selected_model_state_sha256
                        ),
                },

            "stage_a_initialization":
                stage_a_initialization_summary(
                    result
                    .stage_a_initialization
                ),

            "stage_a":
                _stage_summary(
                    result
                    .stage_a_result
                ),

            "stage_b":
                {
                    "backbone_lr":
                        result.backbone_lr,

                    "initialization":
                        stage_b_initialization_summary(
                            result
                            .stage_b_initialization
                        ),

                    "training":
                        _stage_summary(
                            result
                            .stage_b_result
                        ),

                    "protocol_review_required":
                        (
                            result
                            .protocol_review_required
                        ),
                },

            "selection":
                {
                    "checkpoint_source":
                        "raw_argmin_weighted_dev_loss",

                    "selected_backbone_lr":
                        result.backbone_lr,

                    "selected_raw_best_weighted_dev_loss":
                        (
                            result
                            .stage_b_result
                            .raw_best_weighted_dev_loss
                        ),

                    "selected_stage_b_checkpoint_model_state_sha256":
                        (
                            selected_checkpoint_model_sha
                        ),

                    "auroc_changes_checkpoint_selection":
                        False,

                    "protocol_review_required":
                        (
                            result
                            .protocol_review_required
                        ),
                },
        }

        temp = stage_file(
            summary_path
        )

        write_yaml_atomic(
            path=(
                temp
            ),
            value=(
                summary
            ),
        )

        summary_sha = sha256_file(
            temp
        )

        artifact_records.append(
            {
                "path":
                    relative_to_run(
                        run_dir=root,
                        path=(
                            summary_path
                        ),
                    ),

                "sha256":
                    summary_sha,

                "size_bytes":
                    temp.stat().st_size,

                "artifact_type":
                    "selected_lr_stability_summary_yaml",
            }
        )

        # ==============================================================
        # Manifest — deliberately does not list itself.
        # ==============================================================

        manifest = {
            "schema_version":
                1,

            "artifact_type":
                "selected_lr_stability_artifact_manifest",

            "run":
                {
                    "run_id":
                        run_id,

                    "git_commit":
                        git_commit,

                    "resolution_name":
                        result.resolution_name,

                    "run_seed":
                        result.run_seed,

                    "backbone_lr":
                        result.backbone_lr,
                },

            "held_out_test_accessed":
                False,

            "manifest_self_excluded":
                True,

            "artifact_count":
                len(
                    artifact_records
                ),

            "artifacts":
                artifact_records,
        }

        temp = stage_file(
            manifest_path
        )

        write_yaml_atomic(
            path=(
                temp
            ),
            value=(
                manifest
            ),
        )

        manifest_sha = sha256_file(
            temp
        )

        # ==============================================================
        # Promote all staged artifacts.
        # ==============================================================

        for (
            temporary,
            final,
        ) in staged:

            if final.exists():

                raise FileExistsError(
                    final
                )

            os.replace(
                temporary,
                final,
            )

            moved.append(
                final
            )

        return StabilityArtifactBundle(
            summary_path=(
                relative_to_run(
                    run_dir=root,
                    path=(
                        summary_path
                    ),
                )
            ),

            summary_sha256=(
                summary_sha
            ),

            manifest_path=(
                relative_to_run(
                    run_dir=root,
                    path=(
                        manifest_path
                    ),
                )
            ),

            manifest_sha256=(
                manifest_sha
            ),

            selected_checkpoint_path=(
                relative_to_run(
                    run_dir=root,
                    path=(
                        stage_b_checkpoint_path
                    ),
                )
            ),

            selected_checkpoint_file_sha256=(
                selected_checkpoint_file_sha
            ),

            selected_checkpoint_model_state_sha256=(
                selected_checkpoint_model_sha
            ),
        )

    except Exception:

        # Remove staged temporaries.
        for (
            temporary,
            _final,
        ) in staged:

            try:

                if temporary.exists():

                    temporary.unlink()

            except OSError:

                pass

        # Roll back final files created by this persistence call.
        for path in reversed(
            moved
        ):

            try:

                if path.exists():

                    path.unlink()

            except OSError:

                pass

        raise


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen selected-LR ResNet-18 "
            "multi-seed stability experiment."
        )
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
        required=True,
        type=int,
        choices=(
            9,
            10,
        ),
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

    args = parser.parse_args()

    # ==================================================================
    # Pre-run source/provenance gates.
    # ==================================================================

    git_info = (
        require_scientifically_clean_git()
    )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            args.config
        )
    )

    # MUST precede anything that may initialize CUDA.
    cublas_workspace_config = (
        establish_pre_cuda_environment(
            experiment_cfg=(
                experiment_cfg
            )
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
            experiment_path=(
                experiment_path
            ),
            machine_path=(
                machine_path
            ),
        )
    )

    experiment_sha = (
        sha256_file(
            experiment_path
        )
    )

    locked_selection = (
        load_locked_selection(
            resolution_name=(
                args.resolution
            )
        )
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

    runtime_info = (
        runtime_provenance(
            configured_device=(
                configured_device
            )
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
    ] = (
        cublas_workspace_config
    )

    source_hashes = (
        production_source_hashes()
    )

    source_hashes[
        "tools/run_resnet18_lr_screening.py"
    ] = sha256_file(
        REPO_ROOT
        / "tools"
        / "run_resnet18_lr_screening.py"
    )

    source_hashes[
        "tools/run_resnet18_stability.py"
    ] = sha256_file(
        Path(
            __file__
        ).resolve()
    )

    # ==================================================================
    # Immutable run identity.
    # ==================================================================

    created_at = (
        datetime.now(
            timezone.utc
        )
    )

    timestamp = (
        created_at.strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
    )

    run_id = (
        f"{timestamp}_"
        f"resnet18_stability_"
        f"{safe_component(args.resolution)}_"
        f"seed{args.run_seed}_"
        f"{safe_component(machine_id)}_"
        f"{git_info['commit_sha'][:8]}"
    )

    runs_root = (
        resolve_runs_root(
            machine_cfg
        )
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
    # Exact scientific-config snapshot.
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
            "Scientific config snapshot SHA mismatch."
        )

    # ==================================================================
    # Logging.
    # ==================================================================

    log_path = (
        run_dir
        / "logs"
        / "stability.log"
    )

    configure_run_logging(
        log_path=(
            log_path
        )
    )

    run_yaml_path = (
        run_dir
        / "run.yaml"
    )

    # ==================================================================
    # Initial run record.
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
                    "selected_lr_multi_seed_stability",

                "status":
                    "initialized",

                "resolution_name":
                    args.resolution,

                "run_seed":
                    args.run_seed,

                "backbone_lr":
                    locked_selection.backbone_lr,
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

        "locked_lr_source":
            {
                "policy":
                    "reuse_seed8_lr_screening_winner",

                "summary_path":
                    (
                        locked_selection
                        .source_summary_path
                    ),

                "summary_sha256":
                    (
                        locked_selection
                        .source_summary_sha256
                    ),

                "source_run_id":
                    (
                        locked_selection
                        .source_run_id
                    ),

                "source_git_commit":
                    (
                        locked_selection
                        .source_git_commit
                    ),

                "screening_seed":
                    8,

                "backbone_lr":
                    locked_selection.backbone_lr,

                "seed8_selected_weighted_dev_loss":
                    (
                        locked_selection
                        .seed8_selected_weighted_dev_loss
                    ),
            },

        "validator":
            validator_artifact,

        "machine":
            {
                "id":
                    machine_id,

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
                "stability_log":
                    "logs/stability.log",

                "stability_summary":
                    None,

                "artifact_manifest":
                    None,

                "selected_checkpoint":
                    None,
            },

        "timing":
            {
                "training_seconds":
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
        path=(
            run_yaml_path
        ),
        value=(
            run_record
        ),
    )

    LOGGER.info(
        "CUBLAS_WORKSPACE_CONFIG: %s",
        cublas_workspace_config,
    )

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "RESNET-18 SELECTED-LR STABILITY RUN"
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
        "Locked backbone LR: %.8g",
        locked_selection.backbone_lr,
    )

    LOGGER.info(
        "Locked LR source: %s",
        locked_selection.source_summary_path,
    )

    LOGGER.info(
        "Experiment config SHA-256: %s",
        experiment_sha,
    )

    LOGGER.info(
        "Held-out test: NOT ACCESSED"
    )

    started_at = (
        datetime.now(
            timezone.utc
        )
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
    ] = (
        started_at.isoformat()
    )

    write_yaml_atomic(
        path=(
            run_yaml_path
        ),
        value=(
            run_record
        ),
    )

    try:

        training_start = (
            time.perf_counter()
        )

        result = (
            run_selected_lr_stability(
                experiment_cfg=(
                    experiment_cfg
                ),
                machine_cfg=(
                    machine_cfg
                ),
                repo_root=(
                    REPO_ROOT
                ),
                resolution_name=(
                    args.resolution
                ),
                run_seed=(
                    args.run_seed
                ),
                locked_selection=(
                    locked_selection
                ),
            )
        )

        training_seconds = (
            time.perf_counter()
            - training_start
        )

        persistence_start = (
            time.perf_counter()
        )

        artifact_bundle = (
            persist_stability_result(
                result=(
                    result
                ),
                run_dir=(
                    run_dir
                ),
                run_id=(
                    run_id
                ),
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

        total_epochs = (
            result
            .stage_a_result
            .epochs_completed
            +
            result
            .stage_b_result
            .epochs_completed
        )

        mean_seconds_per_epoch = (
            training_seconds
            / total_epochs
        )

        completed_at = (
            datetime.now(
                timezone.utc
            )
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
        ] = (
            completed_at.isoformat()
        )

        run_record[
            "artifacts"
        ][
            "stability_summary"
        ] = {
            "path":
                (
                    artifact_bundle
                    .summary_path
                ),

            "sha256":
                (
                    artifact_bundle
                    .summary_sha256
                ),
        }

        run_record[
            "artifacts"
        ][
            "artifact_manifest"
        ] = {
            "path":
                (
                    artifact_bundle
                    .manifest_path
                ),

            "sha256":
                (
                    artifact_bundle
                    .manifest_sha256
                ),
        }

        run_record[
            "artifacts"
        ][
            "selected_checkpoint"
        ] = {
            "path":
                (
                    artifact_bundle
                    .selected_checkpoint_path
                ),

            "file_sha256":
                (
                    artifact_bundle
                    .selected_checkpoint_file_sha256
                ),

            "model_state_sha256":
                (
                    artifact_bundle
                    .selected_checkpoint_model_state_sha256
                ),

            "backbone_lr":
                result.backbone_lr,

            "checkpoint_epoch":
                (
                    result
                    .stage_b_result
                    .raw_best_epoch
                ),

            "weighted_dev_loss":
                (
                    result
                    .stage_b_result
                    .raw_best_weighted_dev_loss
                ),
        }

        run_record[
            "timing"
        ] = {
            "training_seconds":
                training_seconds,

            "persistence_seconds":
                persistence_seconds,

            "total_completed_training_epochs":
                total_epochs,

            "mean_seconds_per_completed_epoch":
                mean_seconds_per_epoch,
        }

        run_record[
            "result"
        ] = {
            "backbone_lr":
                result.backbone_lr,

            "raw_best_weighted_dev_loss":
                (
                    result
                    .stage_b_result
                    .raw_best_weighted_dev_loss
                ),

            "dev_auroc_at_raw_best_checkpoint":
                (
                    result
                    .stage_b_result
                    .dev_auroc_at_raw_best_checkpoint
                ),

            "best_dev_auroc":
                (
                    result
                    .stage_b_result
                    .best_dev_auroc
                ),

            "protocol_review_required":
                (
                    result
                    .protocol_review_required
                ),
        }

        write_yaml_atomic(
            path=(
                run_yaml_path
            ),
            value=(
                run_record
            ),
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "SELECTED-LR STABILITY RUN: PASS"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Run directory: %s",
            run_dir,
        )

        LOGGER.info(
            "Resolution: %s",
            result.resolution_name,
        )

        LOGGER.info(
            "Seed: %d",
            result.run_seed,
        )

        LOGGER.info(
            "Backbone LR: %.8g",
            result.backbone_lr,
        )

        LOGGER.info(
            "Selected raw-best weighted dev loss: %.12f",
            (
                result
                .stage_b_result
                .raw_best_weighted_dev_loss
            ),
        )

        LOGGER.info(
            "AUROC at selected checkpoint: %.12f",
            (
                result
                .stage_b_result
                .dev_auroc_at_raw_best_checkpoint
            ),
        )

        LOGGER.info(
            "Protocol review required: %s",
            result.protocol_review_required,
        )

        LOGGER.info(
            "Held-out test: NOT ACCESSED"
        )

        return 0

    except Exception as exc:

        failed_at = (
            datetime.now(
                timezone.utc
            )
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
        ] = (
            failed_at.isoformat()
        )

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

        write_yaml_atomic(
            path=(
                run_yaml_path
            ),
            value=(
                run_record
            ),
        )

        LOGGER.exception(
            "SELECTED-LR STABILITY RUN: FAIL"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )