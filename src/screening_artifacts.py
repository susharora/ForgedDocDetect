"""
Durable artifact persistence for one completed Tech-2 resolution LR screen.

Scientific artifact policy
--------------------------
Small scientific evidence is written beneath the run directory:

    metrics/
    manifests/

Large model checkpoint binaries are written beneath:

    checkpoints/

and remain local by normal project Git policy.

For one completed ResolutionLRScreeningResult this module writes:

    checkpoints/
        stage_a_raw_best.pt
        stage_b_lr_3e-05_raw_best.pt
        stage_b_lr_1e-04_raw_best.pt
        stage_b_lr_3e-04_raw_best.pt

    metrics/
        stage_a_history.csv
        stage_b_lr_3e-05_history.csv
        stage_b_lr_1e-04_history.csv
        stage_b_lr_3e-04_history.csv
        lr_screening_summary.yaml

    manifests/
        lr_screening_artifacts.yaml

Checkpoint semantics
--------------------
Checkpoint files contain MODEL STATE ONLY plus immutable metadata.

They deliberately do NOT contain:
- optimizer state;
- Python RNG state;
- NumPy RNG state;
- torch RNG state;
- DataLoader generator state.

This matches the already-frozen raw-argmin checkpoint contract.

Every checkpoint is immediately loaded back from disk and its model-state
SHA-256 is recomputed before the artifact is accepted.

No overwrite
------------
This persistence layer is strict:

- destination files must not already exist;
- all files are first written to unique staging paths;
- every staged file is verified;
- only then are files promoted to their canonical names;
- promotion failure attempts rollback of files created by this call.

This module does NOT:
- create run-level Git provenance;
- launch training;
- derive thresholds;
- access held-out test;
- perform Grad-CAM.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from src.lr_screening import (
    ResolutionLRScreeningResult,
    StageBCandidateScreeningResult,
)

from src.stage_runner import (
    StageEpochRecord,
    StageRunResult,
)

from src.training_control import (
    RawArgminModelCheckpoint,
    state_dict_sha256,
)


# ======================================================================
# Constants
# ======================================================================

ARTIFACT_SCHEMA_VERSION = 1

HISTORY_COLUMNS = (
    "stage",
    "epoch",

    "train_weighted_numerator",
    "train_weight_denominator",
    "train_weighted_loss",
    "train_sample_count",
    "train_batch_count",
    "train_optimizer_steps",

    "dev_weighted_numerator",
    "dev_weight_denominator",
    "dev_weighted_loss",
    "dev_sample_count",
    "dev_batch_count",
    "dev_bonafide_count",
    "dev_attack_count",

    "dev_auroc",

    "raw_checkpoint_updated",
    "raw_best_epoch",
    "raw_best_loss",

    "patience_anchor_initialized",
    "meaningful_improvement",
    "patience_anchor_loss",
    "patience_counter",
    "patience_epochs",

    "stop_due_to_patience",
    "stop_due_to_maximum_epochs",
    "should_stop",

    "transition_note",
)


# ======================================================================
# Small helpers
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


def _bool_text(
    value: bool,
) -> str:

    return (
        "true"
        if value
        else "false"
    )


def _optional_bool_text(
    value: bool | None,
) -> str:

    if value is None:

        return ""

    return _bool_text(
        value
    )


def format_backbone_lr_tag(
    backbone_lr: float,
) -> str:
    """
    Stable filesystem tag for the three frozen Stage-B LRs.

    Examples:
        3e-5 -> 3e-05
        1e-4 -> 1e-04
        3e-4 -> 3e-04
    """

    value = float(
        backbone_lr
    )

    allowed = {
        0.00003:
            "3e-05",

        0.0001:
            "1e-04",

        0.0003:
            "3e-04",
    }

    if value not in allowed:

        raise ValueError(
            "Backbone LR is outside the frozen screening set:\n"
            f"  {value}"
        )

    return allowed[
        value
    ]


def _relative_to_run(
    *,
    run_dir: Path,
    path: Path,
) -> str:

    try:

        relative = path.relative_to(
            run_dir
        )

    except ValueError as exc:

        raise RuntimeError(
            "Artifact path escaped the run directory:\n"
            f"  run_dir={run_dir}\n"
            f"  path={path}"
        ) from exc

    return relative.as_posix()


# ======================================================================
# Artifact record
# ======================================================================

@dataclass(
    frozen=True
)
class PersistedArtifact:

    path: str
    sha256: str
    size_bytes: int
    artifact_type: str

    stage: str | None = None
    backbone_lr: float | None = None

    model_state_sha256: str | None = None
    checkpoint_epoch: int | None = None
    weighted_dev_loss: float | None = None

    selected: bool | None = None


@dataclass(
    frozen=True
)
class ScreeningArtifactBundle:

    run_id: str
    resolution_name: str
    run_seed: int

    run_dir: str

    summary_path: str
    manifest_path: str

    selected_backbone_lr: float

    artifacts: tuple[
        PersistedArtifact,
        ...,
    ]


# ======================================================================
# Stage-history serialization
# ======================================================================

def _history_row(
    record: StageEpochRecord,
) -> dict[str, Any]:

    train = record.train_loss
    dev = record.dev_loss
    control = record.control

    return {
        "stage":
            record.stage,

        "epoch":
            record.epoch,

        "train_weighted_numerator":
            train.weighted_numerator,

        "train_weight_denominator":
            train.weight_denominator,

        "train_weighted_loss":
            train.weighted_loss,

        "train_sample_count":
            train.sample_count,

        "train_batch_count":
            train.batch_count,

        "train_optimizer_steps":
            record.train_optimizer_steps,

        "dev_weighted_numerator":
            dev.weighted_numerator,

        "dev_weight_denominator":
            dev.weight_denominator,

        "dev_weighted_loss":
            dev.weighted_loss,

        "dev_sample_count":
            dev.sample_count,

        "dev_batch_count":
            dev.batch_count,

        "dev_bonafide_count":
            dev.bonafide_count,

        "dev_attack_count":
            dev.attack_count,

        "dev_auroc":
            record.dev_auroc.auroc,

        "raw_checkpoint_updated":
            _bool_text(
                control.raw_checkpoint_updated
            ),

        "raw_best_epoch":
            control.raw_best_epoch,

        "raw_best_loss":
            control.raw_best_loss,

        "patience_anchor_initialized":
            _bool_text(
                control.patience_anchor_initialized
            ),

        "meaningful_improvement":
            _optional_bool_text(
                control.meaningful_improvement
            ),

        "patience_anchor_loss":
            control.patience_anchor_loss,

        "patience_counter":
            control.patience_counter,

        "patience_epochs":
            control.patience_epochs,

        "stop_due_to_patience":
            _bool_text(
                control.stop_due_to_patience
            ),

        "stop_due_to_maximum_epochs":
            _bool_text(
                control.stop_due_to_maximum_epochs
            ),

        "should_stop":
            _bool_text(
                control.should_stop
            ),

        "transition_note":
            (
                ""
                if record.transition_note is None
                else record.transition_note
            ),
    }


def _validate_stage_history(
    result: StageRunResult,
) -> None:

    if result.epochs_completed <= 0:

        raise RuntimeError(
            "Cannot persist a stage with zero completed epochs."
        )

    if (
        len(
            result.history
        )
        != result.epochs_completed
    ):

        raise RuntimeError(
            "Stage history length disagrees with epochs_completed."
        )

    expected_epochs = tuple(
        range(
            1,
            result.epochs_completed
            + 1,
        )
    )

    observed_epochs = tuple(
        record.epoch
        for record
        in result.history
    )

    if observed_epochs != expected_epochs:

        raise RuntimeError(
            "Stage history epochs are not consecutive:\n"
            f"  expected={expected_epochs}\n"
            f"  actual={observed_epochs}"
        )

    raw_best_records = [
        record
        for record
        in result.history
        if record.epoch
        == result.raw_best_epoch
    ]

    if len(
        raw_best_records
    ) != 1:

        raise RuntimeError(
            "Stage raw-best epoch does not map to one history row."
        )

    raw_record = raw_best_records[
        0
    ]

    if (
        raw_record.dev_loss.weighted_loss
        != result.raw_best_weighted_dev_loss
    ):

        raise RuntimeError(
            "Stage raw-best loss disagrees with history."
        )

    if (
        result.raw_best_checkpoint.epoch
        != result.raw_best_epoch
    ):

        raise RuntimeError(
            "Stage checkpoint epoch disagrees with stage result."
        )

    if (
        result.raw_best_checkpoint.weighted_dev_loss
        != result.raw_best_weighted_dev_loss
    ):

        raise RuntimeError(
            "Stage checkpoint loss disagrees with stage result."
        )


def _write_history_csv(
    *,
    path: Path,
    result: StageRunResult,
) -> None:

    _validate_stage_history(
        result
    )

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )

        writer.writeheader()

        for record in result.history:

            writer.writerow(
                _history_row(
                    record
                )
            )


# ======================================================================
# Checkpoint serialization / verification
# ======================================================================

def _checkpoint_payload(
    *,
    checkpoint: RawArgminModelCheckpoint,
    run_id: str,
    resolution_name: str,
    run_seed: int,
    backbone_lr: float | None,
) -> dict[str, Any]:

    current_state_sha = (
        state_dict_sha256(
            checkpoint.model_state_dict
        )
    )

    if (
        current_state_sha
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Checkpoint model-state integrity failure before persistence:\n"
            f"  recorded={checkpoint.model_state_sha256}\n"
            f"  actual={current_state_sha}"
        )

    state_dict = {
        name:
            (
                tensor
                .detach()
                .cpu()
                .contiguous()
                .clone()
            )

        for (
            name,
            tensor,
        ) in checkpoint.model_state_dict.items()
    }

    return {
        "schema_version":
            ARTIFACT_SCHEMA_VERSION,

        "artifact_type":
            "raw_argmin_model_checkpoint",

        "checkpoint_scope":
            "model_parameters_and_buffers_only",

        "run_id":
            run_id,

        "resolution_name":
            resolution_name,

        "run_seed":
            run_seed,

        "stage":
            checkpoint.stage,

        "backbone_lr":
            (
                None
                if backbone_lr is None
                else float(
                    backbone_lr
                )
            ),

        "epoch":
            checkpoint.epoch,

        "weighted_dev_loss":
            checkpoint.weighted_dev_loss,

        "model_state_sha256":
            checkpoint.model_state_sha256,

        "contains_optimizer_state":
            False,

        "contains_global_rng_state":
            False,

        "contains_dataloader_generator_state":
            False,

        "model_state_dict":
            state_dict,
    }


def verify_checkpoint_artifact(
    path: Path,
) -> dict[str, Any]:

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        payload,
        dict,
    ):

        raise TypeError(
            "Persisted checkpoint payload must be a dictionary."
        )

    required = {
        "schema_version",
        "artifact_type",
        "checkpoint_scope",
        "run_id",
        "resolution_name",
        "run_seed",
        "stage",
        "backbone_lr",
        "epoch",
        "weighted_dev_loss",
        "model_state_sha256",
        "contains_optimizer_state",
        "contains_global_rng_state",
        "contains_dataloader_generator_state",
        "model_state_dict",
    }

    missing = (
        required
        - set(
            payload
        )
    )

    if missing:

        raise RuntimeError(
            "Persisted checkpoint is missing required fields:\n"
            f"  {sorted(missing)}"
        )

    if (
        payload[
            "schema_version"
        ]
        != ARTIFACT_SCHEMA_VERSION
    ):

        raise RuntimeError(
            "Persisted checkpoint schema version mismatch."
        )

    if (
        payload[
            "artifact_type"
        ]
        != "raw_argmin_model_checkpoint"
    ):

        raise RuntimeError(
            "Unexpected checkpoint artifact_type."
        )

    if (
        payload[
            "checkpoint_scope"
        ]
        != "model_parameters_and_buffers_only"
    ):

        raise RuntimeError(
            "Persisted checkpoint scope changed."
        )

    if (
        payload[
            "contains_optimizer_state"
        ]
        is not False
    ):

        raise RuntimeError(
            "Persisted raw checkpoint unexpectedly contains optimizer state."
        )

    if (
        payload[
            "contains_global_rng_state"
        ]
        is not False
    ):

        raise RuntimeError(
            "Persisted raw checkpoint unexpectedly contains RNG state."
        )

    if (
        payload[
            "contains_dataloader_generator_state"
        ]
        is not False
    ):

        raise RuntimeError(
            "Persisted raw checkpoint unexpectedly contains "
            "DataLoader generator state."
        )

    state = payload[
        "model_state_dict"
    ]

    if not isinstance(
        state,
        Mapping,
    ):

        raise TypeError(
            "Persisted model_state_dict must be a mapping."
        )

    for (
        name,
        tensor,
    ) in state.items():

        if not isinstance(
            name,
            str,
        ):

            raise TypeError(
                "Persisted checkpoint state keys must be strings."
            )

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            raise TypeError(
                "Persisted checkpoint state values must be tensors."
            )

        if tensor.device.type != "cpu":

            raise RuntimeError(
                "Persisted checkpoint did not load CPU-resident:\n"
                f"  key={name}\n"
                f"  device={tensor.device}"
            )

    actual_state_sha = (
        state_dict_sha256(
            state
        )
    )

    expected_state_sha = str(
        payload[
            "model_state_sha256"
        ]
    )

    if actual_state_sha != expected_state_sha:

        raise RuntimeError(
            "Persisted checkpoint model-state SHA mismatch:\n"
            f"  expected={expected_state_sha}\n"
            f"  actual={actual_state_sha}"
        )

    epoch = payload[
        "epoch"
    ]

    if (
        not isinstance(
            epoch,
            int,
        )
        or isinstance(
            epoch,
            bool,
        )
        or epoch <= 0
    ):

        raise RuntimeError(
            "Persisted checkpoint epoch is invalid."
        )

    loss = float(
        payload[
            "weighted_dev_loss"
        ]
    )

    if (
        not math.isfinite(
            loss
        )
        or loss < 0.0
    ):

        raise RuntimeError(
            "Persisted checkpoint weighted dev loss is invalid."
        )

    return payload


def _write_checkpoint(
    *,
    path: Path,
    checkpoint: RawArgminModelCheckpoint,
    run_id: str,
    resolution_name: str,
    run_seed: int,
    backbone_lr: float | None,
) -> None:

    payload = _checkpoint_payload(
        checkpoint=checkpoint,
        run_id=run_id,
        resolution_name=resolution_name,
        run_seed=run_seed,
        backbone_lr=backbone_lr,
    )

    torch.save(
        payload,
        path,
    )

    reloaded = verify_checkpoint_artifact(
        path
    )

    if (
        reloaded[
            "model_state_sha256"
        ]
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Checkpoint round-trip changed model-state identity."
        )

    if (
        reloaded[
            "epoch"
        ]
        != checkpoint.epoch
    ):

        raise RuntimeError(
            "Checkpoint round-trip changed epoch metadata."
        )

    if (
        float(
            reloaded[
                "weighted_dev_loss"
            ]
        )
        != checkpoint.weighted_dev_loss
    ):

        raise RuntimeError(
            "Checkpoint round-trip changed weighted-dev-loss metadata."
        )


# ======================================================================
# Stage summaries
# ======================================================================

def _stage_summary(
    result: StageRunResult,
) -> dict[str, Any]:

    _validate_stage_history(
        result
    )

    value: dict[
        str,
        Any,
    ] = {
        "stage":
            result.stage,

        "epochs_completed":
            result.epochs_completed,

        "stop_reason":
            result.stop_reason,

        "raw_best_epoch":
            result.raw_best_epoch,

        "raw_best_weighted_dev_loss":
            result.raw_best_weighted_dev_loss,

        "raw_best_model_state_sha256":
            (
                result
                .raw_best_checkpoint
                .model_state_sha256
            ),

        "dev_auroc_at_raw_best_checkpoint":
            (
                result
                .dev_auroc_at_raw_best_checkpoint
            ),

        "best_dev_auroc":
            result.best_dev_auroc,

        "best_dev_auroc_epoch":
            result.best_dev_auroc_epoch,

        "model_left_in_eval_mode":
            result.model_left_in_eval_mode,

        "optimizer_reuse_permitted":
            result.optimizer_reuse_permitted,
    }

    disagreement = (
        result
        .stage_b_auroc_disagreement
    )

    if disagreement is not None:

        value[
            "stage_b_auroc_disagreement"
        ] = {
            "best_dev_auroc":
                disagreement.best_dev_auroc,

            "best_dev_auroc_epoch":
                disagreement.best_dev_auroc_epoch,

            "loss_argmin_checkpoint_epoch":
                (
                    disagreement
                    .loss_argmin_checkpoint_epoch
                ),

            "auroc_at_loss_argmin_checkpoint":
                (
                    disagreement
                    .auroc_at_loss_argmin_checkpoint
                ),

            "difference":
                disagreement.difference,

            "threshold":
                disagreement.threshold,

            "protocol_review_flag":
                disagreement.protocol_review_flag,

            "automatically_switch_checkpoint":
                (
                    disagreement
                    .automatically_switch_checkpoint
                ),
        }

    return value


def _stage_b_candidate_summary(
    candidate: StageBCandidateScreeningResult,
    *,
    selected_lr: float,
) -> dict[str, Any]:

    initialization = (
        candidate
        .initialization
    )

    return {
        "backbone_lr":
            candidate.backbone_lr,

        "selected":
            (
                candidate.backbone_lr
                == selected_lr
            ),

        "protocol_review_flag":
            candidate.protocol_review_flag,

        "initialization":
            {
                "stage_a_checkpoint_epoch":
                    (
                        initialization
                        .stage_a_checkpoint_epoch
                    ),

                "stage_a_checkpoint_weighted_dev_loss":
                    (
                        initialization
                        .stage_a_checkpoint_weighted_dev_loss
                    ),

                "stage_a_checkpoint_model_state_sha256":
                    (
                        initialization
                        .stage_a_checkpoint_model_state_sha256
                    ),

                "fresh_model_state_sha256_before_restore":
                    (
                        initialization
                        .fresh_model_state_sha256_before_restore
                    ),

                "restored_model_state_sha256_cpu":
                    (
                        initialization
                        .restored_model_state_sha256_cpu
                    ),

                "restored_model_state_sha256_after_device_move":
                    (
                        initialization
                        .restored_model_state_sha256_after_device_move
                    ),

                "project_train_generator_initial_state_sha256":
                    (
                        initialization
                        .project_train_generator_initial_state_sha256
                    ),

                "dev_val_generator_initial_state_sha256":
                    (
                        initialization
                        .dev_val_generator_initial_state_sha256
                    ),

                "optimizer_state_entries_at_construction":
                    (
                        initialization
                        .optimizer_state_entries_at_construction
                    ),
            },

        "training":
            _stage_summary(
                candidate
                .stage_b_result
            ),
    }


# ======================================================================
# Staging
# ======================================================================

@dataclass
class _StagedFile:

    temporary_path: Path
    final_path: Path


def _new_staging_path(
    final_path: Path,
) -> Path:

    token = uuid.uuid4().hex

    return final_path.with_name(
        final_path.name
        + ".partial."
        + token
    )


def _assert_final_paths_absent(
    paths: Iterable[
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
            "Refusing to overwrite existing scientific artifacts:\n"
            + "\n".join(
                str(
                    path
                )
                for path
                in existing
            )
        )


def _cleanup_staging(
    staged: Iterable[
        _StagedFile
    ],
) -> None:

    for item in staged:

        try:

            if item.temporary_path.exists():

                item.temporary_path.unlink()

        except OSError:

            pass


def _promote_staged(
    staged: list[
        _StagedFile
    ],
) -> None:

    moved: list[
        Path
    ] = []

    try:

        for item in staged:

            if item.final_path.exists():

                raise FileExistsError(
                    item.final_path
                )

            os.replace(
                item.temporary_path,
                item.final_path,
            )

            moved.append(
                item.final_path
            )

    except Exception:

        # All final paths are guaranteed to have been absent before this
        # operation, so rollback may safely remove files moved by this call.
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
# Main persistence function
# ======================================================================

def persist_resolution_lr_screening_result(
    *,
    result: ResolutionLRScreeningResult,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    experiment_config_sha256: str,
) -> ScreeningArtifactBundle:
    """
    Persist one COMPLETE one-resolution LR-screening result.

    The caller must supply a unique scientific run directory.

    This function writes no held-out-test artifact and has no test-data
    argument or path.
    """

    if not isinstance(
        result,
        ResolutionLRScreeningResult,
    ):

        raise TypeError(
            "result must be ResolutionLRScreeningResult."
        )

    root = Path(
        run_dir
    ).expanduser().resolve()

    if not root.is_dir():

        raise FileNotFoundError(
            "Scientific run directory does not exist:\n"
            f"  {root}"
        )

    if not run_id.strip():

        raise ValueError(
            "run_id must be non-empty."
        )

    if (
        len(
            git_commit
        )
        != 40
        or any(
            character
            not in "0123456789abcdef"
            for character
            in git_commit.lower()
        )
    ):

        raise ValueError(
            "git_commit must be a full 40-character hexadecimal SHA."
        )

    if (
        len(
            experiment_config_sha256
        )
        != 64
        or any(
            character
            not in "0123456789abcdef"
            for character
            in experiment_config_sha256.lower()
        )
    ):

        raise ValueError(
            "experiment_config_sha256 must be a full SHA-256."
        )

    if result.run_seed != 8:

        raise ValueError(
            "Initial LR screening artifacts require frozen seed 8."
        )

    if result.resolution_name not in {
        "r256",
        "r512",
    }:

        raise ValueError(
            "Unexpected LR-screening resolution."
        )

    candidates = (
        result
        .stage_b_candidates
    )

    expected_lrs = (
        0.00003,
        0.0001,
        0.0003,
    )

    observed_lrs = tuple(
        candidate.backbone_lr
        for candidate
        in candidates
    )

    if observed_lrs != expected_lrs:

        raise RuntimeError(
            "Persisted screening candidates do not match frozen LR order:\n"
            f"  expected={expected_lrs}\n"
            f"  actual={observed_lrs}"
        )

    if (
        result.selected_backbone_lr
        not in expected_lrs
    ):

        raise RuntimeError(
            "Selected LR is outside the frozen screening set."
        )

    # ------------------------------------------------------------------
    # Required subdirectories.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Canonical final paths.
    # ------------------------------------------------------------------

    stage_a_checkpoint_path = (
        checkpoints_dir
        / "stage_a_raw_best.pt"
    )

    stage_a_history_path = (
        metrics_dir
        / "stage_a_history.csv"
    )

    summary_path = (
        metrics_dir
        / "lr_screening_summary.yaml"
    )

    manifest_path = (
        manifests_dir
        / "lr_screening_artifacts.yaml"
    )

    stage_b_checkpoint_paths: dict[
        float,
        Path,
    ] = {}

    stage_b_history_paths: dict[
        float,
        Path,
    ] = {}

    for lr in expected_lrs:

        tag = format_backbone_lr_tag(
            lr
        )

        stage_b_checkpoint_paths[
            lr
        ] = (
            checkpoints_dir
            / (
                f"stage_b_lr_{tag}"
                "_raw_best.pt"
            )
        )

        stage_b_history_paths[
            lr
        ] = (
            metrics_dir
            / (
                f"stage_b_lr_{tag}"
                "_history.csv"
            )
        )

    all_final_paths = [
        stage_a_checkpoint_path,
        stage_a_history_path,
        summary_path,
        manifest_path,
        *stage_b_checkpoint_paths.values(),
        *stage_b_history_paths.values(),
    ]

    _assert_final_paths_absent(
        all_final_paths
    )

    staged: list[
        _StagedFile
    ] = []

    artifact_records: list[
        PersistedArtifact
    ] = []

    try:

        # ==============================================================
        # Stage-A checkpoint
        # ==============================================================

        temp = _new_staging_path(
            stage_a_checkpoint_path
        )

        staged.append(
            _StagedFile(
                temporary_path=temp,
                final_path=(
                    stage_a_checkpoint_path
                ),
            )
        )

        _write_checkpoint(
            path=temp,
            checkpoint=(
                result
                .stage_a_result
                .raw_best_checkpoint
            ),
            run_id=run_id,
            resolution_name=(
                result.resolution_name
            ),
            run_seed=result.run_seed,
            backbone_lr=None,
        )

        artifact_records.append(
            PersistedArtifact(
                path=_relative_to_run(
                    run_dir=root,
                    path=stage_a_checkpoint_path,
                ),
                sha256=sha256_file(
                    temp
                ),
                size_bytes=(
                    temp.stat().st_size
                ),
                artifact_type=(
                    "raw_argmin_model_checkpoint"
                ),
                stage="stage_a",
                backbone_lr=None,
                model_state_sha256=(
                    result
                    .stage_a_result
                    .raw_best_checkpoint
                    .model_state_sha256
                ),
                checkpoint_epoch=(
                    result
                    .stage_a_result
                    .raw_best_checkpoint
                    .epoch
                ),
                weighted_dev_loss=(
                    result
                    .stage_a_result
                    .raw_best_checkpoint
                    .weighted_dev_loss
                ),
                selected=True,
            )
        )

        # ==============================================================
        # Stage-A history
        # ==============================================================

        temp = _new_staging_path(
            stage_a_history_path
        )

        staged.append(
            _StagedFile(
                temporary_path=temp,
                final_path=(
                    stage_a_history_path
                ),
            )
        )

        _write_history_csv(
            path=temp,
            result=(
                result
                .stage_a_result
            ),
        )

        artifact_records.append(
            PersistedArtifact(
                path=_relative_to_run(
                    run_dir=root,
                    path=stage_a_history_path,
                ),
                sha256=sha256_file(
                    temp
                ),
                size_bytes=(
                    temp.stat().st_size
                ),
                artifact_type=(
                    "epoch_history_csv"
                ),
                stage="stage_a",
                selected=True,
            )
        )

        # ==============================================================
        # Three Stage-B candidates
        # ==============================================================

        for candidate in candidates:

            lr = float(
                candidate.backbone_lr
            )

            selected = (
                lr
                == result.selected_backbone_lr
            )

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------

            final_checkpoint = (
                stage_b_checkpoint_paths[
                    lr
                ]
            )

            temp = _new_staging_path(
                final_checkpoint
            )

            staged.append(
                _StagedFile(
                    temporary_path=temp,
                    final_path=(
                        final_checkpoint
                    ),
                )
            )

            _write_checkpoint(
                path=temp,
                checkpoint=(
                    candidate
                    .stage_b_result
                    .raw_best_checkpoint
                ),
                run_id=run_id,
                resolution_name=(
                    result.resolution_name
                ),
                run_seed=result.run_seed,
                backbone_lr=lr,
            )

            artifact_records.append(
                PersistedArtifact(
                    path=_relative_to_run(
                        run_dir=root,
                        path=final_checkpoint,
                    ),
                    sha256=sha256_file(
                        temp
                    ),
                    size_bytes=(
                        temp.stat().st_size
                    ),
                    artifact_type=(
                        "raw_argmin_model_checkpoint"
                    ),
                    stage="stage_b",
                    backbone_lr=lr,
                    model_state_sha256=(
                        candidate
                        .stage_b_result
                        .raw_best_checkpoint
                        .model_state_sha256
                    ),
                    checkpoint_epoch=(
                        candidate
                        .stage_b_result
                        .raw_best_checkpoint
                        .epoch
                    ),
                    weighted_dev_loss=(
                        candidate
                        .stage_b_result
                        .raw_best_checkpoint
                        .weighted_dev_loss
                    ),
                    selected=selected,
                )
            )

            # ----------------------------------------------------------
            # History
            # ----------------------------------------------------------

            final_history = (
                stage_b_history_paths[
                    lr
                ]
            )

            temp = _new_staging_path(
                final_history
            )

            staged.append(
                _StagedFile(
                    temporary_path=temp,
                    final_path=(
                        final_history
                    ),
                )
            )

            _write_history_csv(
                path=temp,
                result=(
                    candidate
                    .stage_b_result
                ),
            )

            artifact_records.append(
                PersistedArtifact(
                    path=_relative_to_run(
                        run_dir=root,
                        path=final_history,
                    ),
                    sha256=sha256_file(
                        temp
                    ),
                    size_bytes=(
                        temp.stat().st_size
                    ),
                    artifact_type=(
                        "epoch_history_csv"
                    ),
                    stage="stage_b",
                    backbone_lr=lr,
                    selected=selected,
                )
            )

        # ==============================================================
        # Screening summary
        # ==============================================================

        stage_a_init = (
            result
            .stage_a_initialization
        )

        summary = {
            "schema_version":
                ARTIFACT_SCHEMA_VERSION,

            "artifact_type":
                "resolution_lr_screening_summary",

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

                    "backbone_lr_candidates":
                        list(
                            expected_lrs
                        ),

                    "held_out_test_accessed":
                        False,
                },

            "stage_a_initialization":
                {
                    "fresh_model_state_sha256":
                        (
                            stage_a_init
                            .fresh_model_state_sha256
                        ),

                    "project_train_generator_initial_state_sha256":
                        (
                            stage_a_init
                            .project_train_generator_initial_state_sha256
                        ),

                    "dev_val_generator_initial_state_sha256":
                        (
                            stage_a_init
                            .dev_val_generator_initial_state_sha256
                        ),

                    "optimizer_state_entries_at_construction":
                        (
                            stage_a_init
                            .optimizer_state_entries_at_construction
                        ),

                    "pretrained_checkpoint_sha256":
                        (
                            stage_a_init
                            .model_provenance
                            .pretrained_checkpoint_sha256
                        ),
                },

            "stage_a":
                _stage_summary(
                    result.stage_a_result
                ),

            "stage_b_candidates":
                [
                    _stage_b_candidate_summary(
                        candidate,
                        selected_lr=(
                            result
                            .selected_backbone_lr
                        ),
                    )

                    for candidate
                    in candidates
                ],

            "selection":
                {
                    "metric":
                        "class_weighted_dev_cross_entropy",

                    "source":
                        "raw_argmin_checkpoint",

                    "rule":
                        "lowest_best_weighted_dev_loss",

                    "exact_tie_breaker":
                        "lower_backbone_learning_rate",

                    "selected_backbone_lr":
                        result.selected_backbone_lr,

                    "selected_raw_best_weighted_dev_loss":
                        (
                            result
                            .selected_raw_best_weighted_dev_loss
                        ),

                    "selected_stage_b_checkpoint_model_state_sha256":
                        (
                            result
                            .selected_stage_b_checkpoint_sha256
                        ),

                    "exact_loss_tie_encountered":
                        (
                            result
                            .exact_loss_tie_encountered
                        ),

                    "protocol_review_required":
                        (
                            result
                            .protocol_review_required
                        ),

                    "auroc_changes_selection":
                        False,
                },
        }

        temp = _new_staging_path(
            summary_path
        )

        staged.append(
            _StagedFile(
                temporary_path=temp,
                final_path=summary_path,
            )
        )

        with temp.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file:

            yaml.safe_dump(
                summary,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        artifact_records.append(
            PersistedArtifact(
                path=_relative_to_run(
                    run_dir=root,
                    path=summary_path,
                ),
                sha256=sha256_file(
                    temp
                ),
                size_bytes=(
                    temp.stat().st_size
                ),
                artifact_type=(
                    "lr_screening_summary_yaml"
                ),
                selected=True,
            )
        )

        # ==============================================================
        # Artifact manifest
        # ==============================================================

        manifest = {
            "schema_version":
                ARTIFACT_SCHEMA_VERSION,

            "artifact_type":
                "resolution_lr_screening_artifact_manifest",

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

                    "held_out_test_accessed":
                        False,
                },

            "selection":
                {
                    "selected_backbone_lr":
                        (
                            result
                            .selected_backbone_lr
                        ),

                    "selected_stage_b_checkpoint_model_state_sha256":
                        (
                            result
                            .selected_stage_b_checkpoint_sha256
                        ),
                },

            "artifacts":
                [
                    {
                        "path":
                            artifact.path,

                        "sha256":
                            artifact.sha256,

                        "size_bytes":
                            artifact.size_bytes,

                        "artifact_type":
                            artifact.artifact_type,

                        "stage":
                            artifact.stage,

                        "backbone_lr":
                            artifact.backbone_lr,

                        "model_state_sha256":
                            artifact.model_state_sha256,

                        "checkpoint_epoch":
                            artifact.checkpoint_epoch,

                        "weighted_dev_loss":
                            artifact.weighted_dev_loss,

                        "selected":
                            artifact.selected,
                    }

                    for artifact
                    in artifact_records
                ],
        }

        temp = _new_staging_path(
            manifest_path
        )

        staged.append(
            _StagedFile(
                temporary_path=temp,
                final_path=manifest_path,
            )
        )

        with temp.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file:

            yaml.safe_dump(
                manifest,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        # ==============================================================
        # Promote only after every staged artifact was successfully made.
        # ==============================================================

        _promote_staged(
            staged
        )

    except Exception:

        _cleanup_staging(
            staged
        )

        raise

    # ==================================================================
    # Final post-promotion verification
    # ==================================================================

    for artifact in artifact_records:

        path = (
            root
            / artifact.path
        )

        if not path.is_file():

            raise RuntimeError(
                "Promoted artifact is missing:\n"
                f"  {path}"
            )

        if (
            sha256_file(
                path
            )
            != artifact.sha256
        ):

            raise RuntimeError(
                "Promoted artifact SHA-256 changed:\n"
                f"  {path}"
            )

        if (
            artifact.artifact_type
            == "raw_argmin_model_checkpoint"
        ):

            payload = (
                verify_checkpoint_artifact(
                    path
                )
            )

            if (
                payload[
                    "model_state_sha256"
                ]
                != artifact.model_state_sha256
            ):

                raise RuntimeError(
                    "Promoted checkpoint model-state identity changed."
                )

    if not manifest_path.is_file():

        raise RuntimeError(
            "Artifact manifest was not promoted."
        )

    manifest_sha = sha256_file(
        manifest_path
    )

    manifest_size = (
        manifest_path
        .stat()
        .st_size
    )

    final_artifacts = (
        tuple(
            artifact_records
        )
        +
        (
            PersistedArtifact(
                path=_relative_to_run(
                    run_dir=root,
                    path=manifest_path,
                ),
                sha256=manifest_sha,
                size_bytes=manifest_size,
                artifact_type=(
                    "artifact_manifest_yaml"
                ),
                selected=True,
            ),
        )
    )

    return ScreeningArtifactBundle(
        run_id=run_id,
        resolution_name=(
            result.resolution_name
        ),
        run_seed=result.run_seed,
        run_dir=str(
            root
        ),
        summary_path=(
            _relative_to_run(
                run_dir=root,
                path=summary_path,
            )
        ),
        manifest_path=(
            _relative_to_run(
                run_dir=root,
                path=manifest_path,
            )
        ),
        selected_backbone_lr=(
            result.selected_backbone_lr
        ),
        artifacts=(
            final_artifacts
        ),
    )