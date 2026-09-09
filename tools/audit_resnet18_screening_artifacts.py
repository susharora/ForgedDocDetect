#!/usr/bin/env python3
"""
Synthetic persistence audit for Tech-2 LR-screening artifacts.

The production component under audit is:

    src.screening_artifacts

This audit constructs a fully synthetic but structurally valid
ResolutionLRScreeningResult and verifies:

- Stage-A raw-best checkpoint persistence;
- all three Stage-B raw-best checkpoint files;
- exact model-state SHA-256 round-trip;
- model-only checkpoint scope;
- absence of optimizer/RNG/DataLoader state;
- Stage-A and Stage-B epoch-history CSV contents;
- Stage-B epoch-1 transition note persistence;
- screening-summary YAML contents;
- selected versus losing LR identity;
- AUROC protocol-review flag persistence;
- artifact-manifest contents;
- manifest file hashes and sizes;
- all paths remain relative to the run directory;
- strict no-overwrite behavior;
- existing artifacts remain byte-identical after rejected overwrite;
- staged-write failure leaves no scientific files behind;
- mid-promotion failure rolls back already-promoted files;
- temporary partial files are removed after failures.

The audit uses temporary directories only.

No FantasyID Dataset is constructed.
No image is decoded.
No CUDA computation is performed.
No forward pass is performed.
No backward pass is performed.
No optimizer step is performed.
No FPR10 threshold is derived.
No held-out test is accessed.
No console printing is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import torch
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


from src.config import (
    load_experiment_config,
    load_machine_config,
)

from src.development_metrics import (
    DevAUROCResult,
)

from src.engine import (
    EpochLossSummary,
)

from src.lr_screening import (
    ResolutionLRScreeningResult,
    StageAInitializationEvidence,
    StageBCandidateScreeningResult,
)

from src.screening_artifacts import (
    HISTORY_COLUMNS,
    format_backbone_lr_tag,
    persist_resolution_lr_screening_result,
    verify_checkpoint_artifact,
)

import src.screening_artifacts as screening_artifacts

from src.stage_b_branching import (
    StageBBranchInitializationEvidence,
)

from src.stage_runner import (
    StageBAUROCDisagreement,
    StageEpochRecord,
    StageRunResult,
)

from src.training_control import (
    RawArgminModelCheckpoint,
    StageControlDecision,
    state_dict_sha256,
)


LOGGER = logging.getLogger(
    "audit_resnet18_screening_artifacts"
)


VALIDATION_HANDOFF_PATTERN = (
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


# ======================================================================
# Generic helpers
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


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        value = (
            yaml.safe_load(
                file
            )
            or {}
        )

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            f"{path} must contain a YAML mapping."
        )

    return value


def require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            f"{label} must be a mapping, "
            f"got {type(value).__name__}: {value!r}"
        )

    return value


def require_key(
    mapping: dict[str, Any],
    key: str,
    label: str,
) -> Any:

    if key not in mapping:

        raise KeyError(
            f"Missing required key: "
            f"{label}.{key}"
        )

    return mapping[
        key
    ]


def require_close(
    *,
    label: str,
    actual: float,
    expected: float,
    atol: float = 1.0e-15,
) -> None:

    if not math.isclose(
        float(
            actual
        ),
        float(
            expected
        ),
        rel_tol=0.0,
        abs_tol=atol,
    ):

        raise RuntimeError(
            f"{label} mismatch:\n"
            f"  expected={expected!r}\n"
            f"  actual={actual!r}"
        )


def expect_exception(
    *,
    label: str,
    function: Callable[[], Any],
    exception_types: tuple[
        type[BaseException],
        ...,
    ],
) -> bool:

    try:

        function()

    except exception_types:

        LOGGER.info(
            "[PASS] %s",
            label,
        )

        return True

    raise RuntimeError(
        f"Expected exception was not raised: {label}"
    )


def synthetic_sha(
    label: str,
) -> str:

    return hashlib.sha256(
        label.encode(
            "utf-8"
        )
    ).hexdigest()


# ======================================================================
# Git gate
# ======================================================================

def require_clean_git() -> str:

    commit_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    status_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    status = (
        status_result
        .stdout
        .strip()
    )

    if status:

        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit/remove outstanding files before running "
            "the screening-artifact audit.\n\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Canonical experiment validator
# ======================================================================

def run_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> tuple[
    Path,
    str,
]:

    import re

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
            "Canonical experiment validator failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    lines = [
        line.strip()
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]

    if len(
        lines
    ) != 1:

        raise RuntimeError(
            "Expected exactly one validator handoff line:\n"
            f"{lines}"
        )

    match = re.match(
        VALIDATION_HANDOFF_PATTERN,
        lines[
            0
        ],
    )

    if match is None:

        raise RuntimeError(
            "Could not parse validator handoff:\n"
            f"{lines[0]}"
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

    path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = match.group(
        "sha256"
    )

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    actual_sha = sha256_file(
        path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Validator artifact SHA-256 mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        path,
        expected_sha,
    )


# ======================================================================
# Logging
# ======================================================================

def configure_outputs(
    *,
    tool_cfg: dict[str, Any],
    machine_id: str,
) -> tuple[
    logging.Logger,
    Path,
    Path,
    Path,
]:

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
    )

    logging_cfg = require_mapping(
        require_key(
            tool_cfg,
            "logging",
            "audit_config",
        ),
        "audit_config.logging",
    )

    output_cfg = require_mapping(
        require_key(
            tool_cfg,
            "output",
            "audit_config",
        ),
        "audit_config.output",
    )

    log_directory = resolve_repo_path(
        require_key(
            logging_cfg,
            "directory",
            "audit_config.logging",
        )
    )

    output_directory = resolve_repo_path(
        require_key(
            output_cfg,
            "directory",
            "audit_config.output",
        )
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = {
        "machine_id":
            machine_id,

        "timestamp":
            timestamp,
    }

    log_path = (
        log_directory
        / str(
            require_key(
                logging_cfg,
                "filename",
                "audit_config.logging",
            )
        ).format(
            **values
        )
    )

    result_path = (
        output_directory
        / str(
            require_key(
                output_cfg,
                "filename",
                "audit_config.output",
            )
        ).format(
            **values
        )
    )

    partial_path = Path(
        str(
            result_path
        )
        + ".partial"
    )

    level_name = str(
        require_key(
            logging_cfg,
            "level",
            "audit_config.logging",
        )
    ).upper()

    if not hasattr(
        logging,
        level_name,
    ):

        raise ValueError(
            f"Unknown logging level: {level_name!r}"
        )

    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(
        getattr(
            logging,
            level_name,
        )
    )

    handler = logging.FileHandler(
        log_path,
        mode="x",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    formatter.converter = (
        time.gmtime
    )

    handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        handler
    )

    return (
        LOGGER,
        log_path,
        result_path,
        partial_path,
    )


# ======================================================================
# Synthetic checkpoints
# ======================================================================

def make_checkpoint(
    *,
    stage: str,
    epoch: int,
    weighted_dev_loss: float,
    marker: float,
) -> RawArgminModelCheckpoint:

    state = {
        "layer.weight":
            torch.tensor(
                [
                    [
                        marker,
                        marker
                        + 1.0,
                    ],
                    [
                        marker
                        + 2.0,
                        marker
                        + 3.0,
                    ],
                ],
                dtype=torch.float32,
            ),

        "layer.bias":
            torch.tensor(
                [
                    marker
                    / 10.0,
                    marker
                    / 20.0,
                ],
                dtype=torch.float32,
            ),

        "bn.running_mean":
            torch.tensor(
                [
                    marker
                    / 100.0,
                    marker
                    / 200.0,
                ],
                dtype=torch.float32,
            ),

        "bn.num_batches_tracked":
            torch.tensor(
                int(
                    marker
                ),
                dtype=torch.int64,
            ),
    }

    digest = state_dict_sha256(
        state
    )

    return RawArgminModelCheckpoint(
        stage=stage,
        epoch=epoch,
        weighted_dev_loss=float(
            weighted_dev_loss
        ),
        model_state_sha256=digest,
        model_state_dict=state,
    )


# ======================================================================
# Synthetic epoch history
# ======================================================================

def make_loss_summary(
    *,
    value: float,
    train: bool,
) -> EpochLossSummary:

    if train:

        denominator = 1440.0

        return EpochLossSummary(
            weighted_numerator=(
                float(
                    value
                )
                * denominator
            ),
            weight_denominator=denominator,
            weighted_loss=float(
                value
            ),
            sample_count=1440,
            batch_count=45,
            bonafide_count=480,
            attack_count=960,
        )

    denominator = 459.0

    return EpochLossSummary(
        weighted_numerator=(
            float(
                value
            )
            * denominator
        ),
        weight_denominator=denominator,
        weighted_loss=float(
            value
        ),
        sample_count=459,
        batch_count=15,
        bonafide_count=153,
        attack_count=306,
    )


def make_auroc(
    value: float,
) -> DevAUROCResult:

    pair_count = (
        153
        * 306
    )

    wins = int(
        round(
            float(
                value
            )
            * pair_count
        )
    )

    wins = max(
        0,
        min(
            pair_count,
            wins,
        ),
    )

    return DevAUROCResult(
        auroc=float(
            value
        ),
        sample_count=459,
        bonafide_count=153,
        attack_count=306,
        positive_negative_pair_count=(
            pair_count
        ),
        strict_attack_wins=wins,
        tied_attack_bonafide_pairs=0,
        score_definition=(
            "synthetic_persistence_audit"
        ),
    )


def make_stage_result(
    *,
    stage: str,
    dev_losses: tuple[
        float,
        ...,
    ],
    dev_aurocs: tuple[
        float,
        ...,
    ],
    checkpoint: RawArgminModelCheckpoint,
    transition_note: str | None,
) -> StageRunResult:

    if len(
        dev_losses
    ) != len(
        dev_aurocs
    ):

        raise ValueError(
            "Synthetic dev loss/AUROC trajectory lengths differ."
        )

    if checkpoint.epoch != 2:

        raise ValueError(
            "Synthetic persistence audit expects raw-best epoch 2."
        )

    if (
        dev_losses[
            1
        ]
        != checkpoint.weighted_dev_loss
    ):

        raise ValueError(
            "Synthetic checkpoint loss does not equal epoch-2 loss."
        )

    patience_epochs = (
        2
        if stage == "stage_a"
        else 5
    )

    history: list[
        StageEpochRecord
    ] = []

    current_raw_best_loss: float | None = None
    current_raw_best_epoch = 0

    for (
        index,
        (
            dev_loss,
            auroc,
        ),
    ) in enumerate(
        zip(
            dev_losses,
            dev_aurocs,
        ),
        start=1,
    ):

        raw_updated = False

        if (
            current_raw_best_loss is None
            or dev_loss
            < current_raw_best_loss
        ):

            current_raw_best_loss = (
                float(
                    dev_loss
                )
            )

            current_raw_best_epoch = (
                index
            )

            raw_updated = True

        patience_counter = (
            0
            if index == 1
            else index - 1
        )

        should_stop = (
            index
            == len(
                dev_losses
            )
        )

        control = StageControlDecision(
            stage=stage,
            epoch=index,
            weighted_dev_loss=float(
                dev_loss
            ),
            raw_checkpoint_updated=(
                raw_updated
            ),
            raw_best_epoch=(
                current_raw_best_epoch
            ),
            raw_best_loss=float(
                current_raw_best_loss
            ),
            patience_anchor_initialized=(
                index == 1
            ),
            meaningful_improvement=(
                None
                if index == 1
                else False
            ),
            patience_anchor_loss=float(
                dev_losses[
                    0
                ]
            ),
            patience_counter=(
                patience_counter
            ),
            patience_epochs=(
                patience_epochs
            ),
            stop_due_to_patience=(
                should_stop
            ),
            stop_due_to_maximum_epochs=False,
            should_stop=(
                should_stop
            ),
        )

        epoch_note: (
            str
            | None
        ) = None

        if (
            stage == "stage_b"
            and index == 1
        ):

            epoch_note = (
                transition_note
            )

        history.append(
            StageEpochRecord(
                stage=stage,
                epoch=index,
                train_loss=(
                    make_loss_summary(
                        value=(
                            1.25
                            -
                            0.01
                            * index
                        ),
                        train=True,
                    )
                ),
                train_optimizer_steps=45,
                dev_loss=(
                    make_loss_summary(
                        value=dev_loss,
                        train=False,
                    )
                ),
                dev_auroc=(
                    make_auroc(
                        auroc
                    )
                ),
                control=control,
                transition_note=(
                    epoch_note
                ),
            )
        )

    best_auroc = max(
        dev_aurocs
    )

    best_auroc_epoch = (
        dev_aurocs.index(
            best_auroc
        )
        + 1
    )

    selected_auroc = (
        dev_aurocs[
            checkpoint.epoch
            - 1
        ]
    )

    disagreement: (
        StageBAUROCDisagreement
        | None
    ) = None

    if stage == "stage_b":

        difference = (
            float(
                best_auroc
            )
            -
            float(
                selected_auroc
            )
        )

        disagreement = (
            StageBAUROCDisagreement(
                best_dev_auroc=float(
                    best_auroc
                ),
                best_dev_auroc_epoch=(
                    best_auroc_epoch
                ),
                loss_argmin_checkpoint_epoch=(
                    checkpoint.epoch
                ),
                auroc_at_loss_argmin_checkpoint=float(
                    selected_auroc
                ),
                difference=float(
                    difference
                ),
                threshold=0.01,
                protocol_review_flag=(
                    difference
                    > 0.01
                ),
                automatically_switch_checkpoint=False,
            )
        )

    return StageRunResult(
        stage=stage,
        epochs_completed=len(
            history
        ),
        history=tuple(
            history
        ),
        stop_reason="patience",
        raw_best_checkpoint=(
            checkpoint
        ),
        raw_best_epoch=(
            checkpoint.epoch
        ),
        raw_best_weighted_dev_loss=(
            checkpoint.weighted_dev_loss
        ),
        dev_auroc_at_raw_best_checkpoint=float(
            selected_auroc
        ),
        best_dev_auroc=float(
            best_auroc
        ),
        best_dev_auroc_epoch=(
            best_auroc_epoch
        ),
        stage_b_auroc_disagreement=(
            disagreement
        ),
        restored_selected_model_state_sha256=(
            checkpoint
            .model_state_sha256
        ),
        model_left_in_eval_mode=True,
        optimizer_reuse_permitted=False,
    )


# ======================================================================
# Synthetic complete LR-screening result
# ======================================================================

def build_synthetic_screening_result(
    *,
    resolution_name: str,
    run_seed: int,
) -> ResolutionLRScreeningResult:

    transition_note = (
        "Backbone unfrozen and BatchNorm running statistics begin "
        "FantasyID domain adaptation; transient dev-loss movement "
        "is expected."
    )

    stage_a_checkpoint = make_checkpoint(
        stage="stage_a",
        epoch=2,
        weighted_dev_loss=0.998,
        marker=10.0,
    )

    stage_a_result = make_stage_result(
        stage="stage_a",
        dev_losses=(
            1.0,
            0.998,
            0.999,
        ),
        dev_aurocs=(
            0.55,
            0.56,
            0.57,
        ),
        checkpoint=(
            stage_a_checkpoint
        ),
        transition_note=None,
    )

    stage_a_initialization = (
        StageAInitializationEvidence(
            resolution_name=(
                resolution_name
            ),
            run_seed=run_seed,
            requested_device="cpu",
            actual_model_device="cpu",
            fresh_model_state_sha256=(
                synthetic_sha(
                    "fresh_stage_a_model"
                )
            ),
            project_train_generator_initial_state_sha256=(
                synthetic_sha(
                    "stage_a_train_generator"
                )
            ),
            dev_val_generator_initial_state_sha256=(
                synthetic_sha(
                    "stage_a_dev_generator"
                )
            ),
            optimizer_state_entries_at_construction=0,
            model_provenance=(
                SimpleNamespace(
                    pretrained_checkpoint_sha256=(
                        "f37072fd47e89c5e827621c5baffa7500819f7896"
                        "bbacec160b1a16c560e07ec"
                    )
                )
            ),
            reproducibility_state=(
                SimpleNamespace(
                    run_seed=run_seed
                )
            ),
            optimizer_evidence=(
                SimpleNamespace(
                    state_entries_at_construction=0
                )
            ),
        )
    )

    candidate_specs = (
        (
            0.00003,
            0.548,
            20.0,
            (
                0.60,
                0.61,
                0.612,
                0.613,
                0.614,
                0.615,
            ),
        ),
        (
            0.0001,
            0.498,
            30.0,
            (
                0.70,
                0.71,
                0.715,
                0.716,
                0.717,
                0.718,
            ),
        ),
        (
            0.0003,
            0.528,
            40.0,
            (
                0.65,
                0.66,
                0.67,
                0.675,
                0.678,
                0.68,
            ),
        ),
    )

    candidates: list[
        StageBCandidateScreeningResult
    ] = []

    for (
        lr,
        best_loss,
        marker,
        aurocs,
    ) in candidate_specs:

        checkpoint = make_checkpoint(
            stage="stage_b",
            epoch=2,
            weighted_dev_loss=(
                best_loss
            ),
            marker=marker,
        )

        first_loss = (
            best_loss
            + 0.002
        )

        stage_b_result = (
            make_stage_result(
                stage="stage_b",
                dev_losses=(
                    first_loss,
                    best_loss,
                    best_loss
                    + 0.001,
                    best_loss
                    + 0.002,
                    best_loss
                    + 0.003,
                    best_loss
                    + 0.004,
                ),
                dev_aurocs=(
                    aurocs
                ),
                checkpoint=(
                    checkpoint
                ),
                transition_note=(
                    transition_note
                ),
            )
        )

        disagreement = (
            stage_b_result
            .stage_b_auroc_disagreement
        )

        if disagreement is None:

            raise RuntimeError(
                "Synthetic Stage-B disagreement unexpectedly missing."
            )

        initialization = (
            StageBBranchInitializationEvidence(
                resolution_name=(
                    resolution_name
                ),
                run_seed=run_seed,
                backbone_lr=lr,
                requested_device="cpu",
                actual_model_device="cpu",
                stage_a_checkpoint_epoch=(
                    stage_a_checkpoint
                    .epoch
                ),
                stage_a_checkpoint_weighted_dev_loss=(
                    stage_a_checkpoint
                    .weighted_dev_loss
                ),
                stage_a_checkpoint_model_state_sha256=(
                    stage_a_checkpoint
                    .model_state_sha256
                ),
                fresh_model_state_sha256_before_restore=(
                    synthetic_sha(
                        f"fresh_stage_b_{lr}"
                    )
                ),
                restored_model_state_sha256_cpu=(
                    stage_a_checkpoint
                    .model_state_sha256
                ),
                restored_model_state_sha256_after_device_move=(
                    stage_a_checkpoint
                    .model_state_sha256
                ),
                project_train_generator_initial_state_sha256=(
                    synthetic_sha(
                        f"train_generator_{lr}"
                    )
                ),
                dev_val_generator_initial_state_sha256=(
                    synthetic_sha(
                        f"dev_generator_{lr}"
                    )
                ),
                optimizer_state_entries_at_construction=0,
            )
        )

        candidates.append(
            StageBCandidateScreeningResult(
                backbone_lr=lr,
                initialization=(
                    initialization
                ),
                stage_b_result=(
                    stage_b_result
                ),
                protocol_review_flag=(
                    disagreement
                    .protocol_review_flag
                ),
            )
        )

    selected = candidates[
        1
    ]

    return ResolutionLRScreeningResult(
        resolution_name=(
            resolution_name
        ),
        run_seed=run_seed,
        stage_a_initialization=(
            stage_a_initialization
        ),
        stage_a_result=(
            stage_a_result
        ),
        stage_b_candidates=tuple(
            candidates
        ),
        selected_backbone_lr=0.0001,
        selected_raw_best_weighted_dev_loss=(
            selected
            .stage_b_result
            .raw_best_weighted_dev_loss
        ),
        selected_stage_b_checkpoint_sha256=(
            selected
            .stage_b_result
            .raw_best_checkpoint
            .model_state_sha256
        ),
        exact_loss_tie_encountered=False,
        protocol_review_required=True,
    )


# ======================================================================
# CSV inspection
# ======================================================================

def read_csv_rows(
    path: Path,
) -> tuple[
    list[
        str
    ],
    list[
        dict[
            str,
            str,
        ]
    ],
]:

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        fieldnames = list(
            reader.fieldnames
            or []
        )

        rows = list(
            reader
        )

    return (
        fieldnames,
        rows,
    )


def verify_history_csv(
    *,
    path: Path,
    stage_result: StageRunResult,
    expected_transition_note: str | None,
) -> dict[str, Any]:

    fieldnames, rows = read_csv_rows(
        path
    )

    if tuple(
        fieldnames
    ) != HISTORY_COLUMNS:

        raise RuntimeError(
            "History CSV column contract changed:\n"
            f"  expected={HISTORY_COLUMNS}\n"
            f"  actual={tuple(fieldnames)}"
        )

    if len(
        rows
    ) != stage_result.epochs_completed:

        raise RuntimeError(
            "History CSV row count mismatch."
        )

    expected_epochs = [
        str(
            value
        )
        for value
        in range(
            1,
            stage_result.epochs_completed
            + 1,
        )
    ]

    observed_epochs = [
        row[
            "epoch"
        ]
        for row
        in rows
    ]

    if observed_epochs != expected_epochs:

        raise RuntimeError(
            "History CSV epoch order mismatch."
        )

    if any(
        row[
            "stage"
        ]
        != stage_result.stage
        for row
        in rows
    ):

        raise RuntimeError(
            "History CSV stage identity mismatch."
        )

    raw_best_row = rows[
        stage_result.raw_best_epoch
        - 1
    ]

    require_close(
        label=(
            f"{path.name}: raw-best dev loss"
        ),
        actual=float(
            raw_best_row[
                "dev_weighted_loss"
            ]
        ),
        expected=(
            stage_result
            .raw_best_weighted_dev_loss
        ),
    )

    if (
        raw_best_row[
            "raw_checkpoint_updated"
        ]
        != "true"
    ):

        raise RuntimeError(
            "Raw-best history row is not marked as checkpoint update."
        )

    if rows[
        0
    ][
        "meaningful_improvement"
    ] != "":

        raise RuntimeError(
            "Epoch-1 meaningful_improvement should serialize empty."
        )

    if (
        rows[
            -1
        ][
            "should_stop"
        ]
        != "true"
    ):

        raise RuntimeError(
            "Final history row is not marked stopping."
        )

    notes = [
        row[
            "transition_note"
        ]
        for row
        in rows
    ]

    if expected_transition_note is None:

        if any(
            note
            for note
            in notes
        ):

            raise RuntimeError(
                "Stage-A history unexpectedly contains transition note."
            )

    else:

        if notes[
            0
        ] != expected_transition_note:

            raise RuntimeError(
                "Stage-B epoch-1 transition note mismatch."
            )

        if any(
            notes[
                1:
            ]
        ):

            raise RuntimeError(
                "Stage-B transition note appears after epoch 1."
            )

    return {
        "path":
            path.name,

        "rows":
            len(
                rows
            ),

        "columns":
            len(
                fieldnames
            ),

        "raw_best_epoch":
            stage_result.raw_best_epoch,

        "raw_best_weighted_dev_loss":
            stage_result.raw_best_weighted_dev_loss,

        "transition_note_epoch_1":
            (
                notes[
                    0
                ]
                or None
            ),
    }


# ======================================================================
# Successful persistence verification
# ======================================================================

def verify_successful_persistence(
    *,
    result: ResolutionLRScreeningResult,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    experiment_sha: str,
    expected_counts: dict[str, Any],
) -> dict[str, Any]:

    bundle = (
        persist_resolution_lr_screening_result(
            result=result,
            run_dir=run_dir,
            run_id=run_id,
            git_commit=git_commit,
            experiment_config_sha256=(
                experiment_sha
            ),
        )
    )

    expected_lrs = (
        0.00003,
        0.0001,
        0.0003,
    )

    expected_paths = {
        "checkpoints/stage_a_raw_best.pt",
        "metrics/stage_a_history.csv",

        "checkpoints/stage_b_lr_3e-05_raw_best.pt",
        "metrics/stage_b_lr_3e-05_history.csv",

        "checkpoints/stage_b_lr_1e-04_raw_best.pt",
        "metrics/stage_b_lr_1e-04_history.csv",

        "checkpoints/stage_b_lr_3e-04_raw_best.pt",
        "metrics/stage_b_lr_3e-04_history.csv",

        "metrics/lr_screening_summary.yaml",
        "manifests/lr_screening_artifacts.yaml",
    }

    bundle_paths = {
        artifact.path
        for artifact
        in bundle.artifacts
    }

    if bundle_paths != expected_paths:

        raise RuntimeError(
            "Persisted artifact path set mismatch:\n"
            f"  expected={sorted(expected_paths)}\n"
            f"  actual={sorted(bundle_paths)}"
        )

    expected_bundle_count = int(
        require_key(
            expected_counts,
            "bundle_artifacts_including_manifest",
            "audit.expected_artifact_counts",
        )
    )

    if len(
        bundle.artifacts
    ) != expected_bundle_count:

        raise RuntimeError(
            "Bundle artifact count mismatch."
        )

    if (
        bundle.selected_backbone_lr
        != result.selected_backbone_lr
    ):

        raise RuntimeError(
            "Bundle selected LR mismatch."
        )

    # ------------------------------------------------------------------
    # Check every bundle hash and size.
    # ------------------------------------------------------------------

    bundle_hashes: dict[
        str,
        str,
    ] = {}

    for artifact in bundle.artifacts:

        path = (
            run_dir
            / artifact.path
        )

        if not path.is_file():

            raise RuntimeError(
                f"Bundle artifact missing: {path}"
            )

        actual_sha = sha256_file(
            path
        )

        if actual_sha != artifact.sha256:

            raise RuntimeError(
                "Bundle artifact SHA mismatch:\n"
                f"  path={artifact.path}\n"
                f"  expected={artifact.sha256}\n"
                f"  actual={actual_sha}"
            )

        if (
            path.stat().st_size
            != artifact.size_bytes
        ):

            raise RuntimeError(
                "Bundle artifact byte-size mismatch."
            )

        if Path(
            artifact.path
        ).is_absolute():

            raise RuntimeError(
                "Artifact manifest path unexpectedly absolute."
            )

        bundle_hashes[
            artifact.path
        ] = (
            actual_sha
        )

    # ------------------------------------------------------------------
    # Checkpoints.
    # ------------------------------------------------------------------

    checkpoint_expectations = {
        "checkpoints/stage_a_raw_best.pt":
            (
                result
                .stage_a_result
                .raw_best_checkpoint,
                None,
            ),
    }

    for candidate in (
        result
        .stage_b_candidates
    ):

        tag = format_backbone_lr_tag(
            candidate.backbone_lr
        )

        checkpoint_expectations[
            (
                f"checkpoints/"
                f"stage_b_lr_{tag}_raw_best.pt"
            )
        ] = (
            candidate
            .stage_b_result
            .raw_best_checkpoint,
            candidate.backbone_lr,
        )

    expected_checkpoint_count = int(
        require_key(
            expected_counts,
            "checkpoint_files",
            "audit.expected_artifact_counts",
        )
    )

    if (
        len(
            checkpoint_expectations
        )
        != expected_checkpoint_count
    ):

        raise RuntimeError(
            "Synthetic checkpoint expectation count mismatch."
        )

    checkpoint_evidence: dict[
        str,
        Any,
    ] = {}

    for (
        relative_path,
        (
            expected_checkpoint,
            expected_lr,
        ),
    ) in checkpoint_expectations.items():

        path = (
            run_dir
            / relative_path
        )

        payload = (
            verify_checkpoint_artifact(
                path
            )
        )

        if (
            payload[
                "run_id"
            ]
            != run_id
        ):

            raise RuntimeError(
                "Checkpoint run_id mismatch."
            )

        if (
            payload[
                "resolution_name"
            ]
            != result.resolution_name
        ):

            raise RuntimeError(
                "Checkpoint resolution mismatch."
            )

        if (
            payload[
                "run_seed"
            ]
            != result.run_seed
        ):

            raise RuntimeError(
                "Checkpoint run seed mismatch."
            )

        if (
            payload[
                "stage"
            ]
            != expected_checkpoint.stage
        ):

            raise RuntimeError(
                "Checkpoint stage mismatch."
            )

        if (
            payload[
                "backbone_lr"
            ]
            != expected_lr
        ):

            raise RuntimeError(
                "Checkpoint backbone LR mismatch."
            )

        if (
            payload[
                "epoch"
            ]
            != expected_checkpoint.epoch
        ):

            raise RuntimeError(
                "Checkpoint epoch mismatch."
            )

        require_close(
            label=(
                f"{relative_path}: weighted dev loss"
            ),
            actual=float(
                payload[
                    "weighted_dev_loss"
                ]
            ),
            expected=(
                expected_checkpoint
                .weighted_dev_loss
            ),
        )

        if (
            payload[
                "model_state_sha256"
            ]
            != expected_checkpoint.model_state_sha256
        ):

            raise RuntimeError(
                "Checkpoint model-state SHA mismatch."
            )

        if (
            payload[
                "contains_optimizer_state"
            ]
            is not False
        ):

            raise RuntimeError(
                "Checkpoint unexpectedly contains optimizer state."
            )

        if (
            payload[
                "contains_global_rng_state"
            ]
            is not False
        ):

            raise RuntimeError(
                "Checkpoint unexpectedly contains global RNG state."
            )

        if (
            payload[
                "contains_dataloader_generator_state"
            ]
            is not False
        ):

            raise RuntimeError(
                "Checkpoint unexpectedly contains DataLoader state."
            )

        checkpoint_evidence[
            relative_path
        ] = {
            "stage":
                payload[
                    "stage"
                ],

            "backbone_lr":
                payload[
                    "backbone_lr"
                ],

            "epoch":
                payload[
                    "epoch"
                ],

            "weighted_dev_loss":
                payload[
                    "weighted_dev_loss"
                ],

            "model_state_sha256":
                payload[
                    "model_state_sha256"
                ],

            "file_sha256":
                sha256_file(
                    path
                ),
        }

    # ------------------------------------------------------------------
    # Histories.
    # ------------------------------------------------------------------

    transition_note = (
        "Backbone unfrozen and BatchNorm running statistics begin "
        "FantasyID domain adaptation; transient dev-loss movement "
        "is expected."
    )

    history_evidence: dict[
        str,
        Any,
    ] = {}

    stage_a_history = (
        run_dir
        / "metrics"
        / "stage_a_history.csv"
    )

    history_evidence[
        "stage_a"
    ] = verify_history_csv(
        path=stage_a_history,
        stage_result=(
            result
            .stage_a_result
        ),
        expected_transition_note=None,
    )

    for candidate in (
        result
        .stage_b_candidates
    ):

        tag = format_backbone_lr_tag(
            candidate.backbone_lr
        )

        path = (
            run_dir
            / "metrics"
            / f"stage_b_lr_{tag}_history.csv"
        )

        history_evidence[
            str(
                candidate.backbone_lr
            )
        ] = verify_history_csv(
            path=path,
            stage_result=(
                candidate
                .stage_b_result
            ),
            expected_transition_note=(
                transition_note
            ),
        )

    expected_history_count = int(
        require_key(
            expected_counts,
            "history_csv_files",
            "audit.expected_artifact_counts",
        )
    )

    if len(
        history_evidence
    ) != expected_history_count:

        raise RuntimeError(
            "History file count mismatch."
        )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------

    summary_path = (
        run_dir
        / bundle.summary_path
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        summary = yaml.safe_load(
            file
        )

    if not isinstance(
        summary,
        dict,
    ):

        raise TypeError(
            "Screening summary must be a YAML mapping."
        )

    if (
        summary[
            "artifact_type"
        ]
        != "resolution_lr_screening_summary"
    ):

        raise RuntimeError(
            "Unexpected screening summary artifact type."
        )

    if (
        summary[
            "run"
        ][
            "run_id"
        ]
        != run_id
    ):

        raise RuntimeError(
            "Summary run_id mismatch."
        )

    if (
        summary[
            "run"
        ][
            "git_commit"
        ]
        != git_commit
    ):

        raise RuntimeError(
            "Summary Git commit mismatch."
        )

    if (
        summary[
            "run"
        ][
            "experiment_config_sha256"
        ]
        != experiment_sha
    ):

        raise RuntimeError(
            "Summary experiment-config SHA mismatch."
        )

    if (
        summary[
            "scope"
        ][
            "held_out_test_accessed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Summary does not explicitly record test non-access."
        )

    selection = summary[
        "selection"
    ]

    if (
        selection[
            "selected_backbone_lr"
        ]
        != 0.0001
    ):

        raise RuntimeError(
            "Summary selected LR mismatch."
        )

    require_close(
        label="summary selected loss",
        actual=(
            selection[
                "selected_raw_best_weighted_dev_loss"
            ]
        ),
        expected=0.498,
    )

    if (
        selection[
            "selected_stage_b_checkpoint_model_state_sha256"
        ]
        != result.selected_stage_b_checkpoint_sha256
    ):

        raise RuntimeError(
            "Summary selected checkpoint SHA mismatch."
        )

    if (
        selection[
            "auroc_changes_selection"
        ]
        is not False
    ):

        raise RuntimeError(
            "Summary incorrectly permits AUROC selection."
        )

    if (
        selection[
            "protocol_review_required"
        ]
        is not True
    ):

        raise RuntimeError(
            "Summary lost protocol-review requirement."
        )

    candidate_summaries = (
        summary[
            "stage_b_candidates"
        ]
    )

    candidate_summary_lrs = tuple(
        float(
            candidate[
                "backbone_lr"
            ]
        )
        for candidate
        in candidate_summaries
    )

    if candidate_summary_lrs != expected_lrs:

        raise RuntimeError(
            "Summary Stage-B candidate order mismatch."
        )

    selected_flags = tuple(
        bool(
            candidate[
                "selected"
            ]
        )
        for candidate
        in candidate_summaries
    )

    if selected_flags != (
        False,
        True,
        False,
    ):

        raise RuntimeError(
            "Summary selected/losing candidate markers are wrong."
        )

    review_flags = tuple(
        bool(
            candidate[
                "protocol_review_flag"
            ]
        )
        for candidate
        in candidate_summaries
    )

    if review_flags != (
        False,
        False,
        True,
    ):

        raise RuntimeError(
            "Summary Stage-B review flags are wrong."
        )

    # ------------------------------------------------------------------
    # Manifest.
    # ------------------------------------------------------------------

    manifest_path = (
        run_dir
        / bundle.manifest_path
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = yaml.safe_load(
            file
        )

    if not isinstance(
        manifest,
        dict,
    ):

        raise TypeError(
            "Artifact manifest must be a YAML mapping."
        )

    if (
        manifest[
            "artifact_type"
        ]
        != "resolution_lr_screening_artifact_manifest"
    ):

        raise RuntimeError(
            "Unexpected artifact manifest type."
        )

    if (
        manifest[
            "scope"
        ][
            "held_out_test_accessed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Manifest does not record held-out-test non-access."
        )

    manifest_entries = manifest[
        "artifacts"
    ]

    expected_manifest_count = int(
        require_key(
            expected_counts,
            "manifest_listed_artifacts",
            "audit.expected_artifact_counts",
        )
    )

    if len(
        manifest_entries
    ) != expected_manifest_count:

        raise RuntimeError(
            "Manifest listed-artifact count mismatch."
        )

    manifest_paths = {
        entry[
            "path"
        ]
        for entry
        in manifest_entries
    }

    if bundle.manifest_path in manifest_paths:

        raise RuntimeError(
            "Manifest recursively lists itself."
        )

    expected_non_manifest_paths = (
        expected_paths
        - {
            bundle.manifest_path
        }
    )

    if manifest_paths != expected_non_manifest_paths:

        raise RuntimeError(
            "Manifest artifact path set mismatch."
        )

    for entry in manifest_entries:

        relative_path = entry[
            "path"
        ]

        if Path(
            relative_path
        ).is_absolute():

            raise RuntimeError(
                "Manifest contains an absolute artifact path."
            )

        path = (
            run_dir
            / relative_path
        )

        if (
            entry[
                "sha256"
            ]
            != sha256_file(
                path
            )
        ):

            raise RuntimeError(
                "Manifest file SHA does not match artifact:\n"
                f"  {relative_path}"
            )

        if (
            entry[
                "size_bytes"
            ]
            != path.stat().st_size
        ):

            raise RuntimeError(
                "Manifest size does not match artifact."
            )

    stage_b_checkpoint_entries = [
        entry
        for entry
        in manifest_entries
        if (
            entry[
                "artifact_type"
            ]
            == "raw_argmin_model_checkpoint"
            and entry[
                "stage"
            ]
            == "stage_b"
        )
    ]

    if len(
        stage_b_checkpoint_entries
    ) != 3:

        raise RuntimeError(
            "Manifest does not contain three Stage-B checkpoints."
        )

    selected_checkpoint_entries = [
        entry
        for entry
        in stage_b_checkpoint_entries
        if entry[
            "selected"
        ]
        is True
    ]

    if len(
        selected_checkpoint_entries
    ) != 1:

        raise RuntimeError(
            "Manifest must identify exactly one selected Stage-B checkpoint."
        )

    selected_checkpoint_entry = (
        selected_checkpoint_entries[
            0
        ]
    )

    if (
        selected_checkpoint_entry[
            "backbone_lr"
        ]
        != 0.0001
    ):

        raise RuntimeError(
            "Manifest selected checkpoint belongs to wrong LR."
        )

    if (
        selected_checkpoint_entry[
            "model_state_sha256"
        ]
        != result.selected_stage_b_checkpoint_sha256
    ):

        raise RuntimeError(
            "Manifest selected checkpoint model SHA mismatch."
        )

    # ------------------------------------------------------------------
    # No partial files survived.
    # ------------------------------------------------------------------

    partial_files = [
        path
        for path
        in run_dir.rglob(
            "*"
        )
        if (
            path.is_file()
            and ".partial."
            in path.name
        )
    ]

    if partial_files:

        raise RuntimeError(
            "Successful persistence left staging files:\n"
            f"  {partial_files}"
        )

    LOGGER.info(
        "[PASS] successful persistence wrote expected 10 artifacts"
    )

    LOGGER.info(
        "[PASS] four checkpoint model states round-trip exactly"
    )

    LOGGER.info(
        "[PASS] checkpoint payloads are model-only"
    )

    LOGGER.info(
        "[PASS] four history CSV files preserve epoch evidence"
    )

    LOGGER.info(
        "[PASS] summary preserves selected and losing LR identities"
    )

    LOGGER.info(
        "[PASS] manifest hashes and sizes reconcile to files"
    )

    LOGGER.info(
        "[PASS] successful persistence leaves no staging files"
    )

    return {
        "bundle_artifact_count":
            len(
                bundle.artifacts
            ),

        "manifest_listed_artifact_count":
            len(
                manifest_entries
            ),

        "selected_backbone_lr":
            bundle.selected_backbone_lr,

        "selected_stage_b_checkpoint_model_state_sha256":
            result.selected_stage_b_checkpoint_sha256,

        "checkpoint_evidence":
            checkpoint_evidence,

        "history_evidence":
            history_evidence,

        "bundle_file_sha256":
            bundle_hashes,

        "summary_selection":
            selection,

        "stage_b_selected_flags":
            list(
                selected_flags
            ),

        "stage_b_protocol_review_flags":
            list(
                review_flags
            ),
    }


# ======================================================================
# No-overwrite audit
# ======================================================================

def audit_no_overwrite(
    *,
    result: ResolutionLRScreeningResult,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    experiment_sha: str,
) -> dict[str, Any]:

    before = {
        path.relative_to(
            run_dir
        ).as_posix():
            sha256_file(
                path
            )

        for path
        in run_dir.rglob(
            "*"
        )

        if path.is_file()
    }

    rejected = expect_exception(
        label=(
            "second persistence attempt rejected by no-overwrite guard"
        ),
        function=lambda:
            persist_resolution_lr_screening_result(
                result=result,
                run_dir=run_dir,
                run_id=run_id,
                git_commit=git_commit,
                experiment_config_sha256=(
                    experiment_sha
                ),
            ),
        exception_types=(
            FileExistsError,
        ),
    )

    after = {
        path.relative_to(
            run_dir
        ).as_posix():
            sha256_file(
                path
            )

        for path
        in run_dir.rglob(
            "*"
        )

        if path.is_file()
    }

    if after != before:

        raise RuntimeError(
            "Rejected overwrite attempt changed existing artifacts."
        )

    partial_files = [
        path
        for path
        in run_dir.rglob(
            "*"
        )
        if (
            path.is_file()
            and ".partial."
            in path.name
        )
    ]

    if partial_files:

        raise RuntimeError(
            "Rejected overwrite attempt created staging files."
        )

    LOGGER.info(
        "[PASS] no-overwrite rejection preserves every existing file SHA"
    )

    return {
        "rejected":
            rejected,

        "files_before":
            len(
                before
            ),

        "files_after":
            len(
                after
            ),

        "all_existing_file_hashes_unchanged":
            True,

        "partial_files_created":
            False,
    }


# ======================================================================
# Failure-cleanup helpers
# ======================================================================

def require_no_files(
    *,
    run_dir: Path,
    label: str,
) -> None:

    files = [
        path
        for path
        in run_dir.rglob(
            "*"
        )
        if path.is_file()
    ]

    if files:

        raise RuntimeError(
            f"{label} left files behind:\n"
            + "\n".join(
                str(
                    path
                )
                for path
                in files
            )
        )


# ======================================================================
# Staged-write failure
# ======================================================================

def audit_staged_write_failure(
    *,
    result: ResolutionLRScreeningResult,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    experiment_sha: str,
) -> dict[str, Any]:

    original_writer = (
        screening_artifacts
        ._write_history_csv
    )

    calls = {
        "count":
            0,
    }

    def failing_writer(
        *,
        path: Path,
        result: StageRunResult,
    ) -> None:

        calls[
            "count"
        ] += 1

        # Stage-A history succeeds.
        # Fail at first Stage-B history after multiple staged files exist.
        if calls[
            "count"
        ] == 2:

            raise RuntimeError(
                "synthetic staged-write failure"
            )

        original_writer(
            path=path,
            result=result,
        )

    with patch.object(
        screening_artifacts,
        "_write_history_csv",
        side_effect=(
            failing_writer
        ),
    ):

        rejected = expect_exception(
            label=(
                "synthetic staged-write failure aborts persistence"
            ),
            function=lambda:
                persist_resolution_lr_screening_result(
                    result=result,
                    run_dir=run_dir,
                    run_id=run_id,
                    git_commit=git_commit,
                    experiment_config_sha256=(
                        experiment_sha
                    ),
                ),
            exception_types=(
                RuntimeError,
            ),
        )

    require_no_files(
        run_dir=run_dir,
        label=(
            "Staged-write failure cleanup"
        ),
    )

    LOGGER.info(
        "[PASS] staged-write failure removes all temporary artifacts"
    )

    return {
        "failure_injected_on_history_write_call":
            2,

        "failure_rejected":
            rejected,

        "files_remaining_after_failure":
            0,

        "partial_files_remaining_after_failure":
            0,
    }


# ======================================================================
# Mid-promotion failure rollback
# ======================================================================

def audit_promotion_failure(
    *,
    result: ResolutionLRScreeningResult,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    experiment_sha: str,
) -> dict[str, Any]:

    original_replace = (
        os.replace
    )

    calls = {
        "count":
            0,
    }

    def failing_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:

        calls[
            "count"
        ] += 1

        # Three final files are promoted, then promotion fails.
        if calls[
            "count"
        ] == 4:

            raise OSError(
                "synthetic promotion failure"
            )

        original_replace(
            source,
            destination,
        )

    synthetic_os = SimpleNamespace(
        replace=(
            failing_replace
        )
    )

    with patch.object(
        screening_artifacts,
        "os",
        synthetic_os,
    ):

        rejected = expect_exception(
            label=(
                "mid-promotion failure triggers rollback"
            ),
            function=lambda:
                persist_resolution_lr_screening_result(
                    result=result,
                    run_dir=run_dir,
                    run_id=run_id,
                    git_commit=git_commit,
                    experiment_config_sha256=(
                        experiment_sha
                    ),
                ),
            exception_types=(
                OSError,
            ),
        )

    require_no_files(
        run_dir=run_dir,
        label=(
            "Promotion rollback"
        ),
    )

    if calls[
        "count"
    ] != 4:

        raise RuntimeError(
            "Promotion failure control did not fail at expected call."
        )

    LOGGER.info(
        "[PASS] mid-promotion failure rolls back already-promoted files"
    )

    return {
        "promotion_failure_call":
            4,

        "promotion_calls_observed":
            calls[
                "count"
            ],

        "failure_rejected":
            rejected,

        "files_remaining_after_rollback":
            0,

        "partial_files_remaining_after_rollback":
            0,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit synthetic persistence of "
            "Tech-2 LR-screening artifacts."
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
        "--audit-config",
        default=(
            "tools/"
            "audit_resnet18_screening_artifacts_config.yaml"
        ),
    )

    args = parser.parse_args()

    audit_config_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        audit_config_path
    )

    if (
        tool_cfg.get(
            "schema_version"
        )
        != 1
    ):

        raise ValueError(
            "Screening-artifact audit schema_version must equal 1."
        )

    commit_sha = require_clean_git()

    experiment_cfg, experiment_path = (
        load_experiment_config(
            args.config
        )
    )

    machine_cfg, machine_path = (
        load_machine_config(
            args.machine_config,
            required=True,
        )
    )

    (
        validator_log,
        validator_log_sha,
    ) = run_validator(
        experiment_path=experiment_path,
        machine_path=machine_path,
    )

    machine_section = require_mapping(
        require_key(
            machine_cfg,
            "machine",
            "machine_config",
        ),
        "machine_config.machine",
    )

    machine_id = str(
        require_key(
            machine_section,
            "id",
            "machine_config.machine",
        )
    )

    (
        logger,
        log_path,
        result_path,
        partial_result_path,
    ) = configure_outputs(
        tool_cfg=tool_cfg,
        machine_id=machine_id,
    )

    try:

        audit_cfg = require_mapping(
            require_key(
                tool_cfg,
                "audit",
                "audit_config",
            ),
            "audit_config.audit",
        )

        expected_experiment_sha = str(
            require_key(
                audit_cfg,
                "expected_experiment_config_sha256",
                "audit_config.audit",
            )
        )

        actual_experiment_sha = sha256_file(
            experiment_path
        )

        if (
            actual_experiment_sha
            != expected_experiment_sha
        ):

            raise RuntimeError(
                "Experiment config SHA-256 mismatch:\n"
                f"  expected={expected_experiment_sha}\n"
                f"  actual={actual_experiment_sha}"
            )

        run_id = str(
            require_key(
                audit_cfg,
                "synthetic_run_id",
                "audit_config.audit",
            )
        )

        resolution_name = str(
            require_key(
                audit_cfg,
                "resolution",
                "audit_config.audit",
            )
        )

        run_seed = int(
            require_key(
                audit_cfg,
                "run_seed",
                "audit_config.audit",
            )
        )

        expected_lrs = tuple(
            float(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "backbone_lr_candidates",
                "audit_config.audit",
            )
        )

        selected_lr = float(
            require_key(
                audit_cfg,
                "selected_backbone_lr",
                "audit_config.audit",
            )
        )

        expected_counts = require_mapping(
            require_key(
                audit_cfg,
                "expected_artifact_counts",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_artifact_counts",
        )

        if resolution_name != "r256":

            raise ValueError(
                "Synthetic persistence audit must use r256."
            )

        if run_seed != 8:

            raise ValueError(
                "Synthetic persistence audit must use seed 8."
            )

        if expected_lrs != (
            0.00003,
            0.0001,
            0.0003,
        ):

            raise ValueError(
                "Synthetic persistence audit LR set changed."
            )

        if selected_lr != 0.0001:

            raise ValueError(
                "Synthetic persistence audit selected LR changed."
            )

        synthetic_result = (
            build_synthetic_screening_result(
                resolution_name=(
                    resolution_name
                ),
                run_seed=run_seed,
            )
        )

        if (
            synthetic_result
            .selected_backbone_lr
            != selected_lr
        ):

            raise RuntimeError(
                "Synthetic result selected LR mismatch."
            )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 SCREENING-ARTIFACT PERSISTENCE AUDIT"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Git commit: %s",
            commit_sha,
        )

        logger.info(
            "Git working tree clean before evidence generation: True"
        )

        logger.info(
            "Machine ID: %s",
            machine_id,
        )

        logger.info(
            "Experiment config SHA-256: %s",
            actual_experiment_sha,
        )

        logger.info(
            "src/screening_artifacts.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "screening_artifacts.py"
            ),
        )

        logger.info(
            "Audit script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "Audit config SHA-256: %s",
            sha256_file(
                audit_config_path
            ),
        )

        logger.info(
            "Validator log: %s",
            validator_log,
        )

        logger.info(
            "Validator log SHA-256: %s",
            validator_log_sha,
        )

        logger.info(
            "Execution: synthetic persistence only"
        )

        logger.info(
            "Temporary run directories: system temporary storage"
        )

        logger.info(
            "FantasyID Dataset construction: NONE"
        )

        logger.info(
            "Image decoding: NONE"
        )

        logger.info(
            "Scientific training: NONE"
        )

        logger.info(
            "CUDA computation: NONE"
        )

        logger.info(
            "FPR10 threshold derivation: NONE"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # ==============================================================
        # Successful persistence + no overwrite
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Successful persistence and round-trip verification ---"
        )

        with tempfile.TemporaryDirectory(
            prefix=(
                "tech2_screening_artifact_success_"
            )
        ) as temporary_root:

            run_dir = (
                Path(
                    temporary_root
                )
                / "run"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            success_evidence = (
                verify_successful_persistence(
                    result=synthetic_result,
                    run_dir=run_dir,
                    run_id=run_id,
                    git_commit=commit_sha,
                    experiment_sha=(
                        actual_experiment_sha
                    ),
                    expected_counts=(
                        expected_counts
                    ),
                )
            )

            logger.info(
                ""
            )

            logger.info(
                "--- Strict no-overwrite behavior ---"
            )

            no_overwrite_evidence = (
                audit_no_overwrite(
                    result=synthetic_result,
                    run_dir=run_dir,
                    run_id=run_id,
                    git_commit=commit_sha,
                    experiment_sha=(
                        actual_experiment_sha
                    ),
                )
            )

        # ==============================================================
        # Staged-write failure cleanup
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Staged-write failure cleanup ---"
        )

        with tempfile.TemporaryDirectory(
            prefix=(
                "tech2_screening_artifact_write_failure_"
            )
        ) as temporary_root:

            run_dir = (
                Path(
                    temporary_root
                )
                / "run"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            staged_failure_evidence = (
                audit_staged_write_failure(
                    result=synthetic_result,
                    run_dir=run_dir,
                    run_id=run_id,
                    git_commit=commit_sha,
                    experiment_sha=(
                        actual_experiment_sha
                    ),
                )
            )

        # ==============================================================
        # Promotion rollback
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Mid-promotion failure rollback ---"
        )

        with tempfile.TemporaryDirectory(
            prefix=(
                "tech2_screening_artifact_promotion_failure_"
            )
        ) as temporary_root:

            run_dir = (
                Path(
                    temporary_root
                )
                / "run"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            promotion_failure_evidence = (
                audit_promotion_failure(
                    result=synthetic_result,
                    run_dir=run_dir,
                    run_id=run_id,
                    git_commit=commit_sha,
                    experiment_sha=(
                        actual_experiment_sha
                    ),
                )
            )

        # ==============================================================
        # Result artifact
        # ==============================================================

        result = {
            "schema_version":
                1,

            "status":
                "PASS",

            "machine":
                {
                    "id":
                        machine_id,
                },

            "provenance":
                {
                    "git_commit":
                        commit_sha,

                    "experiment_config_sha256":
                        actual_experiment_sha,

                    "screening_artifacts_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "screening_artifacts.py"
                        ),

                    "audit_script_sha256":
                        sha256_file(
                            Path(
                                __file__
                            ).resolve()
                        ),

                    "audit_config_sha256":
                        sha256_file(
                            audit_config_path
                        ),

                    "validator_log":
                        str(
                            validator_log
                        ),

                    "validator_log_sha256":
                        validator_log_sha,
                },

            "scope":
                {
                    "synthetic_only":
                        True,

                    "temporary_directories_only":
                        True,

                    "resolution":
                        resolution_name,

                    "run_seed":
                        run_seed,

                    "backbone_lr_candidates":
                        list(
                            expected_lrs
                        ),

                    "fantasyid_dataset_constructed":
                        False,

                    "images_decoded":
                        False,

                    "scientific_training_performed":
                        False,

                    "cuda_computation_performed":
                        False,

                    "forward_pass_performed":
                        False,

                    "backward_pass_performed":
                        False,

                    "optimizer_step_performed":
                        False,

                    "fpr10_threshold_derived":
                        False,

                    "held_out_test_accessed":
                        False,
                },

            "successful_persistence":
                success_evidence,

            "no_overwrite":
                no_overwrite_evidence,

            "staged_write_failure":
                staged_failure_evidence,

            "promotion_failure":
                promotion_failure_evidence,

            "interpretation":
                (
                    "PASS establishes durable one-resolution "
                    "LR-screening artifact persistence using synthetic "
                    "model states. Four model-only raw-argmin "
                    "checkpoints round-trip by model-state SHA-256, "
                    "four epoch histories serialize correctly, summary "
                    "and manifest evidence preserve LR selection and "
                    "protocol-review identity, manifest hashes reconcile "
                    "to persisted bytes, overwrites are rejected without "
                    "mutation, staged-write failures leave no files, and "
                    "mid-promotion failures roll back promoted artifacts. "
                    "No scientific training or held-out-test access "
                    "occurred."
                ),
        }

        with partial_result_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file:

            yaml.safe_dump(
                result,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        partial_result_path.replace(
            result_path
        )

        result_sha = sha256_file(
            result_path
        )

        # ==============================================================
        # Summary
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "AUDIT SUMMARY"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "[PASS] four raw-best checkpoints persisted"
        )

        logger.info(
            "[PASS] all checkpoint model-state SHAs round-trip exactly"
        )

        logger.info(
            "[PASS] checkpoint scope excludes optimizer/RNG state"
        )

        logger.info(
            "[PASS] four epoch-history CSV files validated"
        )

        logger.info(
            "[PASS] Stage-B transition note persisted at epoch 1 only"
        )

        logger.info(
            "[PASS] summary preserves selected and losing LR identities"
        )

        logger.info(
            "[PASS] losing-candidate AUROC review flag remains visible"
        )

        logger.info(
            "[PASS] manifest hashes and file sizes reconcile"
        )

        logger.info(
            "[PASS] artifact paths remain run-relative"
        )

        logger.info(
            "[PASS] second persistence attempt rejected"
        )

        logger.info(
            "[PASS] rejected overwrite changes no existing bytes"
        )

        logger.info(
            "[PASS] staged-write failure leaves no files"
        )

        logger.info(
            "[PASS] mid-promotion failure rolls back promoted files"
        )

        logger.info(
            "[PASS] temporary partial artifacts cleaned after failures"
        )

        logger.info(
            "[PASS] held-out test NOT ACCESSED"
        )

        logger.info(
            "Audit result: %s",
            result_path,
        )

        logger.info(
            "Audit result SHA-256: %s",
            result_sha,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 SCREENING-ARTIFACT PERSISTENCE AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 SCREENING-ARTIFACT PERSISTENCE AUDIT: FAIL"
        )

        if partial_result_path.exists():

            partial_result_path.unlink()

        return 1

    finally:

        for handler in list(
            logger.handlers
        ):

            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )