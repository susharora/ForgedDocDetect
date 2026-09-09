#!/usr/bin/env python3
"""
Synthetic orchestration audit for the frozen Tech-2 multi-epoch runner.

Purpose
-------
The one-epoch engine, stopping controller, checkpoint primitive and
development AUROC have already been audited independently.

This tool therefore replaces:

    train_one_epoch
    evaluate_dev_one_epoch
    compute_dev_auroc

with deterministic synthetic stubs and audits only the orchestration in:

    src.stage_runner.run_training_stage

It verifies:

- train -> dev -> AUROC call order for every epoch;
- Stage-A patience stopping;
- Stage-B patience stopping;
- maximum-epoch stopping;
- raw weighted-dev-loss checkpoint selection;
- sub-0.5% raw-loss improvements still update checkpoint;
- AUROC cannot alter checkpoint selection;
- first-occurrence best-AUROC tie behavior;
- Stage-B >0.01 disagreement flag;
- Stage-B zero disagreement does not flag;
- automatic checkpoint switching remains forbidden;
- Stage-B transition note occurs at epoch 1 only;
- Stage-A contains no Stage-B transition note;
- final model is restored to exact raw-best state;
- final model is left in eval mode;
- completed-stage optimizer reuse is forbidden;
- no epoch executes after the controller stops;
- invalid stage names are rejected.

No FantasyID Dataset is constructed.
No image is decoded.
No CUDA is required.
No real forward pass occurs.
No backward pass occurs.
No optimizer step occurs.
No FPR10 threshold is derived.
No held-out test is accessed.
No print() is used.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.nn as nn
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
    DevEpochResult,
    EpochLossSummary,
    TrainingEpochResult,
)

import src.stage_runner as stage_runner

from src.training_control import (
    model_state_sha256,
)


LOGGER = logging.getLogger(
    "audit_resnet18_stage_runner"
)


VALIDATION_HANDOFF_PATTERN = re.compile(
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
        f"Expected exception not raised: {label}"
    )


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


# ======================================================================
# Git
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
            "the stage-runner audit.\n\n"
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

    match = VALIDATION_HANDOFF_PATTERN.match(
        lines[
            0
        ]
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
            "Validator artifact SHA-256 mismatch."
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

    LOGGER.handlers.clear()
    LOGGER.propagate = False

    level_name = str(
        require_key(
            logging_cfg,
            "level",
            "audit_config.logging",
        )
    ).upper()

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
# Tiny deterministic model
# ======================================================================

class ProbeModel(
    nn.Module
):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [
                        1.0,
                        2.0,
                    ],
                    [
                        3.0,
                        4.0,
                    ],
                ],
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "running_marker",
            torch.tensor(
                [
                    0.0,
                    0.0,
                ],
                dtype=torch.float32,
            ),
        )


def mutate_probe_model(
    *,
    model: ProbeModel,
    epoch: int,
) -> None:

    with torch.no_grad():

        model.weight.copy_(
            torch.tensor(
                [
                    [
                        float(
                            epoch
                        ),
                        float(
                            epoch
                            + 1
                        ),
                    ],
                    [
                        float(
                            epoch
                            + 2
                        ),
                        float(
                            epoch
                            + 3
                        ),
                    ],
                ],
                dtype=torch.float32,
            )
        )

        model.running_marker.copy_(
            torch.tensor(
                [
                    float(
                        epoch
                    ),
                    float(
                        epoch
                        * 10
                    ),
                ],
                dtype=torch.float32,
            )
        )

    model.train()


# ======================================================================
# Synthetic result builders
# ======================================================================

def make_train_loss_summary(
    epoch: int,
) -> EpochLossSummary:

    value = (
        2.0
        -
        0.01
        * float(
            epoch
        )
    )

    denominator = 1440.0

    return EpochLossSummary(
        weighted_numerator=(
            value
            * denominator
        ),
        weight_denominator=denominator,
        weighted_loss=value,
        sample_count=1440,
        batch_count=45,
        bonafide_count=480,
        attack_count=960,
    )


def make_dev_loss_summary(
    value: float,
) -> EpochLossSummary:

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


def make_synthetic_auroc(
    value: float,
) -> DevAUROCResult:

    pair_count = (
        153
        * 306
    )

    if value == 0.0:

        wins = 0
        ties = 0

    elif value == 0.5:

        wins = 0
        ties = pair_count

    elif value == 1.0:

        wins = pair_count
        ties = 0

    else:

        raise ValueError(
            "Synthetic stage-runner audit only uses "
            "AUROC values 0.0, 0.5 or 1.0."
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
        tied_attack_bonafide_pairs=ties,
        score_definition=(
            "synthetic_stage_runner_orchestration_stub"
        ),
    )


# ======================================================================
# Expected-history helpers
# ======================================================================

def expected_raw_best_epochs(
    losses: list[
        float
    ],
) -> list[int]:

    best_loss: float | None = None
    best_epoch = 0

    result: list[
        int
    ] = []

    for (
        index,
        loss,
    ) in enumerate(
        losses,
        start=1,
    ):

        if (
            best_loss is None
            or loss < best_loss
        ):

            best_loss = (
                loss
            )

            best_epoch = (
                index
            )

        result.append(
            best_epoch
        )

    return result


# ======================================================================
# One synthetic orchestration case
# ======================================================================

def run_case(
    *,
    experiment_cfg: dict[str, Any],
    case_name: str,
    case_cfg: dict[str, Any],
    expected_transition_note: str,
) -> dict[str, Any]:

    stage = str(
        require_key(
            case_cfg,
            "stage",
            f"audit.cases.{case_name}",
        )
    )

    losses = [
        float(
            value
        )
        for value
        in require_key(
            case_cfg,
            "dev_losses",
            f"audit.cases.{case_name}",
        )
    ]

    aurocs = [
        float(
            value
        )
        for value
        in require_key(
            case_cfg,
            "dev_aurocs",
            f"audit.cases.{case_name}",
        )
    ]

    if len(
        losses
    ) != len(
        aurocs
    ):

        raise ValueError(
            f"{case_name}: loss/AUROC trajectory lengths differ."
        )

    expected_cfg = require_mapping(
        require_key(
            case_cfg,
            "expected",
            f"audit.cases.{case_name}",
        ),
        f"audit.cases.{case_name}.expected",
    )

    model = ProbeModel()

    initial_model_hash = (
        model_state_sha256(
            model
        )
    )

    state = {
        "epoch":
            0,

        "events":
            [],

        "epoch_model_hashes":
            {},
    }

    synthetic_targets = torch.cat(
        [
            torch.zeros(
                153,
                dtype=torch.int64,
            ),
            torch.ones(
                306,
                dtype=torch.int64,
            ),
        ]
    )

    synthetic_paths = tuple(
        f"synthetic_dev_{index:03d}.jpg"
        for index
        in range(
            459
        )
    )

    dummy_train_loader = object()
    dummy_dev_loader = object()
    dummy_optimizer = object()
    dummy_objective = object()

    # ------------------------------------------------------------------
    # Stubs.
    # ------------------------------------------------------------------

    def stub_train_one_epoch(
        **kwargs: Any,
    ) -> TrainingEpochResult:

        if kwargs[
            "model"
        ] is not model:

            raise RuntimeError(
                f"{case_name}: runner passed wrong model to train."
            )

        if kwargs[
            "loader"
        ] is not dummy_train_loader:

            raise RuntimeError(
                f"{case_name}: runner passed wrong train loader."
            )

        if kwargs[
            "optimizer"
        ] is not dummy_optimizer:

            raise RuntimeError(
                f"{case_name}: runner passed wrong optimizer."
            )

        if kwargs[
            "objective"
        ] is not dummy_objective:

            raise RuntimeError(
                f"{case_name}: runner passed wrong objective."
            )

        if kwargs[
            "stage"
        ] != stage:

            raise RuntimeError(
                f"{case_name}: train stage mismatch."
            )

        state[
            "epoch"
        ] += 1

        epoch = int(
            state[
                "epoch"
            ]
        )

        if epoch > len(
            losses
        ):

            raise RuntimeError(
                f"{case_name}: runner executed beyond "
                "synthetic trajectory."
            )

        mutate_probe_model(
            model=model,
            epoch=epoch,
        )

        state[
            "epoch_model_hashes"
        ][
            epoch
        ] = model_state_sha256(
            model
        )

        state[
            "events"
        ].append(
            f"train:{epoch}"
        )

        return TrainingEpochResult(
            stage=stage,
            loss=make_train_loss_summary(
                epoch
            ),
            optimizer_steps=45,
        )

    def stub_evaluate_dev_one_epoch(
        **kwargs: Any,
    ) -> DevEpochResult:

        if kwargs[
            "model"
        ] is not model:

            raise RuntimeError(
                f"{case_name}: runner passed wrong model to dev."
            )

        if kwargs[
            "loader"
        ] is not dummy_dev_loader:

            raise RuntimeError(
                f"{case_name}: runner passed wrong dev loader."
            )

        epoch = int(
            state[
                "epoch"
            ]
        )

        expected_previous = (
            f"train:{epoch}"
        )

        if (
            not state[
                "events"
            ]
            or state[
                "events"
            ][
                -1
            ]
            != expected_previous
        ):

            raise RuntimeError(
                f"{case_name}: dev did not immediately follow train."
            )

        model.eval()

        state[
            "events"
        ].append(
            f"dev:{epoch}"
        )

        return DevEpochResult(
            loss=make_dev_loss_summary(
                losses[
                    epoch
                    - 1
                ]
            ),
            logits=torch.zeros(
                (
                    459,
                    2,
                ),
                dtype=torch.float32,
            ),
            targets=synthetic_targets.clone(),
            image_paths=synthetic_paths,
        )

    def stub_compute_dev_auroc(
        **kwargs: Any,
    ) -> DevAUROCResult:

        epoch = int(
            state[
                "epoch"
            ]
        )

        expected_previous = (
            f"dev:{epoch}"
        )

        if (
            not state[
                "events"
            ]
            or state[
                "events"
            ][
                -1
            ]
            != expected_previous
        ):

            raise RuntimeError(
                f"{case_name}: AUROC did not immediately follow dev."
            )

        logits = kwargs[
            "logits"
        ]

        targets = kwargs[
            "targets"
        ]

        if tuple(
            logits.shape
        ) != (
            459,
            2,
        ):

            raise RuntimeError(
                f"{case_name}: runner passed wrong logits to AUROC."
            )

        if tuple(
            targets.shape
        ) != (
            459,
        ):

            raise RuntimeError(
                f"{case_name}: runner passed wrong targets to AUROC."
            )

        state[
            "events"
        ].append(
            f"auroc:{epoch}"
        )

        return make_synthetic_auroc(
            aurocs[
                epoch
                - 1
            ]
        )

    # ------------------------------------------------------------------
    # Execute actual stage_runner orchestration.
    # ------------------------------------------------------------------

    with (
        patch.object(
            stage_runner,
            "train_one_epoch",
            side_effect=stub_train_one_epoch,
        ),
        patch.object(
            stage_runner,
            "evaluate_dev_one_epoch",
            side_effect=stub_evaluate_dev_one_epoch,
        ),
        patch.object(
            stage_runner,
            "compute_dev_auroc",
            side_effect=stub_compute_dev_auroc,
        ),
    ):

        result = (
            stage_runner.run_training_stage(
                experiment_cfg=experiment_cfg,
                stage=stage,
                model=model,
                project_train_loader=(
                    dummy_train_loader
                ),
                dev_val_loader=(
                    dummy_dev_loader
                ),
                optimizer=(
                    dummy_optimizer
                ),
                objective=(
                    dummy_objective
                ),
                device="cpu",
            )
        )

    # ==================================================================
    # Expected summary
    # ==================================================================

    expected_epochs = int(
        require_key(
            expected_cfg,
            "epochs_completed",
            f"{case_name}.expected",
        )
    )

    expected_stop_reason = str(
        require_key(
            expected_cfg,
            "stop_reason",
            f"{case_name}.expected",
        )
    )

    expected_raw_best_epoch = int(
        require_key(
            expected_cfg,
            "raw_best_epoch",
            f"{case_name}.expected",
        )
    )

    expected_raw_best_loss = float(
        require_key(
            expected_cfg,
            "raw_best_weighted_dev_loss",
            f"{case_name}.expected",
        )
    )

    expected_selected_auroc = float(
        require_key(
            expected_cfg,
            "dev_auroc_at_raw_best_checkpoint",
            f"{case_name}.expected",
        )
    )

    expected_best_auroc = float(
        require_key(
            expected_cfg,
            "best_dev_auroc",
            f"{case_name}.expected",
        )
    )

    expected_best_auroc_epoch = int(
        require_key(
            expected_cfg,
            "best_dev_auroc_epoch",
            f"{case_name}.expected",
        )
    )

    if result.epochs_completed != expected_epochs:

        raise RuntimeError(
            f"{case_name}: epochs_completed mismatch."
        )

    if result.stop_reason != expected_stop_reason:

        raise RuntimeError(
            f"{case_name}: stop_reason mismatch:\n"
            f"  expected={expected_stop_reason}\n"
            f"  actual={result.stop_reason}"
        )

    if result.raw_best_epoch != expected_raw_best_epoch:

        raise RuntimeError(
            f"{case_name}: raw_best_epoch mismatch."
        )

    require_close(
        label=(
            f"{case_name}: raw_best_weighted_dev_loss"
        ),
        actual=result.raw_best_weighted_dev_loss,
        expected=expected_raw_best_loss,
    )

    require_close(
        label=(
            f"{case_name}: selected-checkpoint AUROC"
        ),
        actual=(
            result
            .dev_auroc_at_raw_best_checkpoint
        ),
        expected=expected_selected_auroc,
    )

    require_close(
        label=(
            f"{case_name}: best dev AUROC"
        ),
        actual=result.best_dev_auroc,
        expected=expected_best_auroc,
    )

    if (
        result.best_dev_auroc_epoch
        != expected_best_auroc_epoch
    ):

        raise RuntimeError(
            f"{case_name}: best AUROC epoch mismatch."
        )

    # ==================================================================
    # Exact event sequence.
    # ==================================================================

    expected_events: list[
        str
    ] = []

    for epoch in range(
        1,
        expected_epochs
        + 1,
    ):

        expected_events.extend(
            [
                f"train:{epoch}",
                f"dev:{epoch}",
                f"auroc:{epoch}",
            ]
        )

    if state[
        "events"
    ] != expected_events:

        raise RuntimeError(
            f"{case_name}: orchestration order mismatch:\n"
            f"  expected={expected_events}\n"
            f"  actual={state['events']}"
        )

    # ==================================================================
    # Per-epoch control history.
    # ==================================================================

    raw_updates = [
        bool(
            record
            .control
            .raw_checkpoint_updated
        )
        for record
        in result.history
    ]

    expected_raw_updates = list(
        require_key(
            expected_cfg,
            "raw_checkpoint_updated",
            f"{case_name}.expected",
        )
    )

    if raw_updates != expected_raw_updates:

        raise RuntimeError(
            f"{case_name}: raw checkpoint-update history mismatch."
        )

    meaningful = [
        record
        .control
        .meaningful_improvement
        for record
        in result.history
    ]

    expected_meaningful = list(
        require_key(
            expected_cfg,
            "meaningful_improvement",
            f"{case_name}.expected",
        )
    )

    if meaningful != expected_meaningful:

        raise RuntimeError(
            f"{case_name}: meaningful-improvement history mismatch."
        )

    patience = [
        int(
            record
            .control
            .patience_counter
        )
        for record
        in result.history
    ]

    expected_patience = [
        int(
            value
        )
        for value
        in require_key(
            expected_cfg,
            "patience_counter",
            f"{case_name}.expected",
        )
    ]

    if patience != expected_patience:

        raise RuntimeError(
            f"{case_name}: patience history mismatch."
        )

    expected_running_best_epochs = (
        expected_raw_best_epochs(
            losses[
                :expected_epochs
            ]
        )
    )

    actual_running_best_epochs = [
        int(
            record
            .control
            .raw_best_epoch
        )
        for record
        in result.history
    ]

    if (
        actual_running_best_epochs
        != expected_running_best_epochs
    ):

        raise RuntimeError(
            f"{case_name}: running raw-best epoch history mismatch."
        )

    for record in result.history[
        :-1
    ]:

        if record.control.should_stop:

            raise RuntimeError(
                f"{case_name}: runner continued after stop."
            )

    if (
        result.history[
            -1
        ]
        .control
        .should_stop
        is not True
    ):

        raise RuntimeError(
            f"{case_name}: final epoch is not stopping epoch."
        )

    # ==================================================================
    # Transition note.
    # ==================================================================

    notes = [
        record.transition_note
        for record
        in result.history
    ]

    if stage == "stage_b":

        if notes[
            0
        ] != expected_transition_note:

            raise RuntimeError(
                f"{case_name}: Stage-B epoch-1 transition note mismatch."
            )

        if any(
            note is not None
            for note
            in notes[
                1:
            ]
        ):

            raise RuntimeError(
                f"{case_name}: Stage-B transition note repeated."
            )

    else:

        if any(
            note is not None
            for note
            in notes
        ):

            raise RuntimeError(
                f"{case_name}: Stage-A contains Stage-B transition note."
            )

    # ==================================================================
    # Final exact raw-best restore.
    # ==================================================================

    epoch_model_hashes = state[
        "epoch_model_hashes"
    ]

    expected_selected_hash = (
        epoch_model_hashes[
            expected_raw_best_epoch
        ]
    )

    final_live_hash = model_state_sha256(
        model
    )

    if (
        result
        .restored_selected_model_state_sha256
        != expected_selected_hash
    ):

        raise RuntimeError(
            f"{case_name}: result selected-model SHA mismatch."
        )

    if final_live_hash != expected_selected_hash:

        raise RuntimeError(
            f"{case_name}: live model was not restored "
            "to raw-best epoch."
        )

    if (
        result
        .raw_best_checkpoint
        .model_state_sha256
        != expected_selected_hash
    ):

        raise RuntimeError(
            f"{case_name}: checkpoint SHA differs from "
            "expected raw-best epoch."
        )

    if (
        expected_raw_best_epoch
        != expected_epochs
    ):

        final_epoch_hash = (
            epoch_model_hashes[
                expected_epochs
            ]
        )

        if final_epoch_hash == expected_selected_hash:

            raise RuntimeError(
                f"{case_name}: synthetic final epoch and raw-best "
                "states are not distinguishable."
            )

    if result.model_left_in_eval_mode is not True:

        raise RuntimeError(
            f"{case_name}: result does not declare eval mode."
        )

    if model.training:

        raise RuntimeError(
            f"{case_name}: live selected model not in eval mode."
        )

    if result.optimizer_reuse_permitted is not False:

        raise RuntimeError(
            f"{case_name}: completed optimizer reuse must be forbidden."
        )

    # ==================================================================
    # AUROC must not control selection.
    # ==================================================================

    expected_auroc_difference = bool(
        require_key(
            expected_cfg,
            "auroc_best_differs_from_selected",
            f"{case_name}.expected",
        )
    )

    actual_auroc_difference = (
        result.best_dev_auroc_epoch
        != result.raw_best_epoch
    )

    if (
        actual_auroc_difference
        != expected_auroc_difference
    ):

        raise RuntimeError(
            f"{case_name}: AUROC-vs-loss selection expectation mismatch."
        )

    if expected_auroc_difference:

        if (
            result.raw_best_epoch
            == result.best_dev_auroc_epoch
        ):

            raise RuntimeError(
                f"{case_name}: AUROC appears to have altered "
                "loss-based checkpoint selection."
            )

    # ==================================================================
    # Stage-B disagreement guard.
    # ==================================================================

    guard_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    if stage == "stage_b":

        expected_guard = require_mapping(
            require_key(
                expected_cfg,
                "stage_b_guard",
                f"{case_name}.expected",
            ),
            f"{case_name}.expected.stage_b_guard",
        )

        actual_guard = (
            result
            .stage_b_auroc_disagreement
        )

        if actual_guard is None:

            raise RuntimeError(
                f"{case_name}: missing Stage-B AUROC guard."
            )

        require_close(
            label=(
                f"{case_name}: AUROC disagreement difference"
            ),
            actual=actual_guard.difference,
            expected=float(
                expected_guard[
                    "difference"
                ]
            ),
        )

        require_close(
            label=(
                f"{case_name}: AUROC disagreement threshold"
            ),
            actual=actual_guard.threshold,
            expected=float(
                expected_guard[
                    "threshold"
                ]
            ),
        )

        if (
            actual_guard.protocol_review_flag
            is not bool(
                expected_guard[
                    "protocol_review_flag"
                ]
            )
        ):

            raise RuntimeError(
                f"{case_name}: protocol-review flag mismatch."
            )

        if (
            actual_guard.automatically_switch_checkpoint
            is not bool(
                expected_guard[
                    "automatically_switch_checkpoint"
                ]
            )
        ):

            raise RuntimeError(
                f"{case_name}: automatic-switch policy mismatch."
            )

        if (
            actual_guard
            .loss_argmin_checkpoint_epoch
            != expected_raw_best_epoch
        ):

            raise RuntimeError(
                f"{case_name}: AUROC guard changed selected checkpoint."
            )

        guard_result = {
            "difference":
                actual_guard.difference,

            "threshold":
                actual_guard.threshold,

            "protocol_review_flag":
                actual_guard.protocol_review_flag,

            "automatically_switch_checkpoint":
                (
                    actual_guard
                    .automatically_switch_checkpoint
                ),

            "loss_argmin_checkpoint_epoch":
                (
                    actual_guard
                    .loss_argmin_checkpoint_epoch
                ),
        }

    else:

        if (
            result
            .stage_b_auroc_disagreement
            is not None
        ):

            raise RuntimeError(
                f"{case_name}: Stage-A unexpectedly produced "
                "Stage-B AUROC guard."
            )

    LOGGER.info(
        "[PASS] %s | stage=%s | epochs=%d | stop=%s | "
        "raw_best_epoch=%d | best_auroc_epoch=%d",
        case_name,
        stage,
        result.epochs_completed,
        result.stop_reason,
        result.raw_best_epoch,
        result.best_dev_auroc_epoch,
    )

    return {
        "stage":
            stage,

        "epochs_completed":
            result.epochs_completed,

        "stop_reason":
            result.stop_reason,

        "initial_model_state_sha256":
            initial_model_hash,

        "raw_best_epoch":
            result.raw_best_epoch,

        "raw_best_weighted_dev_loss":
            result.raw_best_weighted_dev_loss,

        "dev_auroc_at_raw_best_checkpoint":
            (
                result
                .dev_auroc_at_raw_best_checkpoint
            ),

        "best_dev_auroc":
            result.best_dev_auroc,

        "best_dev_auroc_epoch":
            result.best_dev_auroc_epoch,

        "raw_checkpoint_updated":
            raw_updates,

        "meaningful_improvement":
            meaningful,

        "patience_counter":
            patience,

        "running_raw_best_epochs":
            actual_running_best_epochs,

        "events":
            list(
                state[
                    "events"
                ]
            ),

        "epoch_model_state_sha256":
            {
                str(
                    key
                ):
                    value
                for (
                    key,
                    value,
                ) in epoch_model_hashes.items()
            },

        "restored_selected_model_state_sha256":
            final_live_hash,

        "transition_notes":
            notes,

        "model_left_in_eval_mode":
            not model.training,

        "optimizer_reuse_permitted":
            result.optimizer_reuse_permitted,

        "stage_b_auroc_disagreement":
            guard_result,
    }


# ======================================================================
# Invalid-stage guard
# ======================================================================

def audit_invalid_stage(
    *,
    experiment_cfg: dict[str, Any],
    invalid_stage: str,
) -> bool:

    model = ProbeModel()

    return expect_exception(
        label=(
            f"invalid stage {invalid_stage!r} rejected"
        ),
        function=lambda:
            stage_runner.run_training_stage(
                experiment_cfg=experiment_cfg,
                stage=invalid_stage,
                model=model,
                project_train_loader=object(),
                dev_val_loader=object(),
                optimizer=object(),
                objective=object(),
                device="cpu",
            ),
        exception_types=(
            ValueError,
        ),
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit synthetic orchestration of the "
            "Tech-2 multi-epoch stage runner."
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
            "audit_resnet18_stage_runner_config.yaml"
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
            "Stage-runner audit schema_version must equal 1."
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
        partial_path,
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

        expected_transition_note = str(
            require_key(
                audit_cfg,
                "expected_transition_note",
                "audit_config.audit",
            )
        )

        invalid_stage = str(
            require_key(
                audit_cfg,
                "invalid_stage",
                "audit_config.audit",
            )
        )

        cases_cfg = require_mapping(
            require_key(
                audit_cfg,
                "cases",
                "audit_config.audit",
            ),
            "audit_config.audit.cases",
        )

        # --------------------------------------------------------------
        # Provenance.
        # --------------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 MULTI-EPOCH STAGE-RUNNER ORCHESTRATION AUDIT"
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
            "Execution device: CPU only"
        )

        logger.info(
            "Experiment config SHA-256: %s",
            actual_experiment_sha,
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

        for source_path in (
            "src/stage_runner.py",
            "src/training_control.py",
            "src/development_metrics.py",
            "src/engine.py",
        ):

            logger.info(
                "%s SHA-256: %s",
                source_path,
                sha256_file(
                    REPO_ROOT
                    / source_path
                ),
            )

        logger.info(
            "One-epoch engine: SYNTHETICALLY STUBBED"
        )

        logger.info(
            "AUROC computation: SYNTHETICALLY STUBBED"
        )

        logger.info(
            "Stopping/checkpoint controller: REAL PRODUCTION IMPLEMENTATION"
        )

        logger.info(
            "Stage runner: REAL PRODUCTION IMPLEMENTATION"
        )

        logger.info(
            "FantasyID Dataset construction: NONE"
        )

        logger.info(
            "Image decoding: NONE"
        )

        logger.info(
            "Forward/backward/optimizer step: NONE"
        )

        logger.info(
            "FPR10 threshold derivation: NONE"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # --------------------------------------------------------------
        # Cases.
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Synthetic orchestration cases ---"
        )

        case_results: dict[
            str,
            Any,
        ] = {}

        for (
            case_name,
            raw_case_cfg,
        ) in cases_cfg.items():

            case_cfg = require_mapping(
                raw_case_cfg,
                f"audit.cases.{case_name}",
            )

            case_results[
                case_name
            ] = run_case(
                experiment_cfg=experiment_cfg,
                case_name=case_name,
                case_cfg=case_cfg,
                expected_transition_note=(
                    expected_transition_note
                ),
            )

        # --------------------------------------------------------------
        # Invalid stage.
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Invalid-stage guard ---"
        )

        invalid_stage_rejected = (
            audit_invalid_stage(
                experiment_cfg=experiment_cfg,
                invalid_stage=invalid_stage,
            )
        )

        # --------------------------------------------------------------
        # Result.
        # --------------------------------------------------------------

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

                    "stage_runner_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "stage_runner.py"
                        ),

                    "training_control_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "training_control.py"
                        ),

                    "development_metrics_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "development_metrics.py"
                        ),

                    "engine_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "engine.py"
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

                    "execution_device":
                        "cpu",

                    "real_one_epoch_engine_used":
                        False,

                    "real_development_auroc_used":
                        False,

                    "real_stage_training_controller_used":
                        True,

                    "real_stage_runner_used":
                        True,

                    "fantasyid_dataset_constructed":
                        False,

                    "images_decoded":
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

            "cases":
                case_results,

            "invalid_stage":
                {
                    "value":
                        invalid_stage,

                    "rejected":
                        invalid_stage_rejected,
                },

            "interpretation":
                (
                    "PASS establishes the orchestration semantics of "
                    "the production multi-epoch stage runner using "
                    "deterministic synthetic epoch results. Weighted "
                    "dev loss alone controls raw checkpoint selection "
                    "and patience; AUROC is diagnostic only; Stage-B "
                    "disagreement can flag review without switching "
                    "the loss-selected checkpoint; the selected "
                    "raw-best model is restored exactly at completion; "
                    "and completed-stage optimizer reuse is forbidden. "
                    "No real scientific training was performed."
                ),
        }

        with partial_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                result,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        partial_path.replace(
            result_path
        )

        result_sha = sha256_file(
            result_path
        )

        # --------------------------------------------------------------
        # Summary.
        # --------------------------------------------------------------

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
            "[PASS] train -> dev -> AUROC ordering"
        )

        logger.info(
            "[PASS] Stage-A patience stopping"
        )

        logger.info(
            "[PASS] Stage-B patience stopping"
        )

        logger.info(
            "[PASS] maximum-epoch stopping"
        )

        logger.info(
            "[PASS] weighted dev loss alone selects raw checkpoint"
        )

        logger.info(
            "[PASS] AUROC cannot switch selected checkpoint"
        )

        logger.info(
            "[PASS] Stage-B >0.01 disagreement flags review"
        )

        logger.info(
            "[PASS] Stage-B zero disagreement does not flag"
        )

        logger.info(
            "[PASS] Stage-B transition note at epoch 1 only"
        )

        logger.info(
            "[PASS] Stage-A contains no Stage-B transition note"
        )

        logger.info(
            "[PASS] raw-best model restored exactly"
        )

        logger.info(
            "[PASS] completed model left in eval mode"
        )

        logger.info(
            "[PASS] optimizer reuse forbidden after selected-state restore"
        )

        logger.info(
            "[PASS] invalid stage rejected"
        )

        logger.info(
            "[PASS] no FantasyID data/images accessed"
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
            "RESNET-18 MULTI-EPOCH STAGE-RUNNER "
            "ORCHESTRATION AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 MULTI-EPOCH STAGE-RUNNER "
            "ORCHESTRATION AUDIT: FAIL"
        )

        if partial_path.exists():

            partial_path.unlink()

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