#!/usr/bin/env python3
"""
Synthetic orchestration audit for Tech-2 one-resolution LR screening.

This audit exercises the REAL production function:

    src.lr_screening.run_resolution_lr_screening

while replacing the expensive already-validated lower layers with
deterministic stubs.

Validated
---------
- frozen LR-screening contract;
- screening seed = 8;
- candidate order = [3e-5, 1e-4, 3e-4];
- Stage A executes exactly once per resolution;
- Stage A completes before any Stage-B branch is built;
- all three Stage-B branches receive the exact same Stage-A checkpoint
  object and checkpoint SHA;
- all three frozen LR candidates execute exactly once in fixed order;
- selection uses raw-best weighted dev CE only;
- exact loss ties select the lower backbone LR;
- tie-break rule is independent of candidate tuple ordering;
- AUROC protocol-review flags cannot alter LR selection;
- protocol-review requirement is surfaced separately;
- selected Stage-B checkpoint belongs to the loss-selected LR;
- invalid screening seed is rejected before runtime construction;
- invalid resolution is rejected before runtime construction;
- production screening API contains no held-out-test route.

Not validated here
------------------
- real DataLoader construction;
- real Stage-A training;
- real Stage-B training;
- real AUROC values;
- artifact persistence.

Those lower components already have their own audits.

No FantasyID Dataset is constructed.
No image is decoded.
No CUDA computation is performed.
No forward/backward/optimizer step is performed.
No FPR10 threshold is derived.
No held-out test is accessed.
No print() is used.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import logging
import math
import re
import subprocess
import sys
import textwrap
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

import src.lr_screening as lr_screening


LOGGER = logging.getLogger(
    "audit_resnet18_lr_screening"
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
        f"Expected exception not raised: {label}"
    )


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
            "the LR-screening orchestration audit.\n\n"
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

    artifact_path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = match.group(
        "sha256"
    )

    if not artifact_path.is_file():

        raise FileNotFoundError(
            artifact_path
        )

    actual_sha = sha256_file(
        artifact_path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Validator artifact SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        artifact_path,
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
# Static held-out-test-route check
# ======================================================================

def audit_screening_api_surface() -> dict[str, Any]:

    function = (
        lr_screening
        .run_resolution_lr_screening
    )

    signature = inspect.signature(
        function
    )

    parameter_names = tuple(
        signature.parameters.keys()
    )

    expected_parameters = (
        "experiment_cfg",
        "machine_cfg",
        "repo_root",
        "resolution_name",
        "run_seed",
    )

    if parameter_names != expected_parameters:

        raise RuntimeError(
            "Unexpected LR-screening API surface:\n"
            f"  expected={expected_parameters}\n"
            f"  actual={parameter_names}"
        )

    if any(
        "test"
        in name.lower()
        for name
        in parameter_names
    ):

        raise RuntimeError(
            "LR-screening API exposes a test-related parameter."
        )

    source = textwrap.dedent(
        inspect.getsource(
            function
        )
    )

    tree = ast.parse(
        source
    )

    called_names: list[
        str
    ] = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        function_node = node.func

        if isinstance(
            function_node,
            ast.Name,
        ):

            called_names.append(
                function_node.id
            )

        elif isinstance(
            function_node,
            ast.Attribute,
        ):

            called_names.append(
                function_node.attr
            )

    test_related_calls = [
        name
        for name
        in called_names
        if "test" in name.lower()
    ]

    if test_related_calls:

        raise RuntimeError(
            "LR-screening function contains test-related calls:\n"
            f"  {test_related_calls}"
        )

    LOGGER.info(
        "[PASS] LR-screening API exposes development inputs only"
    )

    LOGGER.info(
        "[PASS] LR-screening function contains no test-related call route"
    )

    return {
        "parameters":
            list(
                parameter_names
            ),

        "called_functions":
            called_names,

        "test_related_parameters":
            [],

        "test_related_calls":
            [],
    }


# ======================================================================
# Synthetic outer-orchestration case
# ======================================================================

def run_screening_case(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    resolution_name: str,
    run_seed: int,
    backbone_lrs: tuple[
        float,
        ...,
    ],
    case_name: str,
    case_cfg: dict[str, Any],
) -> dict[str, Any]:

    losses = tuple(
        float(
            value
        )
        for value
        in require_key(
            case_cfg,
            "raw_best_weighted_dev_losses",
            f"audit.cases.{case_name}",
        )
    )

    review_flags = tuple(
        bool(
            value
        )
        for value
        in require_key(
            case_cfg,
            "protocol_review_flags",
            f"audit.cases.{case_name}",
        )
    )

    if len(
        losses
    ) != len(
        backbone_lrs
    ):

        raise ValueError(
            f"{case_name}: loss count does not match LR count."
        )

    if len(
        review_flags
    ) != len(
        backbone_lrs
    ):

        raise ValueError(
            f"{case_name}: review-flag count does not match LR count."
        )

    expected_cfg = require_mapping(
        require_key(
            case_cfg,
            "expected",
            f"audit.cases.{case_name}",
        ),
        f"audit.cases.{case_name}.expected",
    )

    # ------------------------------------------------------------------
    # Synthetic Stage-A selected checkpoint.
    # ------------------------------------------------------------------

    stage_a_checkpoint = SimpleNamespace(
        stage="stage_a",
        epoch=2,
        weighted_dev_loss=0.7,
        model_state_sha256=(
            "synthetic_stage_a_checkpoint_sha256"
        ),
    )

    stage_a_result = SimpleNamespace(
        stage="stage_a",
        raw_best_checkpoint=(
            stage_a_checkpoint
        ),
        raw_best_weighted_dev_loss=0.7,
    )

    stage_a_model = SimpleNamespace(
        role="stage_a_model"
    )

    stage_a_loaders = SimpleNamespace(
        project_train=object(),
        dev_val=object(),
    )

    stage_a_optimizer = object()
    stage_a_objective = object()

    stage_a_initialization = SimpleNamespace(
        resolution_name=resolution_name,
        run_seed=run_seed,
        synthetic=True,
    )

    state: dict[str, Any] = {
        "stage_a_runtime_calls":
            0,

        "stage_a_runner_calls":
            0,

        "branch_build_lrs":
            [],

        "stage_b_runner_lrs":
            [],

        "stage_a_checkpoint_object_ids":
            [],

        "stage_a_checkpoint_shas":
            [],

        "objective_construction_count":
            0,

        "events":
            [],

        "branch_by_model_id":
            {},
    }

    # ------------------------------------------------------------------
    # Stage-A runtime stub.
    # ------------------------------------------------------------------

    def fake_build_stage_a_runtime(
        **kwargs: Any,
    ) -> tuple[
        Any,
        Any,
        Any,
        Any,
        torch.device,
        Any,
    ]:

        state[
            "stage_a_runtime_calls"
        ] += 1

        state[
            "events"
        ].append(
            "stage_a_runtime"
        )

        if (
            kwargs[
                "experiment_cfg"
            ]
            is not experiment_cfg
        ):

            raise RuntimeError(
                f"{case_name}: wrong experiment config passed to Stage A."
            )

        if (
            kwargs[
                "machine_cfg"
            ]
            is not machine_cfg
        ):

            raise RuntimeError(
                f"{case_name}: wrong machine config passed to Stage A."
            )

        if (
            Path(
                kwargs[
                    "repo_root"
                ]
            ).resolve()
            != REPO_ROOT
        ):

            raise RuntimeError(
                f"{case_name}: wrong repo root passed to Stage A."
            )

        if (
            kwargs[
                "resolution_name"
            ]
            != resolution_name
        ):

            raise RuntimeError(
                f"{case_name}: wrong resolution passed to Stage A."
            )

        if (
            kwargs[
                "run_seed"
            ]
            != run_seed
        ):

            raise RuntimeError(
                f"{case_name}: wrong seed passed to Stage A."
            )

        return (
            stage_a_model,
            stage_a_loaders,
            stage_a_optimizer,
            stage_a_objective,
            torch.device(
                "cpu"
            ),
            stage_a_initialization,
        )

    # ------------------------------------------------------------------
    # Stage-B branch builder stub.
    # ------------------------------------------------------------------

    def fake_build_stage_b_branch(
        **kwargs: Any,
    ) -> Any:

        lr = float(
            kwargs[
                "backbone_lr"
            ]
        )

        state[
            "events"
        ].append(
            f"build_stage_b:{lr:.8g}"
        )

        state[
            "branch_build_lrs"
        ].append(
            lr
        )

        checkpoint = kwargs[
            "stage_a_checkpoint"
        ]

        state[
            "stage_a_checkpoint_object_ids"
        ].append(
            id(
                checkpoint
            )
        )

        state[
            "stage_a_checkpoint_shas"
        ].append(
            checkpoint.model_state_sha256
        )

        if checkpoint is not stage_a_checkpoint:

            raise RuntimeError(
                f"{case_name}: Stage-B branch received a different "
                "Stage-A checkpoint object."
            )

        if (
            kwargs[
                "resolution_name"
            ]
            != resolution_name
        ):

            raise RuntimeError(
                f"{case_name}: Stage-B resolution mismatch."
            )

        if (
            kwargs[
                "run_seed"
            ]
            != run_seed
        ):

            raise RuntimeError(
                f"{case_name}: Stage-B seed mismatch."
            )

        branch_index = (
            backbone_lrs.index(
                lr
            )
        )

        model = SimpleNamespace(
            role="stage_b_model",
            backbone_lr=lr,
            branch_index=branch_index,
        )

        dataloaders = SimpleNamespace(
            project_train=object(),
            dev_val=object(),
        )

        initialization = SimpleNamespace(
            stage_a_checkpoint_model_state_sha256=(
                stage_a_checkpoint
                .model_state_sha256
            ),
            restored_model_state_sha256_after_device_move=(
                stage_a_checkpoint
                .model_state_sha256
            ),
        )

        branch = SimpleNamespace(
            initialization_evidence=(
                initialization
            ),
            device=torch.device(
                "cpu"
            ),
            model=model,
            dataloaders=dataloaders,
            optimizer=object(),
        )

        state[
            "branch_by_model_id"
        ][
            id(
                model
            )
        ] = branch

        return branch

    # ------------------------------------------------------------------
    # Stage-B objective stub.
    # ------------------------------------------------------------------

    def fake_objective(
        *,
        experiment_cfg: Any,
        device: Any,
    ) -> Any:

        if experiment_cfg is not experiment_cfg_outer:

            raise RuntimeError(
                f"{case_name}: wrong config passed to Stage-B objective."
            )

        if torch.device(
            device
        ) != torch.device(
            "cpu"
        ):

            raise RuntimeError(
                f"{case_name}: synthetic Stage-B objective "
                "must remain CPU-only."
            )

        state[
            "objective_construction_count"
        ] += 1

        return SimpleNamespace(
            synthetic=True
        )

    # Python closure alias avoids shadowing inside fake_objective.
    experiment_cfg_outer = (
        experiment_cfg
    )

    # ------------------------------------------------------------------
    # Shared Stage-A / Stage-B runner stub.
    # ------------------------------------------------------------------

    def fake_run_training_stage(
        **kwargs: Any,
    ) -> Any:

        stage = kwargs[
            "stage"
        ]

        if stage == "stage_a":

            state[
                "events"
            ].append(
                "run_stage_a"
            )

            state[
                "stage_a_runner_calls"
            ] += 1

            if (
                kwargs[
                    "model"
                ]
                is not stage_a_model
            ):

                raise RuntimeError(
                    f"{case_name}: Stage-A runner received wrong model."
                )

            if (
                kwargs[
                    "optimizer"
                ]
                is not stage_a_optimizer
            ):

                raise RuntimeError(
                    f"{case_name}: Stage-A runner received wrong optimizer."
                )

            return stage_a_result

        if stage != "stage_b":

            raise RuntimeError(
                f"{case_name}: unexpected synthetic stage {stage!r}."
            )

        model = kwargs[
            "model"
        ]

        lr = float(
            model.backbone_lr
        )

        state[
            "events"
        ].append(
            f"run_stage_b:{lr:.8g}"
        )

        state[
            "stage_b_runner_lrs"
        ].append(
            lr
        )

        branch = state[
            "branch_by_model_id"
        ][
            id(
                model
            )
        ]

        if (
            kwargs[
                "project_train_loader"
            ]
            is not branch.dataloaders.project_train
        ):

            raise RuntimeError(
                f"{case_name}: Stage-B runner received wrong train loader."
            )

        if (
            kwargs[
                "dev_val_loader"
            ]
            is not branch.dataloaders.dev_val
        ):

            raise RuntimeError(
                f"{case_name}: Stage-B runner received wrong dev loader."
            )

        if (
            kwargs[
                "optimizer"
            ]
            is not branch.optimizer
        ):

            raise RuntimeError(
                f"{case_name}: Stage-B runner received wrong optimizer."
            )

        branch_index = int(
            model.branch_index
        )

        checkpoint = SimpleNamespace(
            stage="stage_b",
            model_state_sha256=(
                f"synthetic_stage_b_checkpoint_{branch_index}"
            ),
        )

        disagreement = SimpleNamespace(
            protocol_review_flag=(
                review_flags[
                    branch_index
                ]
            )
        )

        return SimpleNamespace(
            stage="stage_b",
            raw_best_checkpoint=(
                checkpoint
            ),
            raw_best_weighted_dev_loss=(
                losses[
                    branch_index
                ]
            ),
            stage_b_auroc_disagreement=(
                disagreement
            ),
        )

    # ------------------------------------------------------------------
    # Execute REAL outer orchestration.
    # ------------------------------------------------------------------

    with (
        patch.object(
            lr_screening,
            "_build_stage_a_runtime",
            side_effect=(
                fake_build_stage_a_runtime
            ),
        ),
        patch.object(
            lr_screening,
            "build_stage_b_branch",
            side_effect=(
                fake_build_stage_b_branch
            ),
        ),
        patch.object(
            lr_screening,
            "run_training_stage",
            side_effect=(
                fake_run_training_stage
            ),
        ),
        patch.object(
            lr_screening,
            "WeightedCrossEntropyObjective",
            side_effect=(
                fake_objective
            ),
        ),
        patch.object(
            lr_screening.torch.cuda,
            "is_available",
            return_value=False,
        ),
    ):

        result = (
            lr_screening
            .run_resolution_lr_screening(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                repo_root=REPO_ROOT,
                resolution_name=resolution_name,
                run_seed=run_seed,
            )
        )

    # ==================================================================
    # Stage A exactly once.
    # ==================================================================

    if (
        state[
            "stage_a_runtime_calls"
        ]
        != 1
    ):

        raise RuntimeError(
            f"{case_name}: Stage-A runtime was not built exactly once."
        )

    if (
        state[
            "stage_a_runner_calls"
        ]
        != 1
    ):

        raise RuntimeError(
            f"{case_name}: Stage A did not execute exactly once."
        )

    # ==================================================================
    # Three Stage-B candidates, fixed order.
    # ==================================================================

    if tuple(
        state[
            "branch_build_lrs"
        ]
    ) != backbone_lrs:

        raise RuntimeError(
            f"{case_name}: Stage-B branch construction order changed:\n"
            f"  expected={backbone_lrs}\n"
            f"  actual={state['branch_build_lrs']}"
        )

    if tuple(
        state[
            "stage_b_runner_lrs"
        ]
    ) != backbone_lrs:

        raise RuntimeError(
            f"{case_name}: Stage-B execution order changed."
        )

    if (
        state[
            "objective_construction_count"
        ]
        != 3
    ):

        raise RuntimeError(
            f"{case_name}: expected three fresh Stage-B objectives."
        )

    # ==================================================================
    # Same Stage-A checkpoint for every branch.
    # ==================================================================

    checkpoint_object_ids = set(
        state[
            "stage_a_checkpoint_object_ids"
        ]
    )

    if checkpoint_object_ids != {
        id(
            stage_a_checkpoint
        )
    }:

        raise RuntimeError(
            f"{case_name}: LR branches did not receive "
            "one identical Stage-A checkpoint object."
        )

    checkpoint_shas = set(
        state[
            "stage_a_checkpoint_shas"
        ]
    )

    if checkpoint_shas != {
        stage_a_checkpoint.model_state_sha256
    }:

        raise RuntimeError(
            f"{case_name}: LR branches disagree on Stage-A checkpoint SHA."
        )

    candidate_lrs = tuple(
        candidate.backbone_lr
        for candidate
        in result.stage_b_candidates
    )

    if candidate_lrs != backbone_lrs:

        raise RuntimeError(
            f"{case_name}: returned Stage-B candidate order changed."
        )

    for candidate in (
        result.stage_b_candidates
    ):

        if (
            candidate
            .initialization
            .stage_a_checkpoint_model_state_sha256
            != stage_a_checkpoint.model_state_sha256
        ):

            raise RuntimeError(
                f"{case_name}: candidate Stage-A source SHA mismatch."
            )

        if (
            candidate
            .initialization
            .restored_model_state_sha256_after_device_move
            != stage_a_checkpoint.model_state_sha256
        ):

            raise RuntimeError(
                f"{case_name}: candidate did not restore Stage-A SHA."
            )

    # ==================================================================
    # Selection result.
    # ==================================================================

    expected_lr = float(
        require_key(
            expected_cfg,
            "selected_backbone_lr",
            f"{case_name}.expected",
        )
    )

    expected_loss = float(
        require_key(
            expected_cfg,
            "selected_raw_best_weighted_dev_loss",
            f"{case_name}.expected",
        )
    )

    expected_tie = bool(
        require_key(
            expected_cfg,
            "exact_loss_tie_encountered",
            f"{case_name}.expected",
        )
    )

    expected_review = bool(
        require_key(
            expected_cfg,
            "protocol_review_required",
            f"{case_name}.expected",
        )
    )

    if result.selected_backbone_lr != expected_lr:

        raise RuntimeError(
            f"{case_name}: selected LR mismatch:\n"
            f"  expected={expected_lr}\n"
            f"  actual={result.selected_backbone_lr}"
        )

    require_close(
        label=(
            f"{case_name}: selected weighted dev loss"
        ),
        actual=(
            result
            .selected_raw_best_weighted_dev_loss
        ),
        expected=expected_loss,
    )

    if (
        result.exact_loss_tie_encountered
        is not expected_tie
    ):

        raise RuntimeError(
            f"{case_name}: exact tie flag mismatch."
        )

    if (
        result.protocol_review_required
        is not expected_review
    ):

        raise RuntimeError(
            f"{case_name}: protocol-review aggregation mismatch."
        )

    selected_index = (
        backbone_lrs.index(
            expected_lr
        )
    )

    expected_checkpoint_sha = (
        f"synthetic_stage_b_checkpoint_{selected_index}"
    )

    if (
        result
        .selected_stage_b_checkpoint_sha256
        != expected_checkpoint_sha
    ):

        raise RuntimeError(
            f"{case_name}: selected checkpoint does not belong "
            "to loss-selected LR."
        )

    # ==================================================================
    # Exact orchestration event sequence.
    # ==================================================================

    expected_events = [
        "stage_a_runtime",
        "run_stage_a",
    ]

    for lr in backbone_lrs:

        expected_events.extend(
            [
                f"build_stage_b:{lr:.8g}",
                f"run_stage_b:{lr:.8g}",
            ]
        )

    if (
        state[
            "events"
        ]
        != expected_events
    ):

        raise RuntimeError(
            f"{case_name}: outer orchestration event order mismatch:\n"
            f"  expected={expected_events}\n"
            f"  actual={state['events']}"
        )

    LOGGER.info(
        "[PASS] %s | selected_lr=%.8g | loss=%.12f | "
        "tie=%s | review=%s",
        case_name,
        result.selected_backbone_lr,
        result.selected_raw_best_weighted_dev_loss,
        result.exact_loss_tie_encountered,
        result.protocol_review_required,
    )

    return {
        "stage_a_runtime_calls":
            state[
                "stage_a_runtime_calls"
            ],

        "stage_a_runner_calls":
            state[
                "stage_a_runner_calls"
            ],

        "stage_b_branch_build_order":
            state[
                "branch_build_lrs"
            ],

        "stage_b_run_order":
            state[
                "stage_b_runner_lrs"
            ],

        "same_stage_a_checkpoint_object_for_all_branches":
            True,

        "stage_a_checkpoint_sha":
            stage_a_checkpoint.model_state_sha256,

        "candidate_losses":
            list(
                losses
            ),

        "candidate_protocol_review_flags":
            list(
                review_flags
            ),

        "selected_backbone_lr":
            result.selected_backbone_lr,

        "selected_raw_best_weighted_dev_loss":
            (
                result
                .selected_raw_best_weighted_dev_loss
            ),

        "selected_stage_b_checkpoint_sha256":
            (
                result
                .selected_stage_b_checkpoint_sha256
            ),

        "exact_loss_tie_encountered":
            result.exact_loss_tie_encountered,

        "protocol_review_required":
            result.protocol_review_required,

        "events":
            state[
                "events"
            ],
    }


# ======================================================================
# Tie-break order independence
# ======================================================================

def audit_tie_break_order_independence() -> dict[str, Any]:

    high_lr = SimpleNamespace(
        backbone_lr=0.0001,
        stage_b_result=SimpleNamespace(
            raw_best_weighted_dev_loss=0.4,
        ),
    )

    low_lr = SimpleNamespace(
        backbone_lr=0.00003,
        stage_b_result=SimpleNamespace(
            raw_best_weighted_dev_loss=0.4,
        ),
    )

    # Deliberately reverse numeric LR order.
    candidates = (
        high_lr,
        low_lr,
    )

    selected, tie = (
        lr_screening
        ._select_stage_b_candidate(
            candidates=candidates,
        )
    )

    if tie is not True:

        raise RuntimeError(
            "Exact loss tie was not detected."
        )

    if selected is not low_lr:

        raise RuntimeError(
            "Exact tie selection depends on candidate tuple order "
            "rather than choosing the lower backbone LR."
        )

    LOGGER.info(
        "[PASS] lower-LR exact tie-break is independent "
        "of candidate tuple order"
    )

    return {
        "input_order":
            [
                0.0001,
                0.00003,
            ],

        "equal_loss":
            0.4,

        "selected_backbone_lr":
            selected.backbone_lr,

        "exact_tie":
            tie,
    }


# ======================================================================
# Invalid seed / resolution guards
# ======================================================================

def audit_invalid_guards(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    resolution_name: str,
    screening_seed: int,
    invalid_seed: int,
    invalid_resolution: str,
) -> dict[str, bool]:

    def forbidden_runtime(
        **_: Any,
    ) -> Any:

        raise RuntimeError(
            "Scientific runtime construction occurred before "
            "invalid screening input was rejected."
        )

    with patch.object(
        lr_screening,
        "_build_stage_a_runtime",
        side_effect=forbidden_runtime,
    ):

        invalid_seed_rejected = (
            expect_exception(
                label=(
                    f"non-screening seed {invalid_seed} rejected "
                    "before runtime construction"
                ),
                function=lambda:
                    lr_screening
                    .run_resolution_lr_screening(
                        experiment_cfg=experiment_cfg,
                        machine_cfg=machine_cfg,
                        repo_root=REPO_ROOT,
                        resolution_name=resolution_name,
                        run_seed=invalid_seed,
                    ),
                exception_types=(
                    ValueError,
                ),
            )
        )

        invalid_resolution_rejected = (
            expect_exception(
                label=(
                    f"invalid resolution {invalid_resolution!r} "
                    "rejected before runtime construction"
                ),
                function=lambda:
                    lr_screening
                    .run_resolution_lr_screening(
                        experiment_cfg=experiment_cfg,
                        machine_cfg=machine_cfg,
                        repo_root=REPO_ROOT,
                        resolution_name=(
                            invalid_resolution
                        ),
                        run_seed=screening_seed,
                    ),
                exception_types=(
                    ValueError,
                ),
            )
        )

    return {
        "invalid_seed_rejected":
            invalid_seed_rejected,

        "invalid_resolution_rejected":
            invalid_resolution_rejected,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit synthetic outer orchestration of "
            "Tech-2 resolution LR screening."
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
            "audit_resnet18_lr_screening_config.yaml"
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
            "LR-screening audit schema_version must equal 1."
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

        resolution_name = str(
            require_key(
                audit_cfg,
                "resolution",
                "audit_config.audit",
            )
        )

        screening_seed = int(
            require_key(
                audit_cfg,
                "screening_seed",
                "audit_config.audit",
            )
        )

        backbone_lrs = tuple(
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

        invalid_seed = int(
            require_key(
                audit_cfg,
                "invalid_seed",
                "audit_config.audit",
            )
        )

        invalid_resolution = str(
            require_key(
                audit_cfg,
                "invalid_resolution",
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
        # Reassert exact frozen screening contract.
        # --------------------------------------------------------------

        contract = (
            lr_screening
            .load_lr_screening_contract(
                experiment_cfg=experiment_cfg,
            )
        )

        if contract.screening_seed != screening_seed:

            raise RuntimeError(
                "Audit screening seed disagrees with production contract."
            )

        if (
            contract.backbone_lr_candidates
            != backbone_lrs
        ):

            raise RuntimeError(
                "Audit LR set disagrees with production contract."
            )

        if resolution_name not in (
            contract
            .candidate_resolutions
        ):

            raise RuntimeError(
                "Audit resolution is outside production screening set."
            )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 LR-SCREENING ORCHESTRATION AUDIT"
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
            "src/lr_screening.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "lr_screening.py"
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
            "Execution: synthetic outer orchestration only"
        )

        logger.info(
            "FantasyID Dataset construction: NONE"
        )

        logger.info(
            "Image decoding: NONE"
        )

        logger.info(
            "Real Stage-A/Stage-B training: NONE"
        )

        logger.info(
            "FPR10 threshold derivation: NONE"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # ==============================================================
        # API / held-out-test surface
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Development-only screening API ---"
        )

        api_result = (
            audit_screening_api_surface()
        )

        # ==============================================================
        # Synthetic selection cases
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Synthetic LR-screening cases ---"
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
            ] = run_screening_case(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                resolution_name=(
                    resolution_name
                ),
                run_seed=screening_seed,
                backbone_lrs=backbone_lrs,
                case_name=case_name,
                case_cfg=case_cfg,
            )

        # ==============================================================
        # Independent tie-order control
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Exact-tie order-independence control ---"
        )

        tie_order_result = (
            audit_tie_break_order_independence()
        )

        # ==============================================================
        # Invalid input guards
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Screening input guards ---"
        )

        invalid_guard_result = (
            audit_invalid_guards(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                resolution_name=(
                    resolution_name
                ),
                screening_seed=screening_seed,
                invalid_seed=invalid_seed,
                invalid_resolution=(
                    invalid_resolution
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

                    "lr_screening_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "lr_screening.py"
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

                    "resolution":
                        resolution_name,

                    "screening_seed":
                        screening_seed,

                    "backbone_lr_candidates":
                        list(
                            backbone_lrs
                        ),

                    "fantasyid_dataset_constructed":
                        False,

                    "images_decoded":
                        False,

                    "real_training_performed":
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

            "api_surface":
                api_result,

            "cases":
                case_results,

            "tie_break_order_independence":
                tie_order_result,

            "invalid_input_guards":
                invalid_guard_result,

            "interpretation":
                (
                    "PASS establishes the production outer "
                    "one-resolution LR-screening orchestration: "
                    "Stage A executes exactly once, all three frozen "
                    "Stage-B branches receive the same Stage-A "
                    "checkpoint, LR candidates execute in frozen order, "
                    "raw-best weighted dev CE alone determines the "
                    "selected LR, exact ties select the lower LR, "
                    "AUROC review flags are surfaced without changing "
                    "selection, invalid seed/resolution inputs fail "
                    "before runtime construction, and the screening API "
                    "exposes no held-out-test route. No real scientific "
                    "training was performed."
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
            "[PASS] Stage A executes exactly once per resolution"
        )

        logger.info(
            "[PASS] Stage A completes before Stage-B branching"
        )

        logger.info(
            "[PASS] all three Stage-B branches receive "
            "the same Stage-A checkpoint"
        )

        logger.info(
            "[PASS] Stage-B LR candidate order = "
            "[3e-5, 1e-4, 3e-4]"
        )

        logger.info(
            "[PASS] selection uses raw-best weighted dev CE only"
        )

        logger.info(
            "[PASS] exact loss tie selects lower backbone LR"
        )

        logger.info(
            "[PASS] lower-LR tie-break independent of tuple order"
        )

        logger.info(
            "[PASS] AUROC review flags cannot change LR selection"
        )

        logger.info(
            "[PASS] protocol-review flags remain surfaced separately"
        )

        logger.info(
            "[PASS] selected checkpoint belongs to selected LR"
        )

        logger.info(
            "[PASS] invalid seed rejected before runtime construction"
        )

        logger.info(
            "[PASS] invalid resolution rejected before runtime construction"
        )

        logger.info(
            "[PASS] screening API exposes no held-out-test route"
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
            "RESNET-18 LR-SCREENING ORCHESTRATION AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 LR-SCREENING ORCHESTRATION AUDIT: FAIL"
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