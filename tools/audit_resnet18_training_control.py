#!/usr/bin/env python3
"""
Audit frozen Tech-2 stopping and raw-argmin checkpoint control.

This audit is intentionally CPU-only and does not train ResNet-18.

It verifies:

1. frozen experiment validator passes;
2. clean Git state before evidence generation;
3. Stage-A contract = patience 2 / maximum 10;
4. Stage-B contract = patience 5 / maximum 30;
5. meaningful-improvement fraction = 0.005;
6. raw checkpoint updates on every strict raw dev-loss reduction;
7. raw checkpoint update is independent of patience reset;
8. tiny raw improvements may still consume patience;
9. exact 0.5% improvement does NOT reset patience because the
   frozen comparison is strict "<";
10. >0.5% improvement resets patience and updates anchor;
11. worse dev loss cannot replace raw-best checkpoint;
12. Stage-A stops after two consecutive non-meaningful epochs;
13. Stage-B stops after five consecutive non-meaningful epochs;
14. Stage-A maximum epoch = 10;
15. Stage-B maximum epoch = 30;
16. maximum-epoch stopping works independently of patience;
17. raw checkpoints contain detached CPU model state;
18. checkpoint state is independent of later model mutation;
19. checkpoint restore is exact by SHA-256;
20. model training mode is not silently restored by state_dict;
21. requires_grad state is not silently restored by state_dict;
22. checkpoint-integrity tampering is detected;
23. non-consecutive epoch observation is rejected;
24. observation after controller stop is rejected;
25. invalid checkpoint metadata is rejected.

The probe model is a tiny deterministic state container. There is no
dataset access and no CUDA dependency.

No held-out test data are accessed.
No training run is performed.
No Stage-A -> Stage-B RNG semantics are defined here.

Detailed evidence is written under ./logs/.
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
from typing import Any

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

from src.training_control import (
    RawArgminModelCheckpoint,
    StageTrainingController,
    capture_raw_argmin_model_checkpoint,
    load_stage_stopping_contract,
    model_state_sha256,
    restore_raw_argmin_model_checkpoint,
    state_dict_sha256,
)


LOGGER = logging.getLogger(
    "audit_resnet18_training_control"
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


# ======================================================================
# Clean Git gate
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
            "the training-control audit.\n\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Frozen validator
# ======================================================================

def run_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> tuple[
    Path,
    str,
]:

    command = [
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
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Frozen experiment validator failed.\n"
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
            "Frozen experiment validator did not return PASS."
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
            "Validator artifact SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        path,
        expected_sha,
    )


# ======================================================================
# Logging / output
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

    output_path = (
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

    partial_output_path = Path(
        str(
            output_path
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
        output_path,
        partial_output_path,
    )


# ======================================================================
# Deterministic tiny checkpoint probe model
# ======================================================================

class CheckpointProbeModel(
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
                        0.10,
                        -0.20,
                        0.30,
                    ],
                    [
                        -0.40,
                        0.50,
                        -0.60,
                    ],
                ],
                dtype=torch.float32,
            )
        )

        self.bias = nn.Parameter(
            torch.tensor(
                [
                    0.01,
                    -0.02,
                ],
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "running_marker",
            torch.tensor(
                [
                    1.0,
                    -1.0,
                ],
                dtype=torch.float32,
            ),
        )


def mutate_probe_model(
    *,
    model: CheckpointProbeModel,
    epoch: int,
) -> None:
    """
    Give each synthetic epoch a unique deterministic model state.

    No random numbers are consumed.
    """

    with torch.no_grad():

        model.weight.add_(
            float(
                epoch
            )
            * 0.001
        )

        model.bias.sub_(
            float(
                epoch
            )
            * 0.0001
        )

        model.running_marker.add_(
            torch.tensor(
                [
                    float(
                        epoch
                    )
                    * 0.01,

                    -float(
                        epoch
                    )
                    * 0.02,
                ],
                dtype=torch.float32,
            )
        )


# ======================================================================
# Contract audit
# ======================================================================

def audit_contracts(
    *,
    experiment_cfg: dict[str, Any],
    expected_cfg: dict[str, Any],
) -> dict[str, Any]:

    result: dict[
        str,
        Any,
    ] = {}

    for stage in (
        "stage_a",
        "stage_b",
    ):

        expected = require_mapping(
            require_key(
                expected_cfg,
                stage,
                "audit.expected_contracts",
            ),
            f"audit.expected_contracts.{stage}",
        )

        contract = load_stage_stopping_contract(
            experiment_cfg=experiment_cfg,
            stage=stage,
        )

        expected_patience = int(
            require_key(
                expected,
                "patience_epochs",
                f"audit.expected_contracts.{stage}",
            )
        )

        expected_maximum = int(
            require_key(
                expected,
                "maximum_epochs",
                f"audit.expected_contracts.{stage}",
            )
        )

        expected_fraction = float(
            require_key(
                expected,
                "meaningful_relative_improvement_fraction",
                f"audit.expected_contracts.{stage}",
            )
        )

        if (
            contract.patience_epochs
            != expected_patience
        ):

            raise RuntimeError(
                f"{stage} patience mismatch."
            )

        if (
            contract.maximum_epochs
            != expected_maximum
        ):

            raise RuntimeError(
                f"{stage} maximum-epoch mismatch."
            )

        if (
            contract
            .meaningful_relative_improvement_fraction
            != expected_fraction
        ):

            raise RuntimeError(
                f"{stage} meaningful-improvement fraction mismatch."
            )

        result[
            stage
        ] = {
            "patience_epochs":
                contract.patience_epochs,

            "maximum_epochs":
                contract.maximum_epochs,

            "meaningful_relative_improvement_fraction":
                (
                    contract
                    .meaningful_relative_improvement_fraction
                ),

            "checkpoint_metric":
                contract.checkpoint_metric,

            "checkpoint_policy":
                contract.checkpoint_policy,
        }

        LOGGER.info(
            "[PASS] %s contract | patience=%d | max_epochs=%d | "
            "relative_fraction=%.6g",
            stage,
            contract.patience_epochs,
            contract.maximum_epochs,
            contract.meaningful_relative_improvement_fraction,
        )

    return result


# ======================================================================
# Trajectory audit
# ======================================================================

def normalize_nullable_bool_list(
    values: list[
        Any
    ],
) -> list[
    bool | None
]:

    result: list[
        bool | None
    ] = []

    for value in values:

        if value is None:

            result.append(
                None
            )

        elif isinstance(
            value,
            bool,
        ):

            result.append(
                value
            )

        else:

            raise TypeError(
                "Expected bool/null trajectory value, "
                f"got {value!r}"
            )

    return result


def run_trajectory(
    *,
    experiment_cfg: dict[str, Any],
    trajectory_name: str,
    trajectory_cfg: dict[str, Any],
) -> dict[str, Any]:

    stage = str(
        require_key(
            trajectory_cfg,
            "stage",
            f"audit.trajectories.{trajectory_name}",
        )
    )

    losses = [
        float(
            value
        )
        for value
        in require_key(
            trajectory_cfg,
            "losses",
            f"audit.trajectories.{trajectory_name}",
        )
    ]

    expected_cfg = require_mapping(
        require_key(
            trajectory_cfg,
            "expected",
            f"audit.trajectories.{trajectory_name}",
        ),
        (
            "audit.trajectories."
            f"{trajectory_name}.expected"
        ),
    )

    expected_raw_updated = [
        bool(
            value
        )
        for value
        in require_key(
            expected_cfg,
            "raw_checkpoint_updated",
            (
                "audit.trajectories."
                f"{trajectory_name}.expected"
            ),
        )
    ]

    expected_meaningful = (
        normalize_nullable_bool_list(
            list(
                require_key(
                    expected_cfg,
                    "meaningful_improvement",
                    (
                        "audit.trajectories."
                        f"{trajectory_name}.expected"
                    ),
                )
            )
        )
    )

    expected_patience = [
        int(
            value
        )
        for value
        in require_key(
            expected_cfg,
            "patience_counter",
            (
                "audit.trajectories."
                f"{trajectory_name}.expected"
            ),
        )
    ]

    expected_stop_patience = [
        bool(
            value
        )
        for value
        in require_key(
            expected_cfg,
            "stop_due_to_patience",
            (
                "audit.trajectories."
                f"{trajectory_name}.expected"
            ),
        )
    ]

    expected_raw_best_epoch = [
        int(
            value
        )
        for value
        in require_key(
            expected_cfg,
            "raw_best_epoch",
            (
                "audit.trajectories."
                f"{trajectory_name}.expected"
            ),
        )
    ]

    expected_length = len(
        losses
    )

    for (
        label,
        values,
    ) in (
        (
            "raw_checkpoint_updated",
            expected_raw_updated,
        ),
        (
            "meaningful_improvement",
            expected_meaningful,
        ),
        (
            "patience_counter",
            expected_patience,
        ),
        (
            "stop_due_to_patience",
            expected_stop_patience,
        ),
        (
            "raw_best_epoch",
            expected_raw_best_epoch,
        ),
    ):

        if len(
            values
        ) != expected_length:

            raise RuntimeError(
                f"{trajectory_name}: expected {label} length "
                "does not match losses."
            )

    controller = StageTrainingController(
        experiment_cfg=experiment_cfg,
        stage=stage,
    )

    model = CheckpointProbeModel()

    decisions: list[
        dict[str, Any]
    ] = []

    state_hash_by_epoch: dict[
        int,
        str,
    ] = {}

    observed_stop_epoch: (
        int
        | None
    ) = None

    for (
        index,
        loss,
    ) in enumerate(
        losses,
        start=1,
    ):

        mutate_probe_model(
            model=model,
            epoch=index,
        )

        current_state_hash = (
            model_state_sha256(
                model
            )
        )

        state_hash_by_epoch[
            index
        ] = current_state_hash

        decision = (
            controller.observe_dev_epoch(
                model=model,
                epoch=index,
                weighted_dev_loss=loss,
            )
        )

        position = (
            index
            - 1
        )

        if (
            decision.raw_checkpoint_updated
            != expected_raw_updated[
                position
            ]
        ):

            raise RuntimeError(
                f"{trajectory_name} epoch {index}: "
                "raw-checkpoint-update mismatch."
            )

        if (
            decision.meaningful_improvement
            != expected_meaningful[
                position
            ]
        ):

            raise RuntimeError(
                f"{trajectory_name} epoch {index}: "
                "meaningful-improvement mismatch:\n"
                f"  expected={expected_meaningful[position]!r}\n"
                f"  actual={decision.meaningful_improvement!r}"
            )

        if (
            decision.patience_counter
            != expected_patience[
                position
            ]
        ):

            raise RuntimeError(
                f"{trajectory_name} epoch {index}: "
                "patience-counter mismatch."
            )

        if (
            decision.stop_due_to_patience
            != expected_stop_patience[
                position
            ]
        ):

            raise RuntimeError(
                f"{trajectory_name} epoch {index}: "
                "patience-stop mismatch."
            )

        if (
            decision.raw_best_epoch
            != expected_raw_best_epoch[
                position
            ]
        ):

            raise RuntimeError(
                f"{trajectory_name} epoch {index}: "
                "raw-best-epoch mismatch."
            )

        raw_checkpoint = (
            controller.raw_best_checkpoint
        )

        expected_checkpoint_hash = (
            state_hash_by_epoch[
                decision.raw_best_epoch
            ]
        )

        if (
            raw_checkpoint.model_state_sha256
            != expected_checkpoint_hash
        ):

            raise RuntimeError(
                f"{trajectory_name} epoch {index}: "
                "raw checkpoint does not contain the model "
                "state from raw_best_epoch."
            )

        if decision.should_stop:

            observed_stop_epoch = (
                index
            )

        decisions.append(
            {
                "epoch":
                    decision.epoch,

                "weighted_dev_loss":
                    decision.weighted_dev_loss,

                "raw_checkpoint_updated":
                    decision.raw_checkpoint_updated,

                "raw_best_epoch":
                    decision.raw_best_epoch,

                "raw_best_loss":
                    decision.raw_best_loss,

                "meaningful_improvement":
                    decision.meaningful_improvement,

                "patience_anchor_loss":
                    decision.patience_anchor_loss,

                "patience_counter":
                    decision.patience_counter,

                "stop_due_to_patience":
                    decision.stop_due_to_patience,

                "stop_due_to_maximum_epochs":
                    decision.stop_due_to_maximum_epochs,

                "should_stop":
                    decision.should_stop,

                "raw_checkpoint_sha256":
                    raw_checkpoint.model_state_sha256,
            }
        )

    expected_final_raw_best = int(
        require_key(
            expected_cfg,
            "final_raw_best_epoch",
            (
                "audit.trajectories."
                f"{trajectory_name}.expected"
            ),
        )
    )

    if (
        controller.raw_best_epoch
        != expected_final_raw_best
    ):

        raise RuntimeError(
            f"{trajectory_name}: final raw-best epoch mismatch."
        )

    expected_final_stop = expected_cfg.get(
        "final_stop_epoch"
    )

    if expected_final_stop is not None:

        expected_final_stop = int(
            expected_final_stop
        )

    if (
        observed_stop_epoch
        != expected_final_stop
    ):

        raise RuntimeError(
            f"{trajectory_name}: final stop epoch mismatch:\n"
            f"  expected={expected_final_stop}\n"
            f"  actual={observed_stop_epoch}"
        )

    LOGGER.info(
        "[PASS] trajectory=%s | stage=%s | raw_best_epoch=%d | "
        "stop_epoch=%s",
        trajectory_name,
        stage,
        controller.raw_best_epoch,
        observed_stop_epoch,
    )

    return {
        "stage":
            stage,

        "losses":
            losses,

        "decisions":
            decisions,

        "final_raw_best_epoch":
            controller.raw_best_epoch,

        "final_raw_best_loss":
            controller.raw_best_loss,

        "stop_epoch":
            observed_stop_epoch,
    }


# ======================================================================
# Maximum-epoch audit
# ======================================================================

def audit_maximum_epoch(
    *,
    experiment_cfg: dict[str, Any],
    stage: str,
) -> dict[str, Any]:

    controller = StageTrainingController(
        experiment_cfg=experiment_cfg,
        stage=stage,
    )

    model = CheckpointProbeModel()

    contract = controller.contract

    decisions: list[
        dict[str, Any]
    ] = []

    previous_loss = 1.0

    for epoch in range(
        1,
        contract.maximum_epochs
        + 1,
    ):

        # Epoch 1 establishes 1.0.
        # Each subsequent epoch improves by 1%, comfortably beyond
        # the strict 0.5% threshold, so patience must stay at zero.
        if epoch == 1:

            loss = 1.0

        else:

            loss = (
                previous_loss
                * 0.99
            )

        previous_loss = (
            loss
        )

        mutate_probe_model(
            model=model,
            epoch=epoch,
        )

        decision = (
            controller.observe_dev_epoch(
                model=model,
                epoch=epoch,
                weighted_dev_loss=loss,
            )
        )

        if decision.patience_counter != 0:

            raise RuntimeError(
                f"{stage}: meaningful 1% improvements "
                "unexpectedly consumed patience."
            )

        if epoch < contract.maximum_epochs:

            if decision.should_stop:

                raise RuntimeError(
                    f"{stage}: controller stopped before maximum epoch."
                )

        else:

            if not decision.stop_due_to_maximum_epochs:

                raise RuntimeError(
                    f"{stage}: maximum-epoch stop did not fire."
                )

            if decision.stop_due_to_patience:

                raise RuntimeError(
                    f"{stage}: maximum-epoch trajectory "
                    "unexpectedly stopped due to patience."
                )

            if not decision.should_stop:

                raise RuntimeError(
                    f"{stage}: should_stop false at maximum epoch."
                )

        decisions.append(
            {
                "epoch":
                    epoch,

                "weighted_dev_loss":
                    loss,

                "patience_counter":
                    decision.patience_counter,

                "meaningful_improvement":
                    decision.meaningful_improvement,

                "stop_due_to_maximum_epochs":
                    decision.stop_due_to_maximum_epochs,

                "should_stop":
                    decision.should_stop,
            }
        )

    if (
        controller.raw_best_epoch
        != contract.maximum_epochs
    ):

        raise RuntimeError(
            f"{stage}: raw-best epoch should be final "
            "epoch in monotonically improving trajectory."
        )

    # Once stopped, further observation must fail.
    post_stop_rejected = False

    try:

        controller.observe_dev_epoch(
            model=model,
            epoch=(
                contract.maximum_epochs
                + 1
            ),
            weighted_dev_loss=(
                previous_loss
                * 0.99
            ),
        )

    except RuntimeError:

        post_stop_rejected = (
            True
        )

    if not post_stop_rejected:

        raise RuntimeError(
            f"{stage}: controller accepted an epoch after stopping."
        )

    LOGGER.info(
        "[PASS] %s maximum-epoch contract | maximum=%d",
        stage,
        contract.maximum_epochs,
    )

    return {
        "maximum_epochs":
            contract.maximum_epochs,

        "raw_best_epoch":
            controller.raw_best_epoch,

        "post_stop_observation_rejected":
            post_stop_rejected,

        "decisions":
            decisions,
    }


# ======================================================================
# Checkpoint capture / restore audit
# ======================================================================

def audit_checkpoint_capture_restore() -> dict[str, Any]:

    model = CheckpointProbeModel()

    mutate_probe_model(
        model=model,
        epoch=1,
    )

    expected_hash = model_state_sha256(
        model
    )

    checkpoint = (
        capture_raw_argmin_model_checkpoint(
            model=model,
            stage="stage_a",
            epoch=1,
            weighted_dev_loss=0.75,
        )
    )

    if checkpoint.model_state_sha256 != expected_hash:

        raise RuntimeError(
            "Checkpoint hash differs from source model."
        )

    if checkpoint.epoch != 1:

        raise RuntimeError(
            "Checkpoint epoch metadata mismatch."
        )

    if checkpoint.weighted_dev_loss != 0.75:

        raise RuntimeError(
            "Checkpoint loss metadata mismatch."
        )

    for (
        name,
        value,
    ) in checkpoint.model_state_dict.items():

        if value.device.type != "cpu":

            raise RuntimeError(
                "Checkpoint tensor is not stored on CPU:\n"
                f"  {name}"
            )

        if value.requires_grad:

            raise RuntimeError(
                "Checkpoint tensor unexpectedly requires grad:\n"
                f"  {name}"
            )

    checkpoint_hash_before_mutation = (
        state_dict_sha256(
            checkpoint.model_state_dict
        )
    )

    # Continue mutating the live model.
    mutate_probe_model(
        model=model,
        epoch=2,
    )

    mutated_live_hash = model_state_sha256(
        model
    )

    if mutated_live_hash == expected_hash:

        raise RuntimeError(
            "Probe-model mutation did not change live model state."
        )

    checkpoint_hash_after_mutation = (
        state_dict_sha256(
            checkpoint.model_state_dict
        )
    )

    if (
        checkpoint_hash_after_mutation
        != checkpoint_hash_before_mutation
    ):

        raise RuntimeError(
            "Stored checkpoint changed when live model was mutated."
        )

    # ------------------------------------------------------------------
    # Restore into a model whose mode/requires_grad state is intentionally
    # different. state_dict must restore values only.
    # ------------------------------------------------------------------

    restore_target = (
        CheckpointProbeModel()
    )

    restore_target.eval()

    for parameter in restore_target.parameters():

        parameter.requires_grad_(
            False
        )

    restore_raw_argmin_model_checkpoint(
        model=restore_target,
        checkpoint=checkpoint,
    )

    restored_hash = model_state_sha256(
        restore_target
    )

    if restored_hash != expected_hash:

        raise RuntimeError(
            "Checkpoint restore did not exactly recover "
            "captured model state."
        )

    if restore_target.training:

        raise RuntimeError(
            "state_dict restore unexpectedly changed "
            "model training/eval mode."
        )

    if any(
        parameter.requires_grad
        for parameter
        in restore_target.parameters()
    ):

        raise RuntimeError(
            "state_dict restore unexpectedly changed "
            "requires_grad state."
        )

    # ------------------------------------------------------------------
    # Integrity negative control.
    # ------------------------------------------------------------------

    tamper_model = (
        CheckpointProbeModel()
    )

    tamper_checkpoint = (
        capture_raw_argmin_model_checkpoint(
            model=tamper_model,
            stage="stage_b",
            epoch=2,
            weighted_dev_loss=0.5,
        )
    )

    first_key = next(
        iter(
            tamper_checkpoint
            .model_state_dict
        )
    )

    with torch.no_grad():

        tensor = (
            tamper_checkpoint
            .model_state_dict[
                first_key
            ]
        )

        tensor.view(
            -1
        )[
            0
        ] += 1.0

    tamper_detected = False

    try:

        restore_raw_argmin_model_checkpoint(
            model=tamper_model,
            checkpoint=tamper_checkpoint,
        )

    except RuntimeError:

        tamper_detected = (
            True
        )

    if not tamper_detected:

        raise RuntimeError(
            "Checkpoint integrity tampering was not detected."
        )

    LOGGER.info(
        "[PASS] checkpoint capture is detached CPU state"
    )

    LOGGER.info(
        "[PASS] live-model mutation cannot alter captured checkpoint"
    )

    LOGGER.info(
        "[PASS] checkpoint restore reproduces exact model-state SHA-256"
    )

    LOGGER.info(
        "[PASS] state_dict restore does not alter mode/requires_grad"
    )

    LOGGER.info(
        "[PASS] checkpoint integrity tampering detected"
    )

    return {
        "captured_state_sha256":
            expected_hash,

        "live_state_after_mutation_sha256":
            mutated_live_hash,

        "checkpoint_state_after_live_mutation_sha256":
            checkpoint_hash_after_mutation,

        "restored_state_sha256":
            restored_hash,

        "checkpoint_cpu_only":
            True,

        "checkpoint_detached":
            True,

        "mode_preserved_on_restore":
            True,

        "requires_grad_preserved_on_restore":
            True,

        "tamper_detected":
            tamper_detected,
    }


# ======================================================================
# Controller guard audit
# ======================================================================

def audit_controller_guards(
    *,
    experiment_cfg: dict[str, Any],
) -> dict[str, Any]:

    # --------------------------------------------------------------
    # Raw-best properties before any observation.
    # --------------------------------------------------------------

    controller = StageTrainingController(
        experiment_cfg=experiment_cfg,
        stage="stage_a",
    )

    raw_best_before_observation_rejected = (
        False
    )

    try:

        _ = controller.raw_best_checkpoint

    except RuntimeError:

        raw_best_before_observation_rejected = (
            True
        )

    if not raw_best_before_observation_rejected:

        raise RuntimeError(
            "raw_best_checkpoint was available before "
            "any dev epoch observation."
        )

    # --------------------------------------------------------------
    # Non-consecutive epoch.
    # --------------------------------------------------------------

    model = CheckpointProbeModel()

    mutate_probe_model(
        model=model,
        epoch=1,
    )

    controller.observe_dev_epoch(
        model=model,
        epoch=1,
        weighted_dev_loss=1.0,
    )

    nonconsecutive_epoch_rejected = False

    try:

        controller.observe_dev_epoch(
            model=model,
            epoch=3,
            weighted_dev_loss=0.9,
        )

    except RuntimeError:

        nonconsecutive_epoch_rejected = (
            True
        )

    if not nonconsecutive_epoch_rejected:

        raise RuntimeError(
            "Controller accepted non-consecutive epoch number."
        )

    # --------------------------------------------------------------
    # Invalid checkpoint metadata.
    # --------------------------------------------------------------

    invalid_cases: dict[
        str,
        bool,
    ] = {}

    test_cases = {
        "invalid_stage":
            {
                "stage":
                    "stage_c",

                "epoch":
                    1,

                "loss":
                    0.5,
            },

        "epoch_zero":
            {
                "stage":
                    "stage_a",

                "epoch":
                    0,

                "loss":
                    0.5,
            },

        "negative_loss":
            {
                "stage":
                    "stage_a",

                "epoch":
                    1,

                "loss":
                    -0.1,
            },

        "nan_loss":
            {
                "stage":
                    "stage_a",

                "epoch":
                    1,

                "loss":
                    float(
                        "nan"
                    ),
            },
    }

    for (
        case_name,
        case,
    ) in test_cases.items():

        rejected = False

        try:

            capture_raw_argmin_model_checkpoint(
                model=model,
                stage=str(
                    case[
                        "stage"
                    ]
                ),
                epoch=int(
                    case[
                        "epoch"
                    ]
                ),
                weighted_dev_loss=float(
                    case[
                        "loss"
                    ]
                ),
            )

        except (
            ValueError,
            TypeError,
        ):

            rejected = True

        if not rejected:

            raise RuntimeError(
                "Invalid checkpoint metadata was accepted:\n"
                f"  case={case_name}"
            )

        invalid_cases[
            case_name
        ] = True

    LOGGER.info(
        "[PASS] controller rejects access before first observation"
    )

    LOGGER.info(
        "[PASS] controller rejects non-consecutive epoch observation"
    )

    LOGGER.info(
        "[PASS] invalid checkpoint metadata rejected"
    )

    return {
        "raw_best_before_observation_rejected":
            raw_best_before_observation_rejected,

        "nonconsecutive_epoch_rejected":
            nonconsecutive_epoch_rejected,

        "invalid_checkpoint_cases":
            invalid_cases,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit Tech-2 stopping and raw-argmin "
            "model-checkpoint control."
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
            "audit_resnet18_training_control_config.yaml"
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
            "Training-control audit schema_version must equal 1."
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
        validator_log_path,
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
        output_path,
        partial_output_path,
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

        expected_contracts = require_mapping(
            require_key(
                audit_cfg,
                "expected_contracts",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_contracts",
        )

        trajectories = require_mapping(
            require_key(
                audit_cfg,
                "trajectories",
                "audit_config.audit",
            ),
            "audit_config.audit.trajectories",
        )

        experiment_section = require_mapping(
            require_key(
                experiment_cfg,
                "experiment",
                "experiment_config",
            ),
            "experiment",
        )

        if (
            require_key(
                experiment_section,
                "protocol_status",
                "experiment",
            )
            != "frozen"
        ):

            raise RuntimeError(
                "Training-control audit requires "
                "protocol_status=frozen."
            )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 STOPPING + RAW-CHECKPOINT CONTROL AUDIT"
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
            "Audit script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "src/training_control.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "training_control.py"
            ),
        )

        logger.info(
            "Audit config SHA-256: %s",
            sha256_file(
                audit_config_path
            ),
        )

        logger.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        logger.info(
            "Machine ID: %s",
            machine_id,
        )

        logger.info(
            "Validator log: %s",
            validator_log_path,
        )

        logger.info(
            "Validator log SHA-256: %s",
            validator_log_sha,
        )

        logger.info(
            "Execution device: CPU only"
        )

        logger.info(
            "Dataset access: NONE"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        logger.info(
            "Stage-A -> Stage-B RNG semantics: NOT DEFINED HERE"
        )

        # ==============================================================
        # Frozen contract
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Frozen stopping contracts ---"
        )

        contract_results = audit_contracts(
            experiment_cfg=experiment_cfg,
            expected_cfg=expected_contracts,
        )

        # ==============================================================
        # Synthetic loss trajectories
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Synthetic stopping trajectories ---"
        )

        trajectory_results: dict[
            str,
            Any,
        ] = {}

        for (
            trajectory_name,
            raw_trajectory_cfg,
        ) in trajectories.items():

            trajectory_cfg = require_mapping(
                raw_trajectory_cfg,
                (
                    "audit_config.audit."
                    "trajectories."
                    f"{trajectory_name}"
                ),
            )

            trajectory_results[
                str(
                    trajectory_name
                )
            ] = run_trajectory(
                experiment_cfg=experiment_cfg,
                trajectory_name=str(
                    trajectory_name
                ),
                trajectory_cfg=trajectory_cfg,
            )

        # ==============================================================
        # Maximum epochs
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Maximum-epoch controls ---"
        )

        maximum_epoch_results = {
            stage:
                audit_maximum_epoch(
                    experiment_cfg=experiment_cfg,
                    stage=stage,
                )

            for stage
            in (
                "stage_a",
                "stage_b",
            )
        }

        # ==============================================================
        # Checkpoint capture / restore
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Raw model-checkpoint capture / restore ---"
        )

        checkpoint_results = (
            audit_checkpoint_capture_restore()
        )

        # ==============================================================
        # Guards
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Controller/input guards ---"
        )

        guard_results = audit_controller_guards(
            experiment_cfg=experiment_cfg,
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

                    "experiment_config_sha256":
                        sha256_file(
                            experiment_path
                        ),

                    "training_control_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "training_control.py"
                        ),

                    "validator_log":
                        str(
                            validator_log_path
                        ),

                    "validator_log_sha256":
                        validator_log_sha,
                },

            "contracts":
                contract_results,

            "trajectories":
                trajectory_results,

            "maximum_epochs":
                maximum_epoch_results,

            "checkpoint_capture_restore":
                checkpoint_results,

            "guards":
                guard_results,

            "interpretation":
                (
                    "PASS establishes the frozen distinction between "
                    "raw-argmin checkpoint selection and 0.5%-relative "
                    "patience control, including the strict comparison "
                    "boundary, Stage-A/Stage-B patience and maximum "
                    "epochs, exact detached CPU model-state checkpoint "
                    "capture/restore, and controller input guards. "
                    "This audit does not define Stage-A to Stage-B RNG "
                    "or DataLoader generator continuation semantics."
                ),
        }

        with partial_output_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                result,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        partial_output_path.replace(
            output_path
        )

        output_sha = sha256_file(
            output_path
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
            "[PASS] Stage-A patience=2 / maximum_epochs=10"
        )

        logger.info(
            "[PASS] Stage-B patience=5 / maximum_epochs=30"
        )

        logger.info(
            "[PASS] raw argmin independent from patience threshold"
        )

        logger.info(
            "[PASS] sub-0.5%% raw improvements still update checkpoint"
        )

        logger.info(
            "[PASS] exact 0.5%% improvement does NOT reset patience"
        )

        logger.info(
            "[PASS] >0.5%% meaningful improvement resets patience"
        )

        logger.info(
            "[PASS] worse epoch cannot overwrite raw-best checkpoint"
        )

        logger.info(
            "[PASS] Stage-A patience stop semantics"
        )

        logger.info(
            "[PASS] Stage-B patience stop semantics"
        )

        logger.info(
            "[PASS] Stage-A maximum-epoch stop semantics"
        )

        logger.info(
            "[PASS] Stage-B maximum-epoch stop semantics"
        )

        logger.info(
            "[PASS] raw checkpoint stored as detached CPU model state"
        )

        logger.info(
            "[PASS] checkpoint independent of later live-model mutation"
        )

        logger.info(
            "[PASS] checkpoint restore exact by SHA-256"
        )

        logger.info(
            "[PASS] checkpoint restore preserves mode/requires_grad"
        )

        logger.info(
            "[PASS] checkpoint tampering detected"
        )

        logger.info(
            "[PASS] non-consecutive and post-stop observations rejected"
        )

        logger.info(
            "Audit result: %s",
            output_path,
        )

        logger.info(
            "Audit result SHA-256: %s",
            output_sha,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 STOPPING + RAW-CHECKPOINT CONTROL AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 STOPPING + RAW-CHECKPOINT CONTROL AUDIT: FAIL"
        )

        if partial_output_path.exists():

            partial_output_path.unlink()

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