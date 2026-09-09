#!/usr/bin/env python3
"""
Audit the frozen Tech-2 single-epoch execution engine.

Real FantasyID data are used here.

For each of Stage A and Stage B, independently:

    configure seed
        ->
    construct fresh r256 DataLoaders
        ->
    construct fresh pretrained ResNet-18 + seeded FC
        ->
    construct fresh scientific objective
        ->
    construct fresh stage-specific AdamW
        ->
    run ONE complete project_train epoch
        ->
    run ONE complete dev_val evaluation pass

Each complete stage probe is repeated twice with seed 8.

Validated
---------
1. frozen experiment validator passes;
2. execution begins from clean Git;
3. actual project_train uses all 1,440 images;
4. actual dev_val uses all 459 images;
5. train weighted denominator is exactly 1,440;
6. dev weighted denominator is exactly 459;
7. exact class counts reconcile;
8. train epoch contains exactly 45 optimizer steps;
9. dev epoch contains exactly 15 batches;
10. Stage-A backbone parameters and BN state remain unchanged;
11. Stage-A FC changes;
12. Stage-A Adam state contains only 2 entries;
13. Stage-B backbone parameters change;
14. Stage-B FC changes;
15. Stage-B BatchNorm counters advance exactly 45 times;
16. Stage-B Adam state contains all 62 trainable parameter tensors;
17. train_one_epoch leaves no stale parameter gradients;
18. dev evaluation changes no model parameter or buffer;
19. dev logits are [459,2] CPU float32;
20. dev targets are [459] CPU int64;
21. dev order is exactly frozen manifest order;
22. dev result is finite;
23. repeated same-seed complete Stage-A epoch is deterministic;
24. repeated same-seed complete Stage-B epoch is deterministic.

Important boundary
------------------
Stage A and Stage B are tested as separate fresh same-seed engine probes.

This audit intentionally DOES NOT define Stage-A -> Stage-B checkpoint
or DataLoader-generator-state semantics. Those belong to the future
multi-epoch/checkpoint controller.

No held-out test data are accessed.
No AUROC or FPR10 convention is defined here.
No checkpoint selection is performed.
No early stopping is performed.
No Grad-CAM is performed.

All detailed evidence is written under ./logs/.
No print() is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from src.dataloading import (
    build_development_dataloaders,
)

from src.engine import (
    DevEpochResult,
    EpochLossSummary,
    TrainingEpochResult,
    evaluate_dev_one_epoch,
    train_one_epoch,
)

from src.modeling import (
    apply_stage_a_contract,
    apply_stage_b_contract,
    assert_stage_a_contract,
    assert_stage_b_contract,
    build_resnet18_classifier,
)

from src.objective import (
    WeightedCrossEntropyObjective,
)

from src.optimization import (
    build_stage_a_optimizer,
    build_stage_b_optimizer,
)

from src.reproducibility import (
    configure_run_reproducibility,
)


LOGGER = logging.getLogger(
    "audit_resnet18_epoch_engine"
)


BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


# ======================================================================
# Basic helpers
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


def sha256_bytes(
    value: bytes,
) -> str:

    return hashlib.sha256(
        value
    ).hexdigest()


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
    actual: float,
    expected: float,
    atol: float,
    label: str,
) -> None:

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=atol,
    ):

        raise RuntimeError(
            f"{label} mismatch:\n"
            f"  expected={expected:.17g}\n"
            f"  actual={actual:.17g}\n"
            f"  absolute_error={abs(actual - expected):.17g}\n"
            f"  atol={atol:.17g}"
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
            "the epoch-engine audit.\n\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Frozen experiment validator
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
            "Validator-log SHA mismatch:\n"
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
# Stable tensor/model fingerprints
# ======================================================================

def update_digest_with_tensor(
    digest: Any,
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:

    value = (
        tensor
        .detach()
        .cpu()
        .contiguous()
    )

    digest.update(
        name.encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            value.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        json.dumps(
            list(
                value.shape
            ),
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        value.numpy().tobytes(
            order="C"
        )
    )


def model_state_sha256(
    model: nn.Module,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        value,
    ) in model.state_dict().items():

        update_digest_with_tensor(
            digest,
            name=name,
            tensor=value,
        )

    return digest.hexdigest()


def parameter_sha256(
    model: nn.Module,
    *,
    include_backbone: bool,
    include_classifier: bool,
) -> str:

    digest = hashlib.sha256()

    matched = 0

    for (
        name,
        parameter,
    ) in model.named_parameters():

        is_classifier = (
            name.startswith(
                "fc."
            )
        )

        include = (
            (
                is_classifier
                and include_classifier
            )
            or
            (
                not is_classifier
                and include_backbone
            )
        )

        if not include:

            continue

        matched += 1

        update_digest_with_tensor(
            digest,
            name=name,
            tensor=parameter,
        )

    if matched <= 0:

        raise RuntimeError(
            "Parameter fingerprint matched no parameters."
        )

    return digest.hexdigest()


def tensor_sha256(
    tensor: torch.Tensor,
) -> str:

    digest = hashlib.sha256()

    update_digest_with_tensor(
        digest,
        name="tensor",
        tensor=tensor,
    )

    return digest.hexdigest()


def generator_state_sha256(
    generator: torch.Generator,
) -> str:

    value = (
        generator
        .get_state()
        .contiguous()
        .numpy()
        .tobytes()
    )

    return sha256_bytes(
        value
    )


# ======================================================================
# BatchNorm counters
# ======================================================================

def batchnorm_counters(
    model: nn.Module,
) -> dict[str, int]:

    result: dict[
        str,
        int,
    ] = {}

    for (
        name,
        module,
    ) in model.named_modules():

        if not isinstance(
            module,
            BATCHNORM_TYPES,
        ):

            continue

        if module.num_batches_tracked is None:

            raise RuntimeError(
                "BatchNorm unexpectedly has no num_batches_tracked:\n"
                f"  {name}"
            )

        result[
            name
        ] = int(
            module
            .num_batches_tracked
            .detach()
            .cpu()
            .item()
        )

    if len(
        result
    ) != 20:

        raise RuntimeError(
            "Unexpected BatchNorm module count:\n"
            f"  expected=20\n"
            f"  actual={len(result)}"
        )

    return result


def require_bn_delta(
    *,
    before: dict[str, int],
    after: dict[str, int],
    expected_delta: int,
    label: str,
) -> None:

    if set(
        before
    ) != set(
        after
    ):

        raise RuntimeError(
            f"{label}: BatchNorm module set changed."
        )

    for name in before:

        actual_delta = (
            after[
                name
            ]
            - before[
                name
            ]
        )

        if actual_delta != expected_delta:

            raise RuntimeError(
                f"{label}: BatchNorm counter delta mismatch:\n"
                f"  module={name}\n"
                f"  expected_delta={expected_delta}\n"
                f"  actual_delta={actual_delta}"
            )


# ======================================================================
# Gradient cleanup
# ======================================================================

def require_all_gradients_none(
    model: nn.Module,
) -> None:

    remaining = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.grad is not None
    ]

    if remaining:

        raise RuntimeError(
            "Epoch engine left stale parameter gradients:\n"
            f"  {remaining}"
        )


# ======================================================================
# Adam state checks
# ======================================================================

def validate_optimizer_after_epoch(
    *,
    optimizer: torch.optim.Optimizer,
    expected_state_entries: int,
    expected_steps: int,
) -> dict[str, Any]:

    if len(
        optimizer.state
    ) != expected_state_entries:

        raise RuntimeError(
            "Optimizer state-entry count mismatch:\n"
            f"  expected={expected_state_entries}\n"
            f"  actual={len(optimizer.state)}"
        )

    observed_steps: set[
        float
    ] = set()

    for (
        _,
        state,
    ) in optimizer.state.items():

        if set(
            state
        ) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:

            raise RuntimeError(
                "Unexpected AdamW state keys:\n"
                f"  {sorted(state)}"
            )

        step = state[
            "step"
        ]

        if not isinstance(
            step,
            torch.Tensor,
        ):

            raise RuntimeError(
                "AdamW step is unexpectedly not a tensor."
            )

        step_value = float(
            step.item()
        )

        observed_steps.add(
            step_value
        )

        if step_value != float(
            expected_steps
        ):

            raise RuntimeError(
                "AdamW step counter mismatch:\n"
                f"  expected={expected_steps}\n"
                f"  actual={step_value}"
            )

        for state_name in (
            "exp_avg",
            "exp_avg_sq",
        ):

            value = state[
                state_name
            ]

            if not bool(
                torch.isfinite(
                    value
                ).all()
            ):

                raise RuntimeError(
                    f"AdamW {state_name} contains NaN/Inf."
                )

    return {
        "state_entries":
            len(
                optimizer.state
            ),

        "step_values":
            sorted(
                observed_steps
            ),
    }


def optimizer_state_sha256(
    *,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> str:

    name_by_id = {
        id(
            parameter
        ):
            name

        for (
            name,
            parameter,
        ) in model.named_parameters()
    }

    digest = hashlib.sha256()

    for group in optimizer.param_groups:

        digest.update(
            str(
                group[
                    "name"
                ]
            ).encode(
                "utf-8"
            )
        )

        for key in (
            "lr",
            "weight_decay",
            "betas",
            "eps",
            "amsgrad",
            "foreach",
            "fused",
        ):

            digest.update(
                key.encode(
                    "utf-8"
                )
            )

            digest.update(
                repr(
                    group[
                        key
                    ]
                ).encode(
                    "utf-8"
                )
            )

        names = [
            name_by_id[
                id(
                    parameter
                )
            ]
            for parameter
            in group[
                "params"
            ]
        ]

        digest.update(
            json.dumps(
                names,
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )

    named_state = sorted(
        (
            name_by_id[
                id(
                    parameter
                )
            ],
            state,
        )
        for (
            parameter,
            state,
        ) in optimizer.state.items()
    )

    for (
        parameter_name,
        state,
    ) in named_state:

        digest.update(
            parameter_name.encode(
                "utf-8"
            )
        )

        for state_name in sorted(
            state
        ):

            value = state[
                state_name
            ]

            if isinstance(
                value,
                torch.Tensor,
            ):

                update_digest_with_tensor(
                    digest,
                    name=state_name,
                    tensor=value,
                )

            else:

                digest.update(
                    repr(
                        value
                    ).encode(
                        "utf-8"
                    )
                )

    return digest.hexdigest()


# ======================================================================
# Epoch-result validation
# ======================================================================

def validate_loss_summary(
    *,
    summary: EpochLossSummary,
    expected_cfg: dict[str, Any],
    label: str,
) -> dict[str, Any]:

    expected_rows = int(
        require_key(
            expected_cfg,
            "rows",
            label,
        )
    )

    expected_batches = int(
        require_key(
            expected_cfg,
            "batches",
            label,
        )
    )

    expected_bonafide = int(
        require_key(
            expected_cfg,
            "bonafide",
            label,
        )
    )

    expected_attack = int(
        require_key(
            expected_cfg,
            "attack",
            label,
        )
    )

    expected_denominator = float(
        require_key(
            expected_cfg,
            "weight_denominator",
            label,
        )
    )

    if summary.sample_count != expected_rows:

        raise RuntimeError(
            f"{label}: sample-count mismatch."
        )

    if summary.batch_count != expected_batches:

        raise RuntimeError(
            f"{label}: batch-count mismatch."
        )

    if summary.bonafide_count != expected_bonafide:

        raise RuntimeError(
            f"{label}: bonafide-count mismatch."
        )

    if summary.attack_count != expected_attack:

        raise RuntimeError(
            f"{label}: attack-count mismatch."
        )

    require_close(
        actual=summary.weight_denominator,
        expected=expected_denominator,
        atol=1.0e-12,
        label=(
            f"{label} weight denominator"
        ),
    )

    if not math.isfinite(
        summary.weighted_numerator
    ):

        raise RuntimeError(
            f"{label}: weighted numerator is non-finite."
        )

    if not math.isfinite(
        summary.weighted_loss
    ):

        raise RuntimeError(
            f"{label}: weighted loss is non-finite."
        )

    independently_reduced = (
        summary.weighted_numerator
        / summary.weight_denominator
    )

    require_close(
        actual=summary.weighted_loss,
        expected=independently_reduced,
        atol=1.0e-15,
        label=(
            f"{label} final weighted reduction"
        ),
    )

    return {
        "weighted_numerator":
            summary.weighted_numerator,

        "weight_denominator":
            summary.weight_denominator,

        "weighted_loss":
            summary.weighted_loss,

        "sample_count":
            summary.sample_count,

        "batch_count":
            summary.batch_count,

        "bonafide_count":
            summary.bonafide_count,

        "attack_count":
            summary.attack_count,
    }


# ======================================================================
# Dev-result validation
# ======================================================================

def validate_dev_result(
    *,
    result: DevEpochResult,
    dataset: Any,
    expected_cfg: dict[str, Any],
) -> dict[str, Any]:

    loss_evidence = validate_loss_summary(
        summary=result.loss,
        expected_cfg=expected_cfg,
        label="audit.expected.dev_val",
    )

    expected_rows = int(
        expected_cfg[
            "rows"
        ]
    )

    if tuple(
        result.logits.shape
    ) != (
        expected_rows,
        2,
    ):

        raise RuntimeError(
            "dev logits shape mismatch."
        )

    if result.logits.dtype != torch.float32:

        raise RuntimeError(
            "dev logits are not float32."
        )

    if result.logits.device.type != "cpu":

        raise RuntimeError(
            "dev logits are not stored on CPU."
        )

    if not bool(
        torch.isfinite(
            result.logits
        ).all()
    ):

        raise RuntimeError(
            "dev logits contain NaN/Inf."
        )

    if tuple(
        result.targets.shape
    ) != (
        expected_rows,
    ):

        raise RuntimeError(
            "dev targets shape mismatch."
        )

    if result.targets.dtype != torch.int64:

        raise RuntimeError(
            "dev targets are not int64."
        )

    if result.targets.device.type != "cpu":

        raise RuntimeError(
            "dev targets are not stored on CPU."
        )

    expected_paths = tuple(
        str(
            row[
                "image_path_relative"
            ]
        )
        for row
        in dataset.rows
    )

    if result.image_paths != expected_paths:

        raise RuntimeError(
            "dev result path order differs from frozen manifest."
        )

    expected_targets = torch.tensor(
        [
            int(
                row[
                    "label"
                ]
            )
            for row
            in dataset.rows
        ],
        dtype=torch.int64,
    )

    if not torch.equal(
        result.targets,
        expected_targets,
    ):

        raise RuntimeError(
            "dev targets differ from frozen manifest order."
        )

    paths_hash = sha256_bytes(
        "\n".join(
            result.image_paths
        ).encode(
            "utf-8"
        )
    )

    return {
        "loss":
            loss_evidence,

        "logits_sha256":
            tensor_sha256(
                result.logits
            ),

        "targets_sha256":
            tensor_sha256(
                result.targets
            ),

        "image_paths_sha256":
            paths_hash,
    }


# ======================================================================
# One complete independent stage probe
# ======================================================================

def run_stage_probe(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    device_string: str,
    seed: int,
    resolution_name: str,
    stage: str,
    stage_b_backbone_lr: float,
    expected_train_cfg: dict[str, Any],
    expected_dev_cfg: dict[str, Any],
    expected_state_entries: int,
) -> dict[str, Any]:

    # ------------------------------------------------------------------
    # Establish run RNG state BEFORE loaders/model/CUDA work.
    # ------------------------------------------------------------------

    configure_run_reproducibility(
        experiment_cfg=experiment_cfg,
        run_seed=seed,
    )

    loaders = build_development_dataloaders(
        experiment_cfg=experiment_cfg,
        machine_cfg=machine_cfg,
        repo_root=REPO_ROOT,
        resolution_name=resolution_name,
        run_seed=seed,
    )

    model, provenance = (
        build_resnet18_classifier(
            experiment_cfg=experiment_cfg,
        )
    )

    device = torch.device(
        device_string
    )

    model = model.to(
        device
    )

    objective = WeightedCrossEntropyObjective(
        experiment_cfg=experiment_cfg,
        device=device,
    )

    if stage == "stage_a":

        apply_stage_a_contract(
            model
        )

        assert_stage_a_contract(
            model
        )

        optimizer, _ = (
            build_stage_a_optimizer(
                model=model,
                experiment_cfg=experiment_cfg,
            )
        )

    elif stage == "stage_b":

        apply_stage_b_contract(
            model
        )

        assert_stage_b_contract(
            model
        )

        optimizer, _ = (
            build_stage_b_optimizer(
                model=model,
                experiment_cfg=experiment_cfg,
                backbone_lr=stage_b_backbone_lr,
            )
        )

    else:

        raise ValueError(
            f"Unknown stage: {stage!r}"
        )

    if len(
        optimizer.state
    ) != 0:

        raise RuntimeError(
            f"{stage}: fresh optimizer unexpectedly contains state."
        )

    # ------------------------------------------------------------------
    # Pre-train fingerprints.
    # ------------------------------------------------------------------

    initial_state_hash = (
        model_state_sha256(
            model
        )
    )

    backbone_before = (
        parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=False,
        )
    )

    fc_before = (
        parameter_sha256(
            model,
            include_backbone=False,
            include_classifier=True,
        )
    )

    bn_before = batchnorm_counters(
        model
    )

    train_generator_before = (
        generator_state_sha256(
            loaders.project_train_generator
        )
    )

    # ==================================================================
    # REAL COMPLETE TRAIN EPOCH
    # ==================================================================

    train_result = train_one_epoch(
        model=model,
        loader=loaders.project_train,
        optimizer=optimizer,
        objective=objective,
        device=device,
        stage=stage,
    )

    if not isinstance(
        train_result,
        TrainingEpochResult,
    ):

        raise RuntimeError(
            "train_one_epoch returned unexpected result type."
        )

    train_evidence = validate_loss_summary(
        summary=train_result.loss,
        expected_cfg=expected_train_cfg,
        label="audit.expected.project_train",
    )

    expected_optimizer_steps = int(
        expected_train_cfg[
            "optimizer_steps"
        ]
    )

    if (
        train_result.optimizer_steps
        != expected_optimizer_steps
    ):

        raise RuntimeError(
            f"{stage}: optimizer-step count mismatch:\n"
            f"  expected={expected_optimizer_steps}\n"
            f"  actual={train_result.optimizer_steps}"
        )

    require_all_gradients_none(
        model
    )

    train_generator_after = (
        generator_state_sha256(
            loaders.project_train_generator
        )
    )

    if (
        train_generator_before
        == train_generator_after
    ):

        raise RuntimeError(
            f"{stage}: project_train DataLoader generator "
            "did not advance during the epoch."
        )

    backbone_after_train = (
        parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=False,
        )
    )

    fc_after_train = (
        parameter_sha256(
            model,
            include_backbone=False,
            include_classifier=True,
        )
    )

    bn_after_train = batchnorm_counters(
        model
    )

    optimizer_evidence = (
        validate_optimizer_after_epoch(
            optimizer=optimizer,
            expected_state_entries=expected_state_entries,
            expected_steps=expected_optimizer_steps,
        )
    )

    optimizer_hash = (
        optimizer_state_sha256(
            optimizer=optimizer,
            model=model,
        )
    )

    # ------------------------------------------------------------------
    # Stage-specific scientific behavior after full train epoch.
    # ------------------------------------------------------------------

    if stage == "stage_a":

        assert_stage_a_contract(
            model
        )

        if (
            backbone_after_train
            != backbone_before
        ):

            raise RuntimeError(
                "Stage-A full epoch changed backbone parameters."
            )

        if (
            fc_after_train
            == fc_before
        ):

            raise RuntimeError(
                "Stage-A full epoch did not change FC parameters."
            )

        require_bn_delta(
            before=bn_before,
            after=bn_after_train,
            expected_delta=0,
            label="Stage A training",
        )

    else:

        assert_stage_b_contract(
            model
        )

        if (
            backbone_after_train
            == backbone_before
        ):

            raise RuntimeError(
                "Stage-B full epoch did not change backbone parameters."
            )

        if (
            fc_after_train
            == fc_before
        ):

            raise RuntimeError(
                "Stage-B full epoch did not change FC parameters."
            )

        require_bn_delta(
            before=bn_before,
            after=bn_after_train,
            expected_delta=expected_optimizer_steps,
            label="Stage B training",
        )

    state_before_dev = (
        model_state_sha256(
            model
        )
    )

    bn_before_dev = batchnorm_counters(
        model
    )

    # ==================================================================
    # REAL COMPLETE DEV EVALUATION
    # ==================================================================

    dev_result = evaluate_dev_one_epoch(
        model=model,
        loader=loaders.dev_val,
        objective=objective,
        device=device,
    )

    dev_evidence = validate_dev_result(
        result=dev_result,
        dataset=loaders.dev_val_dataset,
        expected_cfg=expected_dev_cfg,
    )

    state_after_dev = (
        model_state_sha256(
            model
        )
    )

    bn_after_dev = batchnorm_counters(
        model
    )

    if (
        state_after_dev
        != state_before_dev
    ):

        raise RuntimeError(
            f"{stage}: dev evaluation changed model parameter/buffer state."
        )

    require_bn_delta(
        before=bn_before_dev,
        after=bn_after_dev,
        expected_delta=0,
        label=(
            f"{stage} dev evaluation"
        ),
    )

    if model.training:

        raise RuntimeError(
            f"{stage}: dev evaluation did not leave model in eval mode."
        )

    require_all_gradients_none(
        model
    )

    final_state_hash = (
        model_state_sha256(
            model
        )
    )

    result = {
        "stage":
            stage,

        "seed":
            seed,

        "resolution":
            resolution_name,

        "model":
            {
                "pretrained_checkpoint_sha256":
                    provenance.pretrained_checkpoint_sha256,

                "initial_state_sha256":
                    initial_state_hash,

                "backbone_before_sha256":
                    backbone_before,

                "backbone_after_train_sha256":
                    backbone_after_train,

                "fc_before_sha256":
                    fc_before,

                "fc_after_train_sha256":
                    fc_after_train,

                "state_before_dev_sha256":
                    state_before_dev,

                "state_after_dev_sha256":
                    state_after_dev,

                "final_state_sha256":
                    final_state_hash,
            },

        "batchnorm":
            {
                "before_train":
                    bn_before,

                "after_train":
                    bn_after_train,

                "after_dev":
                    bn_after_dev,
            },

        "dataloader":
            {
                "project_train_generator_before_sha256":
                    train_generator_before,

                "project_train_generator_after_sha256":
                    train_generator_after,
            },

        "train":
            {
                **train_evidence,

                "optimizer_steps":
                    train_result.optimizer_steps,
            },

        "dev":
            dev_evidence,

        "optimizer":
            {
                **optimizer_evidence,

                "state_sha256":
                    optimizer_hash,
            },
    }

    del optimizer
    del objective
    del model
    del loaders

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return result


# ======================================================================
# Same-seed repeatability
# ======================================================================

def deterministic_stage_fingerprint(
    result: dict[str, Any],
) -> dict[str, Any]:

    return {
        "initial_model_state_sha256":
            result[
                "model"
            ][
                "initial_state_sha256"
            ],

        "train_weighted_numerator":
            result[
                "train"
            ][
                "weighted_numerator"
            ],

        "train_weighted_loss":
            result[
                "train"
            ][
                "weighted_loss"
            ],

        "backbone_after_train_sha256":
            result[
                "model"
            ][
                "backbone_after_train_sha256"
            ],

        "fc_after_train_sha256":
            result[
                "model"
            ][
                "fc_after_train_sha256"
            ],

        "optimizer_state_sha256":
            result[
                "optimizer"
            ][
                "state_sha256"
            ],

        "dev_weighted_numerator":
            result[
                "dev"
            ][
                "loss"
            ][
                "weighted_numerator"
            ],

        "dev_weighted_loss":
            result[
                "dev"
            ][
                "loss"
            ][
                "weighted_loss"
            ],

        "dev_logits_sha256":
            result[
                "dev"
            ][
                "logits_sha256"
            ],

        "dev_targets_sha256":
            result[
                "dev"
            ][
                "targets_sha256"
            ],

        "dev_paths_sha256":
            result[
                "dev"
            ][
                "image_paths_sha256"
            ],

        "final_model_state_sha256":
            result[
                "model"
            ][
                "final_state_sha256"
            ],

        "train_generator_after_sha256":
            result[
                "dataloader"
            ][
                "project_train_generator_after_sha256"
            ],
    }


def audit_stage_repeats(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    device_string: str,
    seed: int,
    resolution_name: str,
    stage: str,
    repeats: int,
    stage_b_backbone_lr: float,
    expected_train_cfg: dict[str, Any],
    expected_dev_cfg: dict[str, Any],
    expected_state_entries: int,
) -> dict[str, Any]:

    results: list[
        dict[str, Any]
    ] = []

    for repeat_index in range(
        repeats
    ):

        LOGGER.info(
            "%s complete epoch probe | repeat=%d/%d",
            stage,
            repeat_index
            + 1,
            repeats,
        )

        stage_result = run_stage_probe(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            device_string=device_string,
            seed=seed,
            resolution_name=resolution_name,
            stage=stage,
            stage_b_backbone_lr=stage_b_backbone_lr,
            expected_train_cfg=expected_train_cfg,
            expected_dev_cfg=expected_dev_cfg,
            expected_state_entries=expected_state_entries,
        )

        results.append(
            stage_result
        )

        LOGGER.info(
            "[PASS] %s repeat=%d | train_loss=%.12g | dev_loss=%.12g",
            stage,
            repeat_index
            + 1,
            stage_result[
                "train"
            ][
                "weighted_loss"
            ],
            stage_result[
                "dev"
            ][
                "loss"
            ][
                "weighted_loss"
            ],
        )

    reference_fingerprint = (
        deterministic_stage_fingerprint(
            results[
                0
            ]
        )
    )

    for repeat_index, result in enumerate(
        results[
            1:
        ],
        start=2,
    ):

        fingerprint = (
            deterministic_stage_fingerprint(
                result
            )
        )

        if fingerprint != reference_fingerprint:

            raise RuntimeError(
                "Same-seed complete epoch execution is not "
                "bit-reproducible within this machine:\n"
                f"  stage={stage}\n"
                f"  repeat={repeat_index}\n"
                f"  expected={reference_fingerprint}\n"
                f"  actual={fingerprint}"
            )

    LOGGER.info(
        "[PASS] %s complete train+dev epoch is deterministic "
        "across %d same-seed repeats",
        stage,
        repeats,
    )

    return {
        "repeats":
            repeats,

        "deterministic_fingerprint":
            reference_fingerprint,

        "reference_run":
            results[
                0
            ],
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit real FantasyID single-epoch "
            "train/dev execution."
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
            "audit_resnet18_epoch_engine_config.yaml"
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
            "Epoch-engine audit schema_version must equal 1."
        )

    # ------------------------------------------------------------------
    # Capture clean state before validator/audit artifacts are created.
    # ------------------------------------------------------------------

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

    runtime_cfg = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
    )

    device_string = str(
        require_key(
            runtime_cfg,
            "device",
            "machine_config.runtime",
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

        seed = int(
            require_key(
                audit_cfg,
                "seed",
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

        stages = [
            str(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "stages",
                "audit_config.audit",
            )
        ]

        repeats = int(
            require_key(
                audit_cfg,
                "repeats_per_stage",
                "audit_config.audit",
            )
        )

        stage_b_backbone_lr = float(
            require_key(
                audit_cfg,
                "stage_b_backbone_lr",
                "audit_config.audit",
            )
        )

        require_cuda = bool(
            require_key(
                audit_cfg,
                "require_cuda",
                "audit_config.audit",
            )
        )

        expected_cfg = require_mapping(
            require_key(
                audit_cfg,
                "expected",
                "audit_config.audit",
            ),
            "audit_config.audit.expected",
        )

        expected_train_cfg = require_mapping(
            require_key(
                expected_cfg,
                "project_train",
                "audit.expected",
            ),
            "audit.expected.project_train",
        )

        expected_dev_cfg = require_mapping(
            require_key(
                expected_cfg,
                "dev_val",
                "audit.expected",
            ),
            "audit.expected.dev_val",
        )

        expected_state_cfg = require_mapping(
            require_key(
                expected_cfg,
                "optimizer_state_entries",
                "audit.expected",
            ),
            "audit.expected.optimizer_state_entries",
        )

        # --------------------------------------------------------------
        # Strict audit contract.
        # --------------------------------------------------------------

        if seed != 8:

            raise ValueError(
                "Epoch-engine audit must use screening seed 8."
            )

        if resolution_name != "r256":

            raise ValueError(
                "Epoch-engine audit currently uses r256 only."
            )

        if stages != [
            "stage_a",
            "stage_b",
        ]:

            raise ValueError(
                "Audit stages must be exactly "
                "['stage_a', 'stage_b']."
            )

        if repeats != 2:

            raise ValueError(
                "Each stage must be repeated exactly twice."
            )

        if stage_b_backbone_lr != 0.0001:

            raise ValueError(
                "Stage-B epoch probe must use backbone LR 1e-4."
            )

        frozen_train_expected = {
            "rows":
                1440,

            "batches":
                45,

            "bonafide":
                480,

            "attack":
                960,

            "weight_denominator":
                1440.0,

            "optimizer_steps":
                45,
        }

        frozen_dev_expected = {
            "rows":
                459,

            "batches":
                15,

            "bonafide":
                153,

            "attack":
                306,

            "weight_denominator":
                459.0,
        }

        normalized_train = {
            str(
                key
            ):
                (
                    float(
                        value
                    )
                    if key == "weight_denominator"
                    else int(
                        value
                    )
                )

            for (
                key,
                value,
            ) in expected_train_cfg.items()
        }

        normalized_dev = {
            str(
                key
            ):
                (
                    float(
                        value
                    )
                    if key == "weight_denominator"
                    else int(
                        value
                    )
                )

            for (
                key,
                value,
            ) in expected_dev_cfg.items()
        }

        if normalized_train != frozen_train_expected:

            raise ValueError(
                "Configured project_train audit expectations "
                "do not match frozen contract."
            )

        if normalized_dev != frozen_dev_expected:

            raise ValueError(
                "Configured dev_val audit expectations "
                "do not match frozen contract."
            )

        if int(
            expected_state_cfg[
                "stage_a"
            ]
        ) != 2:

            raise ValueError(
                "Expected Stage-A Adam state entries must be 2."
            )

        if int(
            expected_state_cfg[
                "stage_b"
            ]
        ) != 62:

            raise ValueError(
                "Expected Stage-B Adam state entries must be 62."
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
                "Epoch-engine audit requires protocol_status=frozen."
            )

        if require_cuda:

            if not device_string.startswith(
                "cuda"
            ):

                raise RuntimeError(
                    "CUDA required but machine runtime device "
                    f"is {device_string!r}."
                )

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "CUDA required but torch.cuda.is_available() "
                    "is False."
                )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 REAL FANTASYID EPOCH-ENGINE AUDIT"
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

        for source_path in (
            "src/engine.py",
            "src/dataloading.py",
            "src/data.py",
            "src/modeling.py",
            "src/objective.py",
            "src/optimization.py",
            "src/reproducibility.py",
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
            "Device: %s",
            device_string,
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
            "Dataset access: project_train + dev_val only"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        logger.info(
            "Resolution: %s",
            resolution_name,
        )

        logger.info(
            "Run seed: %d",
            seed,
        )

        logger.info(
            "Stage probes are independent fresh same-seed runs; "
            "no Stage-A -> Stage-B RNG/checkpoint semantics are "
            "defined by this audit."
        )

        # ==============================================================
        # Stage A
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Full Stage-A project_train + dev_val probe ---"
        )

        stage_a_results = audit_stage_repeats(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            device_string=device_string,
            seed=seed,
            resolution_name=resolution_name,
            stage="stage_a",
            repeats=repeats,
            stage_b_backbone_lr=stage_b_backbone_lr,
            expected_train_cfg=expected_train_cfg,
            expected_dev_cfg=expected_dev_cfg,
            expected_state_entries=int(
                expected_state_cfg[
                    "stage_a"
                ]
            ),
        )

        # ==============================================================
        # Stage B
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Full Stage-B project_train + dev_val probe ---"
        )

        stage_b_results = audit_stage_repeats(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            device_string=device_string,
            seed=seed,
            resolution_name=resolution_name,
            stage="stage_b",
            repeats=repeats,
            stage_b_backbone_lr=stage_b_backbone_lr,
            expected_train_cfg=expected_train_cfg,
            expected_dev_cfg=expected_dev_cfg,
            expected_state_entries=int(
                expected_state_cfg[
                    "stage_b"
                ]
            ),
        )

        # ==============================================================
        # Same fresh initial model identity between the two independent
        # stage probes.
        # ==============================================================

        stage_a_initial = (
            stage_a_results[
                "deterministic_fingerprint"
            ][
                "initial_model_state_sha256"
            ]
        )

        stage_b_initial = (
            stage_b_results[
                "deterministic_fingerprint"
            ][
                "initial_model_state_sha256"
            ]
        )

        if stage_a_initial != stage_b_initial:

            raise RuntimeError(
                "Independent Stage-A and Stage-B probes did not "
                "start from the same fresh seed-8 model identity."
            )

        logger.info(
            "[PASS] independent Stage-A and Stage-B probes begin "
            "from identical fresh seed-8 model state"
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

                    "device":
                        device_string,

                    "num_workers":
                        int(
                            runtime_cfg[
                                "num_workers"
                            ]
                        ),
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

                    "engine_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "engine.py"
                        ),

                    "validator_log":
                        str(
                            validator_log_path
                        ),

                    "validator_log_sha256":
                        validator_log_sha,
                },

            "scope":
                {
                    "seed":
                        seed,

                    "resolution":
                        resolution_name,

                    "stages_independent":
                        True,

                    "held_out_test_accessed":
                        False,
                },

            "stage_a":
                stage_a_results,

            "stage_b":
                stage_b_results,

            "interpretation":
                (
                    "PASS establishes that the real FantasyID "
                    "single-epoch engine consumes complete "
                    "project_train/dev_val partitions, computes the "
                    "frozen weighted-loss reductions, performs exactly "
                    "one AdamW step per project_train batch, preserves "
                    "the Stage-A backbone/BatchNorm freeze, adapts the "
                    "Stage-B backbone/BatchNorm state, preserves model "
                    "state during dev evaluation, returns frozen-order "
                    "raw dev logits, and is deterministic for repeated "
                    "same-seed execution on this machine. Stage-A to "
                    "Stage-B checkpoint/RNG-state semantics are "
                    "deliberately outside this audit."
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
            "[PASS] Stage-A complete project_train epoch: "
            "1440 samples / 45 optimizer steps"
        )

        logger.info(
            "[PASS] Stage-A complete dev_val pass: "
            "459 samples / 15 batches"
        )

        logger.info(
            "[PASS] Stage-A weighted denominators: train=1440 / dev=459"
        )

        logger.info(
            "[PASS] Stage-A backbone parameters and BN counters frozen"
        )

        logger.info(
            "[PASS] Stage-A FC updated"
        )

        logger.info(
            "[PASS] Stage-A Adam state has 2 parameter entries at step 45"
        )

        logger.info(
            "[PASS] Stage-B complete project_train epoch: "
            "1440 samples / 45 optimizer steps"
        )

        logger.info(
            "[PASS] Stage-B complete dev_val pass: "
            "459 samples / 15 batches"
        )

        logger.info(
            "[PASS] Stage-B weighted denominators: train=1440 / dev=459"
        )

        logger.info(
            "[PASS] Stage-B backbone and FC updated"
        )

        logger.info(
            "[PASS] Stage-B all 20 BN counters advance by 45"
        )

        logger.info(
            "[PASS] Stage-B Adam state has 62 parameter entries at step 45"
        )

        logger.info(
            "[PASS] dev evaluation changes no model parameter/buffer state"
        )

        logger.info(
            "[PASS] dev logits/targets/path order follow frozen manifest"
        )

        logger.info(
            "[PASS] engine clears gradients after each train epoch"
        )

        logger.info(
            "[PASS] repeated same-seed Stage-A full epoch deterministic"
        )

        logger.info(
            "[PASS] repeated same-seed Stage-B full epoch deterministic"
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
            "RESNET-18 REAL FANTASYID EPOCH-ENGINE AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 REAL FANTASYID EPOCH-ENGINE AUDIT: FAIL"
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