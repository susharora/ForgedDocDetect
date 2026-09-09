#!/usr/bin/env python3
"""
Audit frozen Tech-2 Stage-B branch initialization.

Scope
-----
Construct one controlled Stage-A raw-argmin model checkpoint and then
construct all three frozen Stage-B backbone-LR branches:

    3e-5
    1e-4
    3e-4

This audit verifies that the ONLY intended branch difference at
initialization is the Stage-B backbone learning rate.

Validated
---------
1. clean Git before evidence generation;
2. canonical schema-v3 experiment validator passes;
3. exact experiment-config SHA-256;
4. controlled Stage-A checkpoint differs from the fresh seed-8 model;
5. all Stage-B branches restore the exact same Stage-A model SHA-256;
6. all Stage-B branches independently rebuild the same fresh seed-8
   model before checkpoint restore;
7. no Stage-A optimizer state is carried;
8. all Stage-B AdamW optimizers are fresh and empty;
9. all optimizer parameter-group membership is identical;
10. FC LR remains 1e-3 for every branch;
11. only backbone-group LR changes between candidate branches;
12. train/dev DataLoader generators start from fresh run-seed state;
13. Stage-A-like advanced generator state is not inherited;
14. complete first-epoch project_train sampler order is identical across
    all three LR branches;
15. sampler order equals an independent fresh run-seed reference;
16. sampler inspection restores generator state afterward;
17. initial global Python / NumPy / torch CPU / CUDA RNG fingerprints are
    identical across LR branches;
18. Stage-B full-backbone model contract is active;
19. no parameter gradients exist at initialization;
20. invalid backbone LR is rejected.

The project_train sampler is inspected directly without DataLoader
iteration. Therefore no image is decoded by this audit.

No forward pass.
No backward pass.
No optimizer step.
No scientific training.
No dev evaluation.
No held-out test access.
No AUROC.
No LR selection.
No Grad-CAM.

All detailed evidence is written under ./logs/.
No print() is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import pickle
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
)
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

from src.modeling import (
    apply_stage_a_contract,
    assert_stage_a_contract,
    assert_stage_b_contract,
    build_resnet18_classifier,
)

from src.reproducibility import (
    configure_run_reproducibility,
    make_dataloader_generator,
)

from src.stage_b_branching import (
    StageBBranch,
    build_stage_b_branch,
    dataloader_generator_state_sha256,
    load_stage_b_branch_contract,
)

from src.training_control import (
    RawArgminModelCheckpoint,
    capture_raw_argmin_model_checkpoint,
    model_state_sha256,
)


LOGGER = logging.getLogger(
    "audit_resnet18_stage_b_branching"
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


def sha256_bytes(
    value: bytes,
) -> str:

    return hashlib.sha256(
        value
    ).hexdigest()


def sha256_pickle(
    value: Any,
) -> str:

    return sha256_bytes(
        pickle.dumps(
            value,
            protocol=5,
        )
    )


def tensor_sha256(
    tensor: torch.Tensor,
) -> str:

    value = (
        tensor
        .detach()
        .cpu()
        .contiguous()
    )

    digest = hashlib.sha256()

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
            "the Stage-B branching audit.\n\n"
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

    partial_result_path = Path(
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
        partial_result_path,
    )


# ======================================================================
# Global RNG fingerprints
# ======================================================================

def global_rng_fingerprint() -> dict[str, Any]:

    result: dict[
        str,
        Any,
    ] = {
        "python":
            sha256_pickle(
                random.getstate()
            ),

        "numpy":
            sha256_pickle(
                np.random.get_state()
            ),

        "torch_cpu":
            tensor_sha256(
                torch.get_rng_state()
            ),
    }

    if torch.cuda.is_available():

        result[
            "torch_cuda_all"
        ] = [
            tensor_sha256(
                state
            )
            for state
            in torch.cuda.get_rng_state_all()
        ]

    else:

        result[
            "torch_cuda_all"
        ] = []

    return result


# ======================================================================
# Controlled Stage-A checkpoint
# ======================================================================

def build_controlled_stage_a_checkpoint(
    *,
    experiment_cfg: dict[str, Any],
    run_seed: int,
    epoch: int,
    weighted_dev_loss: float,
    classifier_weight_delta: float,
    classifier_bias_delta: float,
) -> tuple[
    RawArgminModelCheckpoint,
    dict[str, Any],
]:

    configure_run_reproducibility(
        experiment_cfg=experiment_cfg,
        run_seed=run_seed,
    )

    model, provenance = (
        build_resnet18_classifier(
            experiment_cfg=experiment_cfg,
        )
    )

    apply_stage_a_contract(
        model
    )

    assert_stage_a_contract(
        model
    )

    fresh_hash = model_state_sha256(
        model
    )

    if not isinstance(
        model.fc,
        nn.Linear,
    ):

        raise RuntimeError(
            "Controlled Stage-A checkpoint requires nn.Linear FC."
        )

    with torch.no_grad():

        model.fc.weight.add_(
            float(
                classifier_weight_delta
            )
        )

        model.fc.bias.add_(
            float(
                classifier_bias_delta
            )
        )

    mutated_hash = model_state_sha256(
        model
    )

    if mutated_hash == fresh_hash:

        raise RuntimeError(
            "Controlled Stage-A FC mutation did not change "
            "the model-state fingerprint."
        )

    checkpoint = (
        capture_raw_argmin_model_checkpoint(
            model=model,
            stage="stage_a",
            epoch=epoch,
            weighted_dev_loss=weighted_dev_loss,
        )
    )

    if (
        checkpoint.model_state_sha256
        != mutated_hash
    ):

        raise RuntimeError(
            "Controlled Stage-A checkpoint does not match "
            "mutated source model."
        )

    for (
        name,
        value,
    ) in checkpoint.model_state_dict.items():

        if value.device.type != "cpu":

            raise RuntimeError(
                "Controlled Stage-A checkpoint is not CPU-resident:\n"
                f"  {name}"
            )

    evidence = {
        "run_seed":
            run_seed,

        "epoch":
            epoch,

        "weighted_dev_loss":
            weighted_dev_loss,

        "classifier_weight_delta":
            classifier_weight_delta,

        "classifier_bias_delta":
            classifier_bias_delta,

        "fresh_seeded_model_state_sha256":
            fresh_hash,

        "controlled_checkpoint_model_state_sha256":
            checkpoint.model_state_sha256,

        "pretrained_checkpoint_sha256":
            provenance.pretrained_checkpoint_sha256,
    }

    del model

    gc.collect()

    return (
        checkpoint,
        evidence,
    )


# ======================================================================
# Independent run-seed generator reference
# ======================================================================

def build_generator_reference(
    *,
    run_seed: int,
    rows: int,
) -> dict[str, Any]:

    generator = make_dataloader_generator(
        run_seed
    )

    initial_hash = (
        dataloader_generator_state_sha256(
            generator
        )
    )

    sampler = RandomSampler(
        range(
            rows
        ),
        replacement=False,
        generator=generator,
    )

    order = [
        int(
            value
        )
        for value
        in sampler
    ]

    advanced_hash = (
        dataloader_generator_state_sha256(
            generator
        )
    )

    if len(
        order
    ) != rows:

        raise RuntimeError(
            "Independent generator reference did not yield "
            "exactly one complete epoch."
        )

    if len(
        set(
            order
        )
    ) != rows:

        raise RuntimeError(
            "Independent generator reference contains duplicate indices."
        )

    if set(
        order
    ) != set(
        range(
            rows
        )
    ):

        raise RuntimeError(
            "Independent generator reference is not a full permutation."
        )

    order_hash = sha256_bytes(
        json.dumps(
            order,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )
    )

    return {
        "initial_generator_state_sha256":
            initial_hash,

        "advanced_one_epoch_generator_state_sha256":
            advanced_hash,

        "order_indices":
            order,

        "order_indices_sha256":
            order_hash,
    }


# ======================================================================
# Branch sampler inspection without image decoding
# ======================================================================

def inspect_project_train_sampler(
    *,
    branch: StageBBranch,
    expected_rows: int,
) -> dict[str, Any]:

    loader = branch.dataloaders.project_train
    dataset = branch.dataloaders.project_train_dataset
    generator = (
        branch
        .dataloaders
        .project_train_generator
    )

    if len(
        dataset
    ) != expected_rows:

        raise RuntimeError(
            "project_train Dataset row-count mismatch."
        )

    sampler = loader.sampler

    if not isinstance(
        sampler,
        RandomSampler,
    ):

        raise RuntimeError(
            "project_train sampler is not RandomSampler."
        )

    if sampler.generator is not generator:

        raise RuntimeError(
            "project_train sampler is not using the exposed "
            "project_train generator object."
        )

    initial_hash = (
        dataloader_generator_state_sha256(
            generator
        )
    )

    initial_state = (
        generator
        .get_state()
        .clone()
    )

    order = [
        int(
            value
        )
        for value
        in sampler
    ]

    advanced_hash = (
        dataloader_generator_state_sha256(
            generator
        )
    )

    if len(
        order
    ) != expected_rows:

        raise RuntimeError(
            "Stage-B project_train sampler did not produce "
            "exactly one complete epoch."
        )

    if len(
        set(
            order
        )
    ) != expected_rows:

        raise RuntimeError(
            "Stage-B project_train sampler produced duplicates."
        )

    if set(
        order
    ) != set(
        range(
            expected_rows
        )
    ):

        raise RuntimeError(
            "Stage-B project_train sampler is not a complete "
            "Dataset permutation."
        )

    # Restore because this audit must not alter the branch's scientific
    # initial state merely by inspecting the intended first epoch.
    generator.set_state(
        initial_state
    )

    restored_hash = (
        dataloader_generator_state_sha256(
            generator
        )
    )

    if restored_hash != initial_hash:

        raise RuntimeError(
            "Audit failed to restore project_train generator state."
        )

    order_hash = sha256_bytes(
        json.dumps(
            order,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )
    )

    paths = [
        str(
            dataset.rows[
                index
            ][
                "image_path_relative"
            ]
        )
        for index
        in order
    ]

    path_hash = sha256_bytes(
        "\n".join(
            paths
        ).encode(
            "utf-8"
        )
    )

    return {
        "initial_generator_state_sha256":
            initial_hash,

        "advanced_one_epoch_generator_state_sha256":
            advanced_hash,

        "restored_generator_state_sha256":
            restored_hash,

        "order_indices":
            order,

        "order_indices_sha256":
            order_hash,

        "order_paths_sha256":
            path_hash,

        "first_10_paths":
            paths[
                :10
            ],

        "last_10_paths":
            paths[
                -10:
            ],
    }


# ======================================================================
# Dev sampler structural check
# ======================================================================

def inspect_dev_loader(
    *,
    branch: StageBBranch,
    expected_rows: int,
    expected_generator_hash: str,
) -> dict[str, Any]:

    loader = branch.dataloaders.dev_val
    dataset = branch.dataloaders.dev_val_dataset

    if len(
        dataset
    ) != expected_rows:

        raise RuntimeError(
            "dev_val Dataset row-count mismatch."
        )

    if not isinstance(
        loader.sampler,
        SequentialSampler,
    ):

        raise RuntimeError(
            "dev_val sampler must be SequentialSampler."
        )

    generator_hash = (
        dataloader_generator_state_sha256(
            branch
            .dataloaders
            .dev_val_generator
        )
    )

    if (
        generator_hash
        != expected_generator_hash
    ):

        raise RuntimeError(
            "dev_val generator was not freshly seeded from run_seed."
        )

    path_hash = sha256_bytes(
        "\n".join(
            str(
                row[
                    "image_path_relative"
                ]
            )
            for row
            in dataset.rows
        ).encode(
            "utf-8"
        )
    )

    return {
        "generator_initial_state_sha256":
            generator_hash,

        "manifest_order_paths_sha256":
            path_hash,
    }


# ======================================================================
# Optimizer inspection
# ======================================================================

def inspect_optimizer(
    *,
    branch: StageBBranch,
    expected_groups_cfg: dict[str, Any],
) -> dict[str, Any]:

    optimizer = branch.optimizer
    model = branch.model

    if not isinstance(
        optimizer,
        torch.optim.AdamW,
    ):

        raise RuntimeError(
            "Stage-B branch optimizer is not AdamW."
        )

    if len(
        optimizer.state
    ) != 0:

        raise RuntimeError(
            "Fresh Stage-B optimizer contains state."
        )

    if (
        branch
        .initialization_evidence
        .optimizer_state_entries_at_construction
        != 0
    ):

        raise RuntimeError(
            "Branch initialization evidence reports non-empty "
            "optimizer state."
        )

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

    result: dict[
        str,
        Any,
    ] = {}

    actual_group_order: list[
        str
    ] = []

    for group in optimizer.param_groups:

        group_name = str(
            group[
                "name"
            ]
        )

        actual_group_order.append(
            group_name
        )

        parameter_names = [
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

        parameter_elements = sum(
            parameter.numel()
            for parameter
            in group[
                "params"
            ]
        )

        result[
            group_name
        ] = {
            "parameter_names":
                parameter_names,

            "parameter_tensors":
                len(
                    group[
                        "params"
                    ]
                ),

            "parameter_elements":
                parameter_elements,

            "lr":
                float(
                    group[
                        "lr"
                    ]
                ),

            "weight_decay":
                float(
                    group[
                        "weight_decay"
                    ]
                ),

            "betas":
                [
                    float(
                        group[
                            "betas"
                        ][
                            0
                        ]
                    ),
                    float(
                        group[
                            "betas"
                        ][
                            1
                        ]
                    ),
                ],

            "eps":
                float(
                    group[
                        "eps"
                    ]
                ),

            "amsgrad":
                bool(
                    group[
                        "amsgrad"
                    ]
                ),

            "foreach":
                group[
                    "foreach"
                ],

            "fused":
                group[
                    "fused"
                ],

            "maximize":
                bool(
                    group[
                        "maximize"
                    ]
                ),

            "capturable":
                bool(
                    group[
                        "capturable"
                    ]
                ),

            "differentiable":
                bool(
                    group[
                        "differentiable"
                    ]
                ),
        }

    expected_group_order = [
        "backbone_decay",
        "backbone_no_decay",
        "fc_decay",
        "fc_no_decay",
    ]

    if actual_group_order != expected_group_order:

        raise RuntimeError(
            "Stage-B optimizer group order mismatch:\n"
            f"  expected={expected_group_order}\n"
            f"  actual={actual_group_order}"
        )

    for group_name in expected_group_order:

        expected = require_mapping(
            require_key(
                expected_groups_cfg,
                group_name,
                "audit.expected.optimizer_groups",
            ),
            (
                "audit.expected.optimizer_groups."
                f"{group_name}"
            ),
        )

        actual = result[
            group_name
        ]

        expected_tensors = int(
            require_key(
                expected,
                "parameter_tensors",
                (
                    "audit.expected.optimizer_groups."
                    f"{group_name}"
                ),
            )
        )

        expected_elements = int(
            require_key(
                expected,
                "parameter_elements",
                (
                    "audit.expected.optimizer_groups."
                    f"{group_name}"
                ),
            )
        )

        expected_decay = float(
            require_key(
                expected,
                "weight_decay",
                (
                    "audit.expected.optimizer_groups."
                    f"{group_name}"
                ),
            )
        )

        if (
            actual[
                "parameter_tensors"
            ]
            != expected_tensors
        ):

            raise RuntimeError(
                f"{group_name}: parameter tensor-count mismatch."
            )

        if (
            actual[
                "parameter_elements"
            ]
            != expected_elements
        ):

            raise RuntimeError(
                f"{group_name}: parameter element-count mismatch."
            )

        if (
            actual[
                "weight_decay"
            ]
            != expected_decay
        ):

            raise RuntimeError(
                f"{group_name}: weight-decay mismatch."
            )

        if group_name.startswith(
            "fc_"
        ):

            expected_lr = float(
                require_key(
                    expected,
                    "lr",
                    (
                        "audit.expected.optimizer_groups."
                        f"{group_name}"
                    ),
                )
            )

            if (
                actual[
                    "lr"
                ]
                != expected_lr
            ):

                raise RuntimeError(
                    f"{group_name}: classifier LR mismatch."
                )

        else:

            if (
                actual[
                    "lr"
                ]
                != branch.backbone_lr
            ):

                raise RuntimeError(
                    f"{group_name}: backbone LR does not match "
                    "branch LR."
                )

        if (
            actual[
                "betas"
            ]
            != [
                0.9,
                0.999,
            ]
        ):

            raise RuntimeError(
                f"{group_name}: AdamW beta mismatch."
            )

        if (
            actual[
                "eps"
            ]
            != 1.0e-8
        ):

            raise RuntimeError(
                f"{group_name}: AdamW eps mismatch."
            )

        if actual[
            "amsgrad"
        ] is not False:

            raise RuntimeError(
                f"{group_name}: amsgrad must be false."
            )

        if actual[
            "foreach"
        ] is not False:

            raise RuntimeError(
                f"{group_name}: foreach must be false."
            )

        if actual[
            "fused"
        ] is not False:

            raise RuntimeError(
                f"{group_name}: fused must be false."
            )

        if actual[
            "maximize"
        ] is not False:

            raise RuntimeError(
                f"{group_name}: maximize must be false."
            )

        if actual[
            "capturable"
        ] is not False:

            raise RuntimeError(
                f"{group_name}: capturable must be false."
            )

        if actual[
            "differentiable"
        ] is not False:

            raise RuntimeError(
                f"{group_name}: differentiable must be false."
            )

    return {
        "state_entries":
            len(
                optimizer.state
            ),

        "group_order":
            actual_group_order,

        "groups":
            result,
    }


def optimizer_structure_without_learning_rates(
    optimizer_evidence: dict[str, Any],
) -> dict[str, Any]:

    result = {
        "state_entries":
            optimizer_evidence[
                "state_entries"
            ],

        "group_order":
            list(
                optimizer_evidence[
                    "group_order"
                ]
            ),

        "groups":
            {},
    }

    for (
        group_name,
        group,
    ) in optimizer_evidence[
        "groups"
    ].items():

        result[
            "groups"
        ][
            group_name
        ] = {
            key:
                value

            for (
                key,
                value,
            ) in group.items()

            if key != "lr"
        }

    return result


# ======================================================================
# One candidate branch
# ======================================================================

def audit_one_branch(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    resolution_name: str,
    run_seed: int,
    backbone_lr: float,
    checkpoint: RawArgminModelCheckpoint,
    controlled_checkpoint_evidence: dict[str, Any],
    expected_train_rows: int,
    expected_dev_rows: int,
    expected_groups_cfg: dict[str, Any],
    reference_generator: dict[str, Any],
) -> dict[str, Any]:

    branch = build_stage_b_branch(
        experiment_cfg=experiment_cfg,
        machine_cfg=machine_cfg,
        repo_root=REPO_ROOT,
        resolution_name=resolution_name,
        run_seed=run_seed,
        stage_a_checkpoint=checkpoint,
        backbone_lr=backbone_lr,
    )

    assert_stage_b_contract(
        branch.model
    )

    if branch.run_seed != run_seed:

        raise RuntimeError(
            "Stage-B branch run_seed mismatch."
        )

    if (
        branch.resolution_name
        != resolution_name
    ):

        raise RuntimeError(
            "Stage-B branch resolution mismatch."
        )

    if branch.backbone_lr != backbone_lr:

        raise RuntimeError(
            "Stage-B branch LR mismatch."
        )

    initialization = (
        branch.initialization_evidence
    )

    if (
        initialization
        .stage_a_checkpoint_model_state_sha256
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Stage-B initialization evidence has wrong "
            "Stage-A checkpoint SHA."
        )

    if (
        initialization
        .restored_model_state_sha256_cpu
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "CPU checkpoint restore does not equal Stage-A checkpoint."
        )

    if (
        initialization
        .restored_model_state_sha256_after_device_move
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Device-resident Stage-B model differs from "
            "Stage-A checkpoint."
        )

    current_model_hash = (
        model_state_sha256(
            branch.model
        )
    )

    if (
        current_model_hash
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Live Stage-B model differs from Stage-A checkpoint "
            "immediately after initialization."
        )

    expected_fresh_hash = (
        controlled_checkpoint_evidence[
            "fresh_seeded_model_state_sha256"
        ]
    )

    if (
        initialization
        .fresh_model_state_sha256_before_restore
        != expected_fresh_hash
    ):

        raise RuntimeError(
            "Fresh branch model before restore does not match "
            "independent same-seed model construction."
        )

    if (
        initialization
        .fresh_model_state_sha256_before_restore
        == checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Controlled Stage-A checkpoint is not distinguishable "
            "from fresh model; restore audit would be ineffective."
        )

    reference_initial_generator_hash = (
        reference_generator[
            "initial_generator_state_sha256"
        ]
    )

    if (
        initialization
        .project_train_generator_initial_state_sha256
        != reference_initial_generator_hash
    ):

        raise RuntimeError(
            "Stage-B project_train generator did not begin "
            "from fresh run-seed state."
        )

    if (
        initialization
        .dev_val_generator_initial_state_sha256
        != reference_initial_generator_hash
    ):

        raise RuntimeError(
            "Stage-B dev_val generator did not begin "
            "from fresh run-seed state."
        )

    # --------------------------------------------------------------
    # Capture global RNG state immediately after branch construction.
    # --------------------------------------------------------------

    rng_before_sampler_inspection = (
        global_rng_fingerprint()
    )

    train_sampler = (
        inspect_project_train_sampler(
            branch=branch,
            expected_rows=expected_train_rows,
        )
    )

    rng_after_sampler_inspection = (
        global_rng_fingerprint()
    )

    if (
        rng_after_sampler_inspection
        != rng_before_sampler_inspection
    ):

        raise RuntimeError(
            "Inspecting the dedicated DataLoader sampler "
            "unexpectedly changed global RNG state."
        )

    if (
        train_sampler[
            "order_indices_sha256"
        ]
        != reference_generator[
            "order_indices_sha256"
        ]
    ):

        raise RuntimeError(
            "Stage-B first-epoch sampler order differs from "
            "independent fresh run-seed reference."
        )

    if (
        train_sampler[
            "order_indices"
        ]
        != reference_generator[
            "order_indices"
        ]
    ):

        raise RuntimeError(
            "Stage-B first-epoch sampler indices differ from "
            "independent reference."
        )

    if (
        train_sampler[
            "advanced_one_epoch_generator_state_sha256"
        ]
        != reference_generator[
            "advanced_one_epoch_generator_state_sha256"
        ]
    ):

        raise RuntimeError(
            "Stage-B sampler generator evolution differs from "
            "independent reference."
        )

    if (
        train_sampler[
            "initial_generator_state_sha256"
        ]
        == reference_generator[
            "advanced_one_epoch_generator_state_sha256"
        ]
    ):

        raise RuntimeError(
            "Stage-B appears to have inherited an advanced "
            "Stage-A-like generator state."
        )

    dev_loader = inspect_dev_loader(
        branch=branch,
        expected_rows=expected_dev_rows,
        expected_generator_hash=(
            reference_initial_generator_hash
        ),
    )

    optimizer = inspect_optimizer(
        branch=branch,
        expected_groups_cfg=expected_groups_cfg,
    )

    remaining_gradients = [
        name
        for (
            name,
            parameter,
        ) in branch.model.named_parameters()
        if parameter.grad is not None
    ]

    if remaining_gradients:

        raise RuntimeError(
            "Fresh Stage-B branch contains parameter gradients:\n"
            f"  {remaining_gradients}"
        )

    result = {
        "backbone_lr":
            backbone_lr,

        "initialization":
            {
                "requested_device":
                    initialization.requested_device,

                "actual_model_device":
                    initialization.actual_model_device,

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

        "global_rng_after_initialization":
            rng_before_sampler_inspection,

        "project_train_sampler":
            {
                key:
                    value

                for (
                    key,
                    value,
                ) in train_sampler.items()

                if key != "order_indices"
            },

        "dev_loader":
            dev_loader,

        "optimizer":
            optimizer,

        "live_model_state_sha256":
            current_model_hash,

        "parameter_gradients_present":
            False,
    }

    del branch

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return result


# ======================================================================
# Cross-branch comparison
# ======================================================================

def compare_branches(
    *,
    results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    if len(
        results
    ) != 3:

        raise RuntimeError(
            "Expected exactly three Stage-B branch results."
        )

    # --------------------------------------------------------------
    # Model identity.
    # --------------------------------------------------------------

    checkpoint_hashes = {
        result[
            "initialization"
        ][
            "stage_a_checkpoint_model_state_sha256"
        ]
        for result
        in results
    }

    restored_hashes = {
        result[
            "live_model_state_sha256"
        ]
        for result
        in results
    }

    fresh_model_hashes = {
        result[
            "initialization"
        ][
            "fresh_model_state_sha256_before_restore"
        ]
        for result
        in results
    }

    if len(
        checkpoint_hashes
    ) != 1:

        raise RuntimeError(
            "LR branches do not reference one identical "
            "Stage-A checkpoint."
        )

    if len(
        restored_hashes
    ) != 1:

        raise RuntimeError(
            "LR branches do not begin with identical restored "
            "model state."
        )

    if checkpoint_hashes != restored_hashes:

        raise RuntimeError(
            "Restored branch model identity differs from "
            "Stage-A checkpoint identity."
        )

    if len(
        fresh_model_hashes
    ) != 1:

        raise RuntimeError(
            "Fresh pre-restore model construction differs "
            "between LR branches."
        )

    # --------------------------------------------------------------
    # Generator identity / first epoch ordering.
    # --------------------------------------------------------------

    train_generator_hashes = {
        result[
            "initialization"
        ][
            "project_train_generator_initial_state_sha256"
        ]
        for result
        in results
    }

    dev_generator_hashes = {
        result[
            "initialization"
        ][
            "dev_val_generator_initial_state_sha256"
        ]
        for result
        in results
    }

    train_order_hashes = {
        result[
            "project_train_sampler"
        ][
            "order_indices_sha256"
        ]
        for result
        in results
    }

    train_path_order_hashes = {
        result[
            "project_train_sampler"
        ][
            "order_paths_sha256"
        ]
        for result
        in results
    }

    if len(
        train_generator_hashes
    ) != 1:

        raise RuntimeError(
            "Stage-B branches have different initial train "
            "generator states."
        )

    if len(
        dev_generator_hashes
    ) != 1:

        raise RuntimeError(
            "Stage-B branches have different initial dev "
            "generator states."
        )

    if len(
        train_order_hashes
    ) != 1:

        raise RuntimeError(
            "Stage-B LR branches have different first-epoch "
            "project_train order."
        )

    if len(
        train_path_order_hashes
    ) != 1:

        raise RuntimeError(
            "Stage-B LR branches have different first-epoch "
            "project_train path order."
        )

    # --------------------------------------------------------------
    # Global RNG identity.
    # --------------------------------------------------------------

    reference_rng = results[
        0
    ][
        "global_rng_after_initialization"
    ]

    for result in results[
        1:
    ]:

        if (
            result[
                "global_rng_after_initialization"
            ]
            != reference_rng
        ):

            raise RuntimeError(
                "Stage-B LR branches do not have identical "
                "post-initialization global RNG state."
            )

    # --------------------------------------------------------------
    # Optimizer structure must be identical after ignoring group LR.
    # --------------------------------------------------------------

    reference_structure = (
        optimizer_structure_without_learning_rates(
            results[
                0
            ][
                "optimizer"
            ]
        )
    )

    for result in results[
        1:
    ]:

        structure = (
            optimizer_structure_without_learning_rates(
                result[
                    "optimizer"
                ]
            )
        )

        if structure != reference_structure:

            raise RuntimeError(
                "Stage-B optimizer branches differ in more "
                "than learning rate."
            )

    # FC learning rates must remain identical.
    fc_decay_lrs = {
        result[
            "optimizer"
        ][
            "groups"
        ][
            "fc_decay"
        ][
            "lr"
        ]
        for result
        in results
    }

    fc_no_decay_lrs = {
        result[
            "optimizer"
        ][
            "groups"
        ][
            "fc_no_decay"
        ][
            "lr"
        ]
        for result
        in results
    }

    if fc_decay_lrs != {
        0.001
    }:

        raise RuntimeError(
            "fc_decay LR differs between Stage-B branches."
        )

    if fc_no_decay_lrs != {
        0.001
    }:

        raise RuntimeError(
            "fc_no_decay LR differs between Stage-B branches."
        )

    observed_backbone_lrs = [
        result[
            "optimizer"
        ][
            "groups"
        ][
            "backbone_decay"
        ][
            "lr"
        ]
        for result
        in results
    ]

    for result in results:

        branch_lr = result[
            "backbone_lr"
        ]

        if (
            result[
                "optimizer"
            ][
                "groups"
            ][
                "backbone_no_decay"
            ][
                "lr"
            ]
            != branch_lr
        ):

            raise RuntimeError(
                "backbone_no_decay LR differs from branch LR."
            )

    return {
        "identical_stage_a_checkpoint":
            True,

        "identical_restored_model_state":
            True,

        "identical_fresh_model_before_restore":
            True,

        "identical_train_generator_initial_state":
            True,

        "identical_dev_generator_initial_state":
            True,

        "identical_first_epoch_project_train_order":
            True,

        "identical_global_rng_after_initialization":
            True,

        "optimizer_structure_identical_except_learning_rate":
            True,

        "classifier_lr_identical":
            True,

        "observed_backbone_lrs":
            observed_backbone_lrs,

        "stage_a_checkpoint_sha256":
            next(
                iter(
                    checkpoint_hashes
                )
            ),

        "first_epoch_order_indices_sha256":
            next(
                iter(
                    train_order_hashes
                )
            ),

        "first_epoch_order_paths_sha256":
            next(
                iter(
                    train_path_order_hashes
                )
            ),
    }


# ======================================================================
# Invalid-LR guard
# ======================================================================

def audit_invalid_lr_rejection(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    resolution_name: str,
    run_seed: int,
    checkpoint: RawArgminModelCheckpoint,
    invalid_lr: float,
) -> bool:

    rejected = False

    try:

        _ = build_stage_b_branch(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=REPO_ROOT,
            resolution_name=resolution_name,
            run_seed=run_seed,
            stage_a_checkpoint=checkpoint,
            backbone_lr=invalid_lr,
        )

    except ValueError:

        rejected = True

    if not rejected:

        raise RuntimeError(
            "Stage-B branch initializer accepted an invalid "
            f"backbone LR: {invalid_lr}"
        )

    return True


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen Stage-B LR branch initialization."
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
            "audit_resnet18_stage_b_branching_config.yaml"
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
            "Stage-B branching audit schema_version must equal 1."
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

    runtime_cfg = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
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
                "Experiment-config SHA mismatch:\n"
                f"  expected={expected_experiment_sha}\n"
                f"  actual={actual_experiment_sha}"
            )

        run_seed = int(
            require_key(
                audit_cfg,
                "run_seed",
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

        backbone_lrs = [
            float(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "backbone_lr_candidates",
                "audit_config.audit",
            )
        ]

        invalid_lr = float(
            require_key(
                audit_cfg,
                "invalid_backbone_lr",
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

        checkpoint_cfg = require_mapping(
            require_key(
                audit_cfg,
                "controlled_stage_a_checkpoint",
                "audit_config.audit",
            ),
            "audit_config.audit.controlled_stage_a_checkpoint",
        )

        checkpoint_epoch = int(
            require_key(
                checkpoint_cfg,
                "epoch",
                "audit.controlled_stage_a_checkpoint",
            )
        )

        checkpoint_loss = float(
            require_key(
                checkpoint_cfg,
                "weighted_dev_loss",
                "audit.controlled_stage_a_checkpoint",
            )
        )

        weight_delta = float(
            require_key(
                checkpoint_cfg,
                "classifier_weight_delta",
                "audit.controlled_stage_a_checkpoint",
            )
        )

        bias_delta = float(
            require_key(
                checkpoint_cfg,
                "classifier_bias_delta",
                "audit.controlled_stage_a_checkpoint",
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

        expected_train_rows = int(
            require_key(
                expected_cfg,
                "project_train_rows",
                "audit.expected",
            )
        )

        expected_dev_rows = int(
            require_key(
                expected_cfg,
                "dev_val_rows",
                "audit.expected",
            )
        )

        expected_groups_cfg = require_mapping(
            require_key(
                expected_cfg,
                "optimizer_groups",
                "audit.expected",
            ),
            "audit.expected.optimizer_groups",
        )

        # --------------------------------------------------------------
        # Strict audit scope.
        # --------------------------------------------------------------

        if run_seed != 8:

            raise ValueError(
                "Stage-B branching audit must use seed 8."
            )

        if resolution_name != "r256":

            raise ValueError(
                "Stage-B branching audit currently uses r256."
            )

        if backbone_lrs != [
            0.00003,
            0.0001,
            0.0003,
        ]:

            raise ValueError(
                "Audit LR candidates do not match frozen Stage-B set."
            )

        if invalid_lr in backbone_lrs:

            raise ValueError(
                "Invalid LR control overlaps the valid LR set."
            )

        if expected_train_rows != 1440:

            raise ValueError(
                "Expected project_train rows must equal 1440."
            )

        if expected_dev_rows != 459:

            raise ValueError(
                "Expected dev_val rows must equal 459."
            )

        branch_contract = (
            load_stage_b_branch_contract(
                experiment_cfg=experiment_cfg,
            )
        )

        if list(
            branch_contract.backbone_lr_candidates
        ) != backbone_lrs:

            raise RuntimeError(
                "Production Stage-B contract LR set differs "
                "from audit configuration."
            )

        configured_device = str(
            require_key(
                runtime_cfg,
                "device",
                "machine_config.runtime",
            )
        )

        if require_cuda:

            if not configured_device.startswith(
                "cuda"
            ):

                raise RuntimeError(
                    "Audit requires CUDA but machine config does not."
                )

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "Audit requires CUDA but CUDA is unavailable."
                )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 STAGE-B BRANCH INITIALIZATION AUDIT"
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
            "Configured device: %s",
            configured_device,
        )

        logger.info(
            "Run seed: %d",
            run_seed,
        )

        logger.info(
            "Resolution: %s",
            resolution_name,
        )

        logger.info(
            "Experiment config SHA-256: %s",
            actual_experiment_sha,
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
            "src/stage_b_branching.py",
            "src/dataloading.py",
            "src/modeling.py",
            "src/optimization.py",
            "src/reproducibility.py",
            "src/training_control.py",
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
            "Dataset access: manifest construction/sampler inspection only"
        )

        logger.info(
            "Image decoding: NOT PERFORMED"
        )

        logger.info(
            "Scientific training: NOT PERFORMED"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # ==============================================================
        # Controlled Stage-A checkpoint
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Controlled Stage-A checkpoint ---"
        )

        (
            stage_a_checkpoint,
            checkpoint_evidence,
        ) = build_controlled_stage_a_checkpoint(
            experiment_cfg=experiment_cfg,
            run_seed=run_seed,
            epoch=checkpoint_epoch,
            weighted_dev_loss=checkpoint_loss,
            classifier_weight_delta=weight_delta,
            classifier_bias_delta=bias_delta,
        )

        logger.info(
            "[PASS] controlled Stage-A checkpoint differs "
            "from fresh seed-8 model"
        )

        logger.info(
            "[PASS] controlled Stage-A checkpoint SHA-256 = %s",
            stage_a_checkpoint.model_state_sha256,
        )

        # ==============================================================
        # Independent generator reference
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Independent run-seed shuffle reference ---"
        )

        generator_reference = (
            build_generator_reference(
                run_seed=run_seed,
                rows=expected_train_rows,
            )
        )

        logger.info(
            "[PASS] independent full project_train permutation built"
        )

        logger.info(
            "[PASS] reference order SHA-256 = %s",
            generator_reference[
                "order_indices_sha256"
            ],
        )

        # ==============================================================
        # Three Stage-B LR branches
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Stage-B LR branches ---"
        )

        branch_results: list[
            dict[str, Any]
        ] = []

        for backbone_lr in backbone_lrs:

            logger.info(
                "Constructing Stage-B branch | backbone_lr=%.8g",
                backbone_lr,
            )

            result = audit_one_branch(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                resolution_name=resolution_name,
                run_seed=run_seed,
                backbone_lr=backbone_lr,
                checkpoint=stage_a_checkpoint,
                controlled_checkpoint_evidence=(
                    checkpoint_evidence
                ),
                expected_train_rows=expected_train_rows,
                expected_dev_rows=expected_dev_rows,
                expected_groups_cfg=expected_groups_cfg,
                reference_generator=generator_reference,
            )

            branch_results.append(
                result
            )

            logger.info(
                "[PASS] branch %.8g restored exact Stage-A model",
                backbone_lr,
            )

            logger.info(
                "[PASS] branch %.8g fresh empty AdamW",
                backbone_lr,
            )

            logger.info(
                "[PASS] branch %.8g first-epoch train order = %s",
                backbone_lr,
                result[
                    "project_train_sampler"
                ][
                    "order_indices_sha256"
                ],
            )

        # ==============================================================
        # Cross-branch comparability
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Cross-branch comparability ---"
        )

        cross_branch = compare_branches(
            results=branch_results,
        )

        logger.info(
            "[PASS] all LR candidates start from identical "
            "Stage-A checkpoint/model state"
        )

        logger.info(
            "[PASS] all LR candidates start from identical "
            "fresh run-seed DataLoader generator state"
        )

        logger.info(
            "[PASS] all LR candidates have identical first-epoch "
            "project_train ordering"
        )

        logger.info(
            "[PASS] all LR candidates have identical initial "
            "global RNG fingerprints"
        )

        logger.info(
            "[PASS] optimizer structure identical except "
            "Stage-B backbone LR"
        )

        logger.info(
            "[PASS] classifier LR fixed at 1e-3 for all branches"
        )

        # ==============================================================
        # Invalid LR guard
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Invalid-LR guard ---"
        )

        invalid_lr_rejected = (
            audit_invalid_lr_rejection(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                resolution_name=resolution_name,
                run_seed=run_seed,
                checkpoint=stage_a_checkpoint,
                invalid_lr=invalid_lr,
            )
        )

        logger.info(
            "[PASS] non-frozen Stage-B LR %.8g rejected",
            invalid_lr,
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
                        configured_device,

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

                    "experiment_config_sha256":
                        actual_experiment_sha,

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

                    "stage_b_branching_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "stage_b_branching.py"
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
                    "run_seed":
                        run_seed,

                    "resolution":
                        resolution_name,

                    "backbone_lr_candidates":
                        backbone_lrs,

                    "project_train_rows":
                        expected_train_rows,

                    "dev_val_rows":
                        expected_dev_rows,

                    "image_decoding_performed":
                        False,

                    "forward_pass_performed":
                        False,

                    "backward_pass_performed":
                        False,

                    "optimizer_step_performed":
                        False,

                    "scientific_training_performed":
                        False,

                    "held_out_test_accessed":
                        False,
                },

            "controlled_stage_a_checkpoint":
                checkpoint_evidence,

            "independent_generator_reference":
                {
                    key:
                        value

                    for (
                        key,
                        value,
                    ) in generator_reference.items()

                    if key != "order_indices"
                },

            "branches":
                branch_results,

            "cross_branch_comparability":
                cross_branch,

            "invalid_backbone_lr":
                {
                    "value":
                        invalid_lr,

                    "rejected":
                        invalid_lr_rejected,
                },

            "interpretation":
                (
                    "PASS establishes that schema-v3 Stage-B branch "
                    "construction carries only the exact Stage-A "
                    "raw-argmin model state, resets RNG/DataLoader "
                    "state from run_seed for every LR candidate, "
                    "constructs fresh empty AdamW state, exposes "
                    "identical corresponding first-epoch project_train "
                    "ordering, and changes only the configured "
                    "backbone learning rate between the three "
                    "screening branches. No branch training was "
                    "performed."
                ),
        }

        with partial_result_path.open(
            "x",
            encoding="utf-8",
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
            "[PASS] schema-v3 branch contract"
        )

        logger.info(
            "[PASS] controlled Stage-A checkpoint restored exactly"
        )

        logger.info(
            "[PASS] all three LR branches same starting model"
        )

        logger.info(
            "[PASS] all three LR branches fresh optimizer state"
        )

        logger.info(
            "[PASS] optimizer membership/settings identical "
            "except backbone LR"
        )

        logger.info(
            "[PASS] all three LR branches fresh run-seed generators"
        )

        logger.info(
            "[PASS] advanced Stage-A-like generator state not inherited"
        )

        logger.info(
            "[PASS] all three LR branches same first-epoch train order"
        )

        logger.info(
            "[PASS] all three LR branches same initial global RNG state"
        )

        logger.info(
            "[PASS] no images decoded"
        )

        logger.info(
            "[PASS] no training/forward/backward/optimizer step"
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
            "RESNET-18 STAGE-B BRANCH INITIALIZATION AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 STAGE-B BRANCH INITIALIZATION AUDIT: FAIL"
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