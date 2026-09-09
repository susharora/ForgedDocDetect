#!/usr/bin/env python3
"""
Audit the frozen Tech-2 ResNet-18 model and transfer-learning contracts.

This audit verifies:

1. frozen experiment validation passes;
2. execution begins from a clean Git commit;
3. actual torchvision IMAGENET1K_V1 checkpoint SHA-256;
4. model parameter counts and BatchNorm count;
5. pretrained backbone is identical across run seeds;
6. FC initialization is reproducible for the same seed;
7. FC initialization differs across seeds 8/9/10;
8. Stage-A requires_grad and module-mode contract;
9. Stage-A BatchNorm running state remains exactly unchanged;
10. Stage-A backward creates gradients only for the classifier;
11. Stage-A parameter grouping is exactly:
        fc_decay
        fc_no_decay
12. Stage-B unfreezes the complete network;
13. Stage-B BatchNorm counters advance during a real forward pass;
14. Stage-B backbone state changes only through BN buffers during
    this no-optimizer probe;
15. Stage-B backward reaches every model parameter;
16. Stage-B parameter grouping is exactly:
        backbone_decay
        backbone_no_decay
        fc_decay
        fc_no_decay
17. every bias and BatchNorm affine parameter receives zero decay;
18. every other trainable parameter receives weight decay;
19. Stage-B LR candidates are restricted to the frozen set;
20. evaluation-mode round trips behave correctly.

No optimizer.step() is executed.
No scientific training loss is used.
No Dataset or held-out test data are accessed.

The scalar used for backward is merely mean(logits^2), solely to
exercise autograd and BN behavior. It is NOT the scientific training
objective.

All detailed evidence is written under ./logs/.
No print() is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
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

from src.modeling import (
    EXPECTED_PRETRAINED_SHA256,
    apply_evaluation_contract,
    apply_stage_a_contract,
    apply_stage_b_contract,
    assert_stage_a_contract,
    assert_stage_b_contract,
    build_resnet18_classifier,
    build_stage_a_parameter_groups,
    build_stage_b_parameter_groups,
)

from src.reproducibility import (
    configure_run_reproducibility,
)


LOGGER = logging.getLogger(
    "audit_resnet18_model_contract"
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
            "Commit/remove outstanding files before "
            "running the model-contract audit.\n\n"
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
            "Validator artifact SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        path,
        expected_sha,
    )


# ======================================================================
# Logging/output
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
# Tensor / model fingerprints
# ======================================================================

def update_digest_with_tensor(
    digest: hashlib._Hash,
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
    *,
    include_fc: bool,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        tensor,
    ) in model.state_dict().items():

        if (
            not include_fc
            and name.startswith(
                "fc."
            )
        ):

            continue

        update_digest_with_tensor(
            digest,
            name=name,
            tensor=tensor,
        )

    return digest.hexdigest()


def classifier_sha256(
    model: nn.Module,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        tensor,
    ) in model.state_dict().items():

        if not name.startswith(
            "fc."
        ):

            continue

        update_digest_with_tensor(
            digest,
            name=name,
            tensor=tensor,
        )

    return digest.hexdigest()


# ======================================================================
# BatchNorm snapshots
# ======================================================================

def snapshot_batchnorm(
    model: nn.Module,
) -> dict[
    str,
    dict[str, torch.Tensor],
]:

    result: dict[
        str,
        dict[str, torch.Tensor],
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

        if (
            module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
        ):

            raise RuntimeError(
                "Frozen ResNet BatchNorm unexpectedly lacks "
                f"running state: {name}"
            )

        result[
            name
        ] = {
            "running_mean":
                (
                    module.running_mean
                    .detach()
                    .cpu()
                    .clone()
                ),

            "running_var":
                (
                    module.running_var
                    .detach()
                    .cpu()
                    .clone()
                ),

            "num_batches_tracked":
                (
                    module.num_batches_tracked
                    .detach()
                    .cpu()
                    .clone()
                ),
        }

    return result


def require_bn_exactly_unchanged(
    *,
    before: dict[
        str,
        dict[str, torch.Tensor],
    ],
    after: dict[
        str,
        dict[str, torch.Tensor],
    ],
) -> None:

    if set(
        before
    ) != set(
        after
    ):

        raise RuntimeError(
            "BatchNorm module set changed."
        )

    for name in before:

        for field in (
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):

            if not torch.equal(
                before[
                    name
                ][
                    field
                ],
                after[
                    name
                ][
                    field
                ],
            ):

                raise RuntimeError(
                    "Stage-A BatchNorm state changed:\n"
                    f"  module={name}\n"
                    f"  field={field}"
                )


def require_bn_stage_b_update(
    *,
    before: dict[
        str,
        dict[str, torch.Tensor],
    ],
    after: dict[
        str,
        dict[str, torch.Tensor],
    ],
) -> dict[str, Any]:

    if set(
        before
    ) != set(
        after
    ):

        raise RuntimeError(
            "BatchNorm module set changed."
        )

    changed_statistics = 0

    for name in before:

        before_count = int(
            before[
                name
            ][
                "num_batches_tracked"
            ].item()
        )

        after_count = int(
            after[
                name
            ][
                "num_batches_tracked"
            ].item()
        )

        if after_count != (
            before_count
            + 1
        ):

            raise RuntimeError(
                "Stage-B BatchNorm counter did not advance "
                "exactly once:\n"
                f"  module={name}\n"
                f"  before={before_count}\n"
                f"  after={after_count}"
            )

        mean_changed = not torch.equal(
            before[
                name
            ][
                "running_mean"
            ],
            after[
                name
            ][
                "running_mean"
            ],
        )

        var_changed = not torch.equal(
            before[
                name
            ][
                "running_var"
            ],
            after[
                name
            ][
                "running_var"
            ],
        )

        if (
            mean_changed
            or var_changed
        ):

            changed_statistics += 1

    if changed_statistics <= 0:

        raise RuntimeError(
            "Stage-B forward did not modify any "
            "BatchNorm running statistics."
        )

    return {
        "batchnorm_modules":
            len(
                before
            ),

        "modules_with_changed_running_statistics":
            changed_statistics,
    }


# ======================================================================
# Parameter group audit
# ======================================================================

def parameter_name_by_id(
    model: nn.Module,
) -> dict[int, str]:

    return {
        id(
            parameter
        ):
            name

        for (
            name,
            parameter,
        ) in model.named_parameters()
    }


def independent_batchnorm_parameter_ids(
    model: nn.Module,
) -> set[int]:

    result: set[int] = set()

    for module in model.modules():

        if not isinstance(
            module,
            BATCHNORM_TYPES,
        ):

            continue

        for parameter in (
            module.parameters(
                recurse=False
            )
        ):

            result.add(
                id(
                    parameter
                )
            )

    return result


def summarize_groups(
    groups: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for group in groups:

        name = str(
            group[
                "name"
            ]
        )

        result[
            name
        ] = {
            "parameter_tensors":
                len(
                    group[
                        "params"
                    ]
                ),

            "parameter_elements":
                sum(
                    parameter.numel()
                    for parameter
                    in group[
                        "params"
                    ]
                ),

            "learning_rate":
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
        }

    return result


def validate_group_membership(
    *,
    model: nn.Module,
    groups: list[
        dict[str, Any]
    ],
    expected_summary: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    dict[str, Any],
]:

    name_by_id = parameter_name_by_id(
        model
    )

    bn_ids = (
        independent_batchnorm_parameter_ids(
            model
        )
    )

    seen: set[int] = set()

    membership: dict[
        str,
        list[str],
    ] = {}

    for group in groups:

        group_name = str(
            group[
                "name"
            ]
        )

        group_weight_decay = float(
            group[
                "weight_decay"
            ]
        )

        group_names: list[str] = []

        for parameter in group[
            "params"
        ]:

            parameter_id = id(
                parameter
            )

            if parameter_id in seen:

                raise RuntimeError(
                    "Parameter appears in more than one group."
                )

            seen.add(
                parameter_id
            )

            parameter_name = name_by_id[
                parameter_id
            ]

            should_have_zero_decay = (
                parameter_name.endswith(
                    ".bias"
                )
                or
                parameter_id in bn_ids
            )

            if should_have_zero_decay:

                if group_weight_decay != 0.0:

                    raise RuntimeError(
                        "Bias/BatchNorm parameter received "
                        "weight decay:\n"
                        f"  {parameter_name}"
                    )

            else:

                if group_weight_decay != 0.0001:

                    raise RuntimeError(
                        "Regular trainable parameter did not "
                        "receive frozen weight decay:\n"
                        f"  {parameter_name}\n"
                        f"  weight_decay={group_weight_decay}"
                    )

            group_names.append(
                parameter_name
            )

        membership[
            group_name
        ] = group_names

    expected_trainable_ids = {
        id(
            parameter
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    }

    if seen != expected_trainable_ids:

        raise RuntimeError(
            "Parameter grouping does not cover every "
            "trainable parameter exactly once:\n"
            f"  grouped={len(seen)}\n"
            f"  expected={len(expected_trainable_ids)}"
        )

    summary = summarize_groups(
        groups
    )

    normalized_expected = {
        str(
            group_name
        ):
            {
                "parameter_tensors":
                    int(
                        values[
                            "parameter_tensors"
                        ]
                    ),

                "parameter_elements":
                    int(
                        values[
                            "parameter_elements"
                        ]
                    ),

                "learning_rate":
                    float(
                        values[
                            "learning_rate"
                        ]
                    ),

                "weight_decay":
                    float(
                        values[
                            "weight_decay"
                        ]
                    ),
            }

        for (
            group_name,
            values,
        ) in expected_summary.items()
    }

    if summary != normalized_expected:

        raise RuntimeError(
            "Parameter-group summary mismatch:\n"
            f"  expected={normalized_expected}\n"
            f"  actual={summary}"
        )

    return {
        group_name:
            {
                **summary[
                    group_name
                ],

                "parameter_names":
                    membership[
                        group_name
                    ],
            }

        for group_name
        in summary
    }


# ======================================================================
# Seed-controlled model construction
# ======================================================================

def audit_model_initialization(
    *,
    experiment_cfg: dict[str, Any],
    seeds: list[int],
    repeats: int,
    expected_model: dict[str, int],
) -> dict[str, Any]:

    results: dict[
        str,
        Any,
    ] = {}

    classifier_hashes_across_seeds: set[
        str
    ] = set()

    backbone_hash_reference: str | None = None

    checkpoint_path_reference: str | None = None

    for seed in seeds:

        repeat_classifier_hashes: list[str] = []
        repeat_backbone_hashes: list[str] = []

        for repeat_index in range(
            repeats
        ):

            configure_run_reproducibility(
                experiment_cfg=experiment_cfg,
                run_seed=seed,
            )

            model, provenance = (
                build_resnet18_classifier(
                    experiment_cfg=experiment_cfg,
                )
            )

            if (
                provenance.pretrained_checkpoint_sha256
                != EXPECTED_PRETRAINED_SHA256
            ):

                raise RuntimeError(
                    "Pretrained checkpoint SHA mismatch."
                )

            model_counts = {
                "total_parameters":
                    int(
                        provenance.total_parameters
                    ),

                "backbone_parameters":
                    int(
                        provenance.backbone_parameters
                    ),

                "classifier_parameters":
                    int(
                        provenance.classifier_parameters
                    ),

                "batchnorm_modules":
                    int(
                        provenance.batchnorm_modules
                    ),
            }

            if model_counts != expected_model:

                raise RuntimeError(
                    "ResNet-18 model-count mismatch:\n"
                    f"  expected={expected_model}\n"
                    f"  actual={model_counts}"
                )

            classifier_hash = (
                classifier_sha256(
                    model
                )
            )

            backbone_hash = (
                model_state_sha256(
                    model,
                    include_fc=False,
                )
            )

            repeat_classifier_hashes.append(
                classifier_hash
            )

            repeat_backbone_hashes.append(
                backbone_hash
            )

            if backbone_hash_reference is None:

                backbone_hash_reference = (
                    backbone_hash
                )

            elif (
                backbone_hash
                != backbone_hash_reference
            ):

                raise RuntimeError(
                    "Pretrained backbone differs across "
                    "seed/repeated model construction."
                )

            checkpoint_path = (
                provenance
                .pretrained_checkpoint_path
            )

            if checkpoint_path_reference is None:

                checkpoint_path_reference = (
                    checkpoint_path
                )

            elif (
                checkpoint_path
                != checkpoint_path_reference
            ):

                raise RuntimeError(
                    "torchvision checkpoint path changed "
                    "during one audit."
                )

            del model

            gc.collect()

        if len(
            set(
                repeat_classifier_hashes
            )
        ) != 1:

            raise RuntimeError(
                "Classifier initialization is not reproducible "
                f"for run seed {seed}."
            )

        if len(
            set(
                repeat_backbone_hashes
            )
        ) != 1:

            raise RuntimeError(
                "Backbone is not reproducible "
                f"for run seed {seed}."
            )

        classifier_hash = (
            repeat_classifier_hashes[
                0
            ]
        )

        if (
            classifier_hash
            in classifier_hashes_across_seeds
        ):

            raise RuntimeError(
                "Different run seeds produced identical "
                "FC initialization."
            )

        classifier_hashes_across_seeds.add(
            classifier_hash
        )

        results[
            str(
                seed
            )
        ] = {
            "classifier_sha256":
                classifier_hash,

            "backbone_state_sha256":
                repeat_backbone_hashes[
                    0
                ],

            "repeated_builds":
                repeats,
        }

        LOGGER.info(
            "[PASS] model initialization | seed=%d | "
            "classifier=%s | backbone=%s",
            seed,
            classifier_hash,
            repeat_backbone_hashes[
                0
            ],
        )

    LOGGER.info(
        "[PASS] same seed reproduces classifier initialization"
    )

    LOGGER.info(
        "[PASS] seeds 8/9/10 produce distinct classifier initialization"
    )

    LOGGER.info(
        "[PASS] pretrained backbone is identical across all seed builds"
    )

    return {
        "checkpoint_path":
            checkpoint_path_reference,

        "pretrained_checkpoint_sha256":
            EXPECTED_PRETRAINED_SHA256,

        "backbone_state_sha256":
            backbone_hash_reference,

        "seeds":
            results,
    }


# ======================================================================
# Deterministic synthetic input
# ======================================================================

def make_probe_input(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Deterministic structured input without consuming any RNG stream.
    """

    x_axis = torch.linspace(
        -1.0,
        1.0,
        steps=width,
        dtype=torch.float32,
        device=device,
    )

    y_axis = torch.linspace(
        -0.5,
        0.5,
        steps=height,
        dtype=torch.float32,
        device=device,
    )

    base = (
        y_axis[
            None,
            None,
            :,
            None,
        ]
        +
        x_axis[
            None,
            None,
            None,
            :,
        ]
    )

    channel_0 = base

    channel_1 = (
        base
        * 0.5
        + 0.1
    )

    channel_2 = (
        -base
        * 0.25
        - 0.2
    )

    image = torch.cat(
        [
            channel_0,
            channel_1,
            channel_2,
        ],
        dim=1,
    )

    return image.repeat(
        batch_size,
        1,
        1,
        1,
    )


# ======================================================================
# Stage-A and Stage-B runtime probes
# ======================================================================

def audit_stage_contracts(
    *,
    experiment_cfg: dict[str, Any],
    device_string: str,
    seed: int,
    resolution_name: str,
    batch_size: int,
    backbone_lr: float,
    expected_stage_a_groups: dict[str, Any],
    expected_stage_b_groups: dict[str, Any],
) -> dict[str, Any]:

    configure_run_reproducibility(
        experiment_cfg=experiment_cfg,
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

    preprocessing = require_mapping(
        require_key(
            experiment_cfg,
            "preprocessing",
            "experiment_config",
        ),
        "preprocessing",
    )

    canvases = require_mapping(
        require_key(
            preprocessing,
            "candidate_canvases",
            "preprocessing",
        ),
        "preprocessing.candidate_canvases",
    )

    canvas = require_mapping(
        require_key(
            canvases,
            resolution_name,
            "preprocessing.candidate_canvases",
        ),
        (
            "preprocessing."
            "candidate_canvases."
            f"{resolution_name}"
        ),
    )

    height = int(
        require_key(
            canvas,
            "canvas_height",
            (
                "preprocessing."
                "candidate_canvases."
                f"{resolution_name}"
            ),
        )
    )

    width = int(
        require_key(
            canvas,
            "canvas_width",
            (
                "preprocessing."
                "candidate_canvases."
                f"{resolution_name}"
            ),
        )
    )

    input_tensor = make_probe_input(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
    )

    # ==================================================================
    # Stage A
    # ==================================================================

    apply_stage_a_contract(
        model
    )

    assert_stage_a_contract(
        model
    )

    stage_a_groups, _ = (
        build_stage_a_parameter_groups(
            model=model,
            experiment_cfg=experiment_cfg,
        )
    )

    stage_a_group_evidence = (
        validate_group_membership(
            model=model,
            groups=stage_a_groups,
            expected_summary=expected_stage_a_groups,
        )
    )

    stage_a_backbone_before = (
        model_state_sha256(
            model,
            include_fc=False,
        )
    )

    stage_a_bn_before = (
        snapshot_batchnorm(
            model
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        input_tensor
    )

    if tuple(
        logits.shape
    ) != (
        batch_size,
        2,
    ):

        raise RuntimeError(
            "Stage-A forward output shape mismatch:\n"
            f"  expected={(batch_size, 2)}\n"
            f"  actual={tuple(logits.shape)}"
        )

    if logits.dtype != torch.float32:

        raise RuntimeError(
            "Stage-A logits are not float32."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):

        raise RuntimeError(
            "Stage-A logits contain NaN/Inf."
        )

    # Autograd probe only; not scientific training loss.
    probe_scalar = (
        logits
        .square()
        .mean()
    )

    probe_scalar.backward()

    stage_a_bn_after = (
        snapshot_batchnorm(
            model
        )
    )

    require_bn_exactly_unchanged(
        before=stage_a_bn_before,
        after=stage_a_bn_after,
    )

    stage_a_backbone_after = (
        model_state_sha256(
            model,
            include_fc=False,
        )
    )

    if (
        stage_a_backbone_before
        != stage_a_backbone_after
    ):

        raise RuntimeError(
            "Stage-A backbone state changed during "
            "forward/backward."
        )

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if name.startswith(
            "fc."
        ):

            if parameter.grad is None:

                raise RuntimeError(
                    "Stage-A classifier parameter has no gradient:\n"
                    f"  {name}"
                )

            if not bool(
                torch.isfinite(
                    parameter.grad
                ).all()
            ):

                raise RuntimeError(
                    "Stage-A classifier gradient is non-finite:\n"
                    f"  {name}"
                )

        else:

            if parameter.grad is not None:

                raise RuntimeError(
                    "Stage-A frozen backbone received gradient:\n"
                    f"  {name}"
                )

    LOGGER.info(
        "[PASS] Stage A forward/backward: classifier-only gradients"
    )

    LOGGER.info(
        "[PASS] Stage A BatchNorm running state exactly unchanged"
    )

    LOGGER.info(
        "[PASS] Stage A backbone state exactly unchanged"
    )

    # ------------------------------------------------------------------
    # Evaluation round-trip from Stage A.
    # ------------------------------------------------------------------

    apply_evaluation_contract(
        model
    )

    if model.fc.training:

        raise RuntimeError(
            "Evaluation contract failed to set FC eval mode."
        )

    apply_stage_a_contract(
        model
    )

    assert_stage_a_contract(
        model
    )

    LOGGER.info(
        "[PASS] Stage A -> evaluation -> Stage A mode round-trip"
    )

    # ==================================================================
    # Stage B
    # ==================================================================

    model.zero_grad(
        set_to_none=True
    )

    apply_stage_b_contract(
        model
    )

    assert_stage_b_contract(
        model
    )

    stage_b_groups, _ = (
        build_stage_b_parameter_groups(
            model=model,
            experiment_cfg=experiment_cfg,
            backbone_lr=backbone_lr,
        )
    )

    stage_b_group_evidence = (
        validate_group_membership(
            model=model,
            groups=stage_b_groups,
            expected_summary=expected_stage_b_groups,
        )
    )

    stage_b_parameters_before = {
        name:
            (
                parameter
                .detach()
                .cpu()
                .clone()
            )

        for (
            name,
            parameter,
        ) in model.named_parameters()
    }

    stage_b_bn_before = (
        snapshot_batchnorm(
            model
        )
    )

    stage_b_backbone_before = (
        model_state_sha256(
            model,
            include_fc=False,
        )
    )

    logits = model(
        input_tensor
    )

    if tuple(
        logits.shape
    ) != (
        batch_size,
        2,
    ):

        raise RuntimeError(
            "Stage-B forward output shape mismatch."
        )

    probe_scalar = (
        logits
        .square()
        .mean()
    )

    probe_scalar.backward()

    stage_b_bn_after = (
        snapshot_batchnorm(
            model
        )
    )

    bn_update_evidence = (
        require_bn_stage_b_update(
            before=stage_b_bn_before,
            after=stage_b_bn_after,
        )
    )

    stage_b_backbone_after = (
        model_state_sha256(
            model,
            include_fc=False,
        )
    )

    if (
        stage_b_backbone_before
        == stage_b_backbone_after
    ):

        raise RuntimeError(
            "Stage-B backbone state did not change despite "
            "BatchNorm running-stat adaptation."
        )

    # No optimizer step occurred, so parameters themselves must not move.
    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not torch.equal(
            stage_b_parameters_before[
                name
            ],
            parameter
            .detach()
            .cpu(),
        ):

            raise RuntimeError(
                "Model parameter changed despite no optimizer step:\n"
                f"  {name}"
            )

        if parameter.grad is None:

            raise RuntimeError(
                "Stage-B trainable parameter did not receive gradient:\n"
                f"  {name}"
            )

        if not bool(
            torch.isfinite(
                parameter.grad
            ).all()
        ):

            raise RuntimeError(
                "Stage-B parameter gradient contains NaN/Inf:\n"
                f"  {name}"
            )

    LOGGER.info(
        "[PASS] Stage B forward/backward reaches all parameters"
    )

    LOGGER.info(
        "[PASS] Stage B BatchNorm counters advance exactly once"
    )

    LOGGER.info(
        "[PASS] Stage B running statistics adapt"
    )

    LOGGER.info(
        "[PASS] No parameter changes without optimizer.step()"
    )

    # ------------------------------------------------------------------
    # Evaluation round-trip from Stage B.
    # ------------------------------------------------------------------

    apply_evaluation_contract(
        model
    )

    apply_stage_b_contract(
        model
    )

    assert_stage_b_contract(
        model
    )

    LOGGER.info(
        "[PASS] Stage B -> evaluation -> Stage B mode round-trip"
    )

    # ------------------------------------------------------------------
    # All three frozen Stage-B LR candidates must be accepted.
    # ------------------------------------------------------------------

    transfer_learning = require_mapping(
        require_key(
            experiment_cfg,
            "transfer_learning",
            "experiment_config",
        ),
        "transfer_learning",
    )

    stage_b_cfg = require_mapping(
        require_key(
            transfer_learning,
            "stage_b",
            "transfer_learning",
        ),
        "transfer_learning.stage_b",
    )

    stage_b_optimizer_cfg = require_mapping(
        require_key(
            stage_b_cfg,
            "optimizer",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.optimizer",
    )

    candidate_lrs = [
        float(
            value
        )
        for value
        in require_key(
            stage_b_optimizer_cfg,
            "backbone_lr_candidates",
            "transfer_learning.stage_b.optimizer",
        )
    ]

    accepted_lrs: list[float] = []

    for candidate_lr in candidate_lrs:

        groups, _ = (
            build_stage_b_parameter_groups(
                model=model,
                experiment_cfg=experiment_cfg,
                backbone_lr=candidate_lr,
            )
        )

        for group in groups:

            if str(
                group[
                    "name"
                ]
            ).startswith(
                "backbone_"
            ):

                if (
                    float(
                        group[
                            "lr"
                        ]
                    )
                    != candidate_lr
                ):

                    raise RuntimeError(
                        "Backbone LR did not propagate into "
                        "Stage-B parameter group."
                    )

            else:

                if (
                    float(
                        group[
                            "lr"
                        ]
                    )
                    != 0.001
                ):

                    raise RuntimeError(
                        "Classifier LR changed during "
                        "Stage-B LR screening."
                    )

        accepted_lrs.append(
            candidate_lr
        )

    invalid_lr_rejected = False

    try:

        build_stage_b_parameter_groups(
            model=model,
            experiment_cfg=experiment_cfg,
            backbone_lr=0.001,
        )

    except ValueError:

        invalid_lr_rejected = True

    if not invalid_lr_rejected:

        raise RuntimeError(
            "Stage-B parameter grouping accepted an LR "
            "outside the frozen candidate set."
        )

    LOGGER.info(
        "[PASS] Frozen Stage-B LR candidates accepted: %s",
        accepted_lrs,
    )

    LOGGER.info(
        "[PASS] Non-frozen Stage-B LR rejected"
    )

    model.zero_grad(
        set_to_none=True
    )

    del input_tensor
    del model

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return {
        "device":
            str(
                device
            ),

        "seed":
            seed,

        "resolution":
            resolution_name,

        "input_shape":
            [
                batch_size,
                3,
                height,
                width,
            ],

        "model_provenance":
            {
                "total_parameters":
                    provenance.total_parameters,

                "backbone_parameters":
                    provenance.backbone_parameters,

                "classifier_parameters":
                    provenance.classifier_parameters,

                "batchnorm_modules":
                    provenance.batchnorm_modules,

                "pretrained_checkpoint_path":
                    provenance.pretrained_checkpoint_path,

                "pretrained_checkpoint_sha256":
                    provenance.pretrained_checkpoint_sha256,
            },

        "stage_a":
            {
                "groups":
                    stage_a_group_evidence,

                "backbone_state_before_sha256":
                    stage_a_backbone_before,

                "backbone_state_after_sha256":
                    stage_a_backbone_after,

                "batchnorm_state_unchanged":
                    True,
            },

        "stage_b":
            {
                "groups":
                    stage_b_group_evidence,

                "backbone_state_before_sha256":
                    stage_b_backbone_before,

                "backbone_state_after_sha256":
                    stage_b_backbone_after,

                "batchnorm":
                    bn_update_evidence,

                "accepted_backbone_lrs":
                    accepted_lrs,

                "invalid_lr_rejected":
                    invalid_lr_rejected,
            },
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen ResNet-18 model and "
            "transfer-learning stage contracts."
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
            "audit_resnet18_model_contract_config.yaml"
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
            "Model-contract audit schema_version must equal 1."
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

    machine_cfg_section = require_mapping(
        require_key(
            machine_cfg,
            "machine",
            "machine_config",
        ),
        "machine_config.machine",
    )

    machine_id = str(
        require_key(
            machine_cfg_section,
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

        seeds = [
            int(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "seeds",
                "audit_config.audit",
            )
        ]

        repeats = int(
            require_key(
                audit_cfg,
                "repeated_builds_per_seed",
                "audit_config.audit",
            )
        )

        stage_probe = require_mapping(
            require_key(
                audit_cfg,
                "stage_probe",
                "audit_config.audit",
            ),
            "audit_config.audit.stage_probe",
        )

        probe_seed = int(
            require_key(
                stage_probe,
                "seed",
                "audit_config.audit.stage_probe",
            )
        )

        probe_resolution = str(
            require_key(
                stage_probe,
                "resolution",
                "audit_config.audit.stage_probe",
            )
        )

        probe_batch_size = int(
            require_key(
                stage_probe,
                "batch_size",
                "audit_config.audit.stage_probe",
            )
        )

        probe_backbone_lr = float(
            require_key(
                stage_probe,
                "stage_b_backbone_lr",
                "audit_config.audit.stage_probe",
            )
        )

        require_cuda = bool(
            require_key(
                stage_probe,
                "require_cuda",
                "audit_config.audit.stage_probe",
            )
        )

        expected_model = {
            str(
                key
            ):
                int(
                    value
                )

            for (
                key,
                value,
            ) in require_mapping(
                require_key(
                    audit_cfg,
                    "expected_model",
                    "audit_config.audit",
                ),
                "audit_config.audit.expected_model",
            ).items()
        }

        expected_stage_a_groups = require_mapping(
            require_key(
                audit_cfg,
                "expected_stage_a_groups",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_stage_a_groups",
        )

        expected_stage_b_groups = require_mapping(
            require_key(
                audit_cfg,
                "expected_stage_b_groups",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_stage_b_groups",
        )

        if seeds != [
            8,
            9,
            10,
        ]:

            raise ValueError(
                "Audit seeds must be exactly [8, 9, 10]."
            )

        if repeats != 2:

            raise ValueError(
                "Model initialization must be repeated "
                "exactly twice per seed in this audit."
            )

        if probe_seed != 8:

            raise ValueError(
                "Stage contract probe must use screening seed 8."
            )

        if probe_resolution != "r256":

            raise ValueError(
                "Stage contract probe must use r256."
            )

        if probe_batch_size != 2:

            raise ValueError(
                "Stage contract probe batch size must equal 2."
            )

        if probe_backbone_lr != 0.0001:

            raise ValueError(
                "Stage-B contract probe must use backbone LR 1e-4."
            )

        frozen_expected_model = {
            "total_parameters":
                11177538,

            "backbone_parameters":
                11176512,

            "classifier_parameters":
                1026,

            "batchnorm_modules":
                20,
        }

        if expected_model != frozen_expected_model:

            raise ValueError(
                "Audit model counts do not match known torchvision "
                "ResNet-18 binary-classifier structure:\n"
                f"  expected={frozen_expected_model}\n"
                f"  configured={expected_model}"
            )

        if require_cuda:

            if not device_string.startswith(
                "cuda"
            ):

                raise RuntimeError(
                    "Model-contract audit requires CUDA but "
                    f"machine device is {device_string!r}."
                )

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "Model-contract audit requires CUDA but "
                    "torch.cuda.is_available() is False."
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
                "Model-contract audit requires protocol_status=frozen."
            )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 MODEL + TRANSFER-LEARNING CONTRACT AUDIT"
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
            "src/modeling.py",
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
            "Configured device: %s",
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
            "Held-out test: NOT ACCESSED"
        )

        logger.info(
            "Scientific training loss: NOT USED"
        )

        logger.info(
            "optimizer.step(): NOT USED"
        )

        # ==============================================================
        # Exact pretrained model / seed initialization
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Model construction / seed-controlled FC initialization ---"
        )

        initialization_results = (
            audit_model_initialization(
                experiment_cfg=experiment_cfg,
                seeds=seeds,
                repeats=repeats,
                expected_model=expected_model,
            )
        )

        # ==============================================================
        # Actual GPU Stage-A / Stage-B behavior
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Stage-A / Stage-B runtime contract ---"
        )

        stage_results = audit_stage_contracts(
            experiment_cfg=experiment_cfg,
            device_string=device_string,
            seed=probe_seed,
            resolution_name=probe_resolution,
            batch_size=probe_batch_size,
            backbone_lr=probe_backbone_lr,
            expected_stage_a_groups=expected_stage_a_groups,
            expected_stage_b_groups=expected_stage_b_groups,
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

                    "modeling_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "modeling.py"
                        ),

                    "reproducibility_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "reproducibility.py"
                        ),

                    "validator_log":
                        str(
                            validator_log_path
                        ),

                    "validator_log_sha256":
                        validator_log_sha,
                },

            "initialization":
                initialization_results,

            "stage_contracts":
                stage_results,

            "interpretation":
                (
                    "PASS establishes the exact pretrained ResNet-18 "
                    "identity, run-seed-controlled binary classifier "
                    "initialization, architecture counts, Stage-A "
                    "backbone/BatchNorm freeze behavior, Stage-B "
                    "full-backbone/BatchNorm adaptation behavior, "
                    "and exact AdamW-ready parameter grouping. "
                    "No optimizer step or scientific training loss "
                    "was executed."
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
            "[PASS] actual IMAGENET1K_V1 checkpoint full SHA-256"
        )

        logger.info(
            "[PASS] ResNet-18 binary-classifier parameter counts"
        )

        logger.info(
            "[PASS] pretrained backbone identical across seeds"
        )

        logger.info(
            "[PASS] same seed reproduces FC initialization"
        )

        logger.info(
            "[PASS] seeds 8/9/10 produce distinct FC initialization"
        )

        logger.info(
            "[PASS] Stage-A backbone frozen and BatchNorm invariant"
        )

        logger.info(
            "[PASS] Stage-A classifier-only gradients"
        )

        logger.info(
            "[PASS] Stage-A exact parameter groups"
        )

        logger.info(
            "[PASS] Stage-B complete network trainable"
        )

        logger.info(
            "[PASS] Stage-B BatchNorm running statistics adapt"
        )

        logger.info(
            "[PASS] Stage-B gradients reach every parameter"
        )

        logger.info(
            "[PASS] Stage-B exact parameter groups"
        )

        logger.info(
            "[PASS] bias and BatchNorm affine decay exclusions"
        )

        logger.info(
            "[PASS] frozen Stage-B LR candidate restriction"
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
            "RESNET-18 MODEL + TRANSFER-LEARNING CONTRACT AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 MODEL + TRANSFER-LEARNING CONTRACT AUDIT: FAIL"
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