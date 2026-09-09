#!/usr/bin/env python3
"""
FP32 training-memory probe for the proposed ResNet-18 transfer-learning
configuration.

Question answered
-----------------
Can this machine safely execute a full Stage-B training step using:

    ResNet-18 / IMAGENET1K_V1
    full backbone trainable
    BatchNorm in train mode
    batch size = 32
    input = 3 x 512 x 864
    FP32
    AdamW

This is deliberately a TRUE training probe:

    forward
    weighted binary cross-entropy
    backward
    optimizer.step()

Multiple steps are executed because AdamW allocates its first/second
moment state lazily during the first optimizer step.

Synthetic input
---------------
Synthetic tensors are used intentionally.

GPU activation / gradient memory is determined by tensor shapes,
architecture, dtype and optimizer state, not by document content.
Using synthetic input isolates the machine-memory question from the
still-to-be-implemented image preprocessing pipeline.

This tool therefore does NOT validate:
    - resize/pad correctness;
    - dataset loading;
    - class-index semantics;
    - final transfer-learning behaviour.

Those belong to the scientific training implementation.

No project_train images are decoded.
dev_val is not accessed.
held-out test is not accessed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from src.config import (
    load_experiment_config,
    load_machine_config,
)


BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


# ======================================================================
# Basic utilities
# ======================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
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
    ) as f:

        value = (
            yaml.safe_load(
                f
            )
            or {}
        )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Top level of YAML must be a mapping: "
            f"{path}"
        )

    return value


def git_commit_sha() -> str:

    result = subprocess.run(
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

    return result.stdout.strip()


def require_clean_git() -> str:
    """
    Run before creating any audit artifact.
    """

    commit = git_commit_sha()

    result = subprocess.run(
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
        result.stdout
        .strip()
    )

    if status:
        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit or stash outstanding changes before "
            "running the memory probe.\n\n"
            f"{status}"
        )

    return commit


def gib(
    byte_count: int | float,
) -> float:

    return (
        float(
            byte_count
        )
        / 1024**3
    )


# ======================================================================
# Logging / result paths
# ======================================================================

def configure_outputs(
    *,
    tool_cfg: dict[str, Any],
    machine_id: str,
) -> tuple[
    logging.Logger,
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

    logging_cfg = tool_cfg[
        "logging"
    ]

    output_cfg = tool_cfg[
        "output"
    ]

    log_dir = resolve_repo_path(
        logging_cfg[
            "directory"
        ]
    )

    output_dir = resolve_repo_path(
        output_cfg[
            "directory"
        ]
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    format_args = {
        "timestamp":
            timestamp,

        "machine_id":
            machine_id,
    }

    log_path = (
        log_dir
        / logging_cfg[
            "filename"
        ].format(
            **format_args
        )
    )

    output_path = (
        output_dir
        / output_cfg[
            "filename"
        ].format(
            **format_args
        )
    )

    level = getattr(
        logging,
        str(
            logging_cfg[
                "level"
            ]
        ).upper(),
    )

    logger = logging.getLogger(
        "probe_resnet18_training_memory"
    )

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(
        level
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

    logger.addHandler(
        handler
    )

    return (
        logger,
        log_path,
        output_path,
    )


# ======================================================================
# Frozen experiment validation
# ======================================================================

def run_frozen_provenance_gate(
    *,
    experiment_path: Path,
    machine_path: Path,
    logger: logging.Logger,
) -> None:

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

    for line in (
        result.stdout
        .splitlines()
    ):
        logger.info(
            "PROVENANCE_GATE | %s",
            line,
        )

    for line in (
        result.stderr
        .splitlines()
    ):
        logger.error(
            "PROVENANCE_GATE_STDERR | %s",
            line,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen Tech-1 provenance gate failed."
        )


# ======================================================================
# Determinism
# ======================================================================

def configure_determinism(
    cfg: dict[str, Any],
) -> int:

    seed = int(
        cfg[
            "seed"
        ]
    )

    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = str(
        cfg[
            "cublas_workspace_config"
        ]
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = bool(
        cfg[
            "cudnn_benchmark"
        ]
    )

    torch.backends.cudnn.deterministic = bool(
        cfg[
            "cudnn_deterministic"
        ]
    )

    torch.backends.cuda.matmul.allow_tf32 = bool(
        cfg[
            "allow_tf32_matmul"
        ]
    )

    torch.backends.cudnn.allow_tf32 = bool(
        cfg[
            "allow_tf32_cudnn"
        ]
    )

    torch.use_deterministic_algorithms(
        bool(
            cfg[
                "deterministic_algorithms"
            ]
        )
    )

    return seed


# ======================================================================
# ResNet / optimizer construction
# ======================================================================

def build_model(
    *,
    num_classes: int,
) -> nn.Module:

    weights = (
        ResNet18_Weights
        .IMAGENET1K_V1
    )

    model = resnet18(
        weights=weights
    )

    in_features = (
        model.fc
        .in_features
    )

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def batchnorm_parameter_ids(
    model: nn.Module,
) -> set[int]:

    result: set[
        int
    ] = set()

    for module in (
        model.modules()
    ):

        if isinstance(
            module,
            BATCHNORM_TYPES,
        ):

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


def build_optimizer(
    *,
    model: nn.Module,
    cfg: dict[str, Any],
) -> tuple[
    torch.optim.Optimizer,
    dict[str, int],
]:

    backbone_lr = float(
        cfg[
            "backbone_lr"
        ]
    )

    fc_lr = float(
        cfg[
            "fc_lr"
        ]
    )

    weight_decay = float(
        cfg[
            "weight_decay"
        ]
    )

    beta1 = float(
        cfg[
            "betas"
        ][0]
    )

    beta2 = float(
        cfg[
            "betas"
        ][1]
    )

    eps = float(
        cfg[
            "eps"
        ]
    )

    exclude_bias = bool(
        cfg[
            "exclude_bias_from_decay"
        ]
    )

    exclude_bn = bool(
        cfg[
            "exclude_batchnorm_from_decay"
        ]
    )

    bn_ids = (
        batchnorm_parameter_ids(
            model
        )
        if exclude_bn
        else set()
    )

    groups: dict[
        str,
        list[
            nn.Parameter
        ],
    ] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "fc_decay": [],
        "fc_no_decay": [],
    }

    seen_ids: set[
        int
    ] = set()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not parameter.requires_grad:
            continue

        parameter_id = id(
            parameter
        )

        if parameter_id in seen_ids:
            raise RuntimeError(
                "A trainable parameter appeared more than once "
                "while constructing optimizer groups."
            )

        seen_ids.add(
            parameter_id
        )

        is_fc = (
            name.startswith(
                "fc."
            )
        )

        is_bias = (
            name.endswith(
                ".bias"
            )
        )

        no_decay = (
            (
                exclude_bias
                and is_bias
            )
            or
            (
                exclude_bn
                and parameter_id
                in bn_ids
            )
        )

        if is_fc:

            key = (
                "fc_no_decay"
                if no_decay
                else "fc_decay"
            )

        else:

            key = (
                "backbone_no_decay"
                if no_decay
                else "backbone_decay"
            )

        groups[
            key
        ].append(
            parameter
        )

    expected_ids = {
        id(
            parameter
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    }

    if seen_ids != expected_ids:
        raise RuntimeError(
            "Optimizer grouping did not reconcile to all "
            "trainable model parameters."
        )

    parameter_groups = [
        {
            "name":
                "backbone_decay",

            "params":
                groups[
                    "backbone_decay"
                ],

            "lr":
                backbone_lr,

            "weight_decay":
                weight_decay,
        },
        {
            "name":
                "backbone_no_decay",

            "params":
                groups[
                    "backbone_no_decay"
                ],

            "lr":
                backbone_lr,

            "weight_decay":
                0.0,
        },
        {
            "name":
                "fc_decay",

            "params":
                groups[
                    "fc_decay"
                ],

            "lr":
                fc_lr,

            "weight_decay":
                weight_decay,
        },
        {
            "name":
                "fc_no_decay",

            "params":
                groups[
                    "fc_no_decay"
                ],

            "lr":
                fc_lr,

            "weight_decay":
                0.0,
        },
    ]

    for group in (
        parameter_groups
    ):

        if not group[
            "params"
        ]:
            raise RuntimeError(
                "Expected optimizer group is empty: "
                f"{group['name']}"
            )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(
            beta1,
            beta2,
        ),
        eps=eps,
        amsgrad=bool(
            cfg[
                "amsgrad"
            ]
        ),
        foreach=bool(
            cfg[
                "foreach"
            ]
        ),
        fused=bool(
            cfg[
                "fused"
            ]
        ),
    )

    group_counts = {
        key:
            sum(
                parameter.numel()
                for parameter
                in parameters
            )
        for (
            key,
            parameters,
        ) in groups.items()
    }

    return (
        optimizer,
        group_counts,
    )


# ======================================================================
# Result artifact
# ======================================================================

def write_result(
    *,
    path: Path,
    result: dict[str, Any],
) -> None:

    with path.open(
        "x",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            result,
            f,
            sort_keys=False,
        )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Probe FP32 ResNet-18 Stage-B training memory "
            "at batch 32 and 512x864."
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
        "--probe-config",
        default=(
            "tools/"
            "probe_resnet18_training_memory_config.yaml"
        ),
    )

    args = parser.parse_args()

    tool_cfg_path = resolve_repo_path(
        args.probe_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    # --------------------------------------------------------------
    # Check code provenance before generating output artifacts.
    # --------------------------------------------------------------

    commit_sha = (
        require_clean_git()
    )

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

    machine_id = str(
        machine_cfg[
            "machine"
        ][
            "id"
        ]
    )

    (
        logger,
        log_path,
        output_path,
    ) = configure_outputs(
        tool_cfg=tool_cfg,
        machine_id=machine_id,
    )

    try:

        probe_cfg = tool_cfg[
            "probe"
        ]

        determinism_cfg = (
            probe_cfg[
                "determinism"
            ]
        )

        seed = configure_determinism(
            determinism_cfg
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 FP32 TRAINING-MEMORY PROBE"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Git commit: %s",
            commit_sha,
        )

        logger.info(
            "Script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "Probe config SHA-256: %s",
            sha256_file(
                tool_cfg_path
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
            "No project_train images decoded."
        )

        logger.info(
            "dev_val: NOT ACCESSED"
        )

        logger.info(
            "held-out test: NOT ACCESSED"
        )

        run_frozen_provenance_gate(
            experiment_path=experiment_path,
            machine_path=machine_path,
            logger=logger,
        )

        logger.info(
            "[PASS] Frozen Tech-1 provenance gate"
        )

        # ----------------------------------------------------------
        # Validate probe contract.
        # ----------------------------------------------------------

        if probe_cfg[
            "architecture"
        ] != "resnet18":
            raise ValueError(
                "Memory probe currently supports only resnet18."
            )

        if probe_cfg[
            "weights"
        ] != "IMAGENET1K_V1":
            raise ValueError(
                "Memory probe is locked to IMAGENET1K_V1."
            )

        if probe_cfg[
            "dtype"
        ] != "float32":
            raise ValueError(
                "This probe is explicitly FP32."
            )

        if bool(
            probe_cfg[
                "amp"
            ]
        ):
            raise ValueError(
                "AMP must be false for the FP32 memory probe."
            )

        if not bool(
            probe_cfg[
                "full_backbone_trainable"
            ]
        ):
            raise ValueError(
                "Worst-case probe requires the full backbone "
                "to be trainable."
            )

        if probe_cfg[
            "model_mode"
        ] != "train":
            raise ValueError(
                "Worst-case probe requires model_mode=train."
            )

        device = torch.device(
            machine_cfg[
                "runtime"
            ][
                "device"
            ]
        )

        if device.type != "cuda":
            raise RuntimeError(
                "Training-memory probe requires CUDA."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

        torch.cuda.set_device(
            device_index
        )

        properties = (
            torch.cuda
            .get_device_properties(
                device_index
            )
        )

        free_before, total_memory = (
            torch.cuda.mem_get_info(
                device_index
            )
        )

        initial_free_fraction = (
            free_before
            / total_memory
        )

        minimum_free_fraction = float(
            probe_cfg[
                "minimum_initial_global_free_fraction"
            ]
        )

        logger.info(
            "GPU: %s",
            properties.name,
        )

        logger.info(
            "Compute capability: %d.%d",
            properties.major,
            properties.minor,
        )

        logger.info(
            "Total VRAM: %.3f GiB",
            gib(
                total_memory
            ),
        )

        logger.info(
            "Global free VRAM before model allocation: %.3f GiB "
            "(%.2f%%)",
            gib(
                free_before
            ),
            100.0
            * initial_free_fraction,
        )

        if (
            initial_free_fraction
            < minimum_free_fraction
        ):
            raise RuntimeError(
                "GPU does not have enough initially free VRAM "
                "for a clean capacity measurement:\n"
                f"  required_fraction={minimum_free_fraction:.3f}\n"
                f"  actual_fraction={initial_free_fraction:.3f}"
            )

        batch_size = int(
            probe_cfg[
                "batch_size"
            ]
        )

        channels = int(
            probe_cfg[
                "input_channels"
            ]
        )

        height = int(
            probe_cfg[
                "input_height"
            ]
        )

        width = int(
            probe_cfg[
                "input_width"
            ]
        )

        num_classes = int(
            probe_cfg[
                "num_classes"
            ]
        )

        training_steps = int(
            probe_cfg[
                "training_steps"
            ]
        )

        logger.info(
            "Probe tensor shape: (%d, %d, %d, %d)",
            batch_size,
            channels,
            height,
            width,
        )

        logger.info(
            "Input dtype: torch.float32"
        )

        logger.info(
            "AMP/autocast: DISABLED"
        )

        logger.info(
            "Training mode: FULL BACKBONE + FC"
        )

        logger.info(
            "BatchNorm mode: TRAIN"
        )

        logger.info(
            "Training steps: %d",
            training_steps,
        )

        logger.info(
            "Seed: %d",
            seed,
        )

        logger.info(
            "cuDNN benchmark: %s",
            torch.backends.cudnn.benchmark,
        )

        logger.info(
            "cuDNN deterministic: %s",
            torch.backends.cudnn.deterministic,
        )

        logger.info(
            "Deterministic algorithms: %s",
            torch.are_deterministic_algorithms_enabled(),
        )

        logger.info(
            "TF32 matmul allowed: %s",
            torch.backends.cuda.matmul.allow_tf32,
        )

        logger.info(
            "TF32 cuDNN allowed: %s",
            torch.backends.cudnn.allow_tf32,
        )

        logger.info(
            "CUBLAS_WORKSPACE_CONFIG: %s",
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        )

        # ----------------------------------------------------------
        # Synthetic batch.
        #
        # Generate on CPU first so CUDA memory measurement remains
        # explicit.
        # ----------------------------------------------------------

        synthetic_generator = (
            torch.Generator(
                device="cpu"
            )
        )

        synthetic_generator.manual_seed(
            seed
        )

        images_cpu = torch.randn(
            (
                batch_size,
                channels,
                height,
                width,
            ),
            dtype=torch.float32,
            generator=synthetic_generator,
        )

        # Ensure both binary classes occur in the synthetic batch.
        targets_cpu = (
            torch.arange(
                batch_size,
                dtype=torch.long,
            )
            % num_classes
        )

        # ----------------------------------------------------------
        # Model and optimizer.
        # ----------------------------------------------------------

        model = build_model(
            num_classes=num_classes
        )

        for parameter in (
            model.parameters()
        ):
            parameter.requires_grad = True

        model.train()

        model.to(
            device
        )

        optimizer_cfg = (
            probe_cfg[
                "optimizer"
            ]
        )

        optimizer, group_counts = (
            build_optimizer(
                model=model,
                cfg=optimizer_cfg,
            )
        )

        logger.info(
            "Optimizer: AdamW"
        )

        logger.info(
            "AdamW backbone LR: %.8g",
            float(
                optimizer_cfg[
                    "backbone_lr"
                ]
            ),
        )

        logger.info(
            "AdamW FC LR: %.8g",
            float(
                optimizer_cfg[
                    "fc_lr"
                ]
            ),
        )

        logger.info(
            "AdamW weight decay: %.8g",
            float(
                optimizer_cfg[
                    "weight_decay"
                ]
            ),
        )

        logger.info(
            "AdamW betas: (%s, %s)",
            optimizer_cfg[
                "betas"
            ][0],
            optimizer_cfg[
                "betas"
            ][1],
        )

        logger.info(
            "AdamW eps: %s",
            optimizer_cfg[
                "eps"
            ],
        )

        logger.info(
            "AdamW foreach: %s",
            optimizer_cfg[
                "foreach"
            ],
        )

        logger.info(
            "AdamW fused: %s",
            optimizer_cfg[
                "fused"
            ],
        )

        for (
            group_name,
            parameter_count,
        ) in group_counts.items():

            logger.info(
                "Optimizer group %-20s parameters=%d",
                group_name,
                parameter_count,
            )

        total_parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )

        if (
            trainable_parameters
            != total_parameters
        ):
            raise RuntimeError(
                "Full Stage-B probe expected every parameter "
                "to be trainable."
            )

        logger.info(
            "Total parameters: %d",
            total_parameters,
        )

        logger.info(
            "Trainable parameters: %d",
            trainable_parameters,
        )

        # ----------------------------------------------------------
        # Transfer synthetic batch to GPU.
        # ----------------------------------------------------------

        images = images_cpu.to(
            device
        )

        targets = targets_cpu.to(
            device
        )

        # Weighted-CE execution path.
        #
        # Unit weights are sufficient for a memory probe: the
        # two-element weight tensor exercises the same loss pathway
        # without asserting any class-index semantics here.
        loss_weights = torch.ones(
            num_classes,
            dtype=torch.float32,
            device=device,
        )

        del images_cpu
        del targets_cpu

        gc.collect()

        torch.cuda.empty_cache()

        torch.cuda.synchronize(
            device
        )

        allocated_before = (
            torch.cuda
            .memory_allocated(
                device
            )
        )

        reserved_before = (
            torch.cuda
            .memory_reserved(
                device
            )
        )

        torch.cuda.reset_peak_memory_stats(
            device
        )

        logger.info(
            "CUDA allocated before first training step: %.3f GiB",
            gib(
                allocated_before
            ),
        )

        logger.info(
            "CUDA reserved before first training step: %.3f GiB",
            gib(
                reserved_before
            ),
        )

        # ----------------------------------------------------------
        # Full training steps.
        # ----------------------------------------------------------

        losses: list[
            float
        ] = []

        start_time = (
            time.perf_counter()
        )

        for step in range(
            1,
            training_steps + 1,
        ):

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            if logits.shape != (
                batch_size,
                num_classes,
            ):
                raise RuntimeError(
                    "Unexpected classifier output shape: "
                    f"{tuple(logits.shape)}"
                )

            loss = F.cross_entropy(
                logits,
                targets,
                weight=loss_weights,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite synthetic training loss."
                )

            loss.backward()

            optimizer.step()

            torch.cuda.synchronize(
                device
            )

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                    .item()
                )
            )

            logger.info(
                "Step %d/%d | "
                "loss=%.6f | "
                "allocated=%.3f GiB | "
                "reserved=%.3f GiB | "
                "peak_allocated=%.3f GiB | "
                "peak_reserved=%.3f GiB",
                step,
                training_steps,
                losses[-1],
                gib(
                    torch.cuda
                    .memory_allocated(
                        device
                    )
                ),
                gib(
                    torch.cuda
                    .memory_reserved(
                        device
                    )
                ),
                gib(
                    torch.cuda
                    .max_memory_allocated(
                        device
                    )
                ),
                gib(
                    torch.cuda
                    .max_memory_reserved(
                        device
                    )
                ),
            )

        torch.cuda.synchronize(
            device
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        peak_allocated = (
            torch.cuda
            .max_memory_allocated(
                device
            )
        )

        peak_reserved = (
            torch.cuda
            .max_memory_reserved(
                device
            )
        )

        free_after, _ = (
            torch.cuda.mem_get_info(
                device_index
            )
        )

        peak_allocated_fraction = (
            peak_allocated
            / total_memory
        )

        peak_reserved_fraction = (
            peak_reserved
            / total_memory
        )

        reserved_headroom = (
            total_memory
            - peak_reserved
        )

        max_peak_reserved_fraction = float(
            probe_cfg[
                "max_peak_reserved_fraction"
            ]
        )

        capacity_pass = (
            peak_reserved_fraction
            <= max_peak_reserved_fraction
        )

        logger.info(
            "-" * 72
        )

        logger.info(
            "MEMORY RESULT"
        )

        logger.info(
            "Peak allocated: %.3f GiB (%.2f%% total VRAM)",
            gib(
                peak_allocated
            ),
            100.0
            * peak_allocated_fraction,
        )

        logger.info(
            "Peak reserved: %.3f GiB (%.2f%% total VRAM)",
            gib(
                peak_reserved
            ),
            100.0
            * peak_reserved_fraction,
        )

        logger.info(
            "Reserved-memory headroom to physical capacity: "
            "%.3f GiB",
            gib(
                reserved_headroom
            ),
        )

        logger.info(
            "Global free VRAM after probe: %.3f GiB",
            gib(
                free_after
            ),
        )

        logger.info(
            "Three-step elapsed time: %.3f s",
            elapsed,
        )

        logger.info(
            "Mean synthetic step time: %.3f s",
            (
                elapsed
                / training_steps
            ),
        )

        logger.info(
            "Capacity guard: peak_reserved_fraction <= %.2f",
            max_peak_reserved_fraction,
        )

        if not capacity_pass:
            raise RuntimeError(
                "Batch 32 executes but does not satisfy "
                "the predeclared VRAM safety margin:\n"
                f"  peak_reserved_fraction="
                f"{peak_reserved_fraction:.4f}\n"
                f"  maximum_allowed="
                f"{max_peak_reserved_fraction:.4f}"
            )

        # ----------------------------------------------------------
        # Persistent result artifact.
        # ----------------------------------------------------------

        result = {
            "schema_version": 1,

            "status":
                "PASS",

            "machine": {
                "id":
                    machine_id,

                "gpu":
                    properties.name,

                "compute_capability":
                    (
                        f"{properties.major}."
                        f"{properties.minor}"
                    ),

                "total_vram_gib":
                    gib(
                        total_memory
                    ),

                "initial_global_free_gib":
                    gib(
                        free_before
                    ),

                "initial_global_free_fraction":
                    initial_free_fraction,
            },

            "provenance": {
                "git_commit":
                    commit_sha,

                "script_sha256":
                    sha256_file(
                        Path(
                            __file__
                        ).resolve()
                    ),

                "probe_config_sha256":
                    sha256_file(
                        tool_cfg_path
                    ),

                "experiment_config_sha256":
                    sha256_file(
                        experiment_path
                    ),
            },

            "probe": {
                "architecture":
                    "resnet18",

                "weights":
                    "IMAGENET1K_V1",

                "batch_size":
                    batch_size,

                "input_shape_nchw": [
                    batch_size,
                    channels,
                    height,
                    width,
                ],

                "dtype":
                    "float32",

                "amp":
                    False,

                "model_mode":
                    "train",

                "full_backbone_trainable":
                    True,

                "training_steps":
                    training_steps,

                "optimizer":
                    "AdamW",

                "optimizer_foreach":
                    bool(
                        optimizer_cfg[
                            "foreach"
                        ]
                    ),

                "optimizer_fused":
                    bool(
                        optimizer_cfg[
                            "fused"
                        ]
                    ),
            },

            "memory": {
                "allocated_before_step_gib":
                    gib(
                        allocated_before
                    ),

                "reserved_before_step_gib":
                    gib(
                        reserved_before
                    ),

                "peak_allocated_gib":
                    gib(
                        peak_allocated
                    ),

                "peak_reserved_gib":
                    gib(
                        peak_reserved
                    ),

                "peak_allocated_fraction":
                    peak_allocated_fraction,

                "peak_reserved_fraction":
                    peak_reserved_fraction,

                "reserved_headroom_gib":
                    gib(
                        reserved_headroom
                    ),

                "max_allowed_peak_reserved_fraction":
                    max_peak_reserved_fraction,
            },

            "runtime": {
                "elapsed_seconds":
                    elapsed,

                "mean_step_seconds":
                    (
                        elapsed
                        / training_steps
                    ),

                "synthetic_losses":
                    losses,
            },

            "interpretation": (
                "PASS establishes that this machine can execute "
                "the proposed worst-case FP32 Stage-B tensor shape "
                "with the configured VRAM safety margin. "
                "It does not validate image preprocessing or "
                "classifier performance."
            ),
        }

        write_result(
            path=output_path,
            result=result,
        )

        logger.info(
            "Result YAML: %s",
            output_path,
        )

        logger.info(
            "Result YAML SHA-256: %s",
            sha256_file(
                output_path
            ),
        )

        logger.info(
            "[PASS] FP32 batch-32 512x864 full Stage-B "
            "training step satisfies VRAM guard."
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 FP32 TRAINING-MEMORY PROBE: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except torch.cuda.OutOfMemoryError:

        logger.exception(
            "CUDA OUT OF MEMORY during FP32 training-memory probe."
        )

        try:
            logger.error(
                "CUDA memory summary:\n%s",
                torch.cuda.memory_summary(),
            )
        except Exception:
            pass

        return 1

    except Exception:

        logger.exception(
            "RESNET-18 FP32 TRAINING-MEMORY PROBE: FAIL"
        )

        return 1

    finally:

        for handler in (
            logger.handlers
        ):
            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )