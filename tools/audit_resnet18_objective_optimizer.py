#!/usr/bin/env python3
"""
Audit the frozen Tech-2 weighted objective and AdamW contract.

Validated here
--------------
1. frozen experiment validator still passes;
2. Git is clean before evidence generation;
3. project_train class weights are independently re-derived;
4. dev_val does not participate in class-weight derivation;
5. explicit weighted CE agrees with PyTorch weighted CE;
6. weighted-CE gradients agree with the independent formulation;
7. epoch loss is numerator/denominator based, not mean-of-batch-means;
8. epoch reduction is invariant to tested batch partitioning;
9. Stage-A AdamW is classifier-only and initially stateless;
10. Stage-A AdamW hyperparameters match the frozen contract;
11. a real Stage-A optimizer.step() changes FC but not backbone;
12. Stage-A Adam state contains only FC weight/bias;
13. Stage-B optimizer is a genuinely fresh optimizer;
14. Stage-B optimizer starts with zero Adam state;
15. Stage-B group LRs and decay values are exact;
16. a real Stage-B optimizer.step() updates backbone and FC;
17. Stage-B Adam state covers every trainable parameter tensor;
18. Adam state structure is consistent with amsgrad=False;
19. repeating the complete two-stage probe with the same run seed
    reproduces model and optimizer-state fingerprints.

Synthetic deterministic inputs are used intentionally.

No Dataset is opened.
No project_train image is decoded.
No dev_val image is decoded.
Held-out test is not accessed.

This is not a training run and does not implement:
- epochs;
- stopping;
- checkpoint selection;
- AUROC;
- resolution selection;
- Grad-CAM.

All detailed evidence is written to ./logs/.
No print() is used.
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
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
    apply_stage_b_contract,
    build_resnet18_classifier,
)

from src.objective import (
    WeightedCrossEntropyObjective,
    WeightedLossAccumulator,
    derive_class_weight_contract,
)

from src.optimization import (
    build_stage_a_optimizer,
    build_stage_b_optimizer,
)

from src.reproducibility import (
    configure_run_reproducibility,
)


LOGGER = logging.getLogger(
    "audit_resnet18_objective_optimizer"
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
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
            f"  abs_error={abs(actual - expected):.17g}\n"
            f"  atol={atol:.17g}"
        )


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
            "Commit/remove outstanding files before "
            "running the objective/optimizer audit.\n\n"
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
# Tensor hashing
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


def model_parameter_sha256(
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
            "Parameter hash matched no parameters."
        )

    return digest.hexdigest()


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

    # ------------------------------------------------------------------
    # Hyperparameter groups.
    # ------------------------------------------------------------------

    for group in optimizer.param_groups:

        group_name = str(
            group.get(
                "name",
                ""
            )
        )

        digest.update(
            group_name.encode(
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
            "maximize",
            "capturable",
            "differentiable",
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

        digest.update(
            json.dumps(
                parameter_names,
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )

    # ------------------------------------------------------------------
    # Adam state by stable model parameter name.
    # ------------------------------------------------------------------

    named_state = sorted(
        (
            name_by_id[
                id(
                    parameter
                )
            ],
            parameter,
            state,
        )
        for (
            parameter,
            state,
        ) in optimizer.state.items()
    )

    for (
        parameter_name,
        _,
        state,
    ) in named_state:

        digest.update(
            parameter_name.encode(
                "utf-8"
            )
        )

        for state_key in sorted(
            state
        ):

            value = state[
                state_key
            ]

            digest.update(
                str(
                    state_key
                ).encode(
                    "utf-8"
                )
            )

            if isinstance(
                value,
                torch.Tensor,
            ):

                update_digest_with_tensor(
                    digest,
                    name=str(
                        state_key
                    ),
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
# Objective contract
# ======================================================================

def audit_class_weights(
    *,
    experiment_cfg: dict[str, Any],
    objective_cfg: dict[str, Any],
) -> dict[str, Any]:

    contract = derive_class_weight_contract(
        experiment_cfg
    )

    expected_counts = require_mapping(
        require_key(
            objective_cfg,
            "expected_project_train_counts",
            "audit.objective",
        ),
        "audit.objective.expected_project_train_counts",
    )

    actual_counts = {
        "bonafide":
            contract.bonafide_count,

        "attack":
            contract.attack_count,

        "total":
            contract.total_count,
    }

    normalized_expected_counts = {
        str(
            key
        ):
            int(
                value
            )
        for (
            key,
            value,
        ) in expected_counts.items()
    }

    if actual_counts != normalized_expected_counts:

        raise RuntimeError(
            "Derived project_train counts mismatch:\n"
            f"  expected={normalized_expected_counts}\n"
            f"  actual={actual_counts}"
        )

    expected_weights = require_mapping(
        require_key(
            objective_cfg,
            "expected_weights",
            "audit.objective",
        ),
        "audit.objective.expected_weights",
    )

    expected_bonafide_weight = float(
        require_key(
            expected_weights,
            "bonafide",
            "audit.objective.expected_weights",
        )
    )

    expected_attack_weight = float(
        require_key(
            expected_weights,
            "attack",
            "audit.objective.expected_weights",
        )
    )

    if contract.bonafide_weight != expected_bonafide_weight:

        raise RuntimeError(
            "Bonafide class weight mismatch."
        )

    if contract.attack_weight != expected_attack_weight:

        raise RuntimeError(
            "Attack class weight mismatch."
        )

    expected_train_denominator = float(
        require_key(
            objective_cfg,
            "expected_project_train_weight_denominator",
            "audit.objective",
        )
    )

    train_denominator = (
        contract.bonafide_count
        * contract.bonafide_weight
        +
        contract.attack_count
        * contract.attack_weight
    )

    if train_denominator != expected_train_denominator:

        raise RuntimeError(
            "project_train weighted denominator mismatch:\n"
            f"  expected={expected_train_denominator}\n"
            f"  actual={train_denominator}"
        )

    class_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "class_contract",
            "experiment_config",
        ),
        "class_contract",
    )

    dev_counts = require_mapping(
        require_key(
            class_cfg,
            "dev_val_counts",
            "class_contract",
        ),
        "class_contract.dev_val_counts",
    )

    dev_denominator = (
        int(
            require_key(
                dev_counts,
                "bonafide",
                "class_contract.dev_val_counts",
            )
        )
        * contract.bonafide_weight
        +
        int(
            require_key(
                dev_counts,
                "attack",
                "class_contract.dev_val_counts",
            )
        )
        * contract.attack_weight
    )

    expected_dev_denominator = float(
        require_key(
            objective_cfg,
            "expected_dev_val_weight_denominator",
            "audit.objective",
        )
    )

    if dev_denominator != expected_dev_denominator:

        raise RuntimeError(
            "dev_val weighted denominator mismatch:\n"
            f"  expected={expected_dev_denominator}\n"
            f"  actual={dev_denominator}"
        )

    # ------------------------------------------------------------------
    # Negative control:
    #
    # Deliberately alter only dev counts IN MEMORY. The derived weights
    # must remain unchanged, proving the implementation has no hidden
    # dependency on dev_val class composition.
    # ------------------------------------------------------------------

    modified_cfg = copy.deepcopy(
        experiment_cfg
    )

    modified_cfg[
        "class_contract"
    ][
        "dev_val_counts"
    ] = {
        "bonafide":
            1,

        "attack":
            1,

        "total":
            2,
    }

    modified_contract = (
        derive_class_weight_contract(
            modified_cfg
        )
    )

    if (
        modified_contract.bonafide_weight
        != contract.bonafide_weight
        or
        modified_contract.attack_weight
        != contract.attack_weight
    ):

        raise RuntimeError(
            "In-memory dev-count perturbation changed "
            "project_train-derived weights."
        )

    LOGGER.info(
        "[PASS] project_train class counts = %s",
        actual_counts,
    )

    LOGGER.info(
        "[PASS] class weights | bonafide=%.12g | attack=%.12g",
        contract.bonafide_weight,
        contract.attack_weight,
    )

    LOGGER.info(
        "[PASS] project_train weighted denominator = %.12g",
        train_denominator,
    )

    LOGGER.info(
        "[PASS] dev_val reuses project_train weights | denominator=%.12g",
        dev_denominator,
    )

    LOGGER.info(
        "[PASS] in-memory dev-count perturbation cannot change weights"
    )

    return {
        "project_train_counts":
            actual_counts,

        "weights":
            {
                "bonafide":
                    contract.bonafide_weight,

                "attack":
                    contract.attack_weight,
            },

        "project_train_weight_denominator":
            train_denominator,

        "dev_val_weight_denominator":
            dev_denominator,

        "dev_count_negative_control":
            "PASS",
    }


# ======================================================================
# Weighted CE algebra / gradient equivalence
# ======================================================================

def audit_batch_objective(
    *,
    experiment_cfg: dict[str, Any],
    loss_atol: float,
    gradient_atol: float,
) -> dict[str, Any]:

    device = torch.device(
        "cpu"
    )

    objective = WeightedCrossEntropyObjective(
        experiment_cfg=experiment_cfg,
        device=device,
    )

    logits_explicit = torch.tensor(
        [
            [1.25, -0.75],
            [-0.20, 0.80],
            [0.10, 0.20],
            [2.00, -1.00],
            [-1.25, 1.50],
            [0.75, 0.25],
            [-0.40, -0.10],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    targets = torch.tensor(
        [
            0,
            1,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    result = objective(
        logits_explicit,
        targets,
    )

    explicit_loss = float(
        result.loss.item()
    )

    result.loss.backward()

    explicit_gradient = (
        logits_explicit
        .grad
        .detach()
        .clone()
    )

    logits_reference = (
        logits_explicit
        .detach()
        .clone()
        .requires_grad_(
            True
        )
    )

    reference_weights = torch.tensor(
        [
            1.5,
            0.75,
        ],
        dtype=torch.float32,
    )

    reference_loss_tensor = F.cross_entropy(
        logits_reference,
        targets,
        weight=reference_weights,
        reduction="mean",
        label_smoothing=0.0,
    )

    reference_loss = float(
        reference_loss_tensor.item()
    )

    reference_loss_tensor.backward()

    reference_gradient = (
        logits_reference
        .grad
        .detach()
        .clone()
    )

    loss_error = abs(
        explicit_loss
        - reference_loss
    )

    gradient_error = float(
        (
            explicit_gradient
            - reference_gradient
        )
        .abs()
        .max()
        .item()
    )

    if loss_error > loss_atol:

        raise RuntimeError(
            "Explicit weighted CE disagrees with "
            "PyTorch weighted CrossEntropyLoss:\n"
            f"  explicit={explicit_loss:.12g}\n"
            f"  reference={reference_loss:.12g}\n"
            f"  abs_error={loss_error:.12g}"
        )

    if gradient_error > gradient_atol:

        raise RuntimeError(
            "Explicit weighted-CE gradient disagrees with "
            "PyTorch weighted CrossEntropyLoss:\n"
            f"  max_abs_error={gradient_error:.12g}"
        )

    # ------------------------------------------------------------------
    # Independently check numerator / denominator.
    # ------------------------------------------------------------------

    per_sample = F.cross_entropy(
        logits_explicit.detach(),
        targets,
        reduction="none",
        label_smoothing=0.0,
    )

    sample_weights = (
        reference_weights[
            targets
        ]
    )

    expected_numerator = float(
        (
            per_sample
            * sample_weights
        )
        .to(
            dtype=torch.float64
        )
        .sum()
        .item()
    )

    expected_denominator = float(
        sample_weights
        .to(
            dtype=torch.float64
        )
        .sum()
        .item()
    )

    require_close(
        actual=result.epoch_weighted_numerator,
        expected=expected_numerator,
        atol=1.0e-12,
        label="batch epoch numerator",
    )

    require_close(
        actual=result.epoch_weight_denominator,
        expected=expected_denominator,
        atol=1.0e-12,
        label="batch epoch denominator",
    )

    LOGGER.info(
        "[PASS] explicit weighted CE agrees with PyTorch reference"
    )

    LOGGER.info(
        "[PASS] weighted-CE gradient equivalence | max_abs_error=%.12g",
        gradient_error,
    )

    return {
        "explicit_loss":
            explicit_loss,

        "reference_loss":
            reference_loss,

        "loss_abs_error":
            loss_error,

        "gradient_max_abs_error":
            gradient_error,

        "epoch_weighted_numerator":
            result.epoch_weighted_numerator,

        "epoch_weight_denominator":
            result.epoch_weight_denominator,
    }


# ======================================================================
# Epoch reduction
# ======================================================================

def accumulate_partition(
    *,
    objective: WeightedCrossEntropyObjective,
    logits: torch.Tensor,
    targets: torch.Tensor,
    sizes: list[int],
) -> WeightedLossAccumulator:

    if sum(
        sizes
    ) != int(
        targets.numel()
    ):

        raise ValueError(
            "Partition sizes do not cover the synthetic dataset."
        )

    accumulator = WeightedLossAccumulator()

    offset = 0

    for size in sizes:

        if size <= 0:

            raise ValueError(
                "Partition sizes must be positive."
            )

        end = (
            offset
            + size
        )

        result = objective(
            logits[
                offset:end
            ],
            targets[
                offset:end
            ],
        )

        accumulator.update(
            result
        )

        offset = end

    return accumulator


def audit_epoch_reduction(
    *,
    experiment_cfg: dict[str, Any],
    atol: float,
) -> dict[str, Any]:

    objective = WeightedCrossEntropyObjective(
        experiment_cfg=experiment_cfg,
        device="cpu",
    )

    logits = torch.linspace(
        -2.25,
        2.75,
        steps=34,
        dtype=torch.float32,
    ).reshape(
        17,
        2,
    )

    targets = torch.tensor(
        [
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    class_weights = torch.tensor(
        [
            1.5,
            0.75,
        ],
        dtype=torch.float32,
    )

    per_sample = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    sample_weights = (
        class_weights[
            targets
        ]
    )

    expected_numerator = float(
        (
            per_sample
            * sample_weights
        )
        .to(
            dtype=torch.float64
        )
        .sum()
        .item()
    )

    expected_denominator = float(
        sample_weights
        .to(
            dtype=torch.float64
        )
        .sum()
        .item()
    )

    expected_value = (
        expected_numerator
        / expected_denominator
    )

    partitions = {
        "single_batch":
            [
                17
            ],

        "uneven_three_batches":
            [
                3,
                5,
                9,
            ],

        "mixed_five_batches":
            [
                2,
                7,
                1,
                4,
                3,
            ],

        "singleton_batches":
            [
                1
            ]
            * 17,
    }

    evidence: dict[
        str,
        Any,
    ] = {}

    for (
        name,
        sizes,
    ) in partitions.items():

        accumulator = accumulate_partition(
            objective=objective,
            logits=logits,
            targets=targets,
            sizes=sizes,
        )

        require_close(
            actual=accumulator.weighted_numerator,
            expected=expected_numerator,
            atol=atol,
            label=(
                f"{name} epoch numerator"
            ),
        )

        require_close(
            actual=accumulator.weight_denominator,
            expected=expected_denominator,
            atol=atol,
            label=(
                f"{name} epoch denominator"
            ),
        )

        require_close(
            actual=accumulator.value,
            expected=expected_value,
            atol=atol,
            label=(
                f"{name} epoch loss"
            ),
        )

        if accumulator.sample_count != 17:

            raise RuntimeError(
                f"{name} sample count is not 17."
            )

        evidence[
            name
        ] = {
            "sizes":
                sizes,

            "weighted_numerator":
                accumulator.weighted_numerator,

            "weight_denominator":
                accumulator.weight_denominator,

            "loss":
                accumulator.value,

            "sample_count":
                accumulator.sample_count,
        }

    LOGGER.info(
        "[PASS] dataset-level weighted loss agrees across "
        "tested batch partitions"
    )

    LOGGER.info(
        "[PASS] epoch loss is numerator/denominator based, "
        "not mean-of-batch-means"
    )

    return {
        "expected_weighted_numerator":
            expected_numerator,

        "expected_weight_denominator":
            expected_denominator,

        "expected_loss":
            expected_value,

        "partitions":
            evidence,
    }


# ======================================================================
# Optimizer structure
# ======================================================================

def validate_optimizer_structure(
    *,
    optimizer: torch.optim.AdamW,
    expected_groups: dict[
        str,
        tuple[
            float,
            float,
        ],
    ],
) -> dict[str, Any]:

    if not isinstance(
        optimizer,
        torch.optim.AdamW,
    ):

        raise RuntimeError(
            "Optimizer is not torch.optim.AdamW."
        )

    defaults = optimizer.defaults

    expected_defaults = {
        "lr":
            0.001,

        "betas":
            (
                0.9,
                0.999,
            ),

        "eps":
            1.0e-8,

        "weight_decay":
            1.0e-4,

        "amsgrad":
            False,

        "foreach":
            False,

        "maximize":
            False,

        "capturable":
            False,

        "differentiable":
            False,

        "fused":
            False,
    }

    for (
        key,
        expected,
    ) in expected_defaults.items():

        actual = defaults[
            key
        ]

        if actual != expected:

            raise RuntimeError(
                "AdamW default mismatch:\n"
                f"  key={key}\n"
                f"  expected={expected!r}\n"
                f"  actual={actual!r}"
            )

    actual_group_names = [
        str(
            group.get(
                "name"
            )
        )
        for group
        in optimizer.param_groups
    ]

    if (
        actual_group_names
        != list(
            expected_groups
        )
    ):

        raise RuntimeError(
            "AdamW group-name/order mismatch:\n"
            f"  expected={list(expected_groups)}\n"
            f"  actual={actual_group_names}"
        )

    evidence: dict[
        str,
        Any,
    ] = {}

    for group in optimizer.param_groups:

        name = str(
            group[
                "name"
            ]
        )

        expected_lr, expected_decay = (
            expected_groups[
                name
            ]
        )

        if float(
            group[
                "lr"
            ]
        ) != expected_lr:

            raise RuntimeError(
                "Optimizer group LR mismatch:\n"
                f"  group={name}"
            )

        if float(
            group[
                "weight_decay"
            ]
        ) != expected_decay:

            raise RuntimeError(
                "Optimizer group weight_decay mismatch:\n"
                f"  group={name}"
            )

        for (
            key,
            expected,
        ) in (
            (
                "betas",
                (
                    0.9,
                    0.999,
                ),
            ),
            (
                "eps",
                1.0e-8,
            ),
            (
                "amsgrad",
                False,
            ),
            (
                "foreach",
                False,
            ),
            (
                "fused",
                False,
            ),
            (
                "maximize",
                False,
            ),
            (
                "capturable",
                False,
            ),
            (
                "differentiable",
                False,
            ),
        ):

            if group[
                key
            ] != expected:

                raise RuntimeError(
                    "Optimizer group inherited unexpected "
                    "AdamW setting:\n"
                    f"  group={name}\n"
                    f"  key={key}\n"
                    f"  expected={expected!r}\n"
                    f"  actual={group[key]!r}"
                )

        evidence[
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
        }

    return {
        "defaults":
            {
                key:
                    (
                        list(
                            value
                        )
                        if isinstance(
                            value,
                            tuple,
                        )
                        else value
                    )

                for (
                    key,
                    value,
                ) in expected_defaults.items()
            },

        "groups":
            evidence,
    }


# ======================================================================
# Adam-state validation
# ======================================================================

def validate_adam_state_after_first_step(
    *,
    optimizer: torch.optim.AdamW,
    expected_entries: int,
) -> dict[str, Any]:

    if len(
        optimizer.state
    ) != expected_entries:

        raise RuntimeError(
            "AdamW state-entry count after first step mismatch:\n"
            f"  expected={expected_entries}\n"
            f"  actual={len(optimizer.state)}"
        )

    for (
        parameter,
        state,
    ) in optimizer.state.items():

        expected_keys = {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }

        if set(
            state
        ) != expected_keys:

            raise RuntimeError(
                "Unexpected AdamW state structure:\n"
                f"  expected={sorted(expected_keys)}\n"
                f"  actual={sorted(state)}"
            )

        step = state[
            "step"
        ]

        if not isinstance(
            step,
            torch.Tensor,
        ):

            raise RuntimeError(
                "AdamW step state is unexpectedly not a tensor."
            )

        if float(
            step.item()
        ) != 1.0:

            raise RuntimeError(
                "AdamW first-step counter is not 1."
            )

        for state_name in (
            "exp_avg",
            "exp_avg_sq",
        ):

            value = state[
                state_name
            ]

            if value.shape != parameter.shape:

                raise RuntimeError(
                    "AdamW state shape mismatch:\n"
                    f"  state={state_name}\n"
                    f"  parameter_shape={tuple(parameter.shape)}\n"
                    f"  state_shape={tuple(value.shape)}"
                )

            if value.dtype != parameter.dtype:

                raise RuntimeError(
                    "AdamW state dtype mismatch."
                )

            if value.device != parameter.device:

                raise RuntimeError(
                    "AdamW state device mismatch."
                )

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

        "state_keys":
            [
                "step",
                "exp_avg",
                "exp_avg_sq",
            ],

        "amsgrad_extra_state":
            False,
    }


# ======================================================================
# Synthetic model input
# ======================================================================

def make_probe_input(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Structured deterministic input that consumes no RNG state.
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

    image = torch.cat(
        [
            base,
            (
                0.5
                * base
                + 0.1
            ),
            (
                -0.25
                * base
                - 0.2
            ),
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
# Complete Stage-A -> Stage-B optimizer probe
# ======================================================================

def run_two_stage_probe(
    *,
    experiment_cfg: dict[str, Any],
    device_string: str,
    seed: int,
    resolution_name: str,
    batch_size: int,
    backbone_lr: float,
    expected_stage_a_state_entries: int,
    expected_stage_b_state_entries: int,
) -> dict[str, Any]:

    configure_run_reproducibility(
        experiment_cfg=experiment_cfg,
        run_seed=seed,
    )

    model, _ = build_resnet18_classifier(
        experiment_cfg=experiment_cfg,
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

    inputs = make_probe_input(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
    )

    # With batch_size=2 this gives one bonafide and one attack.
    targets = torch.tensor(
        [
            index % 2
            for index
            in range(
                batch_size
            )
        ],
        dtype=torch.int64,
        device=device,
    )

    objective = WeightedCrossEntropyObjective(
        experiment_cfg=experiment_cfg,
        device=device,
    )

    # ==================================================================
    # Stage A
    # ==================================================================

    apply_stage_a_contract(
        model
    )

    stage_a_optimizer, _ = (
        build_stage_a_optimizer(
            model=model,
            experiment_cfg=experiment_cfg,
        )
    )

    if len(
        stage_a_optimizer.state
    ) != 0:

        raise RuntimeError(
            "Fresh Stage-A AdamW is not stateless."
        )

    stage_a_structure = (
        validate_optimizer_structure(
            optimizer=stage_a_optimizer,
            expected_groups={
                "fc_decay":
                    (
                        0.001,
                        0.0001,
                    ),

                "fc_no_decay":
                    (
                        0.001,
                        0.0,
                    ),
            },
        )
    )

    if sum(
        group[
            "parameter_tensors"
        ]
        for group
        in stage_a_structure[
            "groups"
        ].values()
    ) != 2:

        raise RuntimeError(
            "Stage-A optimizer does not contain exactly "
            "fc.weight and fc.bias."
        )

    stage_a_backbone_before = (
        model_parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=False,
        )
    )

    stage_a_fc_before = (
        model_parameter_sha256(
            model,
            include_backbone=False,
            include_classifier=True,
        )
    )

    stage_a_optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        inputs
    )

    stage_a_objective = objective(
        logits,
        targets,
    )

    stage_a_loss = float(
        stage_a_objective
        .loss
        .detach()
        .item()
    )

    stage_a_objective.loss.backward()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if name.startswith(
            "fc."
        ):

            if parameter.grad is None:

                raise RuntimeError(
                    "Stage-A FC parameter has no gradient:\n"
                    f"  {name}"
                )

        else:

            if parameter.grad is not None:

                raise RuntimeError(
                    "Frozen Stage-A backbone received a gradient:\n"
                    f"  {name}"
                )

    stage_a_optimizer.step()

    stage_a_backbone_after = (
        model_parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=False,
        )
    )

    stage_a_fc_after = (
        model_parameter_sha256(
            model,
            include_backbone=False,
            include_classifier=True,
        )
    )

    if (
        stage_a_backbone_after
        != stage_a_backbone_before
    ):

        raise RuntimeError(
            "Stage-A AdamW step changed backbone parameters."
        )

    if (
        stage_a_fc_after
        == stage_a_fc_before
    ):

        raise RuntimeError(
            "Stage-A AdamW step did not change FC parameters."
        )

    stage_a_state_evidence = (
        validate_adam_state_after_first_step(
            optimizer=stage_a_optimizer,
            expected_entries=(
                expected_stage_a_state_entries
            ),
        )
    )

    stage_a_optimizer_hash = (
        optimizer_state_sha256(
            optimizer=stage_a_optimizer,
            model=model,
        )
    )

    LOGGER.info(
        "[PASS] Stage-A AdamW step | loss=%.12g | state_entries=%d",
        stage_a_loss,
        len(
            stage_a_optimizer.state
        ),
    )

    LOGGER.info(
        "[PASS] Stage-A optimizer changes FC and leaves backbone unchanged"
    )

    # Clear Stage-A gradients before switching scientific stages.
    model.zero_grad(
        set_to_none=True
    )

    # ==================================================================
    # Stage B
    # ==================================================================

    apply_stage_b_contract(
        model
    )

    stage_b_optimizer, _ = (
        build_stage_b_optimizer(
            model=model,
            experiment_cfg=experiment_cfg,
            backbone_lr=backbone_lr,
        )
    )

    if stage_b_optimizer is stage_a_optimizer:

        raise RuntimeError(
            "Stage B reused the Stage-A optimizer object."
        )

    if len(
        stage_b_optimizer.state
    ) != 0:

        raise RuntimeError(
            "Fresh Stage-B AdamW inherited optimizer state."
        )

    if (
        len(
            stage_a_optimizer.state
        )
        != expected_stage_a_state_entries
    ):

        raise RuntimeError(
            "Constructing Stage-B optimizer altered "
            "Stage-A optimizer state."
        )

    stage_b_structure = (
        validate_optimizer_structure(
            optimizer=stage_b_optimizer,
            expected_groups={
                "backbone_decay":
                    (
                        backbone_lr,
                        0.0001,
                    ),

                "backbone_no_decay":
                    (
                        backbone_lr,
                        0.0,
                    ),

                "fc_decay":
                    (
                        0.001,
                        0.0001,
                    ),

                "fc_no_decay":
                    (
                        0.001,
                        0.0,
                    ),
            },
        )
    )

    stage_b_tensor_count = sum(
        group[
            "parameter_tensors"
        ]
        for group
        in stage_b_structure[
            "groups"
        ].values()
    )

    if (
        stage_b_tensor_count
        != expected_stage_b_state_entries
    ):

        raise RuntimeError(
            "Stage-B optimizer trainable parameter-tensor count "
            "does not match expected Adam-state coverage:\n"
            f"  expected={expected_stage_b_state_entries}\n"
            f"  actual={stage_b_tensor_count}"
        )

    stage_b_backbone_before = (
        model_parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=False,
        )
    )

    stage_b_fc_before = (
        model_parameter_sha256(
            model,
            include_backbone=False,
            include_classifier=True,
        )
    )

    stage_b_optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        inputs
    )

    stage_b_objective = objective(
        logits,
        targets,
    )

    stage_b_loss = float(
        stage_b_objective
        .loss
        .detach()
        .item()
    )

    stage_b_objective.loss.backward()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if parameter.grad is None:

            raise RuntimeError(
                "Stage-B trainable parameter has no gradient:\n"
                f"  {name}"
            )

        if not bool(
            torch.isfinite(
                parameter.grad
            ).all()
        ):

            raise RuntimeError(
                "Stage-B gradient contains NaN/Inf:\n"
                f"  {name}"
            )

    stage_b_optimizer.step()

    stage_b_backbone_after = (
        model_parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=False,
        )
    )

    stage_b_fc_after = (
        model_parameter_sha256(
            model,
            include_backbone=False,
            include_classifier=True,
        )
    )

    if (
        stage_b_backbone_after
        == stage_b_backbone_before
    ):

        raise RuntimeError(
            "Stage-B AdamW step did not change backbone parameters."
        )

    if (
        stage_b_fc_after
        == stage_b_fc_before
    ):

        raise RuntimeError(
            "Stage-B AdamW step did not change FC parameters."
        )

    stage_b_state_evidence = (
        validate_adam_state_after_first_step(
            optimizer=stage_b_optimizer,
            expected_entries=(
                expected_stage_b_state_entries
            ),
        )
    )

    stage_b_optimizer_hash = (
        optimizer_state_sha256(
            optimizer=stage_b_optimizer,
            model=model,
        )
    )

    full_model_parameter_hash = (
        model_parameter_sha256(
            model,
            include_backbone=True,
            include_classifier=True,
        )
    )

    LOGGER.info(
        "[PASS] Stage-B AdamW step | loss=%.12g | state_entries=%d",
        stage_b_loss,
        len(
            stage_b_optimizer.state
        ),
    )

    LOGGER.info(
        "[PASS] Stage-B optimizer is fresh and does not inherit "
        "Stage-A Adam moments"
    )

    LOGGER.info(
        "[PASS] Stage-B optimizer updates backbone and classifier"
    )

    result = {
        "input_shape":
            [
                batch_size,
                3,
                height,
                width,
            ],

        "stage_a":
            {
                "loss":
                    stage_a_loss,

                "optimizer_structure":
                    stage_a_structure,

                "optimizer_state":
                    stage_a_state_evidence,

                "backbone_before_sha256":
                    stage_a_backbone_before,

                "backbone_after_sha256":
                    stage_a_backbone_after,

                "fc_before_sha256":
                    stage_a_fc_before,

                "fc_after_sha256":
                    stage_a_fc_after,

                "optimizer_state_sha256":
                    stage_a_optimizer_hash,
            },

        "stage_b":
            {
                "loss":
                    stage_b_loss,

                "backbone_lr":
                    backbone_lr,

                "optimizer_structure":
                    stage_b_structure,

                "optimizer_state":
                    stage_b_state_evidence,

                "backbone_before_sha256":
                    stage_b_backbone_before,

                "backbone_after_sha256":
                    stage_b_backbone_after,

                "fc_before_sha256":
                    stage_b_fc_before,

                "fc_after_sha256":
                    stage_b_fc_after,

                "optimizer_state_sha256":
                    stage_b_optimizer_hash,
            },

        "final_model_parameters_sha256":
            full_model_parameter_hash,
    }

    del stage_a_optimizer
    del stage_b_optimizer
    del objective
    del inputs
    del model

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return result


# ======================================================================
# Repeated full-sequence determinism
# ======================================================================

def audit_optimizer_determinism(
    *,
    experiment_cfg: dict[str, Any],
    device_string: str,
    seed: int,
    resolution_name: str,
    batch_size: int,
    backbone_lr: float,
    repeats: int,
    expected_stage_a_state_entries: int,
    expected_stage_b_state_entries: int,
) -> dict[str, Any]:

    results: list[
        dict[str, Any]
    ] = []

    for repeat_index in range(
        repeats
    ):

        LOGGER.info(
            "Two-stage optimizer probe repeat %d / %d",
            repeat_index
            + 1,
            repeats,
        )

        result = run_two_stage_probe(
            experiment_cfg=experiment_cfg,
            device_string=device_string,
            seed=seed,
            resolution_name=resolution_name,
            batch_size=batch_size,
            backbone_lr=backbone_lr,
            expected_stage_a_state_entries=(
                expected_stage_a_state_entries
            ),
            expected_stage_b_state_entries=(
                expected_stage_b_state_entries
            ),
        )

        results.append(
            result
        )

    first = results[
        0
    ]

    comparison_keys = {
        "stage_a_fc_after_sha256":
            first[
                "stage_a"
            ][
                "fc_after_sha256"
            ],

        "stage_a_optimizer_state_sha256":
            first[
                "stage_a"
            ][
                "optimizer_state_sha256"
            ],

        "stage_b_backbone_after_sha256":
            first[
                "stage_b"
            ][
                "backbone_after_sha256"
            ],

        "stage_b_fc_after_sha256":
            first[
                "stage_b"
            ][
                "fc_after_sha256"
            ],

        "stage_b_optimizer_state_sha256":
            first[
                "stage_b"
            ][
                "optimizer_state_sha256"
            ],

        "final_model_parameters_sha256":
            first[
                "final_model_parameters_sha256"
            ],
    }

    for repeat_index, result in enumerate(
        results[
            1:
        ],
        start=2,
    ):

        actual = {
            "stage_a_fc_after_sha256":
                result[
                    "stage_a"
                ][
                    "fc_after_sha256"
                ],

            "stage_a_optimizer_state_sha256":
                result[
                    "stage_a"
                ][
                    "optimizer_state_sha256"
                ],

            "stage_b_backbone_after_sha256":
                result[
                    "stage_b"
                ][
                    "backbone_after_sha256"
                ],

            "stage_b_fc_after_sha256":
                result[
                    "stage_b"
                ][
                    "fc_after_sha256"
                ],

            "stage_b_optimizer_state_sha256":
                result[
                    "stage_b"
                ][
                    "optimizer_state_sha256"
                ],

            "final_model_parameters_sha256":
                result[
                    "final_model_parameters_sha256"
                ],
        }

        if actual != comparison_keys:

            raise RuntimeError(
                "Repeated same-seed two-stage optimizer sequence "
                "was not deterministic:\n"
                f"  repeat={repeat_index}\n"
                f"  expected={comparison_keys}\n"
                f"  actual={actual}"
            )

    LOGGER.info(
        "[PASS] repeated same-seed Stage-A -> Stage-B "
        "optimizer sequence is deterministic"
    )

    return {
        "repeats":
            repeats,

        "deterministic_fingerprints":
            comparison_keys,

        "reference_run":
            first,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen weighted objective and AdamW contract."
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
            "audit_resnet18_objective_optimizer_config.yaml"
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
            "Objective/optimizer audit schema_version must equal 1."
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

        objective_cfg = require_mapping(
            require_key(
                audit_cfg,
                "objective",
                "audit_config.audit",
            ),
            "audit_config.audit.objective",
        )

        optimizer_probe_cfg = require_mapping(
            require_key(
                audit_cfg,
                "optimizer_probe",
                "audit_config.audit",
            ),
            "audit_config.audit.optimizer_probe",
        )

        loss_atol = float(
            require_key(
                objective_cfg,
                "loss_abs_tolerance",
                "audit.objective",
            )
        )

        gradient_atol = float(
            require_key(
                objective_cfg,
                "gradient_abs_tolerance",
                "audit.objective",
            )
        )

        epoch_atol = float(
            require_key(
                objective_cfg,
                "epoch_abs_tolerance",
                "audit.objective",
            )
        )

        seed = int(
            require_key(
                optimizer_probe_cfg,
                "seed",
                "audit.optimizer_probe",
            )
        )

        resolution_name = str(
            require_key(
                optimizer_probe_cfg,
                "resolution",
                "audit.optimizer_probe",
            )
        )

        batch_size = int(
            require_key(
                optimizer_probe_cfg,
                "batch_size",
                "audit.optimizer_probe",
            )
        )

        backbone_lr = float(
            require_key(
                optimizer_probe_cfg,
                "stage_b_backbone_lr",
                "audit.optimizer_probe",
            )
        )

        repeats = int(
            require_key(
                optimizer_probe_cfg,
                "repeated_full_sequences",
                "audit.optimizer_probe",
            )
        )

        require_cuda = bool(
            require_key(
                optimizer_probe_cfg,
                "require_cuda",
                "audit.optimizer_probe",
            )
        )

        expected_stage_a_state_entries = int(
            require_key(
                optimizer_probe_cfg,
                "expected_stage_a_state_entries_after_step",
                "audit.optimizer_probe",
            )
        )

        expected_stage_b_state_entries = int(
            require_key(
                optimizer_probe_cfg,
                "expected_stage_b_state_entries_after_step",
                "audit.optimizer_probe",
            )
        )

        # --------------------------------------------------------------
        # Audit config itself is intentionally strict.
        # --------------------------------------------------------------

        if seed != 8:

            raise ValueError(
                "Optimizer probe must use screening seed 8."
            )

        if resolution_name != "r256":

            raise ValueError(
                "Optimizer probe must currently use r256."
            )

        if batch_size != 2:

            raise ValueError(
                "Optimizer probe batch size must equal 2."
            )

        if backbone_lr != 0.0001:

            raise ValueError(
                "Optimizer probe Stage-B backbone LR must equal 1e-4."
            )

        if repeats != 2:

            raise ValueError(
                "Optimizer probe must repeat the full sequence twice."
            )

        if expected_stage_a_state_entries != 2:

            raise ValueError(
                "Stage-A expected Adam-state entry count must be 2."
            )

        if expected_stage_b_state_entries != 62:

            raise ValueError(
                "Stage-B expected Adam-state entry count must be 62."
            )

        if (
            loss_atol <= 0.0
            or gradient_atol <= 0.0
            or epoch_atol <= 0.0
        ):

            raise ValueError(
                "Audit tolerances must be positive."
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
                "Objective/optimizer audit requires "
                "protocol_status=frozen."
            )

        if require_cuda:

            if not device_string.startswith(
                "cuda"
            ):

                raise RuntimeError(
                    "CUDA optimizer probe required but "
                    f"machine device is {device_string!r}."
                )

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "CUDA optimizer probe required but "
                    "torch.cuda.is_available() is False."
                )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 WEIGHTED OBJECTIVE + ADAMW CONTRACT AUDIT"
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
            "src/objective.py",
            "src/optimization.py",
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
            "Dataset access: NONE"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # ==============================================================
        # Objective algebra
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Frozen class-weight contract ---"
        )

        class_weight_results = (
            audit_class_weights(
                experiment_cfg=experiment_cfg,
                objective_cfg=objective_cfg,
            )
        )

        logger.info(
            ""
        )

        logger.info(
            "--- Weighted CE algebra / gradients ---"
        )

        batch_objective_results = (
            audit_batch_objective(
                experiment_cfg=experiment_cfg,
                loss_atol=loss_atol,
                gradient_atol=gradient_atol,
            )
        )

        logger.info(
            ""
        )

        logger.info(
            "--- Dataset-level epoch reduction algebra ---"
        )

        epoch_reduction_results = (
            audit_epoch_reduction(
                experiment_cfg=experiment_cfg,
                atol=epoch_atol,
            )
        )

        # ==============================================================
        # Real deterministic CUDA optimizer steps
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Real Stage-A -> Stage-B AdamW probe ---"
        )

        optimizer_results = (
            audit_optimizer_determinism(
                experiment_cfg=experiment_cfg,
                device_string=device_string,
                seed=seed,
                resolution_name=resolution_name,
                batch_size=batch_size,
                backbone_lr=backbone_lr,
                repeats=repeats,
                expected_stage_a_state_entries=(
                    expected_stage_a_state_entries
                ),
                expected_stage_b_state_entries=(
                    expected_stage_b_state_entries
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

                    "objective_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "objective.py"
                        ),

                    "optimization_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "optimization.py"
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

            "class_weight_contract":
                class_weight_results,

            "batch_objective":
                batch_objective_results,

            "epoch_reduction":
                epoch_reduction_results,

            "optimizer_probe":
                optimizer_results,

            "interpretation":
                (
                    "PASS establishes that the frozen "
                    "project_train-derived weighted cross-entropy "
                    "algebra, dataset-level epoch reduction, "
                    "Stage-A classifier-only AdamW, fresh Stage-B "
                    "AdamW, optimizer hyperparameters, first-step "
                    "Adam state, and deterministic same-seed "
                    "optimizer behavior match the frozen Tech-2 "
                    "development protocol. This audit is not a "
                    "training run and accesses no dataset images."
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
            "[PASS] class weights independently derived "
            "from project_train only"
        )

        logger.info(
            "[PASS] bonafide=1.5 / attack=0.75"
        )

        logger.info(
            "[PASS] explicit weighted CE algebra"
        )

        logger.info(
            "[PASS] explicit weighted CE gradient algebra"
        )

        logger.info(
            "[PASS] dataset-level epoch reduction"
        )

        logger.info(
            "[PASS] tested epoch reduction invariant to "
            "batch partitioning"
        )

        logger.info(
            "[PASS] exact AdamW defaults and group hyperparameters"
        )

        logger.info(
            "[PASS] Stage-A optimizer starts fresh"
        )

        logger.info(
            "[PASS] Stage-A first step changes FC only"
        )

        logger.info(
            "[PASS] Stage-A Adam state = 2 parameter entries"
        )

        logger.info(
            "[PASS] Stage-B optimizer starts fresh after Stage A"
        )

        logger.info(
            "[PASS] Stage-B first step changes backbone + FC"
        )

        logger.info(
            "[PASS] Stage-B Adam state = 62 parameter entries"
        )

        logger.info(
            "[PASS] amsgrad=false state structure"
        )

        logger.info(
            "[PASS] repeated same-seed two-stage optimizer "
            "sequence deterministic"
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
            "RESNET-18 WEIGHTED OBJECTIVE + ADAMW CONTRACT AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 WEIGHTED OBJECTIVE + ADAMW CONTRACT AUDIT: FAIL"
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