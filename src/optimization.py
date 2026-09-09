"""
Frozen AdamW construction for Tech-2 ResNet-18 transfer learning.

This module consumes the already-validated parameter groups from
src.modeling and constructs fresh AdamW instances.

Stage A
-------
- classifier only
- LR = 1e-3
- fresh optimizer

Stage B
-------
- complete backbone + classifier
- backbone LR in {3e-5, 1e-4, 3e-4}
- classifier LR = 1e-3
- fresh optimizer

AdamW
-----
betas       = (0.9, 0.999)
eps         = 1e-8
weight_decay= 1e-4 for decay groups
amsgrad     = False
foreach     = False
fused       = False
schedule    = constant

Bias and BatchNorm affine exclusions are already encoded in the
parameter groups constructed by src.modeling.

No scheduler is constructed because the frozen learning-rate policy is
constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from src.modeling import (
    ParameterGroupSummary,
    assert_stage_a_contract,
    assert_stage_b_contract,
    build_stage_a_parameter_groups,
    build_stage_b_parameter_groups,
)


# ======================================================================
# Config helpers
# ======================================================================

def require_mapping(
    value: Any,
    label: str,
) -> Mapping[str, Any]:

    if not isinstance(
        value,
        Mapping,
    ):

        raise TypeError(
            f"{label} must be a mapping, "
            f"got {type(value).__name__}: {value!r}"
        )

    return value


def require_key(
    mapping: Mapping[str, Any],
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
# Frozen optimizer contract
# ======================================================================

@dataclass(
    frozen=True
)
class AdamWContract:

    betas: tuple[
        float,
        float,
    ]

    eps: float

    weight_decay: float

    amsgrad: bool
    foreach: bool
    fused: bool

    learning_rate_schedule: str


def load_adamw_contract(
    experiment_cfg: Mapping[str, Any],
) -> AdamWContract:

    optimizer_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "optimizer",
            "experiment_config",
        ),
        "optimizer",
    )

    if (
        require_key(
            optimizer_cfg,
            "name",
            "optimizer",
        )
        != "AdamW"
    ):

        raise ValueError(
            "Frozen optimizer must be AdamW."
        )

    betas_raw = list(
        require_key(
            optimizer_cfg,
            "betas",
            "optimizer",
        )
    )

    if len(
        betas_raw
    ) != 2:

        raise ValueError(
            "AdamW betas must contain exactly two values."
        )

    betas = (
        float(
            betas_raw[
                0
            ]
        ),
        float(
            betas_raw[
                1
            ]
        ),
    )

    if betas != (
        0.9,
        0.999,
    ):

        raise ValueError(
            "Frozen AdamW betas must equal (0.9, 0.999)."
        )

    eps = float(
        require_key(
            optimizer_cfg,
            "eps",
            "optimizer",
        )
    )

    if not math.isclose(
        eps,
        1.0e-8,
        rel_tol=0.0,
        abs_tol=0.0,
    ):

        raise ValueError(
            "Frozen AdamW eps must equal 1e-8."
        )

    weight_decay = float(
        require_key(
            optimizer_cfg,
            "weight_decay",
            "optimizer",
        )
    )

    if not math.isclose(
        weight_decay,
        1.0e-4,
        rel_tol=0.0,
        abs_tol=0.0,
    ):

        raise ValueError(
            "Frozen AdamW weight_decay must equal 1e-4."
        )

    amsgrad = require_key(
        optimizer_cfg,
        "amsgrad",
        "optimizer",
    )

    foreach = require_key(
        optimizer_cfg,
        "foreach",
        "optimizer",
    )

    fused = require_key(
        optimizer_cfg,
        "fused",
        "optimizer",
    )

    if amsgrad is not False:

        raise ValueError(
            "Frozen AdamW requires amsgrad=false."
        )

    if foreach is not False:

        raise ValueError(
            "Frozen AdamW requires foreach=false."
        )

    if fused is not False:

        raise ValueError(
            "Frozen AdamW requires fused=false."
        )

    schedule_cfg = require_mapping(
        require_key(
            optimizer_cfg,
            "learning_rate_schedule",
            "optimizer",
        ),
        "optimizer.learning_rate_schedule",
    )

    schedule_type = str(
        require_key(
            schedule_cfg,
            "type",
            "optimizer.learning_rate_schedule",
        )
    )

    if schedule_type != "constant":

        raise ValueError(
            "Frozen learning-rate schedule must be constant."
        )

    return AdamWContract(
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=False,
        foreach=False,
        fused=False,
        learning_rate_schedule=(
            schedule_type
        ),
    )


# ======================================================================
# Optimizer build evidence
# ======================================================================

@dataclass(
    frozen=True
)
class OptimizerBuildEvidence:

    stage: str

    optimizer_class: str

    state_entries_at_construction: int

    group_summaries: tuple[
        ParameterGroupSummary,
        ...,
    ]


# ======================================================================
# Internal constructor
# ======================================================================

def _build_adamw(
    *,
    parameter_groups: list[
        dict[str, Any]
    ],
    group_summaries: tuple[
        ParameterGroupSummary,
        ...,
    ],
    contract: AdamWContract,
    default_lr: float,
    stage: str,
) -> tuple[
    torch.optim.AdamW,
    OptimizerBuildEvidence,
]:
    """
    Construct an explicitly configured AdamW.

    Every group already has its own LR and weight_decay. The top-level
    values are still supplied explicitly so no PyTorch default is relied
    upon if a future group were ever added.
    """

    if not parameter_groups:

        raise RuntimeError(
            "Cannot construct AdamW with no parameter groups."
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(
            default_lr
        ),
        betas=contract.betas,
        eps=contract.eps,
        weight_decay=contract.weight_decay,
        amsgrad=contract.amsgrad,
        foreach=contract.foreach,
        fused=contract.fused,

        # Explicit implementation-stability defaults.
        maximize=False,
        capturable=False,
        differentiable=False,
    )

    if len(
        optimizer.state
    ) != 0:

        raise RuntimeError(
            "Fresh AdamW unexpectedly contains optimizer state."
        )

    evidence = OptimizerBuildEvidence(
        stage=stage,
        optimizer_class=(
            type(
                optimizer
            ).__name__
        ),
        state_entries_at_construction=0,
        group_summaries=group_summaries,
    )

    return (
        optimizer,
        evidence,
    )


# ======================================================================
# Stage A
# ======================================================================

def build_stage_a_optimizer(
    *,
    model: nn.Module,
    experiment_cfg: Mapping[str, Any],
) -> tuple[
    torch.optim.AdamW,
    OptimizerBuildEvidence,
]:
    """
    Construct a NEW classifier-only Stage-A AdamW.
    """

    assert_stage_a_contract(
        model
    )

    transfer_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "transfer_learning",
            "experiment_config",
        ),
        "transfer_learning",
    )

    stage_a_cfg = require_mapping(
        require_key(
            transfer_cfg,
            "stage_a",
            "transfer_learning",
        ),
        "transfer_learning.stage_a",
    )

    optimizer_cfg = require_mapping(
        require_key(
            stage_a_cfg,
            "optimizer",
            "transfer_learning.stage_a",
        ),
        "transfer_learning.stage_a.optimizer",
    )

    if (
        require_key(
            optimizer_cfg,
            "scope",
            "transfer_learning.stage_a.optimizer",
        )
        != "classifier_only"
    ):

        raise ValueError(
            "Stage-A optimizer scope must be classifier_only."
        )

    if (
        require_key(
            optimizer_cfg,
            "fresh_optimizer_instance",
            "transfer_learning.stage_a.optimizer",
        )
        is not True
    ):

        raise ValueError(
            "Stage A requires a fresh optimizer instance."
        )

    classifier_lr = float(
        require_key(
            optimizer_cfg,
            "classifier_lr",
            "transfer_learning.stage_a.optimizer",
        )
    )

    if classifier_lr != 0.001:

        raise ValueError(
            "Frozen Stage-A classifier LR must equal 1e-3."
        )

    parameter_groups, summaries = (
        build_stage_a_parameter_groups(
            model=model,
            experiment_cfg=experiment_cfg,
        )
    )

    contract = load_adamw_contract(
        experiment_cfg
    )

    return _build_adamw(
        parameter_groups=parameter_groups,
        group_summaries=summaries,
        contract=contract,
        default_lr=classifier_lr,
        stage="stage_a",
    )


# ======================================================================
# Stage B
# ======================================================================

def build_stage_b_optimizer(
    *,
    model: nn.Module,
    experiment_cfg: Mapping[str, Any],
    backbone_lr: float,
) -> tuple[
    torch.optim.AdamW,
    OptimizerBuildEvidence,
]:
    """
    Construct a NEW full-network Stage-B AdamW.

    The backbone LR must belong to the frozen candidate set.
    """

    assert_stage_b_contract(
        model
    )

    transfer_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "transfer_learning",
            "experiment_config",
        ),
        "transfer_learning",
    )

    stage_b_cfg = require_mapping(
        require_key(
            transfer_cfg,
            "stage_b",
            "transfer_learning",
        ),
        "transfer_learning.stage_b",
    )

    optimizer_cfg = require_mapping(
        require_key(
            stage_b_cfg,
            "optimizer",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.optimizer",
    )

    if (
        require_key(
            optimizer_cfg,
            "fresh_optimizer_instance",
            "transfer_learning.stage_b.optimizer",
        )
        is not True
    ):

        raise ValueError(
            "Stage B requires a fresh optimizer instance."
        )

    classifier_lr = float(
        require_key(
            optimizer_cfg,
            "classifier_lr",
            "transfer_learning.stage_b.optimizer",
        )
    )

    if classifier_lr != 0.001:

        raise ValueError(
            "Frozen Stage-B classifier LR must equal 1e-3."
        )

    parameter_groups, summaries = (
        build_stage_b_parameter_groups(
            model=model,
            experiment_cfg=experiment_cfg,
            backbone_lr=float(
                backbone_lr
            ),
        )
    )

    contract = load_adamw_contract(
        experiment_cfg
    )

    return _build_adamw(
        parameter_groups=parameter_groups,
        group_summaries=summaries,
        contract=contract,
        default_lr=classifier_lr,
        stage="stage_b",
    )