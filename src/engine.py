"""
Single-epoch execution primitives for frozen Tech-2 development.

This module connects the already-validated components:

    DataLoader
        ->
    ResNet-18
        ->
    weighted CE
        ->
    backward / AdamW step

Implemented
-----------
- one complete project_train training epoch;
- one complete dev_val evaluation epoch;
- Stage-A / Stage-B mode restoration at epoch start;
- exact dataset-level weighted-loss accumulation;
- complete-sample / complete-batch reconciliation;
- class-count reconciliation;
- optimizer/trainable-parameter reconciliation;
- raw dev logits retained in frozen manifest order.

Not implemented
---------------
- multi-epoch training;
- patience;
- checkpoint selection;
- checkpoint serialization;
- AUROC;
- FPR10 thresholding;
- LR screening;
- resolution selection;
- Stage-A -> Stage-B controller;
- held-out test;
- Grad-CAM.

Important
---------
Training loss reported here is accumulated from each batch immediately
before that batch's optimizer update.

Dev loss is evaluated on one fixed post-training model state.

Raw dev logits are returned rather than defining an attack score here.
The later metrics module will apply the exact frozen Tech-1 evaluation
convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.modeling import (
    apply_evaluation_contract,
    apply_stage_a_contract,
    apply_stage_b_contract,
    assert_stage_a_contract,
    assert_stage_b_contract,
)

from src.objective import (
    WeightedCrossEntropyObjective,
    WeightedLossAccumulator,
)


# ======================================================================
# Public stage type
# ======================================================================

TrainingStage = Literal[
    "stage_a",
    "stage_b",
]


# ======================================================================
# Epoch results
# ======================================================================

@dataclass(
    frozen=True
)
class EpochLossSummary:
    """
    Exact dataset-level weighted CE evidence for one complete epoch.
    """

    weighted_numerator: float
    weight_denominator: float
    weighted_loss: float

    sample_count: int
    batch_count: int

    bonafide_count: int
    attack_count: int


@dataclass(
    frozen=True
)
class TrainingEpochResult:
    """
    Result of one complete project_train optimization epoch.
    """

    stage: TrainingStage

    loss: EpochLossSummary

    optimizer_steps: int


@dataclass(
    frozen=True
)
class DevEpochResult:
    """
    Result of one complete dev_val evaluation epoch.

    logits:
        CPU float32 tensor with shape [N, 2]

    targets:
        CPU int64 tensor with shape [N]

    image_paths:
        frozen dev manifest order

    No probability/score convention is imposed here.
    """

    loss: EpochLossSummary

    logits: Tensor
    targets: Tensor

    image_paths: tuple[
        str,
        ...,
    ]


# ======================================================================
# Dataset / loader structural contract
# ======================================================================

def _require_dataset_role(
    *,
    loader: DataLoader,
    expected_role: str,
) -> Any:

    dataset = loader.dataset

    if not hasattr(
        dataset,
        "expected_role",
    ):

        raise TypeError(
            "Frozen epoch engine requires a manifest-backed Dataset "
            "with expected_role."
        )

    actual_role = str(
        dataset.expected_role
    )

    if actual_role != expected_role:

        raise RuntimeError(
            "DataLoader partition role mismatch:\n"
            f"  expected={expected_role!r}\n"
            f"  actual={actual_role!r}"
        )

    if not hasattr(
        dataset,
        "rows",
    ):

        raise TypeError(
            "Frozen epoch engine requires Dataset.rows "
            "for reconciliation."
        )

    if len(
        dataset
    ) <= 0:

        raise RuntimeError(
            "Empty Dataset is not permitted."
        )

    if len(
        loader
    ) <= 0:

        raise RuntimeError(
            "Empty DataLoader is not permitted."
        )

    return dataset


# ======================================================================
# Device contract
# ======================================================================

def _validate_execution_device(
    *,
    model: nn.Module,
    objective: WeightedCrossEntropyObjective,
    device: torch.device,
) -> None:

    parameters = list(
        model.parameters()
    )

    if not parameters:

        raise RuntimeError(
            "Model has no parameters."
        )

    parameter_devices = {
        parameter.device
        for parameter
        in parameters
    }

    if parameter_devices != {
        device
    }:

        raise RuntimeError(
            "Model parameters are not entirely on the requested device:\n"
            f"  requested={device}\n"
            f"  actual={sorted(str(value) for value in parameter_devices)}"
        )

    parameter_dtypes = {
        parameter.dtype
        for parameter
        in parameters
    }

    if parameter_dtypes != {
        torch.float32
    }:

        raise RuntimeError(
            "Frozen model execution requires every parameter "
            "to be float32:\n"
            f"  actual={parameter_dtypes}"
        )

    if objective.class_weights.device != device:

        raise RuntimeError(
            "Objective class weights are on the wrong device:\n"
            f"  requested={device}\n"
            f"  actual={objective.class_weights.device}"
        )

    if objective.class_weights.dtype != torch.float32:

        raise RuntimeError(
            "Objective class weights must be float32."
        )


# ======================================================================
# Optimizer / trainable-parameter reconciliation
# ======================================================================

def _assert_optimizer_matches_trainable_parameters(
    *,
    model: nn.Module,
    optimizer: Optimizer,
) -> None:
    """
    Require optimizer coverage to equal exactly the currently trainable
    model parameters.

    This catches, for example:
    - accidentally using Stage-A AdamW after unfreezing Stage B;
    - missing parameters;
    - duplicated parameters across optimizer groups.
    """

    trainable_ids = {
        id(
            parameter
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    }

    optimizer_ids: set[
        int
    ] = set()

    for group in optimizer.param_groups:

        for parameter in group[
            "params"
        ]:

            parameter_id = id(
                parameter
            )

            if parameter_id in optimizer_ids:

                raise RuntimeError(
                    "A model parameter appears more than once "
                    "across optimizer groups."
                )

            optimizer_ids.add(
                parameter_id
            )

    if optimizer_ids != trainable_ids:

        missing = (
            trainable_ids
            - optimizer_ids
        )

        extra = (
            optimizer_ids
            - trainable_ids
        )

        raise RuntimeError(
            "Optimizer parameters do not exactly match "
            "currently trainable model parameters:\n"
            f"  trainable={len(trainable_ids)}\n"
            f"  optimizer={len(optimizer_ids)}\n"
            f"  missing={len(missing)}\n"
            f"  extra={len(extra)}"
        )


# ======================================================================
# Batch preparation
# ======================================================================

def _prepare_batch(
    *,
    batch: Mapping[
        str,
        Any,
    ],
    device: torch.device,
) -> tuple[
    Tensor,
    Tensor,
]:

    if "image" not in batch:

        raise KeyError(
            "Batch missing 'image'."
        )

    if "label" not in batch:

        raise KeyError(
            "Batch missing 'label'."
        )

    images = batch[
        "image"
    ]

    targets = batch[
        "label"
    ]

    if not isinstance(
        images,
        Tensor,
    ):

        raise TypeError(
            "Batch 'image' must be a torch.Tensor."
        )

    if not isinstance(
        targets,
        Tensor,
    ):

        raise TypeError(
            "Batch 'label' must be a torch.Tensor."
        )

    if images.ndim != 4:

        raise RuntimeError(
            "Images must have NCHW shape, got "
            f"{tuple(images.shape)}"
        )

    if images.shape[
        1
    ] != 3:

        raise RuntimeError(
            "Frozen RGB model requires 3 input channels."
        )

    if targets.ndim != 1:

        raise RuntimeError(
            "Targets must have shape [N]."
        )

    if (
        images.shape[
            0
        ]
        != targets.shape[
            0
        ]
    ):

        raise RuntimeError(
            "Image/target batch-size mismatch."
        )

    if images.shape[
        0
    ] <= 0:

        raise RuntimeError(
            "Empty batch is not permitted."
        )

    if images.dtype != torch.float32:

        raise RuntimeError(
            "DataLoader images must already be float32."
        )

    if targets.dtype != torch.int64:

        raise RuntimeError(
            "DataLoader labels must be torch.int64."
        )

    if images.device.type != "cpu":

        raise RuntimeError(
            "Frozen DataLoader is expected to emit CPU tensors."
        )

    if targets.device.type != "cpu":

        raise RuntimeError(
            "Frozen DataLoader is expected to emit CPU labels."
        )

    # Do not silently change dtype here. A dtype contract failure should
    # fail before scientific training rather than be repaired implicitly.
    images = images.to(
        device=device,
        non_blocking=False,
    )

    targets = targets.to(
        device=device,
        non_blocking=False,
    )

    return (
        images,
        targets,
    )


# ======================================================================
# Forward-output contract
# ======================================================================

def _validate_logits(
    *,
    logits: Tensor,
    batch_size: int,
    device: torch.device,
) -> None:

    expected_shape = (
        batch_size,
        2,
    )

    if tuple(
        logits.shape
    ) != expected_shape:

        raise RuntimeError(
            "Model output shape mismatch:\n"
            f"  expected={expected_shape}\n"
            f"  actual={tuple(logits.shape)}"
        )

    if logits.dtype != torch.float32:

        raise RuntimeError(
            "Frozen model logits must be float32."
        )

    if logits.device != device:

        raise RuntimeError(
            "Model logits are on the wrong device."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):

        raise RuntimeError(
            "Model logits contain NaN or Inf."
        )


# ======================================================================
# Epoch reconciliation
# ======================================================================

def _expected_label_counts(
    dataset: Any,
) -> tuple[
    int,
    int,
]:

    bonafide = 0
    attack = 0

    for row in dataset.rows:

        label = int(
            row[
                "label"
            ]
        )

        if label == 0:

            bonafide += 1

        elif label == 1:

            attack += 1

        else:

            raise RuntimeError(
                "Dataset contains label outside frozen {0,1}."
            )

    if (
        bonafide
        + attack
        != len(
            dataset
        )
    ):

        raise RuntimeError(
            "Dataset label counts do not reconcile to Dataset length."
        )

    return (
        bonafide,
        attack,
    )


def _expected_weight_denominator(
    *,
    bonafide_count: int,
    attack_count: int,
    objective: WeightedCrossEntropyObjective,
) -> float:

    return (
        bonafide_count
        * objective.contract.bonafide_weight
        +
        attack_count
        * objective.contract.attack_weight
    )


def _finalize_epoch_summary(
    *,
    loader: DataLoader,
    dataset: Any,
    objective: WeightedCrossEntropyObjective,
    accumulator: WeightedLossAccumulator,
    batch_count: int,
    bonafide_count: int,
    attack_count: int,
) -> EpochLossSummary:

    expected_samples = len(
        dataset
    )

    expected_batches = len(
        loader
    )

    if accumulator.sample_count != expected_samples:

        raise RuntimeError(
            "Epoch did not process exactly the complete Dataset:\n"
            f"  expected_samples={expected_samples}\n"
            f"  observed_samples={accumulator.sample_count}"
        )

    if batch_count != expected_batches:

        raise RuntimeError(
            "Epoch batch-count mismatch:\n"
            f"  expected_batches={expected_batches}\n"
            f"  observed_batches={batch_count}"
        )

    expected_bonafide, expected_attack = (
        _expected_label_counts(
            dataset
        )
    )

    if (
        bonafide_count
        != expected_bonafide
        or
        attack_count
        != expected_attack
    ):

        raise RuntimeError(
            "Observed epoch class counts do not match "
            "the frozen manifest:\n"
            f"  expected_bonafide={expected_bonafide}\n"
            f"  observed_bonafide={bonafide_count}\n"
            f"  expected_attack={expected_attack}\n"
            f"  observed_attack={attack_count}"
        )

    expected_denominator = (
        _expected_weight_denominator(
            bonafide_count=expected_bonafide,
            attack_count=expected_attack,
            objective=objective,
        )
    )

    if not math.isclose(
        accumulator.weight_denominator,
        expected_denominator,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):

        raise RuntimeError(
            "Epoch weighted denominator does not reconcile "
            "to manifest class counts:\n"
            f"  expected={expected_denominator:.17g}\n"
            f"  observed={accumulator.weight_denominator:.17g}"
        )

    value = accumulator.value

    if not math.isfinite(
        value
    ):

        raise RuntimeError(
            "Final epoch weighted loss is non-finite."
        )

    return EpochLossSummary(
        weighted_numerator=(
            accumulator.weighted_numerator
        ),
        weight_denominator=(
            accumulator.weight_denominator
        ),
        weighted_loss=value,
        sample_count=(
            accumulator.sample_count
        ),
        batch_count=batch_count,
        bonafide_count=bonafide_count,
        attack_count=attack_count,
    )


# ======================================================================
# Training stage handling
# ======================================================================

def _apply_training_stage(
    *,
    model: nn.Module,
    stage: TrainingStage,
) -> None:

    if stage == "stage_a":

        apply_stage_a_contract(
            model
        )

        assert_stage_a_contract(
            model
        )

        return

    if stage == "stage_b":

        apply_stage_b_contract(
            model
        )

        assert_stage_b_contract(
            model
        )

        return

    raise ValueError(
        f"Unsupported training stage: {stage!r}"
    )


# ======================================================================
# Complete project_train epoch
# ======================================================================

def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    objective: WeightedCrossEntropyObjective,
    device: torch.device | str,
    stage: TrainingStage,
) -> TrainingEpochResult:
    """
    Execute exactly one complete project_train optimization epoch.

    Stage mode is reapplied here intentionally.

    This matters because dev evaluation calls model.eval(); therefore
    every subsequent training epoch must explicitly restore either:

        Stage A:
            backbone eval/frozen + FC train

    or:

        Stage B:
            complete model train/unfrozen
    """

    execution_device = torch.device(
        device
    )

    dataset = _require_dataset_role(
        loader=loader,
        expected_role="project_train",
    )

    _validate_execution_device(
        model=model,
        objective=objective,
        device=execution_device,
    )

    _apply_training_stage(
        model=model,
        stage=stage,
    )

    _assert_optimizer_matches_trainable_parameters(
        model=model,
        optimizer=optimizer,
    )

    accumulator = (
        WeightedLossAccumulator()
    )

    batch_count = 0
    optimizer_steps = 0

    bonafide_count = 0
    attack_count = 0

    for batch in loader:

        images, targets = _prepare_batch(
            batch=batch,
            device=execution_device,
        )

        target_counts = torch.bincount(
            targets.detach().cpu(),
            minlength=2,
        )

        bonafide_count += int(
            target_counts[
                0
            ].item()
        )

        attack_count += int(
            target_counts[
                1
            ].item()
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        _validate_logits(
            logits=logits,
            batch_size=int(
                targets.shape[
                    0
                ]
            ),
            device=execution_device,
        )

        batch_objective = objective(
            logits,
            targets,
        )

        # --------------------------------------------------------------
        # Scientific gradient objective.
        #
        # No AMP.
        # No gradient clipping.
        # No accumulation across batches.
        # Exactly one optimizer step per DataLoader batch.
        # --------------------------------------------------------------

        batch_objective.loss.backward()

        optimizer.step()

        accumulator.update(
            batch_objective
        )

        batch_count += 1
        optimizer_steps += 1

    # Remove stale final-batch gradients before dev evaluation or
    # checkpoint serialization.
    optimizer.zero_grad(
        set_to_none=True
    )

    summary = _finalize_epoch_summary(
        loader=loader,
        dataset=dataset,
        objective=objective,
        accumulator=accumulator,
        batch_count=batch_count,
        bonafide_count=bonafide_count,
        attack_count=attack_count,
    )

    if optimizer_steps != len(
        loader
    ):

        raise RuntimeError(
            "Optimizer step count does not equal "
            "project_train batch count:\n"
            f"  expected={len(loader)}\n"
            f"  actual={optimizer_steps}"
        )

    return TrainingEpochResult(
        stage=stage,
        loss=summary,
        optimizer_steps=optimizer_steps,
    )


# ======================================================================
# Complete dev_val epoch
# ======================================================================

def evaluate_dev_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    objective: WeightedCrossEntropyObjective,
    device: torch.device | str,
) -> DevEpochResult:
    """
    Evaluate exactly one complete dev_val epoch.

    The model is left in eval mode on return.

    The next call to train_one_epoch() explicitly reapplies the relevant
    Stage-A or Stage-B training contract.

    Raw two-class logits are retained in frozen manifest order so the
    future metrics implementation can apply the exact Tech-1 score and
    AUROC/FPR10 conventions without redefining them here.
    """

    execution_device = torch.device(
        device
    )

    dataset = _require_dataset_role(
        loader=loader,
        expected_role="dev_val",
    )

    _validate_execution_device(
        model=model,
        objective=objective,
        device=execution_device,
    )

    apply_evaluation_contract(
        model
    )

    accumulator = (
        WeightedLossAccumulator()
    )

    batch_count = 0

    bonafide_count = 0
    attack_count = 0

    logits_parts: list[
        Tensor
    ] = []

    target_parts: list[
        Tensor
    ] = []

    image_paths: list[
        str
    ] = []

    with torch.inference_mode():

        for batch in loader:

            images, targets = _prepare_batch(
                batch=batch,
                device=execution_device,
            )

            target_counts = torch.bincount(
                targets.detach().cpu(),
                minlength=2,
            )

            bonafide_count += int(
                target_counts[
                    0
                ].item()
            )

            attack_count += int(
                target_counts[
                    1
                ].item()
            )

            logits = model(
                images
            )

            _validate_logits(
                logits=logits,
                batch_size=int(
                    targets.shape[
                        0
                    ]
                ),
                device=execution_device,
            )

            batch_objective = objective(
                logits,
                targets,
            )

            accumulator.update(
                batch_objective
            )

            logits_parts.append(
                logits
                .detach()
                .cpu()
                .contiguous()
            )

            target_parts.append(
                targets
                .detach()
                .cpu()
                .contiguous()
            )

            batch_paths = list(
                batch[
                    "image_path"
                ]
            )

            if len(
                batch_paths
            ) != int(
                targets.shape[
                    0
                ]
            ):

                raise RuntimeError(
                    "dev_val image-path batch length mismatch."
                )

            image_paths.extend(
                str(
                    value
                )
                for value
                in batch_paths
            )

            batch_count += 1

    summary = _finalize_epoch_summary(
        loader=loader,
        dataset=dataset,
        objective=objective,
        accumulator=accumulator,
        batch_count=batch_count,
        bonafide_count=bonafide_count,
        attack_count=attack_count,
    )

    all_logits = torch.cat(
        logits_parts,
        dim=0,
    )

    all_targets = torch.cat(
        target_parts,
        dim=0,
    )

    expected_rows = len(
        dataset
    )

    if tuple(
        all_logits.shape
    ) != (
        expected_rows,
        2,
    ):

        raise RuntimeError(
            "Concatenated dev logits shape mismatch:\n"
            f"  expected={(expected_rows, 2)}\n"
            f"  actual={tuple(all_logits.shape)}"
        )

    if tuple(
        all_targets.shape
    ) != (
        expected_rows,
    ):

        raise RuntimeError(
            "Concatenated dev target shape mismatch."
        )

    if all_logits.dtype != torch.float32:

        raise RuntimeError(
            "Stored dev logits must be CPU float32."
        )

    if all_targets.dtype != torch.int64:

        raise RuntimeError(
            "Stored dev targets must be CPU int64."
        )

    # ------------------------------------------------------------------
    # Frozen dev order must equal manifest order exactly.
    # ------------------------------------------------------------------

    expected_paths = [
        str(
            row[
                "image_path_relative"
            ]
        )
        for row
        in dataset.rows
    ]

    if image_paths != expected_paths:

        raise RuntimeError(
            "dev_val evaluation order differs from frozen "
            "manifest order."
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
        all_targets,
        expected_targets,
    ):

        raise RuntimeError(
            "dev_val targets differ from frozen manifest order."
        )

    if model.training:

        raise RuntimeError(
            "Dev evaluation unexpectedly left model in train mode."
        )

    return DevEpochResult(
        loss=summary,
        logits=all_logits,
        targets=all_targets,
        image_paths=tuple(
            image_paths
        ),
    )