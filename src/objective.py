"""
Frozen class-weighted cross-entropy objective for Tech-2.

Scientific contract
-------------------
Positive class:
    attack = 1

Frozen project_train counts:
    bonafide = 480
    attack   = 960
    total    = 1440

Class-weight formula:
    w_c = N / (2 * n_c)

Therefore:
    bonafide = 1.5
    attack   = 0.75

The same project_train-derived weights are used for:
- Stage-A training;
- Stage-A dev loss;
- Stage-B training;
- Stage-B dev loss;
- checkpoint selection;
- learning-rate selection;
- resolution comparison.

Epoch loss
----------
The frozen epoch metric is:

    sum(w_y * CE_i) / sum(w_y)

over the complete dataset.

This module deliberately keeps the numerator and denominator explicit
instead of averaging already-reduced batch losses.

No held-out test logic exists here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor


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
# Frozen class-weight contract
# ======================================================================

@dataclass(
    frozen=True
)
class ClassWeightContract:

    bonafide_count: int
    attack_count: int
    total_count: int

    bonafide_weight: float
    attack_weight: float


def derive_class_weight_contract(
    experiment_cfg: Mapping[str, Any],
) -> ClassWeightContract:
    """
    Independently derive the class weights from project_train counts
    and reconcile them against the frozen YAML values.
    """

    class_contract = require_mapping(
        require_key(
            experiment_cfg,
            "class_contract",
            "experiment_config",
        ),
        "class_contract",
    )

    train_counts = require_mapping(
        require_key(
            class_contract,
            "project_train_counts",
            "class_contract",
        ),
        "class_contract.project_train_counts",
    )

    bonafide_count = int(
        require_key(
            train_counts,
            "bonafide",
            "class_contract.project_train_counts",
        )
    )

    attack_count = int(
        require_key(
            train_counts,
            "attack",
            "class_contract.project_train_counts",
        )
    )

    total_count = int(
        require_key(
            train_counts,
            "total",
            "class_contract.project_train_counts",
        )
    )

    if (
        bonafide_count
        + attack_count
        != total_count
    ):

        raise RuntimeError(
            "project_train class counts do not reconcile:\n"
            f"  bonafide={bonafide_count}\n"
            f"  attack={attack_count}\n"
            f"  total={total_count}"
        )

    if (
        bonafide_count <= 0
        or attack_count <= 0
    ):

        raise RuntimeError(
            "Both frozen classes must contain samples."
        )

    bonafide_weight = (
        total_count
        /
        (
            2.0
            * bonafide_count
        )
    )

    attack_weight = (
        total_count
        /
        (
            2.0
            * attack_count
        )
    )

    # ------------------------------------------------------------------
    # Reconcile against frozen loss configuration.
    # ------------------------------------------------------------------

    loss_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "loss",
            "experiment_config",
        ),
        "loss",
    )

    if (
        require_key(
            loss_cfg,
            "type",
            "loss",
        )
        != "CrossEntropyLoss"
    ):

        raise ValueError(
            "Frozen loss type must be CrossEntropyLoss."
        )

    weighting_cfg = require_mapping(
        require_key(
            loss_cfg,
            "class_weighting",
            "loss",
        ),
        "loss.class_weighting",
    )

    if (
        require_key(
            weighting_cfg,
            "enabled",
            "loss.class_weighting",
        )
        is not True
    ):

        raise ValueError(
            "Frozen class weighting must be enabled."
        )

    if (
        require_key(
            weighting_cfg,
            "source",
            "loss.class_weighting",
        )
        != "frozen_project_train_only"
    ):

        raise ValueError(
            "Class weights must come only from frozen project_train."
        )

    if (
        require_key(
            weighting_cfg,
            "formula",
            "loss.class_weighting",
        )
        != "w_c = N / (2 * n_c)"
    ):

        raise ValueError(
            "Frozen class-weight formula changed."
        )

    by_name = require_mapping(
        require_key(
            weighting_cfg,
            "expected_by_class_name",
            "loss.class_weighting",
        ),
        "loss.class_weighting.expected_by_class_name",
    )

    configured_bonafide = float(
        require_key(
            by_name,
            "bonafide",
            (
                "loss.class_weighting."
                "expected_by_class_name"
            ),
        )
    )

    configured_attack = float(
        require_key(
            by_name,
            "attack",
            (
                "loss.class_weighting."
                "expected_by_class_name"
            ),
        )
    )

    for (
        label,
        actual,
        expected,
    ) in (
        (
            "bonafide",
            configured_bonafide,
            bonafide_weight,
        ),
        (
            "attack",
            configured_attack,
            attack_weight,
        ),
    ):

        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):

            raise RuntimeError(
                "Configured class weight disagrees with "
                "project_train-derived value:\n"
                f"  class={label}\n"
                f"  configured={actual}\n"
                f"  derived={expected}"
            )

    by_index = require_mapping(
        require_key(
            weighting_cfg,
            "expected_by_class_index",
            "loss.class_weighting",
        ),
        "loss.class_weighting.expected_by_class_index",
    )

    def indexed_value(
        index: int,
    ) -> float:

        if index in by_index:

            return float(
                by_index[
                    index
                ]
            )

        string_index = str(
            index
        )

        if string_index in by_index:

            return float(
                by_index[
                    string_index
                ]
            )

        raise KeyError(
            "Missing frozen class-index weight: "
            f"{index}"
        )

    if not math.isclose(
        indexed_value(
            0
        ),
        bonafide_weight,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):

        raise RuntimeError(
            "Class-index 0 weight does not equal "
            "derived bonafide weight."
        )

    if not math.isclose(
        indexed_value(
            1
        ),
        attack_weight,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):

        raise RuntimeError(
            "Class-index 1 weight does not equal "
            "derived attack weight."
        )

    training_loss = require_mapping(
        require_key(
            loss_cfg,
            "training_loss",
            "loss",
        ),
        "loss.training_loss",
    )

    if (
        require_key(
            training_loss,
            "weighted",
            "loss.training_loss",
        )
        is not True
    ):

        raise ValueError(
            "Frozen training loss must be weighted."
        )

    dev_loss = require_mapping(
        require_key(
            loss_cfg,
            "dev_selection_loss",
            "loss",
        ),
        "loss.dev_selection_loss",
    )

    if (
        require_key(
            dev_loss,
            "weighted",
            "loss.dev_selection_loss",
        )
        is not True
    ):

        raise ValueError(
            "Frozen dev-selection loss must be weighted."
        )

    if (
        require_key(
            dev_loss,
            "reuse_project_train_weights",
            "loss.dev_selection_loss",
        )
        is not True
    ):

        raise ValueError(
            "dev_val must reuse project_train-derived weights."
        )

    if float(
        require_key(
            loss_cfg,
            "label_smoothing",
            "loss",
        )
    ) != 0.0:

        raise ValueError(
            "Frozen label_smoothing must equal 0."
        )

    epoch_cfg = require_mapping(
        require_key(
            loss_cfg,
            "epoch_reduction",
            "loss",
        ),
        "loss.epoch_reduction",
    )

    expected_epoch_contract = {
        "definition":
            "weighted_cross_entropy_over_entire_dataset",

        "numerator":
            "sum_sample_weight_times_per_sample_ce",

        "denominator":
            "sum_sample_weights",
    }

    for (
        key,
        expected,
    ) in expected_epoch_contract.items():

        actual = require_key(
            epoch_cfg,
            key,
            "loss.epoch_reduction",
        )

        if actual != expected:

            raise ValueError(
                "Frozen epoch-loss reduction changed:\n"
                f"  {key}\n"
                f"  expected={expected!r}\n"
                f"  actual={actual!r}"
            )

    return ClassWeightContract(
        bonafide_count=bonafide_count,
        attack_count=attack_count,
        total_count=total_count,
        bonafide_weight=bonafide_weight,
        attack_weight=attack_weight,
    )


# ======================================================================
# Per-batch result
# ======================================================================

@dataclass
class WeightedCrossEntropyBatch:
    """
    One differentiable training/dev loss plus detached epoch statistics.
    """

    loss: Tensor

    epoch_weighted_numerator: float
    epoch_weight_denominator: float

    sample_count: int


# ======================================================================
# Scientific objective
# ======================================================================

class WeightedCrossEntropyObjective:
    """
    Explicit class-weighted CE implementation.

    Batch backward objective:
        sum(w_y * CE_i) / sum(w_y)

    Epoch metric:
        accumulate every sample's weighted numerator and denominator
        across all batches, then divide once at epoch end.

    The per-sample CE is computed without hidden class weighting and the
    frozen weights are multiplied explicitly, making the configured
    scientific formula directly inspectable.
    """

    def __init__(
        self,
        *,
        experiment_cfg: Mapping[str, Any],
        device: torch.device | str,
    ) -> None:

        self.contract = (
            derive_class_weight_contract(
                experiment_cfg
            )
        )

        self.device = torch.device(
            device
        )

        self.class_weights = torch.tensor(
            [
                self.contract.bonafide_weight,
                self.contract.attack_weight,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.per_sample_ce = (
            nn.CrossEntropyLoss(
                reduction="none",
                label_smoothing=0.0,
            )
        )

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> WeightedCrossEntropyBatch:

        if logits.ndim != 2:

            raise ValueError(
                "logits must have shape [N, C], got "
                f"{tuple(logits.shape)}"
            )

        if logits.shape[
            1
        ] != 2:

            raise ValueError(
                "Frozen classifier requires exactly 2 logits."
            )

        if targets.ndim != 1:

            raise ValueError(
                "targets must have shape [N]."
            )

        if (
            logits.shape[
                0
            ]
            != targets.shape[
                0
            ]
        ):

            raise ValueError(
                "logit/target batch-size mismatch."
            )

        if targets.dtype != torch.int64:

            raise TypeError(
                "Cross-entropy targets must be torch.int64, "
                f"got {targets.dtype}."
            )

        if logits.dtype != torch.float32:

            raise TypeError(
                "Frozen objective requires float32 logits, "
                f"got {logits.dtype}."
            )

        if logits.device != targets.device:

            raise RuntimeError(
                "logits and targets are on different devices."
            )

        if logits.device != self.class_weights.device:

            raise RuntimeError(
                "Objective class weights are on the wrong device:\n"
                f"  logits={logits.device}\n"
                f"  weights={self.class_weights.device}"
            )

        if targets.numel() == 0:

            raise ValueError(
                "Empty target batch is not permitted."
            )

        if not bool(
            torch.all(
                (
                    targets == 0
                )
                |
                (
                    targets == 1
                )
            )
        ):

            raise ValueError(
                "Targets outside frozen binary class range {0,1}."
            )

        if not bool(
            torch.isfinite(
                logits
            ).all()
        ):

            raise RuntimeError(
                "Logits contain NaN or Inf."
            )

        # --------------------------------------------------------------
        # Ordinary per-sample CE.
        # --------------------------------------------------------------

        per_sample_ce = (
            self.per_sample_ce(
                logits,
                targets,
            )
        )

        # --------------------------------------------------------------
        # Explicit frozen sample weights.
        # --------------------------------------------------------------

        sample_weights = (
            self.class_weights[
                targets
            ]
        )

        weighted_losses = (
            sample_weights
            * per_sample_ce
        )

        weighted_sum = (
            weighted_losses
            .sum()
        )

        weight_sum = (
            sample_weights
            .sum()
        )

        if not bool(
            weight_sum > 0
        ):

            raise RuntimeError(
                "Batch class-weight denominator is not positive."
            )

        loss = (
            weighted_sum
            / weight_sum
        )

        if not bool(
            torch.isfinite(
                loss
            )
        ):

            raise RuntimeError(
                "Weighted CE produced a non-finite loss."
            )

        # --------------------------------------------------------------
        # Epoch evidence is accumulated in float64 after detaching.
        #
        # This avoids averaging batch means and minimizes sensitivity to
        # how the dataset is partitioned into batches.
        # --------------------------------------------------------------

        epoch_numerator = float(
            weighted_losses
            .detach()
            .to(
                dtype=torch.float64
            )
            .sum()
            .item()
        )

        epoch_denominator = float(
            sample_weights
            .detach()
            .to(
                dtype=torch.float64
            )
            .sum()
            .item()
        )

        return WeightedCrossEntropyBatch(
            loss=loss,
            epoch_weighted_numerator=epoch_numerator,
            epoch_weight_denominator=epoch_denominator,
            sample_count=int(
                targets.numel()
            ),
        )


# ======================================================================
# Dataset-level epoch accumulator
# ======================================================================

@dataclass
class WeightedLossAccumulator:
    """
    Accumulate the frozen epoch metric without averaging batch means.
    """

    weighted_numerator: float = 0.0
    weight_denominator: float = 0.0
    sample_count: int = 0

    def update(
        self,
        result: WeightedCrossEntropyBatch,
    ) -> None:

        if result.sample_count <= 0:

            raise ValueError(
                "Cannot accumulate an empty batch."
            )

        if (
            not math.isfinite(
                result.epoch_weighted_numerator
            )
            or
            not math.isfinite(
                result.epoch_weight_denominator
            )
        ):

            raise RuntimeError(
                "Cannot accumulate non-finite loss statistics."
            )

        if (
            result.epoch_weight_denominator
            <= 0.0
        ):

            raise RuntimeError(
                "Batch loss denominator must be positive."
            )

        self.weighted_numerator += (
            result.epoch_weighted_numerator
        )

        self.weight_denominator += (
            result.epoch_weight_denominator
        )

        self.sample_count += (
            result.sample_count
        )

    @property
    def value(
        self,
    ) -> float:

        if self.sample_count <= 0:

            raise RuntimeError(
                "Cannot compute loss for an empty accumulator."
            )

        if self.weight_denominator <= 0.0:

            raise RuntimeError(
                "Epoch weight denominator is not positive."
            )

        result = (
            self.weighted_numerator
            / self.weight_denominator
        )

        if not math.isfinite(
            result
        ):

            raise RuntimeError(
                "Epoch weighted loss is non-finite."
            )

        return result