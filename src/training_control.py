"""
Early-stopping and raw-argmin model-checkpoint control for Tech-2.

This module implements the frozen distinction between:

1. raw model selection:
       checkpoint whenever weighted dev CE reaches a new raw minimum;

2. patience control:
       reset patience only for a >=0.5% relative improvement against
       the current patience anchor.

These are deliberately independent.

Example
-------
Suppose the patience anchor is:

    0.7000

and a later epoch reaches:

    0.6980

That is a new raw minimum, so it MUST become the selected checkpoint.

But:

    0.6980 < 0.7000 * (1 - 0.005)

is false, so it is NOT a meaningful 0.5% improvement and patience still
increments.

This distinction is part of the frozen scientific protocol.

Checkpoint scope
----------------
The checkpoint implemented here contains MODEL STATE ONLY:

- parameters;
- BatchNorm buffers;
- epoch;
- weighted dev loss.

It deliberately does NOT contain:

- optimizer state;
- Python RNG state;
- NumPy RNG state;
- torch RNG state;
- DataLoader generator state.

Therefore this object is a model-selection checkpoint, not yet a
full resumable training checkpoint.

This boundary is intentional. The frozen protocol requires all Stage-B
LR candidates to start from the exact same Stage-A raw-best MODEL
checkpoint, but detailed Stage-A -> Stage-B RNG continuation/reset
semantics have not yet been encoded. They must not be silently invented
inside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


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
# Frozen stage stopping contract
# ======================================================================

@dataclass(
    frozen=True
)
class StageStoppingContract:

    stage: str

    patience_epochs: int
    maximum_epochs: int

    meaningful_relative_improvement_fraction: float

    checkpoint_metric: str
    checkpoint_policy: str


def load_stage_stopping_contract(
    *,
    experiment_cfg: Mapping[str, Any],
    stage: str,
) -> StageStoppingContract:
    """
    Load and reassert the frozen Stage-A / Stage-B stopping contract.
    """

    if stage not in {
        "stage_a",
        "stage_b",
    }:

        raise ValueError(
            f"Unsupported stage: {stage!r}"
        )

    stopping_definition = require_mapping(
        require_key(
            experiment_cfg,
            "stopping_definition",
            "experiment_config",
        ),
        "stopping_definition",
    )

    shared_fraction = float(
        require_key(
            stopping_definition,
            "meaningful_relative_improvement_fraction",
            "stopping_definition",
        )
    )

    if shared_fraction != 0.005:

        raise ValueError(
            "Frozen meaningful relative improvement "
            "fraction must equal 0.005."
        )

    patience_logic = require_mapping(
        require_key(
            stopping_definition,
            "patience_logic",
            "stopping_definition",
        ),
        "stopping_definition.patience_logic",
    )

    expected_comparison = (
        "current_dev_loss < patience_anchor_loss * (1 - 0.005)"
    )

    if (
        require_key(
            patience_logic,
            "comparison",
            "stopping_definition.patience_logic",
        )
        != expected_comparison
    ):

        raise ValueError(
            "Frozen patience comparison changed."
        )

    meaningful_cfg = require_mapping(
        require_key(
            patience_logic,
            "on_meaningful_improvement",
            "stopping_definition.patience_logic",
        ),
        (
            "stopping_definition."
            "patience_logic."
            "on_meaningful_improvement"
        ),
    )

    if (
        require_key(
            meaningful_cfg,
            "reset_patience",
            (
                "stopping_definition."
                "patience_logic."
                "on_meaningful_improvement"
            ),
        )
        is not True
    ):

        raise ValueError(
            "Meaningful improvement must reset patience."
        )

    if (
        require_key(
            meaningful_cfg,
            "update_patience_anchor",
            (
                "stopping_definition."
                "patience_logic."
                "on_meaningful_improvement"
            ),
        )
        is not True
    ):

        raise ValueError(
            "Meaningful improvement must update patience anchor."
        )

    otherwise_cfg = require_mapping(
        require_key(
            patience_logic,
            "otherwise",
            "stopping_definition.patience_logic",
        ),
        (
            "stopping_definition."
            "patience_logic."
            "otherwise"
        ),
    )

    if (
        require_key(
            otherwise_cfg,
            "increment_patience",
            (
                "stopping_definition."
                "patience_logic."
                "otherwise"
            ),
        )
        is not True
    ):

        raise ValueError(
            "Non-meaningful epoch must increment patience."
        )

    checkpoint_logic = require_mapping(
        require_key(
            stopping_definition,
            "checkpoint_logic",
            "stopping_definition",
        ),
        "stopping_definition.checkpoint_logic",
    )

    if (
        require_key(
            checkpoint_logic,
            "metric",
            "stopping_definition.checkpoint_logic",
        )
        != "class_weighted_dev_cross_entropy"
    ):

        raise ValueError(
            "Frozen checkpoint metric must be "
            "class_weighted_dev_cross_entropy."
        )

    if (
        require_key(
            checkpoint_logic,
            "policy",
            "stopping_definition.checkpoint_logic",
        )
        != "raw_argmin"
    ):

        raise ValueError(
            "Frozen checkpoint policy must be raw_argmin."
        )

    if (
        require_key(
            checkpoint_logic,
            "minimum_improvement_required_for_checkpoint",
            "stopping_definition.checkpoint_logic",
        )
        is not False
    ):

        raise ValueError(
            "Raw checkpoint updates must not require "
            "the 0.5% meaningful-improvement threshold."
        )

    transfer_learning = require_mapping(
        require_key(
            experiment_cfg,
            "transfer_learning",
            "experiment_config",
        ),
        "transfer_learning",
    )

    stage_cfg = require_mapping(
        require_key(
            transfer_learning,
            stage,
            "transfer_learning",
        ),
        f"transfer_learning.{stage}",
    )

    stage_stopping = require_mapping(
        require_key(
            stage_cfg,
            "stopping",
            f"transfer_learning.{stage}",
        ),
        f"transfer_learning.{stage}.stopping",
    )

    patience_epochs = int(
        require_key(
            stage_stopping,
            "patience_epochs",
            f"transfer_learning.{stage}.stopping",
        )
    )

    maximum_epochs = int(
        require_key(
            stage_stopping,
            "maximum_epochs",
            f"transfer_learning.{stage}.stopping",
        )
    )

    stage_fraction = float(
        require_key(
            stage_stopping,
            "meaningful_relative_improvement_fraction",
            f"transfer_learning.{stage}.stopping",
        )
    )

    if stage_fraction != shared_fraction:

        raise ValueError(
            "Stage stopping fraction disagrees with shared "
            "stopping_definition."
        )

    if (
        require_key(
            stage_stopping,
            "metric",
            f"transfer_learning.{stage}.stopping",
        )
        != "class_weighted_dev_cross_entropy"
    ):

        raise ValueError(
            f"{stage} stopping metric changed."
        )

    stage_checkpoint = require_mapping(
        require_key(
            stage_cfg,
            "checkpoint",
            f"transfer_learning.{stage}",
        ),
        f"transfer_learning.{stage}.checkpoint",
    )

    checkpoint_policy = str(
        require_key(
            stage_checkpoint,
            "policy",
            f"transfer_learning.{stage}.checkpoint",
        )
    )

    if (
        checkpoint_policy
        != "raw_argmin_class_weighted_dev_cross_entropy"
    ):

        raise ValueError(
            f"{stage} checkpoint policy changed."
        )

    expected_values = {
        "stage_a":
            {
                "patience":
                    2,

                "maximum_epochs":
                    10,
            },

        "stage_b":
            {
                "patience":
                    5,

                "maximum_epochs":
                    30,
            },
    }

    expected = expected_values[
        stage
    ]

    if patience_epochs != expected[
        "patience"
    ]:

        raise ValueError(
            f"Frozen {stage} patience must equal "
            f"{expected['patience']}."
        )

    if maximum_epochs != expected[
        "maximum_epochs"
    ]:

        raise ValueError(
            f"Frozen {stage} maximum_epochs must equal "
            f"{expected['maximum_epochs']}."
        )

    return StageStoppingContract(
        stage=stage,
        patience_epochs=patience_epochs,
        maximum_epochs=maximum_epochs,
        meaningful_relative_improvement_fraction=(
            shared_fraction
        ),
        checkpoint_metric=(
            "class_weighted_dev_cross_entropy"
        ),
        checkpoint_policy=(
            checkpoint_policy
        ),
    )


# ======================================================================
# Stable model-state hashing
# ======================================================================

def _update_digest_with_tensor(
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

        _update_digest_with_tensor(
            digest,
            name=name,
            tensor=value,
        )

    return digest.hexdigest()


def state_dict_sha256(
    state_dict: Mapping[
        str,
        torch.Tensor,
    ],
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        value,
    ) in state_dict.items():

        _update_digest_with_tensor(
            digest,
            name=name,
            tensor=value,
        )

    return digest.hexdigest()


# ======================================================================
# Raw-argmin model checkpoint
# ======================================================================

def _clone_model_state_to_cpu(
    model: nn.Module,
) -> dict[
    str,
    torch.Tensor,
]:
    """
    Make a detached CPU copy so the checkpoint cannot be mutated by
    continued training and does not consume persistent GPU memory.
    """

    return {
        name:
            (
                value
                .detach()
                .cpu()
                .clone()
            )

        for (
            name,
            value,
        ) in model.state_dict().items()
    }


@dataclass(
    frozen=True
)
class RawArgminModelCheckpoint:

    stage: str

    epoch: int
    weighted_dev_loss: float

    model_state_sha256: str

    model_state_dict: dict[
        str,
        torch.Tensor,
    ]


def capture_raw_argmin_model_checkpoint(
    *,
    model: nn.Module,
    stage: str,
    epoch: int,
    weighted_dev_loss: float,
) -> RawArgminModelCheckpoint:

    if stage not in {
        "stage_a",
        "stage_b",
    }:

        raise ValueError(
            f"Unsupported checkpoint stage: {stage!r}"
        )

    if (
        not isinstance(
            epoch,
            int,
        )
        or isinstance(
            epoch,
            bool,
        )
        or epoch <= 0
    ):

        raise ValueError(
            f"Checkpoint epoch must be positive integer: {epoch!r}"
        )

    if (
        not math.isfinite(
            weighted_dev_loss
        )
        or weighted_dev_loss < 0.0
    ):

        raise ValueError(
            "Checkpoint weighted dev loss must be finite "
            "and non-negative."
        )

    state = _clone_model_state_to_cpu(
        model
    )

    state_hash = state_dict_sha256(
        state
    )

    current_hash = model_state_sha256(
        model
    )

    if state_hash != current_hash:

        raise RuntimeError(
            "CPU checkpoint copy does not reproduce current "
            "model-state fingerprint:\n"
            f"  current={current_hash}\n"
            f"  checkpoint={state_hash}"
        )

    return RawArgminModelCheckpoint(
        stage=stage,
        epoch=epoch,
        weighted_dev_loss=float(
            weighted_dev_loss
        ),
        model_state_sha256=state_hash,
        model_state_dict=state,
    )


def restore_raw_argmin_model_checkpoint(
    *,
    model: nn.Module,
    checkpoint: RawArgminModelCheckpoint,
) -> None:
    """
    Restore only model parameters/buffers.

    Training/eval mode and requires_grad state are deliberately not
    restored by state_dict; the caller must subsequently apply the
    appropriate Stage-A, Stage-B or evaluation contract.
    """

    checkpoint_hash = state_dict_sha256(
        checkpoint.model_state_dict
    )

    if (
        checkpoint_hash
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Stored checkpoint state has changed since capture:\n"
            f"  recorded={checkpoint.model_state_sha256}\n"
            f"  current={checkpoint_hash}"
        )

    incompatible = model.load_state_dict(
        checkpoint.model_state_dict,
        strict=True,
    )

    if incompatible.missing_keys:

        raise RuntimeError(
            "Checkpoint restore produced missing keys:\n"
            f"  {incompatible.missing_keys}"
        )

    if incompatible.unexpected_keys:

        raise RuntimeError(
            "Checkpoint restore produced unexpected keys:\n"
            f"  {incompatible.unexpected_keys}"
        )

    restored_hash = model_state_sha256(
        model
    )

    if (
        restored_hash
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Restored model does not match checkpoint fingerprint:\n"
            f"  expected={checkpoint.model_state_sha256}\n"
            f"  actual={restored_hash}"
        )


# ======================================================================
# Per-epoch control decision
# ======================================================================

@dataclass(
    frozen=True
)
class StageControlDecision:

    stage: str
    epoch: int

    weighted_dev_loss: float

    raw_checkpoint_updated: bool

    raw_best_epoch: int
    raw_best_loss: float

    patience_anchor_initialized: bool
    meaningful_improvement: bool | None

    patience_anchor_loss: float
    patience_counter: int
    patience_epochs: int

    stop_due_to_patience: bool
    stop_due_to_maximum_epochs: bool

    should_stop: bool


# ======================================================================
# Stateful stage controller
# ======================================================================

class StageTrainingController:
    """
    Track frozen Stage-A / Stage-B stopping and raw checkpoint behavior.

    The first dev observation:
        - establishes the patience anchor;
        - establishes the raw-best checkpoint;
        - starts patience at zero.

    Later observations:
        raw checkpoint:
            update for ANY strict reduction in weighted dev loss.

        patience:
            reset only if:
                current < anchor * (1 - 0.005)

            otherwise increment.

    An exact equality with the current raw minimum keeps the earlier
    epoch. This is the ordinary first-occurrence interpretation of argmin
    over the ordered epoch history.
    """

    def __init__(
        self,
        *,
        experiment_cfg: Mapping[str, Any],
        stage: str,
    ) -> None:

        self.contract = (
            load_stage_stopping_contract(
                experiment_cfg=experiment_cfg,
                stage=stage,
            )
        )

        self._last_epoch = 0

        self._patience_anchor_loss: (
            float
            | None
        ) = None

        self._patience_counter = 0

        self._raw_best_loss: (
            float
            | None
        ) = None

        self._raw_best_epoch: (
            int
            | None
        ) = None

        self._raw_best_checkpoint: (
            RawArgminModelCheckpoint
            | None
        ) = None

        self._stopped = False

    @property
    def raw_best_checkpoint(
        self,
    ) -> RawArgminModelCheckpoint:

        if self._raw_best_checkpoint is None:

            raise RuntimeError(
                "No raw-best checkpoint has been observed yet."
            )

        return self._raw_best_checkpoint

    @property
    def raw_best_loss(
        self,
    ) -> float:

        if self._raw_best_loss is None:

            raise RuntimeError(
                "No raw-best loss has been observed yet."
            )

        return self._raw_best_loss

    @property
    def raw_best_epoch(
        self,
    ) -> int:

        if self._raw_best_epoch is None:

            raise RuntimeError(
                "No raw-best epoch has been observed yet."
            )

        return self._raw_best_epoch

    @property
    def patience_counter(
        self,
    ) -> int:

        return self._patience_counter

    @property
    def stopped(
        self,
    ) -> bool:

        return self._stopped

    def observe_dev_epoch(
        self,
        *,
        model: nn.Module,
        epoch: int,
        weighted_dev_loss: float,
    ) -> StageControlDecision:
        """
        Consume exactly one completed dev evaluation.

        Must be called once per epoch, in increasing consecutive order.
        """

        if self._stopped:

            raise RuntimeError(
                "Cannot observe additional epochs after "
                "the stage controller has stopped."
            )

        if (
            not isinstance(
                epoch,
                int,
            )
            or isinstance(
                epoch,
                bool,
            )
        ):

            raise TypeError(
                f"epoch must be integer, got {epoch!r}"
            )

        expected_epoch = (
            self._last_epoch
            + 1
        )

        if epoch != expected_epoch:

            raise RuntimeError(
                "Stage epochs must be observed consecutively:\n"
                f"  expected={expected_epoch}\n"
                f"  actual={epoch}"
            )

        if (
            not math.isfinite(
                weighted_dev_loss
            )
            or weighted_dev_loss < 0.0
        ):

            raise ValueError(
                "weighted_dev_loss must be finite "
                "and non-negative."
            )

        current_loss = float(
            weighted_dev_loss
        )

        # ==============================================================
        # RAW ARGMIN
        #
        # Completely independent from the 0.5% patience rule.
        # ==============================================================

        raw_checkpoint_updated = False

        if (
            self._raw_best_loss is None
            or current_loss
            < self._raw_best_loss
        ):

            checkpoint = (
                capture_raw_argmin_model_checkpoint(
                    model=model,
                    stage=self.contract.stage,
                    epoch=epoch,
                    weighted_dev_loss=current_loss,
                )
            )

            self._raw_best_loss = (
                current_loss
            )

            self._raw_best_epoch = (
                epoch
            )

            self._raw_best_checkpoint = (
                checkpoint
            )

            raw_checkpoint_updated = True

        # ==============================================================
        # PATIENCE
        # ==============================================================

        patience_anchor_initialized = False

        meaningful_improvement: (
            bool
            | None
        )

        if self._patience_anchor_loss is None:

            self._patience_anchor_loss = (
                current_loss
            )

            self._patience_counter = 0

            patience_anchor_initialized = (
                True
            )

            # Epoch 1 establishes the reference; it is not logically an
            # improvement relative to a previous anchor.
            meaningful_improvement = None

        else:

            threshold = (
                self._patience_anchor_loss
                *
                (
                    1.0
                    -
                    self.contract
                    .meaningful_relative_improvement_fraction
                )
            )

            if current_loss < threshold:

                meaningful_improvement = (
                    True
                )

                self._patience_anchor_loss = (
                    current_loss
                )

                self._patience_counter = 0

            else:

                meaningful_improvement = (
                    False
                )

                self._patience_counter += 1

        # ==============================================================
        # STOPPING
        #
        # "patience_epochs = N" means stop once N consecutive observed
        # epochs fail the meaningful-improvement criterion.
        # ==============================================================

        stop_due_to_patience = (
            self._patience_counter
            >= self.contract.patience_epochs
        )

        stop_due_to_maximum_epochs = (
            epoch
            >= self.contract.maximum_epochs
        )

        should_stop = (
            stop_due_to_patience
            or stop_due_to_maximum_epochs
        )

        self._last_epoch = epoch

        if should_stop:

            self._stopped = True

        if self._raw_best_loss is None:

            raise RuntimeError(
                "Raw-best state unexpectedly missing."
            )

        if self._raw_best_epoch is None:

            raise RuntimeError(
                "Raw-best epoch unexpectedly missing."
            )

        if self._patience_anchor_loss is None:

            raise RuntimeError(
                "Patience anchor unexpectedly missing."
            )

        return StageControlDecision(
            stage=self.contract.stage,
            epoch=epoch,
            weighted_dev_loss=current_loss,
            raw_checkpoint_updated=(
                raw_checkpoint_updated
            ),
            raw_best_epoch=(
                self._raw_best_epoch
            ),
            raw_best_loss=(
                self._raw_best_loss
            ),
            patience_anchor_initialized=(
                patience_anchor_initialized
            ),
            meaningful_improvement=(
                meaningful_improvement
            ),
            patience_anchor_loss=(
                self._patience_anchor_loss
            ),
            patience_counter=(
                self._patience_counter
            ),
            patience_epochs=(
                self.contract.patience_epochs
            ),
            stop_due_to_patience=(
                stop_due_to_patience
            ),
            stop_due_to_maximum_epochs=(
                stop_due_to_maximum_epochs
            ),
            should_stop=(
                should_stop
            ),
        )