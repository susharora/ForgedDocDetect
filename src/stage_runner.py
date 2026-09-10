"""
Multi-epoch Stage-A / Stage-B execution for frozen Tech-2 development.

This module connects the already-validated primitives:

    train_one_epoch()
        ->
    evaluate_dev_one_epoch()
        ->
    compute_dev_auroc()
        ->
    StageTrainingController
        ->
    raw-argmin checkpoint restore

Scientific responsibilities
---------------------------
For either Stage A or Stage B:

1. execute complete project_train epochs;
2. execute complete dev_val evaluation after every training epoch;
3. record weighted dev CE;
4. record diagnostic dev AUROC every epoch;
5. update the raw-argmin model checkpoint on every strict dev-loss
   reduction;
6. control patience using the independent frozen 0.5% rule;
7. stop on patience or the frozen maximum epoch;
8. restore the exact raw-argmin model state after the stage finishes.

Additional Stage-B responsibilities
-----------------------------------
- put the frozen BatchNorm-domain-adaptation transition note in epoch 1
  history;
- compute:

      best_dev_auroc_over_run
          -
      dev_auroc_at_loss_argmin_checkpoint

- flag protocol review only when the difference is strictly > 0.01;
- NEVER switch the selected checkpoint because of AUROC.

Important boundaries
--------------------
This module does NOT:

- construct the initial Stage-A model;
- construct Stage-A AdamW;
- perform the Stage-A -> Stage-B branch reset;
- construct Stage-B branches;
- screen backbone learning rates;
- select a learning rate;
- select a resolution;
- derive an FPR10 threshold;
- access held-out test;
- serialize checkpoints;
- perform Grad-CAM.

The optimizer passed into this function belongs only to the executed
stage.

After the stage finishes, the model is restored to an earlier raw-best
checkpoint while the optimizer remains at its final-epoch state.
Therefore the optimizer MUST NOT be reused after this function returns.

That is consistent with the frozen protocol:
- Stage B always gets a fresh optimizer;
- no continuation occurs after a completed Stage-B run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import logging
import time

PROGRESS_LOGGER = logging.getLogger(
    "tech2.resnet18"
)

from src.development_metrics import (
    DevAUROCResult,
    compute_dev_auroc,
    load_dev_auroc_contract,
)

from src.engine import (
    EpochLossSummary,
    TrainingEpochResult,
    evaluate_dev_one_epoch,
    train_one_epoch,
)

from src.modeling import (
    apply_evaluation_contract,
)

from src.objective import (
    WeightedCrossEntropyObjective,
)

from src.training_control import (
    RawArgminModelCheckpoint,
    StageControlDecision,
    StageTrainingController,
    model_state_sha256,
    restore_raw_argmin_model_checkpoint,
)


# ======================================================================
# Public types
# ======================================================================

TrainingStage = Literal[
    "stage_a",
    "stage_b",
]


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
# Stage-B transition note
# ======================================================================

def load_stage_b_transition_note(
    *,
    experiment_cfg: Mapping[str, Any],
) -> str:
    """
    Load and validate the required Stage-B epoch-1 transition note.
    """

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

    note_cfg = require_mapping(
        require_key(
            stage_b_cfg,
            "epoch_1_transition_note",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.epoch_1_transition_note",
    )

    required = require_key(
        note_cfg,
        "required_in_history",
        "transfer_learning.stage_b.epoch_1_transition_note",
    )

    if required is not True:

        raise ValueError(
            "Frozen Stage-B epoch-1 transition note "
            "must be required in history."
        )

    text = str(
        require_key(
            note_cfg,
            "text",
            "transfer_learning.stage_b.epoch_1_transition_note",
        )
    ).strip()

    if not text:

        raise ValueError(
            "Frozen Stage-B epoch-1 transition note is empty."
        )

    expected = (
        "Backbone unfrozen and BatchNorm running statistics begin "
        "FantasyID domain adaptation; transient dev-loss movement "
        "is expected."
    )

    if text != expected:

        raise ValueError(
            "Frozen Stage-B epoch-1 transition note changed:\n"
            f"  expected={expected!r}\n"
            f"  actual={text!r}"
        )

    return text


# ======================================================================
# Per-epoch immutable history
# ======================================================================

@dataclass(
    frozen=True
)
class StageEpochRecord:

    stage: TrainingStage
    epoch: int

    train_loss: EpochLossSummary
    train_optimizer_steps: int

    dev_loss: EpochLossSummary
    dev_auroc: DevAUROCResult

    control: StageControlDecision

    transition_note: str | None


# ======================================================================
# Stage-B AUROC disagreement result
# ======================================================================

@dataclass(
    frozen=True
)
class StageBAUROCDisagreement:

    best_dev_auroc: float
    best_dev_auroc_epoch: int

    loss_argmin_checkpoint_epoch: int
    auroc_at_loss_argmin_checkpoint: float

    difference: float
    threshold: float

    protocol_review_flag: bool

    automatically_switch_checkpoint: bool


# ======================================================================
# Complete stage result
# ======================================================================

@dataclass(
    frozen=True
)
class StageRunResult:

    stage: TrainingStage

    epochs_completed: int

    history: tuple[
        StageEpochRecord,
        ...,
    ]

    stop_reason: str

    raw_best_checkpoint: RawArgminModelCheckpoint

    raw_best_epoch: int
    raw_best_weighted_dev_loss: float
    dev_auroc_at_raw_best_checkpoint: float

    best_dev_auroc: float
    best_dev_auroc_epoch: int

    stage_b_auroc_disagreement: (
        StageBAUROCDisagreement
        | None
    )

    restored_selected_model_state_sha256: str

    model_left_in_eval_mode: bool

    optimizer_reuse_permitted: bool


# ======================================================================
# Stage validation
# ======================================================================

def _validate_stage(
    stage: str,
) -> TrainingStage:

    if stage not in {
        "stage_a",
        "stage_b",
    }:

        raise ValueError(
            f"Unsupported training stage: {stage!r}"
        )

    return stage


# ======================================================================
# Stop reason
# ======================================================================

def _stop_reason(
    decision: StageControlDecision,
) -> str:

    if not decision.should_stop:

        raise RuntimeError(
            "Cannot derive stop reason from a non-stopping decision."
        )

    if (
        decision.stop_due_to_patience
        and decision.stop_due_to_maximum_epochs
    ):

        return (
            "patience_and_maximum_epochs"
        )

    if decision.stop_due_to_patience:

        return "patience"

    if decision.stop_due_to_maximum_epochs:

        return "maximum_epochs"

    raise RuntimeError(
        "Stopping decision has no stopping cause."
    )


# ======================================================================
# Stage-B AUROC disagreement guard
# ======================================================================

def _compute_stage_b_auroc_disagreement(
    *,
    experiment_cfg: Mapping[str, Any],
    best_dev_auroc: float,
    best_dev_auroc_epoch: int,
    raw_best_checkpoint: RawArgminModelCheckpoint,
    dev_auroc_at_raw_best_checkpoint: float,
) -> StageBAUROCDisagreement:

    contract = load_dev_auroc_contract(
        experiment_cfg=experiment_cfg,
    )

    if (
        contract
        .stage_b_disagreement_guard_enabled
        is not True
    ):

        raise RuntimeError(
            "Stage-B AUROC disagreement guard unexpectedly disabled."
        )

    if (
        contract
        .automatically_switch_checkpoint
        is not False
    ):

        raise RuntimeError(
            "Frozen protocol forbids automatic AUROC checkpoint switch."
        )

    for (
        label,
        value,
    ) in (
        (
            "best_dev_auroc",
            best_dev_auroc,
        ),
        (
            "dev_auroc_at_raw_best_checkpoint",
            dev_auroc_at_raw_best_checkpoint,
        ),
    ):

        if not math.isfinite(
            value
        ):

            raise ValueError(
                f"{label} must be finite."
            )

        if not (
            0.0
            <= value
            <= 1.0
        ):

            raise ValueError(
                f"{label} outside [0,1]: {value}"
            )

    if (
        not isinstance(
            best_dev_auroc_epoch,
            int,
        )
        or isinstance(
            best_dev_auroc_epoch,
            bool,
        )
        or best_dev_auroc_epoch <= 0
    ):

        raise ValueError(
            "best_dev_auroc_epoch must be a positive integer."
        )

    difference = (
        float(
            best_dev_auroc
        )
        -
        float(
            dev_auroc_at_raw_best_checkpoint
        )
    )

    # The selected checkpoint is one of the epochs contributing to
    # best_dev_auroc, so a materially negative difference is impossible.
    if difference < -1.0e-15:

        raise RuntimeError(
            "Stage-B AUROC disagreement is negative:\n"
            f"  best_dev_auroc={best_dev_auroc}\n"
            f"  selected_checkpoint_auroc="
            f"{dev_auroc_at_raw_best_checkpoint}\n"
            f"  difference={difference}"
        )

    # Protect against tiny floating subtraction noise.
    if difference < 0.0:

        difference = 0.0

    threshold = float(
        contract
        .stage_b_disagreement_threshold
    )

    if threshold != 0.01:

        raise RuntimeError(
            "Frozen Stage-B AUROC disagreement threshold "
            "must equal 0.01."
        )

    protocol_review_flag = (
        difference
        > threshold
    )

    return StageBAUROCDisagreement(
        best_dev_auroc=float(
            best_dev_auroc
        ),
        best_dev_auroc_epoch=(
            best_dev_auroc_epoch
        ),
        loss_argmin_checkpoint_epoch=(
            raw_best_checkpoint.epoch
        ),
        auroc_at_loss_argmin_checkpoint=float(
            dev_auroc_at_raw_best_checkpoint
        ),
        difference=difference,
        threshold=threshold,
        protocol_review_flag=(
            protocol_review_flag
        ),
        automatically_switch_checkpoint=False,
    )


# ======================================================================
# Cross-result reconciliation
# ======================================================================

def _reconcile_history(
    *,
    stage: TrainingStage,
    history: list[
        StageEpochRecord
    ],
    controller: StageTrainingController,
    selected_auroc: float,
    best_auroc: float,
    best_auroc_epoch: int,
) -> None:

    if not history:

        raise RuntimeError(
            "Completed stage unexpectedly has empty history."
        )

    epochs = [
        record.epoch
        for record
        in history
    ]

    expected_epochs = list(
        range(
            1,
            len(
                history
            )
            + 1,
        )
    )

    if epochs != expected_epochs:

        raise RuntimeError(
            "Stage history epochs are not consecutive:\n"
            f"  expected={expected_epochs}\n"
            f"  actual={epochs}"
        )

    if (
        history[
            -1
        ]
        .control
        .should_stop
        is not True
    ):

        raise RuntimeError(
            "Final stage history record is not a stopping epoch."
        )

    for record in history[
        :-1
    ]:

        if record.control.should_stop:

            raise RuntimeError(
                "Stage contains history after an earlier stop decision."
            )

    if not controller.stopped:

        raise RuntimeError(
            "Stage loop ended but controller is not stopped."
        )

    raw_best_epoch = (
        controller.raw_best_epoch
    )

    raw_best_records = [
        record
        for record
        in history
        if record.epoch
        == raw_best_epoch
    ]

    if len(
        raw_best_records
    ) != 1:

        raise RuntimeError(
            "Raw-best epoch does not map to exactly one history record."
        )

    raw_record = raw_best_records[
        0
    ]

    if (
        raw_record
        .dev_loss
        .weighted_loss
        != controller.raw_best_loss
    ):

        raise RuntimeError(
            "Raw-best history loss disagrees with controller."
        )

    if (
        raw_record
        .dev_auroc
        .auroc
        != selected_auroc
    ):

        raise RuntimeError(
            "Stored AUROC-at-loss-argmin does not match history."
        )

    history_best_auroc = max(
        record.dev_auroc.auroc
        for record
        in history
    )

    if history_best_auroc != best_auroc:

        raise RuntimeError(
            "Tracked best AUROC disagrees with epoch history."
        )

    # First occurrence is intentionally retained on exact AUROC ties.
    expected_best_epoch = next(
        record.epoch
        for record
        in history
        if record.dev_auroc.auroc
        == history_best_auroc
    )

    if (
        expected_best_epoch
        != best_auroc_epoch
    ):

        raise RuntimeError(
            "Best-AUROC epoch does not use first-occurrence tie rule."
        )

    if stage == "stage_b":

        notes = [
            record.transition_note
            for record
            in history
        ]

        if notes[
            0
        ] is None:

            raise RuntimeError(
                "Stage-B epoch 1 is missing required transition note."
            )

        if any(
            note is not None
            for note
            in notes[
                1:
            ]
        ):

            raise RuntimeError(
                "Stage-B transition note must occur only at epoch 1."
            )

    else:

        if any(
            record.transition_note
            is not None
            for record
            in history
        ):

            raise RuntimeError(
                "Stage-A history must not contain Stage-B transition note."
            )


# ======================================================================
# Multi-epoch stage runner
# ======================================================================

def run_training_stage(
    *,
    experiment_cfg: Mapping[str, Any],
    stage: TrainingStage,
    model: nn.Module,
    project_train_loader: DataLoader,
    dev_val_loader: DataLoader,
    optimizer: Optimizer,
    objective: WeightedCrossEntropyObjective,
    device: torch.device | str,
) -> StageRunResult:
    """
    Execute one complete frozen Stage-A or Stage-B training stage.

    The caller is responsible for constructing the correct starting:

        model
        DataLoaders
        optimizer
        objective

    This function owns only repeated epoch execution, development
    diagnostics, stopping/checkpoint control and final model selection.

    On successful return:
        - the model contains the exact raw-best selected state;
        - the model is in evaluation mode;
        - the passed optimizer MUST be discarded.
    """

    validated_stage = _validate_stage(
        stage
    )

    execution_device = torch.device(
        device
    )

    controller = StageTrainingController(
        experiment_cfg=experiment_cfg,
        stage=validated_stage,
    )

    # Reassert the AUROC scientific contract before beginning any
    # training epoch.
    auroc_contract = load_dev_auroc_contract(
        experiment_cfg=experiment_cfg,
    )

    if (
        auroc_contract
        .diagnostic_enabled
        is not True
    ):

        raise RuntimeError(
            "Dev AUROC unexpectedly disabled."
        )

    if (
        auroc_contract
        .participates_in_checkpoint_selection
        is not False
    ):

        raise RuntimeError(
            "AUROC must not participate in checkpoint selection."
        )

    transition_note: (
        str
        | None
    ) = None

    if validated_stage == "stage_b":

        transition_note = (
            load_stage_b_transition_note(
                experiment_cfg=experiment_cfg,
            )
        )

    history: list[
        StageEpochRecord
    ] = []

    selected_checkpoint_auroc: (
        float
        | None
    ) = None

    best_dev_auroc: (
        float
        | None
    ) = None

    best_dev_auroc_epoch: (
        int
        | None
    ) = None

    final_decision: (
        StageControlDecision
        | None
    ) = None

    # ------------------------------------------------------------------
    # Frozen maximum epoch comes directly from the validated controller.
    #
    # The loop still checks decision.should_stop every epoch. The range
    # bound is a second structural guarantee against accidental overrun.
    # ------------------------------------------------------------------

    maximum_epochs = (
        controller
        .contract
        .maximum_epochs
    )

    PROGRESS_LOGGER.info(
        "Stage start | stage=%s | maximum_epochs=%d | patience=%d",
        validated_stage,
        maximum_epochs,
        controller.contract.patience_epochs,
        )


    for epoch in range(
        1,
        maximum_epochs
        + 1,
    ):

        epoch_started = (
        time.perf_counter()
    )

    # ==============================================================
    # Complete project_train epoch
    # ==============================================================

        train_started = (
            time.perf_counter()
        )

        train_result: TrainingEpochResult = (
            train_one_epoch(
                model=model,
                loader=project_train_loader,
                optimizer=optimizer,
                objective=objective,
                device=execution_device,
                stage=validated_stage,
            )
        )

        train_seconds = (
            time.perf_counter()
            - train_started
        )

        if (
            train_result.stage
            != validated_stage
        ):

            raise RuntimeError(
                "Training epoch reported wrong stage."
            )

        # ==============================================================
        # Complete dev_val evaluation
        # ==============================================================

        dev_started = (
            time.perf_counter()
        )

        dev_result = (
            evaluate_dev_one_epoch(
                model=model,
                loader=dev_val_loader,
                objective=objective,
                device=execution_device,
            )
        )

        dev_seconds = (
            time.perf_counter()
            - dev_started
        )
    

        # ==============================================================
        # Diagnostic AUROC
        # ==============================================================

        dev_auroc = compute_dev_auroc(
            experiment_cfg=experiment_cfg,
            logits=dev_result.logits,
            targets=dev_result.targets,
        )

        current_auroc = float(
            dev_auroc.auroc
        )

        if (
            best_dev_auroc is None
            or current_auroc
            > best_dev_auroc
        ):

            best_dev_auroc = (
                current_auroc
            )

            best_dev_auroc_epoch = (
                epoch
            )

        # ==============================================================
        # Raw checkpoint + patience
        #
        # IMPORTANT:
        # This consumes weighted DEV LOSS only.
        # AUROC is never passed to StageTrainingController.
        # ==============================================================

        decision = (
            controller.observe_dev_epoch(
                model=model,
                epoch=epoch,
                weighted_dev_loss=(
                    dev_result
                    .loss
                    .weighted_loss
                ),
            )
        )

        if decision.raw_checkpoint_updated:

            selected_checkpoint_auroc = (
                current_auroc
            )

        # First epoch always establishes raw best.
        if selected_checkpoint_auroc is None:

            raise RuntimeError(
                "Raw-best AUROC was not initialized."
            )

        epoch_note: (
            str
            | None
        ) = None

        if (
            validated_stage
            == "stage_b"
            and epoch == 1
        ):

            epoch_note = (
                transition_note
            )

        history.append(
            StageEpochRecord(
                stage=validated_stage,
                epoch=epoch,
                train_loss=(
                    train_result.loss
                ),
                train_optimizer_steps=(
                    train_result.optimizer_steps
                ),
                dev_loss=(
                    dev_result.loss
                ),
                dev_auroc=dev_auroc,
                control=decision,
                transition_note=(
                    epoch_note
                ),
            )
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started
            )

        PROGRESS_LOGGER.info(
            "Epoch complete | stage=%s | epoch=%d/%d | "
            "train_loss=%.12f | dev_loss=%.12f | AUROC=%.12f | "
            "raw_best_epoch=%d | raw_best_loss=%.12f | "
            "checkpoint_updated=%s | patience=%d/%d | "
            "meaningful_improvement=%s | stop=%s | "
            "train_seconds=%.3f | dev_seconds=%.3f | epoch_seconds=%.3f",
            validated_stage,
            epoch,
            maximum_epochs,
            train_result.loss.weighted_loss,
            dev_result.loss.weighted_loss,
            dev_auroc.auroc,
            decision.raw_best_epoch,
            decision.raw_best_loss,
            decision.raw_checkpoint_updated,
            decision.patience_counter,
            decision.patience_epochs,
            decision.meaningful_improvement,
            decision.should_stop,
            train_seconds,
            dev_seconds,
            epoch_seconds,
        )

        if epoch_note is not None:

            PROGRESS_LOGGER.info(
                "Stage transition | stage=%s | epoch=%d | %s",
                validated_stage,
                epoch,
                epoch_note,
            )


        final_decision = (
            decision
        )

        # Raw dev logits are deliberately not retained in multi-epoch
        # history. AUROC has now been computed and the single-epoch
        # engine already validated their manifest order.
        del dev_result

        if decision.should_stop:

            break

    # ==================================================================
    # Stage completion checks
    # ==================================================================

    if final_decision is None:

        raise RuntimeError(
            "Stage executed zero epochs."
        )

    if not final_decision.should_stop:

        raise RuntimeError(
            "Stage exhausted loop without a frozen stop decision."
        )

    if not controller.stopped:

        raise RuntimeError(
            "Stage controller did not enter stopped state."
        )

    if selected_checkpoint_auroc is None:

        raise RuntimeError(
            "Selected checkpoint AUROC missing after stage completion."
        )

    if best_dev_auroc is None:

        raise RuntimeError(
            "Best dev AUROC missing after stage completion."
        )

    if best_dev_auroc_epoch is None:

        raise RuntimeError(
            "Best dev AUROC epoch missing after stage completion."
        )

    _reconcile_history(
        stage=validated_stage,
        history=history,
        controller=controller,
        selected_auroc=(
            selected_checkpoint_auroc
        ),
        best_auroc=(
            best_dev_auroc
        ),
        best_auroc_epoch=(
            best_dev_auroc_epoch
        ),
    )

    # ==================================================================
    # Exact raw-best restore
    # ==================================================================

    selected_checkpoint = (
        controller
        .raw_best_checkpoint
    )

    restore_raw_argmin_model_checkpoint(
        model=model,
        checkpoint=selected_checkpoint,
    )

    restored_hash = (
        model_state_sha256(
            model
        )
    )

    if (
        restored_hash
        != selected_checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Final selected model restore failed:\n"
            f"  expected="
            f"{selected_checkpoint.model_state_sha256}\n"
            f"  actual={restored_hash}"
        )

    # Leave selected model in deterministic inference/evaluation mode.
    # model.eval() changes module modes only; parameters and BN buffers
    # must remain exactly the selected checkpoint state.
    apply_evaluation_contract(
        model
    )

    restored_hash_after_eval = (
        model_state_sha256(
            model
        )
    )

    if (
        restored_hash_after_eval
        != selected_checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Applying evaluation mode changed selected model state."
        )

    if model.training:

        raise RuntimeError(
            "Completed stage must leave selected model in eval mode."
        )

    # ==================================================================
    # Stage-B AUROC disagreement guard
    # ==================================================================

    stage_b_disagreement: (
        StageBAUROCDisagreement
        | None
    ) = None

    if validated_stage == "stage_b":

        stage_b_disagreement = (
            _compute_stage_b_auroc_disagreement(
                experiment_cfg=experiment_cfg,
                best_dev_auroc=(
                    best_dev_auroc
                ),
                best_dev_auroc_epoch=(
                    best_dev_auroc_epoch
                ),
                raw_best_checkpoint=(
                    selected_checkpoint
                ),
                dev_auroc_at_raw_best_checkpoint=(
                    selected_checkpoint_auroc
                ),
            )
        )

        # Explicitly reassert that AUROC did not alter checkpoint choice.
        if (
            selected_checkpoint.epoch
            != controller.raw_best_epoch
        ):

            raise RuntimeError(
                "AUROC guard unexpectedly changed checkpoint selection."
            )

    stop_reason = _stop_reason(
        final_decision
    )

    PROGRESS_LOGGER.info(
        "Stage complete | stage=%s | epochs=%d | stop=%s | "
        "raw_best_epoch=%d | raw_best_loss=%.12f | "
        "AUROC_at_raw_best=%.12f | best_AUROC=%.12f | "
        "best_AUROC_epoch=%d",
        validated_stage,
        len(
            history
        ),
        stop_reason,
        controller.raw_best_epoch,
        controller.raw_best_loss,
        selected_checkpoint_auroc,
        best_dev_auroc,
        best_dev_auroc_epoch,
    )


    return StageRunResult(
        stage=validated_stage,
        epochs_completed=len(
            history
        ),
        history=tuple(
            history
        ),
        stop_reason=stop_reason,
        raw_best_checkpoint=(
            selected_checkpoint
        ),
        raw_best_epoch=(
            controller.raw_best_epoch
        ),
        raw_best_weighted_dev_loss=(
            controller.raw_best_loss
        ),
        dev_auroc_at_raw_best_checkpoint=float(
            selected_checkpoint_auroc
        ),
        best_dev_auroc=float(
            best_dev_auroc
        ),
        best_dev_auroc_epoch=(
            best_dev_auroc_epoch
        ),
        stage_b_auroc_disagreement=(
            stage_b_disagreement
        ),
        restored_selected_model_state_sha256=(
            restored_hash_after_eval
        ),
        model_left_in_eval_mode=True,

        # Critical:
        # the optimizer corresponds to the FINAL executed epoch model,
        # whereas model has now been restored to RAW BEST.
        optimizer_reuse_permitted=False,
    )