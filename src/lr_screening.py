"""
One-resolution backbone-LR screening orchestration for frozen Tech-2.

For one resolution at screening seed 8:

    configure run reproducibility
        ->
    build fresh development DataLoaders
        ->
    build fresh ResNet-18
        ->
    Stage A
        ->
    exact Stage-A raw-best checkpoint
        ->
    Stage-B branch LR = 3e-5
    Stage-B branch LR = 1e-4
    Stage-B branch LR = 3e-4
        ->
    select lowest raw-best weighted dev CE
        ->
    exact loss tie: choose lower backbone LR

Critical scientific properties
------------------------------
- Stage A runs exactly once per resolution.
- Every Stage-B candidate starts from the same Stage-A raw-best MODEL.
- Every Stage-B candidate independently resets RNG/DataLoader state
  from run_seed through src.stage_b_branching.
- AUROC never changes LR selection.
- AUROC disagreement flags are retained for protocol review.
- held-out test is unsupported.

This module performs the scientific computation in memory.

It deliberately does NOT:
- write checkpoint files;
- write run.yaml;
- write metric CSV/YAML artifacts;
- compare r256 against r512;
- perform multi-seed confirmation;
- derive FPR10 thresholds;
- access held-out test;
- perform Grad-CAM.

Durable artifact writing is the next layer and must be implemented
before launching the real screening experiment.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from src.dataloading import (
    DevelopmentDataLoaders,
    build_development_dataloaders,
)

from src.modeling import (
    ModelBuildProvenance,
    apply_stage_a_contract,
    assert_stage_a_contract,
    build_resnet18_classifier,
)

from src.objective import (
    WeightedCrossEntropyObjective,
)

from src.optimization import (
    OptimizerBuildEvidence,
    build_stage_a_optimizer,
)

from src.reproducibility import (
    ReproducibilityState,
    configure_run_reproducibility,
)

from src.stage_b_branching import (
    StageBBranchInitializationEvidence,
    build_stage_b_branch,
    dataloader_generator_state_sha256,
    load_stage_b_branch_contract,
)

from src.stage_runner import (
    StageRunResult,
    run_training_stage,
)

from src.training_control import (
    model_state_sha256,
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
# Screening contract
# ======================================================================

@dataclass(
    frozen=True
)
class LRScreeningContract:

    screening_seed: int

    candidate_resolutions: tuple[
        str,
        ...,
    ]

    backbone_lr_candidates: tuple[
        float,
        ...,
    ]

    classifier_lr_fixed: float

    stage_a_run_once_per_resolution: bool

    stage_b_same_stage_a_checkpoint: bool

    selection_metric: str
    selection_metric_source: str

    selection_rule: str
    exact_tie_breaker: str

    auroc_does_not_change_selection: bool


def load_lr_screening_contract(
    *,
    experiment_cfg: Mapping[str, Any],
) -> LRScreeningContract:

    development_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "development_selection",
            "experiment_config",
        ),
        "development_selection",
    )

    screening_cfg = require_mapping(
        require_key(
            development_cfg,
            "lr_screening",
            "development_selection",
        ),
        "development_selection.lr_screening",
    )

    screening_seed = int(
        require_key(
            screening_cfg,
            "seed",
            "development_selection.lr_screening",
        )
    )

    if screening_seed != 8:

        raise ValueError(
            "Frozen LR screening seed must equal 8."
        )

    candidate_resolutions = tuple(
        str(
            value
        )
        for value
        in require_key(
            screening_cfg,
            "candidate_resolutions",
            "development_selection.lr_screening",
        )
    )

    if candidate_resolutions != (
        "r256",
        "r512",
    ):

        raise ValueError(
            "Frozen LR-screening resolutions changed:\n"
            f"  actual={candidate_resolutions}"
        )

    backbone_lr_candidates = tuple(
        float(
            value
        )
        for value
        in require_key(
            screening_cfg,
            "backbone_lr_candidates",
            "development_selection.lr_screening",
        )
    )

    expected_lrs = (
        0.00003,
        0.0001,
        0.0003,
    )

    if backbone_lr_candidates != expected_lrs:

        raise ValueError(
            "Frozen LR-screening candidate set changed:\n"
            f"  expected={expected_lrs}\n"
            f"  actual={backbone_lr_candidates}"
        )

    classifier_lr_fixed = float(
        require_key(
            screening_cfg,
            "classifier_lr_fixed",
            "development_selection.lr_screening",
        )
    )

    if classifier_lr_fixed != 0.001:

        raise ValueError(
            "Frozen screening classifier LR must equal 1e-3."
        )

    stage_a_cfg = require_mapping(
        require_key(
            screening_cfg,
            "stage_a",
            "development_selection.lr_screening",
        ),
        "development_selection.lr_screening.stage_a",
    )

    stage_a_once = require_key(
        stage_a_cfg,
        "run_once_per_resolution",
        "development_selection.lr_screening.stage_a",
    )

    if stage_a_once is not True:

        raise ValueError(
            "Stage A must run once per screening resolution."
        )

    stage_b_cfg = require_mapping(
        require_key(
            screening_cfg,
            "stage_b",
            "development_selection.lr_screening",
        ),
        "development_selection.lr_screening.stage_b",
    )

    same_stage_a_checkpoint = require_key(
        stage_b_cfg,
        "branch_all_lr_candidates_from_same_stage_a_checkpoint",
        "development_selection.lr_screening.stage_b",
    )

    if same_stage_a_checkpoint is not True:

        raise ValueError(
            "All Stage-B LR candidates must fork from "
            "one Stage-A checkpoint."
        )

    metric_cfg = require_mapping(
        require_key(
            screening_cfg,
            "lr_selection_metric",
            "development_selection.lr_screening",
        ),
        "development_selection.lr_screening.lr_selection_metric",
    )

    metric_name = str(
        require_key(
            metric_cfg,
            "name",
            (
                "development_selection.lr_screening."
                "lr_selection_metric"
            ),
        )
    )

    if (
        metric_name
        != "class_weighted_dev_cross_entropy"
    ):

        raise ValueError(
            "Frozen LR-selection metric changed."
        )

    metric_source = str(
        require_key(
            metric_cfg,
            "source",
            (
                "development_selection.lr_screening."
                "lr_selection_metric"
            ),
        )
    )

    if metric_source != "raw_argmin_checkpoint":

        raise ValueError(
            "LR selection must use each branch's raw-argmin checkpoint."
        )

    select_cfg = require_mapping(
        require_key(
            screening_cfg,
            "select",
            "development_selection.lr_screening",
        ),
        "development_selection.lr_screening.select",
    )

    selection_rule = str(
        require_key(
            select_cfg,
            "rule",
            "development_selection.lr_screening.select",
        )
    )

    if (
        selection_rule
        != "lowest_best_weighted_dev_loss"
    ):

        raise ValueError(
            "Frozen LR selection rule changed."
        )

    tie_cfg = require_mapping(
        require_key(
            select_cfg,
            "exact_tie_breaker",
            "development_selection.lr_screening.select",
        ),
        (
            "development_selection.lr_screening."
            "select.exact_tie_breaker"
        ),
    )

    exact_tie_breaker = str(
        require_key(
            tie_cfg,
            "rule",
            (
                "development_selection.lr_screening."
                "select.exact_tie_breaker"
            ),
        )
    )

    if (
        exact_tie_breaker
        != "lower_backbone_learning_rate"
    ):

        raise ValueError(
            "Frozen LR exact-tie rule changed."
        )

    auroc_cfg = require_mapping(
        require_key(
            screening_cfg,
            "auroc_guard",
            "development_selection.lr_screening",
        ),
        "development_selection.lr_screening.auroc_guard",
    )

    auroc_no_change = require_key(
        auroc_cfg,
        "does_not_change_selection_rule",
        "development_selection.lr_screening.auroc_guard",
    )

    if auroc_no_change is not True:

        raise ValueError(
            "AUROC must not change the LR selection rule."
        )

    # ------------------------------------------------------------------
    # Cross-check with production Stage-B branch contract.
    # ------------------------------------------------------------------

    branch_contract = (
        load_stage_b_branch_contract(
            experiment_cfg=experiment_cfg,
        )
    )

    if (
        branch_contract.backbone_lr_candidates
        != backbone_lr_candidates
    ):

        raise RuntimeError(
            "LR-screening candidates disagree with "
            "Stage-B branch candidates."
        )

    return LRScreeningContract(
        screening_seed=screening_seed,
        candidate_resolutions=(
            candidate_resolutions
        ),
        backbone_lr_candidates=(
            backbone_lr_candidates
        ),
        classifier_lr_fixed=(
            classifier_lr_fixed
        ),
        stage_a_run_once_per_resolution=True,
        stage_b_same_stage_a_checkpoint=True,
        selection_metric=metric_name,
        selection_metric_source=metric_source,
        selection_rule=selection_rule,
        exact_tie_breaker=(
            exact_tie_breaker
        ),
        auroc_does_not_change_selection=True,
    )


# ======================================================================
# Runtime device
# ======================================================================

def _resolve_runtime_device(
    machine_cfg: Mapping[str, Any],
) -> torch.device:

    runtime_cfg = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
    )

    device = torch.device(
        str(
            require_key(
                runtime_cfg,
                "device",
                "machine_config.runtime",
            )
        )
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "Machine configuration requests CUDA but CUDA is unavailable."
        )

    return device


def _device_matches(
    *,
    actual: torch.device,
    requested: torch.device,
) -> bool:

    if actual.type != requested.type:

        return False

    if requested.index is None:

        return True

    return actual.index == requested.index


# ======================================================================
# Stage-A initialization evidence
# ======================================================================

@dataclass(
    frozen=True
)
class StageAInitializationEvidence:

    resolution_name: str
    run_seed: int

    requested_device: str
    actual_model_device: str

    fresh_model_state_sha256: str

    project_train_generator_initial_state_sha256: str
    dev_val_generator_initial_state_sha256: str

    optimizer_state_entries_at_construction: int

    model_provenance: ModelBuildProvenance
    reproducibility_state: ReproducibilityState
    optimizer_evidence: OptimizerBuildEvidence


# ======================================================================
# Stage-B candidate result
# ======================================================================

@dataclass(
    frozen=True
)
class StageBCandidateScreeningResult:

    backbone_lr: float

    initialization: (
        StageBBranchInitializationEvidence
    )

    stage_b_result: StageRunResult

    protocol_review_flag: bool


# ======================================================================
# Complete one-resolution screening result
# ======================================================================

@dataclass(
    frozen=True
)
class ResolutionLRScreeningResult:

    resolution_name: str
    run_seed: int

    stage_a_initialization: (
        StageAInitializationEvidence
    )

    stage_a_result: StageRunResult

    stage_b_candidates: tuple[
        StageBCandidateScreeningResult,
        ...,
    ]

    selected_backbone_lr: float

    selected_raw_best_weighted_dev_loss: float

    selected_stage_b_checkpoint_sha256: str

    exact_loss_tie_encountered: bool

    protocol_review_required: bool


# ======================================================================
# Stage-A construction
# ======================================================================

def _build_stage_a_runtime(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    repo_root: Path,
    resolution_name: str,
    run_seed: int,
) -> tuple[
    nn.Module,
    DevelopmentDataLoaders,
    torch.optim.AdamW,
    WeightedCrossEntropyObjective,
    torch.device,
    StageAInitializationEvidence,
]:
    """
    Construct the complete Stage-A runtime.

    Frozen order:

        reproducibility reset
        -> fresh development DataLoaders
        -> fresh ResNet-18
        -> Stage-A mode/freeze
        -> runtime device
        -> weighted CE objective
        -> fresh Stage-A AdamW
    """

    reproducibility_state = (
        configure_run_reproducibility(
            experiment_cfg=experiment_cfg,
            run_seed=run_seed,
        )
    )

    dataloaders = (
        build_development_dataloaders(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=repo_root,
            resolution_name=resolution_name,
            run_seed=run_seed,
        )
    )

    train_generator_hash = (
        dataloader_generator_state_sha256(
            dataloaders.project_train_generator
        )
    )

    dev_generator_hash = (
        dataloader_generator_state_sha256(
            dataloaders.dev_val_generator
        )
    )

    model, model_provenance = (
        build_resnet18_classifier(
            experiment_cfg=experiment_cfg,
        )
    )

    fresh_model_hash = (
        model_state_sha256(
            model
        )
    )

    apply_stage_a_contract(
        model
    )

    assert_stage_a_contract(
        model
    )

    device = _resolve_runtime_device(
        machine_cfg
    )

    model = model.to(
        device=device
    )

    assert_stage_a_contract(
        model
    )

    parameter_devices = {
        parameter.device
        for parameter
        in model.parameters()
    }

    if len(
        parameter_devices
    ) != 1:

        raise RuntimeError(
            "Stage-A model parameters span multiple devices."
        )

    actual_device = next(
        iter(
            parameter_devices
        )
    )

    if not _device_matches(
        actual=actual_device,
        requested=device,
    ):

        raise RuntimeError(
            "Stage-A model device does not match machine runtime:\n"
            f"  requested={device}\n"
            f"  actual={actual_device}"
        )

    if (
        model_state_sha256(
            model
        )
        != fresh_model_hash
    ):

        raise RuntimeError(
            "Stage-A device placement changed model-state identity."
        )

    objective = (
        WeightedCrossEntropyObjective(
            experiment_cfg=experiment_cfg,
            device=device,
        )
    )

    optimizer, optimizer_evidence = (
        build_stage_a_optimizer(
            model=model,
            experiment_cfg=experiment_cfg,
        )
    )

    if len(
        optimizer.state
    ) != 0:

        raise RuntimeError(
            "Fresh Stage-A AdamW unexpectedly contains state."
        )

    if (
        optimizer_evidence
        .state_entries_at_construction
        != 0
    ):

        raise RuntimeError(
            "Stage-A optimizer evidence reports non-empty state."
        )

    evidence = (
        StageAInitializationEvidence(
            resolution_name=resolution_name,
            run_seed=run_seed,
            requested_device=str(
                device
            ),
            actual_model_device=str(
                actual_device
            ),
            fresh_model_state_sha256=(
                fresh_model_hash
            ),
            project_train_generator_initial_state_sha256=(
                train_generator_hash
            ),
            dev_val_generator_initial_state_sha256=(
                dev_generator_hash
            ),
            optimizer_state_entries_at_construction=0,
            model_provenance=(
                model_provenance
            ),
            reproducibility_state=(
                reproducibility_state
            ),
            optimizer_evidence=(
                optimizer_evidence
            ),
        )
    )

    return (
        model,
        dataloaders,
        optimizer,
        objective,
        device,
        evidence,
    )


# ======================================================================
# LR selection
# ======================================================================

def _select_stage_b_candidate(
    *,
    candidates: tuple[
        StageBCandidateScreeningResult,
        ...,
    ],
) -> tuple[
    StageBCandidateScreeningResult,
    bool,
]:
    """
    Select by:

        minimum raw-best weighted dev CE

    Exact numerical loss tie:

        lower backbone LR
    """

    if not candidates:

        raise RuntimeError(
            "Cannot select from zero Stage-B candidates."
        )

    losses = [
        candidate
        .stage_b_result
        .raw_best_weighted_dev_loss

        for candidate
        in candidates
    ]

    minimum_loss = min(
        losses
    )

    tied = [
        candidate
        for candidate
        in candidates
        if (
            candidate
            .stage_b_result
            .raw_best_weighted_dev_loss
            == minimum_loss
        )
    ]

    exact_tie = (
        len(
            tied
        )
        > 1
    )

    selected = min(
        tied,
        key=lambda candidate:
            candidate.backbone_lr,
    )

    return (
        selected,
        exact_tie,
    )


# ======================================================================
# One-resolution screening
# ======================================================================

def run_resolution_lr_screening(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    repo_root: Path,
    resolution_name: str,
    run_seed: int,
) -> ResolutionLRScreeningResult:
    """
    Run the complete frozen LR screen for ONE candidate resolution.

    Real use during initial screening:

        resolution_name="r256", run_seed=8

    or:

        resolution_name="r512", run_seed=8

    This function performs real model training when called.
    """

    contract = load_lr_screening_contract(
        experiment_cfg=experiment_cfg,
    )

    if run_seed != contract.screening_seed:

        raise ValueError(
            "LR screening must use the frozen screening seed:\n"
            f"  expected={contract.screening_seed}\n"
            f"  actual={run_seed}"
        )

    if (
        resolution_name
        not in contract.candidate_resolutions
    ):

        raise ValueError(
            "Resolution is outside frozen LR-screening set:\n"
            f"  resolution={resolution_name!r}\n"
            f"  allowed={contract.candidate_resolutions}"
        )

    root = Path(
        repo_root
    ).resolve()

    # ==================================================================
    # Stage A — ONCE for this resolution + seed.
    # ==================================================================

    (
        stage_a_model,
        stage_a_loaders,
        stage_a_optimizer,
        stage_a_objective,
        device,
        stage_a_initialization,
    ) = _build_stage_a_runtime(
        experiment_cfg=experiment_cfg,
        machine_cfg=machine_cfg,
        repo_root=root,
        resolution_name=resolution_name,
        run_seed=run_seed,
    )

    stage_a_result = run_training_stage(
        experiment_cfg=experiment_cfg,
        stage="stage_a",
        model=stage_a_model,
        project_train_loader=(
            stage_a_loaders.project_train
        ),
        dev_val_loader=(
            stage_a_loaders.dev_val
        ),
        optimizer=stage_a_optimizer,
        objective=stage_a_objective,
        device=device,
    )

    if stage_a_result.stage != "stage_a":

        raise RuntimeError(
            "Stage-A runner returned wrong stage identity."
        )

    if (
        stage_a_result
        .raw_best_checkpoint
        .stage
        != "stage_a"
    ):

        raise RuntimeError(
            "LR screening did not obtain a Stage-A raw-best checkpoint."
        )

    stage_a_checkpoint = (
        stage_a_result
        .raw_best_checkpoint
    )

    stage_a_checkpoint_sha = (
        stage_a_checkpoint
        .model_state_sha256
    )

    # Stage-A selected model is currently restored in stage_a_model,
    # but Stage B must NOT continue from this live model/runtime.
    #
    # Every branch will be reconstructed by build_stage_b_branch(),
    # which re-seeds from run_seed and restores only the Stage-A
    # checkpoint state.
    del stage_a_optimizer
    del stage_a_objective
    del stage_a_model
    del stage_a_loaders

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ==================================================================
    # Stage B — three independent branches.
    # ==================================================================

    candidate_results: list[
        StageBCandidateScreeningResult
    ] = []

    for backbone_lr in (
        contract.backbone_lr_candidates
    ):

        branch = build_stage_b_branch(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=root,
            resolution_name=resolution_name,
            run_seed=run_seed,
            stage_a_checkpoint=(
                stage_a_checkpoint
            ),
            backbone_lr=backbone_lr,
        )

        initialization = (
            branch.initialization_evidence
        )

        # --------------------------------------------------------------
        # Every candidate MUST demonstrably restore the same Stage-A
        # checkpoint.
        # --------------------------------------------------------------

        if (
            initialization
            .stage_a_checkpoint_model_state_sha256
            != stage_a_checkpoint_sha
        ):

            raise RuntimeError(
                "Stage-B branch references wrong Stage-A checkpoint."
            )

        if (
            initialization
            .restored_model_state_sha256_after_device_move
            != stage_a_checkpoint_sha
        ):

            raise RuntimeError(
                "Stage-B branch did not restore exact Stage-A state."
            )

        branch_objective = (
            WeightedCrossEntropyObjective(
                experiment_cfg=experiment_cfg,
                device=branch.device,
            )
        )

        stage_b_result = (
            run_training_stage(
                experiment_cfg=experiment_cfg,
                stage="stage_b",
                model=branch.model,
                project_train_loader=(
                    branch
                    .dataloaders
                    .project_train
                ),
                dev_val_loader=(
                    branch
                    .dataloaders
                    .dev_val
                ),
                optimizer=branch.optimizer,
                objective=branch_objective,
                device=branch.device,
            )
        )

        if stage_b_result.stage != "stage_b":

            raise RuntimeError(
                "Stage-B runner returned wrong stage identity."
            )

        if (
            stage_b_result
            .raw_best_checkpoint
            .stage
            != "stage_b"
        ):

            raise RuntimeError(
                "Stage-B candidate did not produce a Stage-B checkpoint."
            )

        disagreement = (
            stage_b_result
            .stage_b_auroc_disagreement
        )

        if disagreement is None:

            raise RuntimeError(
                "Stage-B candidate is missing AUROC disagreement result."
            )

        protocol_review_flag = bool(
            disagreement
            .protocol_review_flag
        )

        candidate_results.append(
            StageBCandidateScreeningResult(
                backbone_lr=float(
                    backbone_lr
                ),
                initialization=(
                    initialization
                ),
                stage_b_result=(
                    stage_b_result
                ),
                protocol_review_flag=(
                    protocol_review_flag
                ),
            )
        )

        # Live branch model/optimizer/loaders are no longer needed.
        # StageRunResult contains the selected CPU checkpoint/history.
        del branch_objective
        del branch

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    candidates = tuple(
        candidate_results
    )

    # ==================================================================
    # Cross-branch checkpoint identity.
    # ==================================================================

    stage_a_sources = {
        candidate
        .initialization
        .stage_a_checkpoint_model_state_sha256

        for candidate
        in candidates
    }

    if stage_a_sources != {
        stage_a_checkpoint_sha
    }:

        raise RuntimeError(
            "Stage-B candidates did not all fork from "
            "the exact same Stage-A checkpoint."
        )

    if len(
        candidates
    ) != len(
        contract.backbone_lr_candidates
    ):

        raise RuntimeError(
            "LR screening did not execute every frozen candidate."
        )

    observed_lrs = tuple(
        candidate.backbone_lr
        for candidate
        in candidates
    )

    if (
        observed_lrs
        != contract.backbone_lr_candidates
    ):

        raise RuntimeError(
            "Stage-B candidates executed in unexpected LR set/order:\n"
            f"  expected={contract.backbone_lr_candidates}\n"
            f"  actual={observed_lrs}"
        )

    # ==================================================================
    # Frozen loss-based LR selection.
    # ==================================================================

    selected, exact_tie = (
        _select_stage_b_candidate(
            candidates=candidates,
        )
    )

    selected_loss = (
        selected
        .stage_b_result
        .raw_best_weighted_dev_loss
    )

    selected_checkpoint_sha = (
        selected
        .stage_b_result
        .raw_best_checkpoint
        .model_state_sha256
    )

    # AUROC flags are surfaced but MUST NOT alter the loss-based choice.
    protocol_review_required = any(
        candidate.protocol_review_flag
        for candidate
        in candidates
    )

    # Independent selection check:
    expected_selected_lr = min(
        (
            (
                candidate
                .stage_b_result
                .raw_best_weighted_dev_loss,
                candidate.backbone_lr,
            )
            for candidate
            in candidates
        )
    )[
        1
    ]

    if selected.backbone_lr != expected_selected_lr:

        raise RuntimeError(
            "LR selection does not equal frozen "
            "(loss, lower-LR tie-break) rule."
        )

    return ResolutionLRScreeningResult(
        resolution_name=resolution_name,
        run_seed=run_seed,
        stage_a_initialization=(
            stage_a_initialization
        ),
        stage_a_result=stage_a_result,
        stage_b_candidates=candidates,
        selected_backbone_lr=(
            selected.backbone_lr
        ),
        selected_raw_best_weighted_dev_loss=(
            selected_loss
        ),
        selected_stage_b_checkpoint_sha256=(
            selected_checkpoint_sha
        ),
        exact_loss_tie_encountered=(
            exact_tie
        ),
        protocol_review_required=(
            protocol_review_required
        ),
    )