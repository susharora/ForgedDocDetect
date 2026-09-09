"""
Stage-B branch construction for the frozen Tech-2 protocol.

One Stage-B branch is defined by:

    resolution
    + run_seed
    + Stage-A raw-argmin model checkpoint
    + Stage-B backbone learning rate

Frozen schema-v3 branch policy
------------------------------
For EVERY Stage-B LR branch:

1. reset global reproducibility state from run_seed;
2. construct fresh development DataLoaders from run_seed;
3. construct a fresh ImageNet-pretrained ResNet-18;
4. restore the exact Stage-A raw-argmin MODEL state;
5. apply the Stage-B full-backbone training contract;
6. construct a fresh Stage-B AdamW optimizer.

Only MODEL state crosses the Stage-A -> Stage-B boundary.

The following are deliberately NOT carried from Stage A:

- AdamW state;
- Python RNG state;
- NumPy RNG state;
- torch RNG state;
- project_train DataLoader-generator state;
- dev_val DataLoader-generator state.

Consequently, for one resolution + run_seed, the three Stage-B backbone
LR candidates begin from:

- exactly the same Stage-A model-state SHA-256;
- fresh generators seeded identically from run_seed;
- the same project_train permutation stream;
- fresh empty AdamW state.

This module does NOT:
- execute training epochs;
- evaluate dev_val;
- implement stopping;
- select an LR;
- compute AUROC;
- access held-out test;
- serialize artifacts.
"""

from __future__ import annotations

import hashlib
import math
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
    apply_stage_b_contract,
    assert_stage_b_contract,
    build_resnet18_classifier,
)

from src.optimization import (
    OptimizerBuildEvidence,
    build_stage_b_optimizer,
)

from src.reproducibility import (
    ReproducibilityState,
    configure_run_reproducibility,
)

from src.training_control import (
    RawArgminModelCheckpoint,
    model_state_sha256,
    restore_raw_argmin_model_checkpoint,
    state_dict_sha256,
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
# Frozen schema-v3 branch contract
# ======================================================================

@dataclass(
    frozen=True
)
class StageBBranchContract:

    policy: str

    ordered_steps: tuple[
        str,
        ...,
    ]

    checkpoint_source: str

    global_rng_policy: str
    dataloader_rng_policy: str

    project_train_generator_source: str
    dev_val_generator_source: str

    backbone_lr_candidates: tuple[
        float,
        ...,
    ]


def load_stage_b_branch_contract(
    *,
    experiment_cfg: Mapping[str, Any],
) -> StageBBranchContract:
    """
    Reassert the complete frozen Stage-B branch-initialisation policy.

    This duplicates the important production-facing invariants from the
    canonical configuration validator so drift cannot silently enter the
    runtime branch constructor.
    """

    schema_version = require_key(
        experiment_cfg,
        "schema_version",
        "experiment_config",
    )

    if schema_version != 3:

        raise ValueError(
            "Stage-B branch construction requires frozen "
            f"experiment schema_version=3, got {schema_version!r}."
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

    stage_b_cfg = require_mapping(
        require_key(
            transfer_cfg,
            "stage_b",
            "transfer_learning",
        ),
        "transfer_learning.stage_b",
    )

    branch_cfg = require_mapping(
        require_key(
            stage_b_cfg,
            "branch_initialisation",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.branch_initialisation",
    )

    # ------------------------------------------------------------------
    # Primary policy.
    # ------------------------------------------------------------------

    policy = str(
        require_key(
            branch_cfg,
            "policy",
            "transfer_learning.stage_b.branch_initialisation",
        )
    )

    expected_policy = (
        "reset_from_run_seed_before_each_stage_b_branch"
    )

    if policy != expected_policy:

        raise ValueError(
            "Frozen Stage-B branch policy mismatch:\n"
            f"  expected={expected_policy!r}\n"
            f"  actual={policy!r}"
        )

    # ------------------------------------------------------------------
    # Exact scientific ordering.
    # ------------------------------------------------------------------

    ordered_steps = tuple(
        str(
            value
        )
        for value
        in require_key(
            branch_cfg,
            "ordered_steps",
            "transfer_learning.stage_b.branch_initialisation",
        )
    )

    expected_steps = (
        "configure_run_reproducibility_from_run_seed",
        "build_fresh_development_dataloaders_from_run_seed",
        "build_fresh_resnet18_classifier",
        "restore_exact_stage_a_raw_argmin_model_state",
        "apply_stage_b_train_contract",
        "construct_fresh_stage_b_optimizer",
    )

    if ordered_steps != expected_steps:

        raise ValueError(
            "Frozen Stage-B branch ordered steps changed:\n"
            f"  expected={expected_steps}\n"
            f"  actual={ordered_steps}"
        )

    # ------------------------------------------------------------------
    # Model checkpoint carry-over.
    # ------------------------------------------------------------------

    checkpoint_cfg = require_mapping(
        require_key(
            branch_cfg,
            "model_checkpoint",
            "transfer_learning.stage_b.branch_initialisation",
        ),
        (
            "transfer_learning.stage_b."
            "branch_initialisation.model_checkpoint"
        ),
    )

    checkpoint_source = str(
        require_key(
            checkpoint_cfg,
            "source",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.model_checkpoint"
            ),
        )
    )

    expected_checkpoint_source = (
        "exact_stage_a_raw_argmin_checkpoint"
    )

    if (
        checkpoint_source
        != expected_checkpoint_source
    ):

        raise ValueError(
            "Frozen Stage-B checkpoint source changed."
        )

    contents_cfg = require_mapping(
        require_key(
            checkpoint_cfg,
            "contents_carried",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.model_checkpoint"
            ),
        ),
        (
            "transfer_learning.stage_b."
            "branch_initialisation."
            "model_checkpoint.contents_carried"
        ),
    )

    expected_contents = {
        "model_parameters_and_buffers":
            True,

        "optimizer_state":
            False,

        "global_rng_state":
            False,

        "dataloader_generator_state":
            False,
    }

    actual_contents = {
        key:
            require_key(
                contents_cfg,
                key,
                (
                    "transfer_learning.stage_b."
                    "branch_initialisation."
                    "model_checkpoint.contents_carried"
                ),
            )

        for key
        in expected_contents
    }

    if actual_contents != expected_contents:

        raise ValueError(
            "Frozen Stage-B checkpoint carry-over contract changed:\n"
            f"  expected={expected_contents}\n"
            f"  actual={actual_contents}"
        )

    initial_checkpoint_cfg = require_mapping(
        require_key(
            stage_b_cfg,
            "initial_checkpoint",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.initial_checkpoint",
    )

    original_checkpoint_source = str(
        require_key(
            initial_checkpoint_cfg,
            "source",
            "transfer_learning.stage_b.initial_checkpoint",
        )
    )

    if (
        original_checkpoint_source
        != checkpoint_source
    ):

        raise ValueError(
            "Stage-B branch checkpoint source disagrees with "
            "transfer_learning.stage_b.initial_checkpoint.source."
        )

    # ------------------------------------------------------------------
    # Global RNG reset.
    # ------------------------------------------------------------------

    global_rng_cfg = require_mapping(
        require_key(
            branch_cfg,
            "global_rng",
            "transfer_learning.stage_b.branch_initialisation",
        ),
        (
            "transfer_learning.stage_b."
            "branch_initialisation.global_rng"
        ),
    )

    global_rng_policy = str(
        require_key(
            global_rng_cfg,
            "policy",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.global_rng"
            ),
        )
    )

    if (
        global_rng_policy
        != "reset_from_run_seed"
    ):

        raise ValueError(
            "Frozen Stage-B global RNG policy changed."
        )

    if (
        require_key(
            global_rng_cfg,
            "carry_stage_a_state",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.global_rng"
            ),
        )
        is not False
    ):

        raise ValueError(
            "Stage-B must not carry Stage-A global RNG state."
        )

    # ------------------------------------------------------------------
    # DataLoader generator reset.
    # ------------------------------------------------------------------

    dataloader_rng_cfg = require_mapping(
        require_key(
            branch_cfg,
            "dataloader_rng",
            "transfer_learning.stage_b.branch_initialisation",
        ),
        (
            "transfer_learning.stage_b."
            "branch_initialisation.dataloader_rng"
        ),
    )

    dataloader_rng_policy = str(
        require_key(
            dataloader_rng_cfg,
            "policy",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.dataloader_rng"
            ),
        )
    )

    if (
        dataloader_rng_policy
        != "fresh_generators_seeded_from_run_seed"
    ):

        raise ValueError(
            "Frozen Stage-B DataLoader RNG policy changed."
        )

    project_train_generator_source = str(
        require_key(
            dataloader_rng_cfg,
            "project_train_generator",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.dataloader_rng"
            ),
        )
    )

    dev_val_generator_source = str(
        require_key(
            dataloader_rng_cfg,
            "dev_val_generator",
            (
                "transfer_learning.stage_b."
                "branch_initialisation.dataloader_rng"
            ),
        )
    )

    if (
        project_train_generator_source
        != "run_seed"
    ):

        raise ValueError(
            "Stage-B project_train generator must be seeded "
            "from run_seed."
        )

    if (
        dev_val_generator_source
        != "run_seed"
    ):

        raise ValueError(
            "Stage-B dev_val generator must be seeded "
            "from run_seed."
        )

    for key in (
        "carry_stage_a_project_train_generator_state",
        "carry_stage_a_dev_val_generator_state",
    ):

        if (
            require_key(
                dataloader_rng_cfg,
                key,
                (
                    "transfer_learning.stage_b."
                    "branch_initialisation.dataloader_rng"
                ),
            )
            is not False
        ):

            raise ValueError(
                "Stage-B must not carry Stage-A DataLoader "
                f"generator state: {key}"
            )

    # ------------------------------------------------------------------
    # LR-branch comparability.
    # ------------------------------------------------------------------

    comparability_cfg = require_mapping(
        require_key(
            branch_cfg,
            "lr_branch_comparability",
            "transfer_learning.stage_b.branch_initialisation",
        ),
        (
            "transfer_learning.stage_b."
            "branch_initialisation.lr_branch_comparability"
        ),
    )

    for key in (
        "all_lr_candidates_same_stage_a_model_checkpoint",
        "all_lr_candidates_same_initial_global_rng_state",
        "same_project_train_order_for_corresponding_stage_b_epochs",
    ):

        if (
            require_key(
                comparability_cfg,
                key,
                (
                    "transfer_learning.stage_b."
                    "branch_initialisation."
                    "lr_branch_comparability"
                ),
            )
            is not True
        ):

            raise ValueError(
                "Frozen Stage-B LR comparability contract changed:\n"
                f"  {key}"
            )

    # ------------------------------------------------------------------
    # Existing Stage-A fork requirement.
    # ------------------------------------------------------------------

    stage_a_execution_cfg = require_mapping(
        require_key(
            stage_a_cfg,
            "execution_scope",
            "transfer_learning.stage_a",
        ),
        "transfer_learning.stage_a.execution_scope",
    )

    if (
        require_key(
            stage_a_execution_cfg,
            "reuse_checkpoint_for_all_stage_b_lr_candidates",
            "transfer_learning.stage_a.execution_scope",
        )
        is not True
    ):

        raise ValueError(
            "Stage-A raw-best checkpoint must be reused for "
            "all Stage-B LR branches."
        )

    # ------------------------------------------------------------------
    # Fresh Stage-B optimizer and frozen LR candidates.
    # ------------------------------------------------------------------

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
            "Stage-B branch construction requires a fresh optimizer."
        )

    backbone_lr_candidates = tuple(
        float(
            value
        )
        for value
        in require_key(
            optimizer_cfg,
            "backbone_lr_candidates",
            "transfer_learning.stage_b.optimizer",
        )
    )

    expected_lrs = (
        0.00003,
        0.0001,
        0.0003,
    )

    if backbone_lr_candidates != expected_lrs:

        raise ValueError(
            "Frozen Stage-B backbone LR candidates changed:\n"
            f"  expected={expected_lrs}\n"
            f"  actual={backbone_lr_candidates}"
        )

    return StageBBranchContract(
        policy=policy,
        ordered_steps=ordered_steps,
        checkpoint_source=checkpoint_source,
        global_rng_policy=global_rng_policy,
        dataloader_rng_policy=dataloader_rng_policy,
        project_train_generator_source=(
            project_train_generator_source
        ),
        dev_val_generator_source=(
            dev_val_generator_source
        ),
        backbone_lr_candidates=(
            backbone_lr_candidates
        ),
    )


# ======================================================================
# Stable generator fingerprint
# ======================================================================

def dataloader_generator_state_sha256(
    generator: torch.Generator,
) -> str:

    state = (
        generator
        .get_state()
        .detach()
        .cpu()
        .contiguous()
    )

    return hashlib.sha256(
        state.numpy().tobytes(
            order="C"
        )
    ).hexdigest()


# ======================================================================
# Machine runtime device
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

    device_string = str(
        require_key(
            runtime_cfg,
            "device",
            "machine_config.runtime",
        )
    )

    device = torch.device(
        device_string
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "Machine config requests CUDA but "
            "torch.cuda.is_available() is False."
        )

    return device


def _device_matches(
    *,
    actual: torch.device,
    requested: torch.device,
) -> bool:

    if actual.type != requested.type:

        return False

    # An unindexed "cuda" means the current/default CUDA device.
    if requested.index is None:

        return True

    return (
        actual.index
        == requested.index
    )


# ======================================================================
# Stage-A checkpoint input guard
# ======================================================================

def _validate_stage_a_checkpoint(
    checkpoint: RawArgminModelCheckpoint,
) -> None:

    if not isinstance(
        checkpoint,
        RawArgminModelCheckpoint,
    ):

        raise TypeError(
            "Stage-B branch requires a "
            "RawArgminModelCheckpoint."
        )

    if checkpoint.stage != "stage_a":

        raise ValueError(
            "Stage-B branch must start from a Stage-A "
            "raw-argmin checkpoint:\n"
            f"  checkpoint.stage={checkpoint.stage!r}"
        )

    if (
        not isinstance(
            checkpoint.epoch,
            int,
        )
        or isinstance(
            checkpoint.epoch,
            bool,
        )
        or checkpoint.epoch <= 0
    ):

        raise ValueError(
            "Stage-A checkpoint epoch must be a positive integer."
        )

    if (
        not math.isfinite(
            checkpoint.weighted_dev_loss
        )
        or checkpoint.weighted_dev_loss < 0.0
    ):

        raise ValueError(
            "Stage-A checkpoint weighted dev loss must be "
            "finite and non-negative."
        )

    current_hash = state_dict_sha256(
        checkpoint.model_state_dict
    )

    if (
        current_hash
        != checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Stage-A checkpoint integrity failure:\n"
            f"  recorded={checkpoint.model_state_sha256}\n"
            f"  actual={current_hash}"
        )

    for (
        name,
        value,
    ) in checkpoint.model_state_dict.items():

        if not isinstance(
            value,
            torch.Tensor,
        ):

            raise TypeError(
                "Stage-A checkpoint state must contain tensors only:\n"
                f"  key={name!r}\n"
                f"  type={type(value).__name__}"
            )

        if value.device.type != "cpu":

            raise RuntimeError(
                "Frozen raw-argmin checkpoint must be CPU-resident:\n"
                f"  key={name!r}\n"
                f"  device={value.device}"
            )


# ======================================================================
# Branch evidence
# ======================================================================

@dataclass(
    frozen=True
)
class StageBBranchInitializationEvidence:

    resolution_name: str
    run_seed: int
    backbone_lr: float

    requested_device: str
    actual_model_device: str

    stage_a_checkpoint_epoch: int
    stage_a_checkpoint_weighted_dev_loss: float
    stage_a_checkpoint_model_state_sha256: str

    fresh_model_state_sha256_before_restore: str

    restored_model_state_sha256_cpu: str
    restored_model_state_sha256_after_device_move: str

    project_train_generator_initial_state_sha256: str
    dev_val_generator_initial_state_sha256: str

    optimizer_state_entries_at_construction: int


@dataclass
class StageBBranch:
    """
    Mutable runtime objects for one Stage-B branch.

    The container itself is intentionally simple; model, optimizer and
    DataLoader state will evolve once the later training runner consumes
    this branch.
    """

    resolution_name: str
    run_seed: int
    backbone_lr: float

    device: torch.device

    reproducibility_state: ReproducibilityState

    dataloaders: DevelopmentDataLoaders

    model: nn.Module
    model_provenance: ModelBuildProvenance

    optimizer: torch.optim.AdamW
    optimizer_evidence: OptimizerBuildEvidence

    initialization_evidence: StageBBranchInitializationEvidence


# ======================================================================
# Branch constructor
# ======================================================================

def build_stage_b_branch(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    repo_root: Path,
    resolution_name: str,
    run_seed: int,
    stage_a_checkpoint: RawArgminModelCheckpoint,
    backbone_lr: float,
) -> StageBBranch:
    """
    Build exactly one fresh Stage-B LR branch.

    Scientific ordering is deliberately aligned to schema-v3 D25.

    The caller may invoke this function once for each frozen backbone LR.
    Each invocation resets global reproducibility state from run_seed,
    so branch construction is independent of previously executed Stage-B
    branches.
    """

    # ------------------------------------------------------------------
    # Validate frozen branch policy before mutating any RNG state.
    # ------------------------------------------------------------------

    contract = load_stage_b_branch_contract(
        experiment_cfg=experiment_cfg,
    )

    _validate_stage_a_checkpoint(
        stage_a_checkpoint
    )

    requested_backbone_lr = float(
        backbone_lr
    )

    if (
        requested_backbone_lr
        not in contract.backbone_lr_candidates
    ):

        raise ValueError(
            "Stage-B backbone LR is outside the frozen branch set:\n"
            f"  requested={requested_backbone_lr}\n"
            f"  allowed={contract.backbone_lr_candidates}"
        )

    root = Path(
        repo_root
    ).resolve()

    # ==================================================================
    # D25 STEP 1
    # Reset all run-level RNG/backend state from run_seed.
    # ==================================================================

    reproducibility_state = (
        configure_run_reproducibility(
            experiment_cfg=experiment_cfg,
            run_seed=run_seed,
        )
    )

    # ==================================================================
    # D25 STEP 2
    # Fresh DataLoaders + fresh independent generators from run_seed.
    # ==================================================================

    dataloaders = (
        build_development_dataloaders(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=root,
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

    # ==================================================================
    # D25 STEP 3
    # Fresh pretrained ResNet-18.
    # ==================================================================

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

    if next(
        model.parameters()
    ).device.type != "cpu":

        raise RuntimeError(
            "Fresh branch model must be CPU-resident "
            "before checkpoint restoration."
        )

    # ==================================================================
    # D25 STEP 4
    # Restore exact Stage-A raw-argmin MODEL state only.
    # ==================================================================

    restore_raw_argmin_model_checkpoint(
        model=model,
        checkpoint=stage_a_checkpoint,
    )

    restored_cpu_hash = (
        model_state_sha256(
            model
        )
    )

    if (
        restored_cpu_hash
        != stage_a_checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Stage-B branch did not restore the exact "
            "Stage-A raw-argmin model state:\n"
            f"  expected={stage_a_checkpoint.model_state_sha256}\n"
            f"  actual={restored_cpu_hash}"
        )

    # ==================================================================
    # D25 STEP 5
    # Apply full-backbone Stage-B train contract.
    # ==================================================================

    apply_stage_b_contract(
        model
    )

    assert_stage_b_contract(
        model
    )

    # ------------------------------------------------------------------
    # Runtime-only device placement.
    #
    # Device placement is intentionally not a scientific branch-selection
    # variable, but it must occur before optimizer construction so the
    # optimizer is built directly over the parameters used for training.
    # ------------------------------------------------------------------

    device = _resolve_runtime_device(
        machine_cfg
    )

    model = model.to(
        device=device
    )

    assert_stage_b_contract(
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
            "Stage-B model parameters span multiple devices:\n"
            f"  devices="
            f"{sorted(str(value) for value in parameter_devices)}"
        )

    actual_model_device = next(
        iter(
            parameter_devices
        )
    )

    if not _device_matches(
        actual=actual_model_device,
        requested=device,
    ):

        raise RuntimeError(
            "Stage-B model placement does not match machine runtime:\n"
            f"  requested={device}\n"
            f"  actual={actual_model_device}"
        )

    restored_device_hash = (
        model_state_sha256(
            model
        )
    )

    if (
        restored_device_hash
        != stage_a_checkpoint.model_state_sha256
    ):

        raise RuntimeError(
            "Device placement changed restored model-state identity:\n"
            f"  expected={stage_a_checkpoint.model_state_sha256}\n"
            f"  actual={restored_device_hash}"
        )

    # ==================================================================
    # D25 STEP 6
    # Fresh Stage-B AdamW.
    # ==================================================================

    optimizer, optimizer_evidence = (
        build_stage_b_optimizer(
            model=model,
            experiment_cfg=experiment_cfg,
            backbone_lr=requested_backbone_lr,
        )
    )

    if len(
        optimizer.state
    ) != 0:

        raise RuntimeError(
            "Fresh Stage-B branch optimizer unexpectedly "
            "contains Adam state."
        )

    if (
        optimizer_evidence.state_entries_at_construction
        != 0
    ):

        raise RuntimeError(
            "Stage-B optimizer construction evidence does not "
            "report empty initial state."
        )

    # ------------------------------------------------------------------
    # Final branch evidence.
    # ------------------------------------------------------------------

    initialization_evidence = (
        StageBBranchInitializationEvidence(
            resolution_name=resolution_name,
            run_seed=run_seed,
            backbone_lr=requested_backbone_lr,
            requested_device=str(
                device
            ),
            actual_model_device=str(
                actual_model_device
            ),
            stage_a_checkpoint_epoch=(
                stage_a_checkpoint.epoch
            ),
            stage_a_checkpoint_weighted_dev_loss=(
                stage_a_checkpoint.weighted_dev_loss
            ),
            stage_a_checkpoint_model_state_sha256=(
                stage_a_checkpoint.model_state_sha256
            ),
            fresh_model_state_sha256_before_restore=(
                fresh_model_hash
            ),
            restored_model_state_sha256_cpu=(
                restored_cpu_hash
            ),
            restored_model_state_sha256_after_device_move=(
                restored_device_hash
            ),
            project_train_generator_initial_state_sha256=(
                train_generator_hash
            ),
            dev_val_generator_initial_state_sha256=(
                dev_generator_hash
            ),
            optimizer_state_entries_at_construction=0,
        )
    )

    return StageBBranch(
        resolution_name=resolution_name,
        run_seed=run_seed,
        backbone_lr=requested_backbone_lr,
        device=device,
        reproducibility_state=(
            reproducibility_state
        ),
        dataloaders=dataloaders,
        model=model,
        model_provenance=model_provenance,
        optimizer=optimizer,
        optimizer_evidence=optimizer_evidence,
        initialization_evidence=(
            initialization_evidence
        ),
    )