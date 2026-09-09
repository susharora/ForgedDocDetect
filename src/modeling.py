"""
Frozen ResNet-18 model and transfer-learning stage contracts.

This module implements the model-facing portion of the frozen Tech-2
scientific protocol.

Implemented
-----------
- torchvision ResNet-18 / IMAGENET1K_V1 construction;
- full SHA-256 verification of the actual pretrained checkpoint file;
- replacement of ImageNet FC with a 512 -> 2 Linear classifier;
- Stage-A backbone freeze / eval contract;
- Stage-B full-backbone train contract;
- BatchNorm affine/running-statistics mode validation;
- AdamW parameter grouping;
- bias and BatchNorm weight-decay exclusions;
- Stage-A classifier-only groups;
- Stage-B backbone + classifier groups;
- exact parameter-coverage checks.

Not implemented
---------------
- optimizer construction;
- loss;
- training loop;
- checkpoint saving;
- early stopping;
- AUROC;
- Grad-CAM.

Important
---------
configure_run_reproducibility(...) must be called BEFORE
build_resnet18_classifier(...).

The new nn.Linear classifier uses PyTorch's default initialization and
therefore consumes the already-established torch RNG stream associated
with the run seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


# ======================================================================
# Frozen architecture constants
# ======================================================================

BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


EXPECTED_PRETRAINED_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


# ======================================================================
# Small config helpers
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
# Hash utilities
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


# ======================================================================
# Model provenance
# ======================================================================

@dataclass(
    frozen=True
)
class ModelBuildProvenance:
    """
    Exact identity of one freshly constructed primary model.
    """

    architecture: str

    pretrained_weights_enum: str
    pretrained_checkpoint_path: str
    pretrained_checkpoint_sha256: str

    classifier_in_features: int
    classifier_out_features: int

    total_parameters: int
    backbone_parameters: int
    classifier_parameters: int

    batchnorm_modules: int


def torchvision_checkpoint_path(
    weights: ResNet18_Weights,
) -> Path:
    """
    Return the expected local torch.hub checkpoint path.

    torchvision's ResNet-18 weights are cached under:

        <torch.hub.get_dir()>/checkpoints/<URL filename>

    We verify the complete SHA-256 ourselves rather than relying only on
    torchvision's filename hash prefix.
    """

    parsed = urlparse(
        weights.url
    )

    filename = Path(
        parsed.path
    ).name

    if not filename:

        raise RuntimeError(
            "Could not derive torchvision checkpoint filename "
            f"from URL: {weights.url!r}"
        )

    return (
        Path(
            torch.hub.get_dir()
        )
        / "checkpoints"
        / filename
    ).resolve()


def verify_pretrained_checkpoint(
    *,
    checkpoint_path: Path,
    expected_sha256: str,
) -> str:
    """
    Verify the complete pretrained checkpoint digest.
    """

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            "Expected torchvision pretrained checkpoint "
            "does not exist after model construction:\n"
            f"  {checkpoint_path}"
        )

    actual = sha256_file(
        checkpoint_path
    )

    if actual != expected_sha256:

        raise RuntimeError(
            "Pretrained ResNet-18 checkpoint SHA-256 mismatch:\n"
            f"  path={checkpoint_path}\n"
            f"  expected={expected_sha256}\n"
            f"  actual={actual}"
        )

    return actual


# ======================================================================
# Frozen model-config validation
# ======================================================================

def validate_model_config(
    experiment_cfg: Mapping[str, Any],
) -> Mapping[str, Any]:

    model_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "model",
            "experiment_config",
        ),
        "model",
    )

    if (
        require_key(
            model_cfg,
            "architecture",
            "model",
        )
        != "resnet18"
    ):

        raise ValueError(
            "Frozen model architecture must be resnet18."
        )

    weights_cfg = require_mapping(
        require_key(
            model_cfg,
            "pretrained_weights",
            "model",
        ),
        "model.pretrained_weights",
    )

    if (
        require_key(
            weights_cfg,
            "source",
            "model.pretrained_weights",
        )
        != "torchvision"
    ):

        raise ValueError(
            "Frozen pretrained-weight source must be torchvision."
        )

    if (
        require_key(
            weights_cfg,
            "enum",
            "model.pretrained_weights",
        )
        != "IMAGENET1K_V1"
    ):

        raise ValueError(
            "Frozen pretrained-weight enum must be IMAGENET1K_V1."
        )

    configured_sha = str(
        require_key(
            weights_cfg,
            "checkpoint_sha256",
            "model.pretrained_weights",
        )
    )

    if configured_sha != EXPECTED_PRETRAINED_SHA256:

        raise ValueError(
            "Configured pretrained checkpoint SHA does not match "
            "the frozen ResNet-18 identity:\n"
            f"  expected={EXPECTED_PRETRAINED_SHA256}\n"
            f"  actual={configured_sha}"
        )

    classifier_cfg = require_mapping(
        require_key(
            model_cfg,
            "classifier",
            "model",
        ),
        "model.classifier",
    )

    required_classifier_values = {
        "replace_original_fc":
            True,

        "type":
            "Linear",

        "in_features":
            512,

        "out_features":
            2,
    }

    for (
        key,
        expected,
    ) in required_classifier_values.items():

        actual = require_key(
            classifier_cfg,
            key,
            "model.classifier",
        )

        if actual != expected:

            raise ValueError(
                "Frozen classifier configuration mismatch:\n"
                f"  model.classifier.{key}\n"
                f"  expected={expected!r}\n"
                f"  actual={actual!r}"
            )

    initialization = require_mapping(
        require_key(
            classifier_cfg,
            "initialisation",
            "model.classifier",
        ),
        "model.classifier.initialisation",
    )

    if (
        require_key(
            initialization,
            "policy",
            "model.classifier.initialisation",
        )
        != "pytorch_nn_linear_default"
    ):

        raise ValueError(
            "Frozen classifier initialization must use "
            "PyTorch nn.Linear default initialization."
        )

    if (
        require_key(
            initialization,
            "controlled_by",
            "model.classifier.initialisation",
        )
        != "run_seed"
    ):

        raise ValueError(
            "Classifier initialization must be controlled by run_seed."
        )

    if (
        require_key(
            model_cfg,
            "dtype",
            "model",
        )
        != "float32"
    ):

        raise ValueError(
            "Frozen primary model dtype must be float32."
        )

    if (
        require_key(
            model_cfg,
            "amp",
            "model",
        )
        is not False
    ):

        raise ValueError(
            "Frozen primary model requires amp=false."
        )

    return model_cfg


# ======================================================================
# Model construction
# ======================================================================

def build_resnet18_classifier(
    *,
    experiment_cfg: Mapping[str, Any],
) -> tuple[
    nn.Module,
    ModelBuildProvenance,
]:
    """
    Construct the frozen primary ResNet-18 classifier.

    Required calling order:

        configure_run_reproducibility(...)
        build_resnet18_classifier(...)

    This function intentionally does NOT reseed RNGs.

    torchvision constructs the base ResNet and loads IMAGENET1K_V1.
    The original 1000-class FC is then replaced by a new default-
    initialized nn.Linear(512, 2).
    """

    validate_model_config(
        experiment_cfg
    )

    weights = (
        ResNet18_Weights
        .IMAGENET1K_V1
    )

    checkpoint_path = (
        torchvision_checkpoint_path(
            weights
        )
    )

    # ------------------------------------------------------------------
    # If a cache entry already exists, reject corruption BEFORE loading.
    #
    # torchvision may otherwise reuse an existing cache file without
    # computing our full frozen SHA-256.
    # ------------------------------------------------------------------

    if checkpoint_path.is_file():

        verify_pretrained_checkpoint(
            checkpoint_path=checkpoint_path,
            expected_sha256=EXPECTED_PRETRAINED_SHA256,
        )

    # ------------------------------------------------------------------
    # Construct/load pretrained network.
    #
    # If the file was not cached, torchvision downloads it here.
    # ------------------------------------------------------------------

    model = resnet18(
        weights=weights
    )

    # ------------------------------------------------------------------
    # Verify the actual file after load/download as well.
    # ------------------------------------------------------------------

    actual_checkpoint_sha = (
        verify_pretrained_checkpoint(
            checkpoint_path=checkpoint_path,
            expected_sha256=EXPECTED_PRETRAINED_SHA256,
        )
    )

    if not isinstance(
        model.fc,
        nn.Linear,
    ):

        raise RuntimeError(
            "torchvision ResNet-18 original classifier "
            "is unexpectedly not nn.Linear."
        )

    pretrained_in_features = int(
        model.fc.in_features
    )

    if pretrained_in_features != 512:

        raise RuntimeError(
            "Unexpected torchvision ResNet-18 FC width:\n"
            f"  expected=512\n"
            f"  actual={pretrained_in_features}"
        )

    # ------------------------------------------------------------------
    # New binary classifier.
    #
    # No explicit initialization call is made. This intentionally uses
    # nn.Linear.reset_parameters(), controlled by the run's torch RNG.
    # ------------------------------------------------------------------

    model.fc = nn.Linear(
        in_features=512,
        out_features=2,
        bias=True,
    )

    # Model is intentionally kept on CPU here. Device placement belongs
    # to the future run/training entry point.
    if next(
        model.parameters()
    ).device.type != "cpu":

        raise RuntimeError(
            "Fresh model construction unexpectedly produced "
            "a non-CPU model."
        )

    if any(
        parameter.dtype
        != torch.float32
        for parameter
        in model.parameters()
    ):

        raise RuntimeError(
            "Fresh ResNet-18 contains non-float32 parameters."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    classifier_parameters = sum(
        parameter.numel()
        for parameter
        in model.fc.parameters()
    )

    backbone_parameters = (
        total_parameters
        - classifier_parameters
    )

    batchnorm_modules = sum(
        1
        for module
        in model.modules()
        if isinstance(
            module,
            BATCHNORM_TYPES,
        )
    )

    provenance = ModelBuildProvenance(
        architecture="resnet18",
        pretrained_weights_enum="IMAGENET1K_V1",
        pretrained_checkpoint_path=str(
            checkpoint_path
        ),
        pretrained_checkpoint_sha256=(
            actual_checkpoint_sha
        ),
        classifier_in_features=512,
        classifier_out_features=2,
        total_parameters=total_parameters,
        backbone_parameters=backbone_parameters,
        classifier_parameters=classifier_parameters,
        batchnorm_modules=batchnorm_modules,
    )

    return (
        model,
        provenance,
    )


# ======================================================================
# Model component helpers
# ======================================================================

def backbone_named_parameters(
    model: nn.Module,
):
    """
    Yield every model parameter except fc.*.
    """

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not name.startswith(
            "fc."
        ):

            yield (
                name,
                parameter,
            )


def classifier_named_parameters(
    model: nn.Module,
):

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if name.startswith(
            "fc."
        ):

            yield (
                name,
                parameter,
            )


def batchnorm_modules(
    model: nn.Module,
):
    """
    Yield named BatchNorm modules.
    """

    for (
        name,
        module,
    ) in model.named_modules():

        if isinstance(
            module,
            BATCHNORM_TYPES,
        ):

            yield (
                name,
                module,
            )


def batchnorm_parameter_ids(
    model: nn.Module,
) -> set[int]:

    result: set[int] = set()

    for (
        _,
        module,
    ) in batchnorm_modules(
        model
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


# ======================================================================
# Stage-A contract
# ======================================================================

def apply_stage_a_contract(
    model: nn.Module,
) -> None:
    """
    Apply the frozen head-only Stage-A state.

    Required scientific state
    -------------------------
    backbone:
        requires_grad=False
        module mode=eval

    BatchNorm:
        affine requires_grad=False
        running statistics do not update

    classifier:
        requires_grad=True
        module mode=train

    Calling this helper at the beginning of every Stage-A training epoch
    makes the intended state explicit and resistant to a previous
    model.train() call.
    """

    if not hasattr(
        model,
        "fc",
    ):

        raise TypeError(
            "Stage-A contract requires a ResNet-style .fc classifier."
        )

    # Freeze everything first.
    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )

    # Put the complete backbone into evaluation behavior.
    model.eval()

    # Re-enable only the binary classifier.
    for parameter in model.fc.parameters():

        parameter.requires_grad_(
            True
        )

    # Linear train/eval behavior is identical, but this encodes the
    # frozen scientific contract explicitly.
    model.fc.train()

    assert_stage_a_contract(
        model
    )


def assert_stage_a_contract(
    model: nn.Module,
) -> None:
    """
    Fail if the actual module/gradient state violates Stage A.
    """

    backbone_parameters = list(
        backbone_named_parameters(
            model
        )
    )

    classifier_parameters = list(
        classifier_named_parameters(
            model
        )
    )

    if not backbone_parameters:

        raise RuntimeError(
            "No backbone parameters found."
        )

    if not classifier_parameters:

        raise RuntimeError(
            "No classifier parameters found."
        )

    for (
        name,
        parameter,
    ) in backbone_parameters:

        if parameter.requires_grad:

            raise RuntimeError(
                "Stage-A backbone parameter is trainable:\n"
                f"  {name}"
            )

    for (
        name,
        parameter,
    ) in classifier_parameters:

        if not parameter.requires_grad:

            raise RuntimeError(
                "Stage-A classifier parameter is frozen:\n"
                f"  {name}"
            )

    if model.fc.training is not True:

        raise RuntimeError(
            "Stage-A classifier must be in train mode."
        )

    # Parent ResNet remains eval; fc has independently been set train.
    if model.training is not False:

        raise RuntimeError(
            "Stage-A ResNet parent module must remain in eval mode."
        )

    for (
        name,
        module,
    ) in batchnorm_modules(
        model
    ):

        if module.training:

            raise RuntimeError(
                "Stage-A BatchNorm unexpectedly in train mode:\n"
                f"  {name}"
            )

        if not module.track_running_stats:

            raise RuntimeError(
                "Stage-A BatchNorm unexpectedly has "
                "track_running_stats=False:\n"
                f"  {name}"
            )

        if module.affine:

            if (
                module.weight is None
                or
                module.bias is None
            ):

                raise RuntimeError(
                    "Affine BatchNorm missing weight/bias:\n"
                    f"  {name}"
                )

            if module.weight.requires_grad:

                raise RuntimeError(
                    "Stage-A BatchNorm weight is trainable:\n"
                    f"  {name}"
                )

            if module.bias.requires_grad:

                raise RuntimeError(
                    "Stage-A BatchNorm bias is trainable:\n"
                    f"  {name}"
                )


# ======================================================================
# Stage-B contract
# ======================================================================

def apply_stage_b_contract(
    model: nn.Module,
) -> None:
    """
    Apply the frozen full-backbone Stage-B state.

    Required state
    --------------
    all parameters:
        requires_grad=True

    model:
        train mode

    BatchNorm:
        train mode
        affine parameters trainable
        running statistics update
    """

    for parameter in model.parameters():

        parameter.requires_grad_(
            True
        )

    model.train()

    assert_stage_b_contract(
        model
    )


def assert_stage_b_contract(
    model: nn.Module,
) -> None:

    if not model.training:

        raise RuntimeError(
            "Stage-B model must be in train mode."
        )

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not parameter.requires_grad:

            raise RuntimeError(
                "Stage-B parameter remains frozen:\n"
                f"  {name}"
            )

    if not model.fc.training:

        raise RuntimeError(
            "Stage-B classifier must be in train mode."
        )

    for (
        name,
        module,
    ) in batchnorm_modules(
        model
    ):

        if not module.training:

            raise RuntimeError(
                "Stage-B BatchNorm unexpectedly in eval mode:\n"
                f"  {name}"
            )

        if not module.track_running_stats:

            raise RuntimeError(
                "Stage-B requires BatchNorm running-stat "
                "adaptation, but track_running_stats=False:\n"
                f"  {name}"
            )

        if module.affine:

            if (
                module.weight is None
                or
                module.bias is None
            ):

                raise RuntimeError(
                    "Affine BatchNorm missing weight/bias:\n"
                    f"  {name}"
                )

            if not module.weight.requires_grad:

                raise RuntimeError(
                    "Stage-B BatchNorm weight remains frozen:\n"
                    f"  {name}"
                )

            if not module.bias.requires_grad:

                raise RuntimeError(
                    "Stage-B BatchNorm bias remains frozen:\n"
                    f"  {name}"
                )


# ======================================================================
# Evaluation state
# ======================================================================

def apply_evaluation_contract(
    model: nn.Module,
) -> None:
    """
    Put the complete classifier into inference/evaluation mode.

    This will be used for dev evaluation in both Stage A and Stage B.

    After dev evaluation, the training loop must reapply the relevant
    Stage-A or Stage-B contract before the next training epoch.
    """

    model.eval()

    if model.training:

        raise RuntimeError(
            "Evaluation contract failed to set model.eval()."
        )

    for (
        name,
        module,
    ) in batchnorm_modules(
        model
    ):

        if module.training:

            raise RuntimeError(
                "Evaluation contract left BatchNorm in train mode:\n"
                f"  {name}"
            )


# ======================================================================
# Optimizer-policy config
# ======================================================================

def validate_optimizer_grouping_config(
    experiment_cfg: Mapping[str, Any],
) -> tuple[
    float,
    float,
]:

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

    weight_decay = float(
        require_key(
            optimizer_cfg,
            "weight_decay",
            "optimizer",
        )
    )

    if weight_decay != 0.0001:

        raise ValueError(
            "Frozen weight_decay must equal 1e-4."
        )

    exclusions = require_mapping(
        require_key(
            optimizer_cfg,
            "weight_decay_exclusions",
            "optimizer",
        ),
        "optimizer.weight_decay_exclusions",
    )

    if (
        require_key(
            exclusions,
            "all_bias_parameters",
            "optimizer.weight_decay_exclusions",
        )
        is not True
    ):

        raise ValueError(
            "Frozen protocol excludes every bias parameter "
            "from weight decay."
        )

    if (
        require_key(
            exclusions,
            "batchnorm_affine_parameters",
            "optimizer.weight_decay_exclusions",
        )
        is not True
    ):

        raise ValueError(
            "Frozen protocol excludes BatchNorm affine "
            "parameters from weight decay."
        )

    stage_a_cfg = require_mapping(
        require_key(
            require_mapping(
                require_key(
                    experiment_cfg,
                    "transfer_learning",
                    "experiment_config",
                ),
                "transfer_learning",
            ),
            "stage_a",
            "transfer_learning",
        ),
        "transfer_learning.stage_a",
    )

    stage_a_optimizer = require_mapping(
        require_key(
            stage_a_cfg,
            "optimizer",
            "transfer_learning.stage_a",
        ),
        "transfer_learning.stage_a.optimizer",
    )

    classifier_lr = float(
        require_key(
            stage_a_optimizer,
            "classifier_lr",
            "transfer_learning.stage_a.optimizer",
        )
    )

    if classifier_lr != 0.001:

        raise ValueError(
            "Frozen classifier learning rate must equal 1e-3."
        )

    return (
        weight_decay,
        classifier_lr,
    )


# ======================================================================
# Parameter grouping
# ======================================================================

@dataclass(
    frozen=True
)
class ParameterGroupSummary:

    name: str

    parameter_tensors: int
    parameter_elements: int

    learning_rate: float
    weight_decay: float


def _group_trainable_parameters(
    *,
    model: nn.Module,
    backbone_lr: float | None,
    classifier_lr: float,
    weight_decay: float,
) -> tuple[
    list[dict[str, Any]],
    tuple[
        ParameterGroupSummary,
        ...,
    ],
]:
    """
    Build exact AdamW-ready parameter groups.

    Rules
    -----
    - every trainable parameter appears exactly once;
    - every bias receives weight_decay=0;
    - every BatchNorm affine parameter receives weight_decay=0;
    - classifier and backbone keep separate LR namespaces.
    """

    bn_parameter_ids = (
        batchnorm_parameter_ids(
            model
        )
    )

    grouped: dict[
        str,
        list[nn.Parameter],
    ] = {
        "backbone_decay":
            [],

        "backbone_no_decay":
            [],

        "fc_decay":
            [],

        "fc_no_decay":
            [],
    }

    seen_parameter_ids: set[int] = set()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not parameter.requires_grad:

            continue

        parameter_id = id(
            parameter
        )

        if parameter_id in seen_parameter_ids:

            raise RuntimeError(
                "Trainable parameter encountered more than once "
                "while building optimizer groups:\n"
                f"  {name}"
            )

        seen_parameter_ids.add(
            parameter_id
        )

        is_classifier = (
            name.startswith(
                "fc."
            )
        )

        is_bias = (
            name.endswith(
                ".bias"
            )
        )

        is_batchnorm_affine = (
            parameter_id
            in bn_parameter_ids
        )

        no_decay = (
            is_bias
            or is_batchnorm_affine
        )

        if is_classifier:

            group_name = (
                "fc_no_decay"
                if no_decay
                else "fc_decay"
            )

        else:

            group_name = (
                "backbone_no_decay"
                if no_decay
                else "backbone_decay"
            )

        grouped[
            group_name
        ].append(
            parameter
        )

    expected_parameter_ids = {
        id(
            parameter
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    }

    if seen_parameter_ids != expected_parameter_ids:

        missing = (
            expected_parameter_ids
            - seen_parameter_ids
        )

        extra = (
            seen_parameter_ids
            - expected_parameter_ids
        )

        raise RuntimeError(
            "Optimizer parameter grouping failed coverage check:\n"
            f"  missing={len(missing)}\n"
            f"  extra={len(extra)}"
        )

    groups: list[
        dict[str, Any]
    ] = []

    summaries: list[
        ParameterGroupSummary
    ] = []

    ordered_group_names = (
        "backbone_decay",
        "backbone_no_decay",
        "fc_decay",
        "fc_no_decay",
    )

    for group_name in ordered_group_names:

        parameters = grouped[
            group_name
        ]

        # Stage A legitimately has no backbone groups because the
        # complete backbone is frozen. Empty groups are omitted.
        if not parameters:

            continue

        if group_name.startswith(
            "backbone_"
        ):

            if backbone_lr is None:

                raise RuntimeError(
                    "Trainable backbone parameters exist but "
                    "backbone_lr=None."
                )

            lr = float(
                backbone_lr
            )

        else:

            lr = float(
                classifier_lr
            )

        group_weight_decay = (
            0.0
            if group_name.endswith(
                "_no_decay"
            )
            else float(
                weight_decay
            )
        )

        groups.append(
            {
                "name":
                    group_name,

                "params":
                    parameters,

                "lr":
                    lr,

                "weight_decay":
                    group_weight_decay,
            }
        )

        summaries.append(
            ParameterGroupSummary(
                name=group_name,
                parameter_tensors=len(
                    parameters
                ),
                parameter_elements=sum(
                    parameter.numel()
                    for parameter
                    in parameters
                ),
                learning_rate=lr,
                weight_decay=(
                    group_weight_decay
                ),
            )
        )

    return (
        groups,
        tuple(
            summaries
        ),
    )


def build_stage_a_parameter_groups(
    *,
    model: nn.Module,
    experiment_cfg: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    tuple[
        ParameterGroupSummary,
        ...,
    ],
]:
    """
    Build Stage-A classifier-only parameter groups.

    Expected group names:
        fc_decay
        fc_no_decay
    """

    assert_stage_a_contract(
        model
    )

    (
        weight_decay,
        classifier_lr,
    ) = validate_optimizer_grouping_config(
        experiment_cfg
    )

    groups, summaries = (
        _group_trainable_parameters(
            model=model,
            backbone_lr=None,
            classifier_lr=classifier_lr,
            weight_decay=weight_decay,
        )
    )

    actual_names = [
        str(
            group[
                "name"
            ]
        )
        for group
        in groups
    ]

    expected_names = [
        "fc_decay",
        "fc_no_decay",
    ]

    if actual_names != expected_names:

        raise RuntimeError(
            "Unexpected Stage-A parameter groups:\n"
            f"  expected={expected_names}\n"
            f"  actual={actual_names}"
        )

    return (
        groups,
        summaries,
    )


def build_stage_b_parameter_groups(
    *,
    model: nn.Module,
    experiment_cfg: Mapping[str, Any],
    backbone_lr: float,
) -> tuple[
    list[dict[str, Any]],
    tuple[
        ParameterGroupSummary,
        ...,
    ],
]:
    """
    Build the four frozen Stage-B parameter groups.

    backbone_lr must be one of the predeclared screening candidates.
    """

    assert_stage_b_contract(
        model
    )

    (
        weight_decay,
        classifier_lr,
    ) = validate_optimizer_grouping_config(
        experiment_cfg
    )

    transfer_learning = require_mapping(
        require_key(
            experiment_cfg,
            "transfer_learning",
            "experiment_config",
        ),
        "transfer_learning",
    )

    stage_b = require_mapping(
        require_key(
            transfer_learning,
            "stage_b",
            "transfer_learning",
        ),
        "transfer_learning.stage_b",
    )

    stage_b_optimizer = require_mapping(
        require_key(
            stage_b,
            "optimizer",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.optimizer",
    )

    configured_candidates = [
        float(
            value
        )
        for value
        in require_key(
            stage_b_optimizer,
            "backbone_lr_candidates",
            "transfer_learning.stage_b.optimizer",
        )
    ]

    candidate = float(
        backbone_lr
    )

    if candidate not in configured_candidates:

        raise ValueError(
            "Stage-B backbone LR is outside the frozen candidate set:\n"
            f"  requested={candidate}\n"
            f"  allowed={configured_candidates}"
        )

    configured_classifier_lr = float(
        require_key(
            stage_b_optimizer,
            "classifier_lr",
            "transfer_learning.stage_b.optimizer",
        )
    )

    if configured_classifier_lr != classifier_lr:

        raise RuntimeError(
            "Stage-A/Stage-B classifier learning-rate "
            "configuration disagrees."
        )

    groups, summaries = (
        _group_trainable_parameters(
            model=model,
            backbone_lr=candidate,
            classifier_lr=classifier_lr,
            weight_decay=weight_decay,
        )
    )

    actual_names = [
        str(
            group[
                "name"
            ]
        )
        for group
        in groups
    ]

    expected_names = [
        "backbone_decay",
        "backbone_no_decay",
        "fc_decay",
        "fc_no_decay",
    ]

    if actual_names != expected_names:

        raise RuntimeError(
            "Unexpected Stage-B parameter groups:\n"
            f"  expected={expected_names}\n"
            f"  actual={actual_names}"
        )

    return (
        groups,
        summaries,
    )