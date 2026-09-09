"""
Development-only diagnostic metrics for the frozen Tech-2 ResNet-18
pipeline.

Implemented
-----------
- binary dev_val AUROC;
- attack = positive class = 1;
- deterministic exact pairwise AUROC definition;
- explicit tie handling;
- Stage-B AUROC-disagreement guard contract metadata.

Not implemented
---------------
- checkpoint selection;
- early stopping;
- threshold derivation;
- FPR10;
- held-out test evaluation;
- calibration;
- Grad-CAM.

Scientific role
---------------
Weighted dev cross-entropy remains the ONLY checkpoint-selection metric.

AUROC is diagnostic only.

For every Stage-B run the later training controller will compute:

    best_dev_auroc_in_run
        -
    dev_auroc_at_loss_argmin_checkpoint

and flag protocol review if the difference is strictly greater than
0.01.

Score representation
--------------------
For two-class logits:

    z0 = bonafide logit
    z1 = attack logit

the AUROC ranking score used here is:

    z1 - z0

For a two-class softmax, attack probability is a strictly monotonic
function of this margin in exact arithmetic. Therefore the ranking
represented by the margin is the binary classifier ranking relevant to
AUROC.

Importantly, this does NOT freeze the later FPR10 threshold-score
representation. Threshold derivation remains deferred to the exact
frozen Tech-1 evaluation convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
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
# Frozen AUROC contract
# ======================================================================

@dataclass(
    frozen=True
)
class DevAUROCContract:

    positive_class_name: str
    positive_class_index: int

    diagnostic_enabled: bool
    participates_in_checkpoint_selection: bool

    stage_b_disagreement_guard_enabled: bool
    stage_b_disagreement_threshold: float

    automatically_switch_checkpoint: bool


def load_dev_auroc_contract(
    *,
    experiment_cfg: Mapping[str, Any],
) -> DevAUROCContract:

    class_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "class_contract",
            "experiment_config",
        ),
        "class_contract",
    )

    positive_class_name = str(
        require_key(
            class_cfg,
            "positive_class",
            "class_contract",
        )
    )

    positive_class_index = int(
        require_key(
            class_cfg,
            "positive_class_index",
            "class_contract",
        )
    )

    if positive_class_name != "attack":

        raise ValueError(
            "Frozen positive class must be attack."
        )

    if positive_class_index != 1:

        raise ValueError(
            "Frozen attack class index must equal 1."
        )

    evaluation_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "evaluation",
            "experiment_config",
        ),
        "evaluation",
    )

    diagnostic_cfg = require_mapping(
        require_key(
            evaluation_cfg,
            "diagnostic_metrics",
            "evaluation",
        ),
        "evaluation.diagnostic_metrics",
    )

    dev_auroc_cfg = require_mapping(
        require_key(
            diagnostic_cfg,
            "dev_auroc",
            "evaluation.diagnostic_metrics",
        ),
        "evaluation.diagnostic_metrics.dev_auroc",
    )

    diagnostic_enabled = require_key(
        dev_auroc_cfg,
        "enabled",
        "evaluation.diagnostic_metrics.dev_auroc",
    )

    if diagnostic_enabled is not True:

        raise ValueError(
            "Frozen protocol requires dev AUROC enabled."
        )

    participates = require_key(
        dev_auroc_cfg,
        "participates_in_checkpoint_selection",
        "evaluation.diagnostic_metrics.dev_auroc",
    )

    if participates is not False:

        raise ValueError(
            "Dev AUROC must not participate in checkpoint selection."
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

    guard_cfg = require_mapping(
        require_key(
            stage_b_cfg,
            "auroc_disagreement_guard",
            "transfer_learning.stage_b",
        ),
        "transfer_learning.stage_b.auroc_disagreement_guard",
    )

    guard_enabled = require_key(
        guard_cfg,
        "enabled",
        "transfer_learning.stage_b.auroc_disagreement_guard",
    )

    if guard_enabled is not True:

        raise ValueError(
            "Frozen Stage-B AUROC disagreement guard must be enabled."
        )

    guard_threshold = float(
        require_key(
            guard_cfg,
            "flag_if_strictly_greater_than",
            "transfer_learning.stage_b.auroc_disagreement_guard",
        )
    )

    if guard_threshold != 0.01:

        raise ValueError(
            "Frozen Stage-B AUROC disagreement threshold "
            "must equal 0.01."
        )

    action_cfg = require_mapping(
        require_key(
            guard_cfg,
            "action",
            "transfer_learning.stage_b.auroc_disagreement_guard",
        ),
        (
            "transfer_learning.stage_b."
            "auroc_disagreement_guard.action"
        ),
    )

    if (
        require_key(
            action_cfg,
            "flag_for_protocol_review",
            (
                "transfer_learning.stage_b."
                "auroc_disagreement_guard.action"
            ),
        )
        is not True
    ):

        raise ValueError(
            "AUROC disagreement must flag protocol review."
        )

    auto_switch = require_key(
        action_cfg,
        "automatically_switch_checkpoint",
        (
            "transfer_learning.stage_b."
            "auroc_disagreement_guard.action"
        ),
    )

    if auto_switch is not False:

        raise ValueError(
            "AUROC disagreement must not automatically "
            "switch checkpoints."
        )

    return DevAUROCContract(
        positive_class_name=positive_class_name,
        positive_class_index=positive_class_index,
        diagnostic_enabled=True,
        participates_in_checkpoint_selection=False,
        stage_b_disagreement_guard_enabled=True,
        stage_b_disagreement_threshold=guard_threshold,
        automatically_switch_checkpoint=False,
    )


# ======================================================================
# Result
# ======================================================================

@dataclass(
    frozen=True
)
class DevAUROCResult:

    auroc: float

    sample_count: int
    bonafide_count: int
    attack_count: int

    positive_negative_pair_count: int

    strict_attack_wins: int
    tied_attack_bonafide_pairs: int

    score_definition: str


# ======================================================================
# Input validation
# ======================================================================

def _validate_dev_tensors(
    *,
    experiment_cfg: Mapping[str, Any],
    logits: Tensor,
    targets: Tensor,
) -> tuple[
    int,
    int,
    int,
]:

    if not isinstance(
        logits,
        Tensor,
    ):

        raise TypeError(
            "logits must be a torch.Tensor."
        )

    if not isinstance(
        targets,
        Tensor,
    ):

        raise TypeError(
            "targets must be a torch.Tensor."
        )

    if logits.ndim != 2:

        raise ValueError(
            "Dev logits must have shape [N, 2]."
        )

    if logits.shape[
        1
    ] != 2:

        raise ValueError(
            "Dev logits must contain exactly two classes."
        )

    if targets.ndim != 1:

        raise ValueError(
            "Dev targets must have shape [N]."
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
            "Dev logits/targets sample counts differ."
        )

    if logits.dtype != torch.float32:

        raise TypeError(
            "Frozen epoch engine must supply dev logits as float32."
        )

    if targets.dtype != torch.int64:

        raise TypeError(
            "Frozen epoch engine must supply dev targets as int64."
        )

    if logits.device.type != "cpu":

        raise ValueError(
            "Development metric input logits must be CPU-resident."
        )

    if targets.device.type != "cpu":

        raise ValueError(
            "Development metric input targets must be CPU-resident."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):

        raise ValueError(
            "Dev logits contain NaN or Inf."
        )

    unique_targets = set(
        int(
            value
        )
        for value
        in torch.unique(
            targets
        ).tolist()
    )

    if not unique_targets.issubset(
        {
            0,
            1,
        }
    ):

        raise ValueError(
            "Dev targets contain labels outside frozen {0,1}."
        )

    class_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "class_contract",
            "experiment_config",
        ),
        "class_contract",
    )

    dev_counts_cfg = require_mapping(
        require_key(
            class_cfg,
            "dev_val_counts",
            "class_contract",
        ),
        "class_contract.dev_val_counts",
    )

    expected_bonafide = int(
        require_key(
            dev_counts_cfg,
            "bonafide",
            "class_contract.dev_val_counts",
        )
    )

    expected_attack = int(
        require_key(
            dev_counts_cfg,
            "attack",
            "class_contract.dev_val_counts",
        )
    )

    expected_total = int(
        require_key(
            dev_counts_cfg,
            "total",
            "class_contract.dev_val_counts",
        )
    )

    if (
        expected_bonafide,
        expected_attack,
        expected_total,
    ) != (
        153,
        306,
        459,
    ):

        raise ValueError(
            "Frozen dev_val class-count contract changed."
        )

    actual_bonafide = int(
        (
            targets
            == 0
        )
        .sum()
        .item()
    )

    actual_attack = int(
        (
            targets
            == 1
        )
        .sum()
        .item()
    )

    actual_total = int(
        targets.numel()
    )

    if actual_total != expected_total:

        raise ValueError(
            "Development AUROC requires the complete frozen dev_val "
            "partition:\n"
            f"  expected={expected_total}\n"
            f"  actual={actual_total}"
        )

    if actual_bonafide != expected_bonafide:

        raise ValueError(
            "dev_val bonafide count mismatch:\n"
            f"  expected={expected_bonafide}\n"
            f"  actual={actual_bonafide}"
        )

    if actual_attack != expected_attack:

        raise ValueError(
            "dev_val attack count mismatch:\n"
            f"  expected={expected_attack}\n"
            f"  actual={actual_attack}"
        )

    return (
        actual_total,
        actual_bonafide,
        actual_attack,
    )


# ======================================================================
# Attack ranking score
# ======================================================================

def attack_ranking_score_from_logits(
    logits: Tensor,
) -> Tensor:
    """
    Return deterministic float64 attack-vs-bonafide ranking margin:

        attack_logit - bonafide_logit

    Input logits remain unmodified.
    """

    if not isinstance(
        logits,
        Tensor,
    ):

        raise TypeError(
            "logits must be a torch.Tensor."
        )

    if (
        logits.ndim != 2
        or logits.shape[
            1
        ] != 2
    ):

        raise ValueError(
            "Expected logits with shape [N,2]."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):

        raise ValueError(
            "Logits contain NaN or Inf."
        )

    # Cast before subtraction so score arithmetic itself is float64.
    values = logits.to(
        dtype=torch.float64,
        device="cpu",
    )

    scores = (
        values[
            :,
            1
        ]
        -
        values[
            :,
            0
        ]
    )

    if not bool(
        torch.isfinite(
            scores
        ).all()
    ):

        raise RuntimeError(
            "Attack ranking scores contain NaN or Inf."
        )

    return scores.contiguous()


# ======================================================================
# Exact pairwise binary AUROC
# ======================================================================

def binary_auroc_pairwise(
    *,
    scores: Tensor,
    targets: Tensor,
    positive_class_index: int = 1,
) -> tuple[
    float,
    int,
    int,
    int,
]:
    """
    Compute binary AUROC using its pairwise probability definition.

    For every positive-negative pair:

        positive score > negative score -> 1
        equal scores                   -> 0.5
        positive score < negative score -> 0

    AUROC = accumulated credit / number of positive-negative pairs.

    Returns:
        auroc
        pair_count
        strict_positive_wins
        tied_pairs
    """

    if positive_class_index != 1:

        raise ValueError(
            "Frozen positive class index must equal 1."
        )

    if scores.ndim != 1:

        raise ValueError(
            "scores must have shape [N]."
        )

    if targets.ndim != 1:

        raise ValueError(
            "targets must have shape [N]."
        )

    if scores.shape != targets.shape:

        raise ValueError(
            "scores/targets shapes differ."
        )

    if scores.device.type != "cpu":

        raise ValueError(
            "Pairwise AUROC scores must be on CPU."
        )

    if targets.device.type != "cpu":

        raise ValueError(
            "Pairwise AUROC targets must be on CPU."
        )

    if scores.dtype != torch.float64:

        raise TypeError(
            "Pairwise AUROC scores must be float64."
        )

    if targets.dtype != torch.int64:

        raise TypeError(
            "Pairwise AUROC targets must be int64."
        )

    if not bool(
        torch.isfinite(
            scores
        ).all()
    ):

        raise ValueError(
            "AUROC scores contain NaN or Inf."
        )

    positive_scores = scores[
        targets
        == positive_class_index
    ]

    negative_scores = scores[
        targets
        != positive_class_index
    ]

    positive_count = int(
        positive_scores.numel()
    )

    negative_count = int(
        negative_scores.numel()
    )

    if positive_count <= 0:

        raise ValueError(
            "AUROC requires at least one positive example."
        )

    if negative_count <= 0:

        raise ValueError(
            "AUROC requires at least one negative example."
        )

    pair_count = (
        positive_count
        * negative_count
    )

    comparisons = (
        positive_scores[
            :,
            None
        ],
        negative_scores[
            None,
            :,
        ],
    )

    positive_grid, negative_grid = (
        comparisons
    )

    strict_wins = int(
        (
            positive_grid
            > negative_grid
        )
        .sum()
        .item()
    )

    tied_pairs = int(
        (
            positive_grid
            == negative_grid
        )
        .sum()
        .item()
    )

    auroc = (
        (
            float(
                strict_wins
            )
            +
            0.5
            * float(
                tied_pairs
            )
        )
        /
        float(
            pair_count
        )
    )

    if not math.isfinite(
        auroc
    ):

        raise RuntimeError(
            "Computed AUROC is non-finite."
        )

    if not (
        0.0
        <= auroc
        <= 1.0
    ):

        raise RuntimeError(
            f"Computed AUROC outside [0,1]: {auroc}"
        )

    return (
        auroc,
        pair_count,
        strict_wins,
        tied_pairs,
    )


# ======================================================================
# Complete frozen dev AUROC
# ======================================================================

def compute_dev_auroc(
    *,
    experiment_cfg: Mapping[str, Any],
    logits: Tensor,
    targets: Tensor,
) -> DevAUROCResult:
    """
    Compute the diagnostic AUROC for one COMPLETE dev_val evaluation.

    This function deliberately refuses partial-dev inputs.
    """

    contract = load_dev_auroc_contract(
        experiment_cfg=experiment_cfg,
    )

    (
        sample_count,
        bonafide_count,
        attack_count,
    ) = _validate_dev_tensors(
        experiment_cfg=experiment_cfg,
        logits=logits,
        targets=targets,
    )

    scores = attack_ranking_score_from_logits(
        logits
    )

    (
        auroc,
        pair_count,
        strict_wins,
        tied_pairs,
    ) = binary_auroc_pairwise(
        scores=scores,
        targets=targets,
        positive_class_index=(
            contract.positive_class_index
        ),
    )

    expected_pair_count = (
        bonafide_count
        * attack_count
    )

    if pair_count != expected_pair_count:

        raise RuntimeError(
            "AUROC positive-negative pair count does not reconcile:\n"
            f"  expected={expected_pair_count}\n"
            f"  actual={pair_count}"
        )

    return DevAUROCResult(
        auroc=auroc,
        sample_count=sample_count,
        bonafide_count=bonafide_count,
        attack_count=attack_count,
        positive_negative_pair_count=(
            pair_count
        ),
        strict_attack_wins=(
            strict_wins
        ),
        tied_attack_bonafide_pairs=(
            tied_pairs
        ),
        score_definition=(
            "attack_logit_minus_bonafide_logit"
        ),
    )