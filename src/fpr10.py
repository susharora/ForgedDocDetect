"""
Frozen Tech-2 score and empirical-FPR threshold primitives.

Native binary convention
------------------------
    bonafide = 0
    attack   = 1
    positive class = attack

Threshold score
---------------
Stored model logits are CPU float32:

    z0 = bonafide logit
    z1 = attack logit

For threshold decisions:

    p_attack = softmax(float64([z0, z1]))[1]

The returned score is CPU float64.

Fixed operating point
---------------------
    attack iff p_attack >= 0.5

Controlled-FPR operating point
------------------------------
Threshold derivation receives ONLY the complete frozen dev_val
bona-fide score vector.

Let:

    N = number of dev bona-fide scores
    K = floor(0.10 * N)

Sort descending:

    b1 >= b2 >= ... >= bN

The first score that is not permitted to remain on the attack side is:

    b_(K+1)

The frozen threshold is:

    nextafter(b_(K+1), +infinity)

Because prediction uses >=, all scores equal to the boundary are
excluded from the attack side.

No interpolation.
No random tie handling.
No threshold clamping.

This module contains no dataset path, checkpoint path or test route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


# ======================================================================
# Frozen contract
# ======================================================================

@dataclass(
    frozen=True
)
class FPR10Contract:

    bonafide_index: int
    attack_index: int
    positive_class: str

    stored_logit_dtype: str

    threshold_score_representation: str
    threshold_score_dtype: str

    fixed_threshold: float

    threshold_source_partition: str
    threshold_selection_population: str

    expected_dev_bonafide_count: int

    target_false_positive_rate: float
    expected_allowed_false_positive_count: int

    prediction_rule: str

    score_order: str
    first_disallowed_rank_rule: str

    threshold_rule: str

    interpolation: bool
    randomized_tie_handling: bool

    attack_scores_participate: bool
    threshold_clamping: bool


@dataclass(
    frozen=True
)
class FPR10ThresholdResult:

    threshold: float

    boundary_score: float
    boundary_rank_one_based: int

    target_false_positive_rate: float

    bonafide_count: int

    allowed_false_positive_count: int

    achieved_false_positive_count: int
    achieved_false_positive_rate: float

    boundary_tie_count: int
    strictly_above_boundary_count: int

    threshold_above_one: bool


# ======================================================================
# Mapping helpers
# ======================================================================

def _require_mapping(
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


def _require_key(
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
# Contract parser
# ======================================================================

def load_fpr10_contract(
    *,
    evaluation_cfg: Mapping[str, Any],
) -> FPR10Contract:

    if int(
        _require_key(
            evaluation_cfg,
            "schema_version",
            "evaluation_protocol",
        )
    ) != 1:

        raise RuntimeError(
            "Final-detection protocol schema_version must equal 1."
        )

    if str(
        _require_key(
            evaluation_cfg,
            "artifact_type",
            "evaluation_protocol",
        )
    ) != "resnet18_final_detection_protocol":

        raise RuntimeError(
            "Unexpected final-detection artifact_type."
        )

    if str(
        _require_key(
            evaluation_cfg,
            "status",
            "evaluation_protocol",
        )
    ) != "frozen_before_held_out_test_access":

        raise RuntimeError(
            "Final-detection protocol is not frozen."
        )

    class_cfg = _require_mapping(
        _require_key(
            evaluation_cfg,
            "class_contract",
            "evaluation_protocol",
        ),
        "class_contract",
    )

    index_by_name = _require_mapping(
        _require_key(
            class_cfg,
            "index_by_name",
            "class_contract",
        ),
        "class_contract.index_by_name",
    )

    bonafide_index = int(
        _require_key(
            index_by_name,
            "bonafide",
            "class_contract.index_by_name",
        )
    )

    attack_index = int(
        _require_key(
            index_by_name,
            "attack",
            "class_contract.index_by_name",
        )
    )

    positive_class = str(
        _require_key(
            class_cfg,
            "positive_class",
            "class_contract",
        )
    )

    positive_class_index = int(
        _require_key(
            class_cfg,
            "positive_class_index",
            "class_contract",
        )
    )

    if bonafide_index != 0:

        raise RuntimeError(
            "Frozen bonafide index must equal 0."
        )

    if attack_index != 1:

        raise RuntimeError(
            "Frozen attack index must equal 1."
        )

    if positive_class != "attack":

        raise RuntimeError(
            "Frozen positive class must be attack."
        )

    if positive_class_index != 1:

        raise RuntimeError(
            "Frozen positive-class index must equal 1."
        )

    score_cfg = _require_mapping(
        _require_key(
            evaluation_cfg,
            "score",
            "evaluation_protocol",
        ),
        "score",
    )

    score_source_cfg = _require_mapping(
        _require_key(
            score_cfg,
            "source",
            "score",
        ),
        "score.source",
    )

    threshold_score_cfg = _require_mapping(
        _require_key(
            score_cfg,
            "threshold_decisions",
            "score",
        ),
        "score.threshold_decisions",
    )

    stored_logit_dtype = str(
        _require_key(
            score_source_cfg,
            "stored_dtype",
            "score.source",
        )
    )

    if stored_logit_dtype != "float32":

        raise RuntimeError(
            "Stored classifier-logit dtype must be float32."
        )

    threshold_score_representation = str(
        _require_key(
            threshold_score_cfg,
            "representation",
            "score.threshold_decisions",
        )
    )

    if (
        threshold_score_representation
        != "attack_softmax_probability"
    ):

        raise RuntimeError(
            "Threshold score must be attack softmax probability."
        )

    threshold_score_dtype = str(
        _require_key(
            threshold_score_cfg,
            "computation_dtype",
            "score.threshold_decisions",
        )
    )

    if threshold_score_dtype != "float64":

        raise RuntimeError(
            "Threshold-score computation dtype must be float64."
        )

    operating_cfg = _require_mapping(
        _require_key(
            evaluation_cfg,
            "operating_points",
            "evaluation_protocol",
        ),
        "operating_points",
    )

    fixed_cfg = _require_mapping(
        _require_key(
            operating_cfg,
            "fixed_0_5",
            "operating_points",
        ),
        "operating_points.fixed_0_5",
    )

    fixed_threshold = float(
        _require_key(
            fixed_cfg,
            "threshold",
            "operating_points.fixed_0_5",
        )
    )

    if fixed_threshold != 0.5:

        raise RuntimeError(
            "Frozen fixed operating threshold must equal 0.5."
        )

    if str(
        _require_key(
            fixed_cfg,
            "prediction_rule",
            "operating_points.fixed_0_5",
        )
    ) != "attack iff p_attack >= 0.5":

        raise RuntimeError(
            "Frozen fixed-threshold prediction rule changed."
        )

    fpr_cfg = _require_mapping(
        _require_key(
            operating_cfg,
            "controlled_fpr10",
            "operating_points",
        ),
        "operating_points.controlled_fpr10",
    )

    threshold_source_partition = str(
        _require_key(
            fpr_cfg,
            "threshold_source_partition",
            "operating_points.controlled_fpr10",
        )
    )

    if threshold_source_partition != "dev_val":

        raise RuntimeError(
            "Controlled-FPR threshold source must be dev_val."
        )

    selection_population = str(
        _require_key(
            fpr_cfg,
            "threshold_selection_population",
            "operating_points.controlled_fpr10",
        )
    )

    if (
        selection_population
        != "complete_frozen_dev_val_bonafide_subset_only"
    ):

        raise RuntimeError(
            "FPR10 threshold population must be dev bona-fide only."
        )

    expected_dev_bonafide_count = int(
        _require_key(
            fpr_cfg,
            "expected_dev_bonafide_count",
            "operating_points.controlled_fpr10",
        )
    )

    if expected_dev_bonafide_count != 153:

        raise RuntimeError(
            "Frozen dev bona-fide count must equal 153."
        )

    target_fpr = float(
        _require_key(
            fpr_cfg,
            "target_empirical_false_positive_rate",
            "operating_points.controlled_fpr10",
        )
    )

    if target_fpr != 0.10:

        raise RuntimeError(
            "Frozen target empirical FPR must equal 0.10."
        )

    expected_allowed_fp = int(
        _require_key(
            fpr_cfg,
            "expected_allowed_false_positive_count",
            "operating_points.controlled_fpr10",
        )
    )

    if expected_allowed_fp != 15:

        raise RuntimeError(
            "Frozen allowed FP count must equal 15."
        )

    attack_scores_participate = _require_key(
        fpr_cfg,
        "attack_scores_participate_in_threshold_selection",
        "operating_points.controlled_fpr10",
    )

    if attack_scores_participate is not False:

        raise RuntimeError(
            "Attack scores must not participate in FPR10 selection."
        )

    prediction_rule = str(
        _require_key(
            fpr_cfg,
            "prediction_rule",
            "operating_points.controlled_fpr10",
        )
    )

    if (
        prediction_rule
        != "attack iff p_attack >= threshold_FPR10"
    ):

        raise RuntimeError(
            "Controlled-FPR prediction rule changed."
        )

    score_order = str(
        _require_key(
            fpr_cfg,
            "score_order",
            "operating_points.controlled_fpr10",
        )
    )

    if score_order != "descending":

        raise RuntimeError(
            "FPR10 score order must be descending."
        )

    first_disallowed_rank_rule = str(
        _require_key(
            fpr_cfg,
            "first_disallowed_rank_rule",
            "operating_points.controlled_fpr10",
        )
    )

    if first_disallowed_rank_rule != "K + 1":

        raise RuntimeError(
            "Frozen first-disallowed-rank rule changed."
        )

    expected_rank = int(
        _require_key(
            fpr_cfg,
            "expected_first_disallowed_rank",
            "operating_points.controlled_fpr10",
        )
    )

    if expected_rank != 16:

        raise RuntimeError(
            "Frozen first disallowed rank must equal 16."
        )

    threshold_rule = str(
        _require_key(
            fpr_cfg,
            "threshold_rule",
            "operating_points.controlled_fpr10",
        )
    )

    if (
        threshold_rule
        != "threshold_FPR10 = nextafter(boundary_score, +infinity)"
    ):

        raise RuntimeError(
            "Frozen FPR10 nextafter rule changed."
        )

    threshold_dtype = str(
        _require_key(
            fpr_cfg,
            "threshold_dtype",
            "operating_points.controlled_fpr10",
        )
    )

    if threshold_dtype != "float64":

        raise RuntimeError(
            "FPR10 threshold dtype must be float64."
        )

    interpolation = _require_key(
        fpr_cfg,
        "interpolation",
        "operating_points.controlled_fpr10",
    )

    if interpolation is not False:

        raise RuntimeError(
            "FPR10 interpolation must remain disabled."
        )

    randomized_tie_handling = _require_key(
        fpr_cfg,
        "randomized_tie_handling",
        "operating_points.controlled_fpr10",
    )

    if randomized_tie_handling is not False:

        raise RuntimeError(
            "Randomized FPR10 tie handling must remain disabled."
        )

    boundary_tie_policy = str(
        _require_key(
            fpr_cfg,
            "boundary_tie_policy",
            "operating_points.controlled_fpr10",
        )
    )

    if (
        boundary_tie_policy
        != "exclude_entire_boundary_tie_group_from_attack_side"
    ):

        raise RuntimeError(
            "Frozen boundary-tie policy changed."
        )

    threshold_clamping = _require_key(
        fpr_cfg,
        "threshold_clamping_to_probability_range",
        "operating_points.controlled_fpr10",
    )

    if threshold_clamping is not False:

        raise RuntimeError(
            "Frozen FPR10 threshold must not be clamped."
        )

    return FPR10Contract(
        bonafide_index=bonafide_index,
        attack_index=attack_index,
        positive_class=positive_class,

        stored_logit_dtype=stored_logit_dtype,

        threshold_score_representation=(
            threshold_score_representation
        ),

        threshold_score_dtype=(
            threshold_score_dtype
        ),

        fixed_threshold=fixed_threshold,

        threshold_source_partition=(
            threshold_source_partition
        ),

        threshold_selection_population=(
            selection_population
        ),

        expected_dev_bonafide_count=(
            expected_dev_bonafide_count
        ),

        target_false_positive_rate=(
            target_fpr
        ),

        expected_allowed_false_positive_count=(
            expected_allowed_fp
        ),

        prediction_rule=(
            prediction_rule
        ),

        score_order=(
            score_order
        ),

        first_disallowed_rank_rule=(
            first_disallowed_rank_rule
        ),

        threshold_rule=(
            threshold_rule
        ),

        interpolation=False,
        randomized_tie_handling=False,

        attack_scores_participate=False,
        threshold_clamping=False,
    )


# ======================================================================
# FP32 two-class logits -> float64 p_attack
# ======================================================================

def attack_probability_from_logits(
    logits: Tensor,
) -> Tensor:
    """
    Convert stored CPU float32 [N,2] logits into CPU float64 p_attack.
    """

    if not isinstance(
        logits,
        Tensor,
    ):

        raise TypeError(
            "logits must be a torch.Tensor."
        )

    if logits.ndim != 2:

        raise RuntimeError(
            "logits must have shape [N,2]."
        )

    if tuple(
        logits.shape[
            1:
        ]
    ) != (
        2,
    ):

        raise RuntimeError(
            "logits must contain exactly two class columns."
        )

    if logits.shape[
        0
    ] <= 0:

        raise RuntimeError(
            "logits must contain at least one sample."
        )

    if logits.dtype != torch.float32:

        raise RuntimeError(
            "Stored classifier logits must be float32."
        )

    if logits.device.type != "cpu":

        raise RuntimeError(
            "Stored classifier logits must be CPU resident."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):

        raise RuntimeError(
            "Classifier logits contain NaN or Inf."
        )

    # Clone so this operation can never mutate persisted logits.
    logits64 = (
        logits
        .detach()
        .to(
            dtype=torch.float64
        )
        .clone()
    )

    probabilities = torch.softmax(
        logits64,
        dim=1,
    )

    attack_scores = (
        probabilities[
            :,
            1,
        ]
        .contiguous()
    )

    if attack_scores.dtype != torch.float64:

        raise RuntimeError(
            "p_attack must be float64."
        )

    if attack_scores.device.type != "cpu":

        raise RuntimeError(
            "p_attack must remain CPU resident."
        )

    if not bool(
        torch.isfinite(
            attack_scores
        ).all()
    ):

        raise RuntimeError(
            "p_attack contains NaN or Inf."
        )

    if bool(
        (
            attack_scores
            < 0.0
        ).any()
    ):

        raise RuntimeError(
            "p_attack contains value below 0."
        )

    if bool(
        (
            attack_scores
            > 1.0
        ).any()
    ):

        raise RuntimeError(
            "p_attack contains value above 1."
        )

    return attack_scores


# ======================================================================
# Probability-score validation
# ======================================================================

def _validate_probability_scores(
    *,
    scores: Tensor,
    label: str,
) -> None:

    if not isinstance(
        scores,
        Tensor,
    ):

        raise TypeError(
            f"{label} must be a torch.Tensor."
        )

    if scores.ndim != 1:

        raise RuntimeError(
            f"{label} must have shape [N]."
        )

    if scores.numel() <= 0:

        raise RuntimeError(
            f"{label} must not be empty."
        )

    if scores.dtype != torch.float64:

        raise RuntimeError(
            f"{label} must be float64."
        )

    if scores.device.type != "cpu":

        raise RuntimeError(
            f"{label} must be CPU resident."
        )

    if not bool(
        torch.isfinite(
            scores
        ).all()
    ):

        raise RuntimeError(
            f"{label} contains NaN or Inf."
        )

    if bool(
        (
            scores
            < 0.0
        ).any()
    ):

        raise RuntimeError(
            f"{label} contains probability below 0."
        )

    if bool(
        (
            scores
            > 1.0
        ).any()
    ):

        raise RuntimeError(
            f"{label} contains probability above 1."
        )


# ======================================================================
# Threshold application
# ======================================================================

def apply_attack_threshold(
    *,
    scores: Tensor,
    threshold: float,
) -> Tensor:
    """
    Apply the frozen attack-positive decision rule.

        attack iff p_attack >= threshold

    Threshold is intentionally permitted to be slightly >1.0.
    """

    _validate_probability_scores(
        scores=scores,
        label="scores",
    )

    threshold_value = float(
        threshold
    )

    if not math.isfinite(
        threshold_value
    ):

        raise RuntimeError(
            "threshold must be finite."
        )

    return (
        scores
        >= threshold_value
    ).to(
        dtype=torch.int64
    )


# ======================================================================
# Frozen empirical-FPR threshold
# ======================================================================

def derive_fpr10_threshold(
    *,
    bonafide_scores: Tensor,
    contract: FPR10Contract,
) -> FPR10ThresholdResult:
    """
    Derive FPR10 from the complete dev bona-fide score population only.

    No attack scores or attack labels are accepted by this API.
    """

    if not isinstance(
        contract,
        FPR10Contract,
    ):

        raise TypeError(
            "contract must be FPR10Contract."
        )

    _validate_probability_scores(
        scores=bonafide_scores,
        label="bonafide_scores",
    )

    bonafide_count = int(
        bonafide_scores.numel()
    )

    if (
        bonafide_count
        != contract.expected_dev_bonafide_count
    ):

        raise RuntimeError(
            "Frozen dev bona-fide count mismatch:\n"
            f"  expected="
            f"{contract.expected_dev_bonafide_count}\n"
            f"  actual={bonafide_count}"
        )

    target_fpr = float(
        contract.target_false_positive_rate
    )

    allowed_false_positive_count = math.floor(
        target_fpr
        * bonafide_count
    )

    if (
        allowed_false_positive_count
        != contract.expected_allowed_false_positive_count
    ):

        raise RuntimeError(
            "Frozen allowed-FP calculation mismatch:\n"
            f"  expected="
            f"{contract.expected_allowed_false_positive_count}\n"
            f"  actual={allowed_false_positive_count}"
        )

    if allowed_false_positive_count < 0:

        raise RuntimeError(
            "Allowed false-positive count became negative."
        )

    if (
        allowed_false_positive_count
        >= bonafide_count
    ):

        raise RuntimeError(
            "Frozen order-statistic rule requires K < N."
        )

    # Descending order:
    #
    # Python / tensor index:
    #   0     -> b1
    #   K-1   -> bK
    #   K     -> b(K+1)
    #
    # With N=153 and target=.10:
    #
    #   K=15
    #   index 15 -> b16
    sorted_scores = torch.sort(
        bonafide_scores,
        descending=True,
    ).values

    boundary_index_zero_based = (
        allowed_false_positive_count
    )

    boundary_rank_one_based = (
        boundary_index_zero_based
        + 1
    )

    boundary_score = float(
        sorted_scores[
            boundary_index_zero_based
        ].item()
    )

    if (
        boundary_rank_one_based
        != (
            allowed_false_positive_count
            + 1
        )
    ):

        raise RuntimeError(
            "Internal boundary-rank inconsistency."
        )

    # Smallest binary64 value strictly larger than b_(K+1).
    #
    # Do NOT clamp this to 1.0.
    threshold = math.nextafter(
        boundary_score,
        math.inf,
    )

    if not math.isfinite(
        threshold
    ):

        raise RuntimeError(
            "Derived FPR10 threshold is non-finite."
        )

    if not (
        threshold
        > boundary_score
    ):

        raise RuntimeError(
            "nextafter threshold is not strictly above boundary."
        )

    predictions = apply_attack_threshold(
        scores=bonafide_scores,
        threshold=threshold,
    )

    achieved_false_positive_count = int(
        predictions.sum().item()
    )

    achieved_false_positive_rate = (
        achieved_false_positive_count
        / bonafide_count
    )

    if (
        achieved_false_positive_count
        > allowed_false_positive_count
    ):

        raise RuntimeError(
            "Derived threshold violates allowed FP count:\n"
            f"  allowed={allowed_false_positive_count}\n"
            f"  achieved={achieved_false_positive_count}"
        )

    if (
        achieved_false_positive_rate
        > target_fpr
    ):

        raise RuntimeError(
            "Derived threshold violates target empirical FPR."
        )

    boundary_tie_count = int(
        (
            bonafide_scores
            == boundary_score
        )
        .sum()
        .item()
    )

    strictly_above_boundary_count = int(
        (
            bonafide_scores
            > boundary_score
        )
        .sum()
        .item()
    )

    # For float64 scores and nextafter(boundary,+inf), these quantities
    # must be identical because no binary64 value exists between them.
    if (
        achieved_false_positive_count
        != strictly_above_boundary_count
    ):

        raise RuntimeError(
            "nextafter boundary semantics are inconsistent:\n"
            f"  predicted_attack="
            f"{achieved_false_positive_count}\n"
            f"  scores_strictly_above_boundary="
            f"{strictly_above_boundary_count}"
        )

    return FPR10ThresholdResult(
        threshold=threshold,

        boundary_score=boundary_score,

        boundary_rank_one_based=(
            boundary_rank_one_based
        ),

        target_false_positive_rate=(
            target_fpr
        ),

        bonafide_count=(
            bonafide_count
        ),

        allowed_false_positive_count=(
            allowed_false_positive_count
        ),

        achieved_false_positive_count=(
            achieved_false_positive_count
        ),

        achieved_false_positive_rate=(
            achieved_false_positive_rate
        ),

        boundary_tie_count=(
            boundary_tie_count
        ),

        strictly_above_boundary_count=(
            strictly_above_boundary_count
        ),

        threshold_above_one=(
            threshold
            > 1.0
        ),
    )