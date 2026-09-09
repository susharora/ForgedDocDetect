#!/usr/bin/env python3
"""
Audit Tech-2 development-only AUROC implementation.

This audit validates:

- attack = positive class = 1;
- dev AUROC is diagnostic only;
- AUROC does not participate in checkpoint selection;
- Stage-B disagreement guard threshold is 0.01;
- guard comparison is strictly greater than 0.01;
- perfect ranking -> AUROC 1;
- reversed ranking -> AUROC 0;
- all ties -> AUROC 0.5;
- mixed wins/ties/losses use 0.5 tie credit;
- pairwise implementation agrees with an independently implemented
  rank-sum AUROC calculation;
- complete frozen dev_val count enforcement;
- class-count enforcement;
- dtype / shape / finite-value guards;
- attack-logit-minus-bonafide-logit polarity;
- logit margin and two-class softmax attack probability produce the
  same ranking and AUROC;
- score calculation does not mutate logits;
- frozen contract drift is rejected.

Important boundary
------------------
This audit validates the encoded Stage-B AUROC disagreement CONTRACT.

The later multi-epoch controller will implement the run-level operation:

    best_dev_auroc_in_run
        -
    dev_auroc_at_loss_argmin_checkpoint

and apply the already-frozen strict > 0.01 rule.

This audit does NOT implement or validate:
- FPR10;
- threshold derivation;
- held-out test evaluation;
- checkpoint selection;
- training;
- Grad-CAM.

No FantasyID images are accessed.
No held-out test is accessed.
No print() is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


# ======================================================================
# Repository imports
# ======================================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    REPO_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            REPO_ROOT
        ),
    )


from src.config import (
    load_experiment_config,
    load_machine_config,
)

from src.development_metrics import (
    attack_ranking_score_from_logits,
    binary_auroc_pairwise,
    compute_dev_auroc,
    load_dev_auroc_contract,
)


LOGGER = logging.getLogger(
    "audit_resnet18_development_metrics"
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


# ======================================================================
# Generic helpers
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


def resolve_repo_path(
    value: str | Path,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():

        path = (
            REPO_ROOT
            / path
        )

    return path.resolve()


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        value = (
            yaml.safe_load(
                file
            )
            or {}
        )

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            f"{path} must contain a YAML mapping."
        )

    return value


def require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            f"{label} must be a mapping, "
            f"got {type(value).__name__}: {value!r}"
        )

    return value


def require_key(
    mapping: dict[str, Any],
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


def expect_exception(
    *,
    label: str,
    function: Callable[[], Any],
    exception_types: tuple[
        type[BaseException],
        ...,
    ],
) -> bool:

    try:

        function()

    except exception_types:

        LOGGER.info(
            "[PASS] %s",
            label,
        )

        return True

    raise RuntimeError(
        f"Expected exception was not raised: {label}"
    )


# ======================================================================
# Clean Git gate
# ======================================================================

def require_clean_git() -> str:

    commit_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    status_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    status = (
        status_result
        .stdout
        .strip()
    )

    if status:

        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit/remove outstanding files before running "
            "the development-metrics audit.\n\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Canonical validator
# ======================================================================

def run_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> tuple[
    Path,
    str,
]:

    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "tools"
            / "validate_experiment_config.py"
        ),
        "--config",
        str(
            experiment_path
        ),
        "--machine-config",
        str(
            machine_path
        ),
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Frozen experiment validator failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    lines = [
        line.strip()
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]

    if len(
        lines
    ) != 1:

        raise RuntimeError(
            "Expected exactly one validator handoff line:\n"
            f"{lines}"
        )

    match = VALIDATION_HANDOFF_PATTERN.match(
        lines[
            0
        ]
    )

    if match is None:

        raise RuntimeError(
            "Could not parse validator handoff:\n"
            f"{lines[0]}"
        )

    if (
        match.group(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Frozen experiment validator did not return PASS."
        )

    path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = match.group(
        "sha256"
    )

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    actual_sha = sha256_file(
        path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Validator artifact SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        path,
        expected_sha,
    )


# ======================================================================
# Logging / result output
# ======================================================================

def configure_outputs(
    *,
    tool_cfg: dict[str, Any],
    machine_id: str,
) -> tuple[
    logging.Logger,
    Path,
    Path,
    Path,
]:

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
    )

    logging_cfg = require_mapping(
        require_key(
            tool_cfg,
            "logging",
            "audit_config",
        ),
        "audit_config.logging",
    )

    output_cfg = require_mapping(
        require_key(
            tool_cfg,
            "output",
            "audit_config",
        ),
        "audit_config.output",
    )

    log_directory = resolve_repo_path(
        require_key(
            logging_cfg,
            "directory",
            "audit_config.logging",
        )
    )

    output_directory = resolve_repo_path(
        require_key(
            output_cfg,
            "directory",
            "audit_config.output",
        )
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = {
        "machine_id":
            machine_id,

        "timestamp":
            timestamp,
    }

    log_path = (
        log_directory
        / str(
            require_key(
                logging_cfg,
                "filename",
                "audit_config.logging",
            )
        ).format(
            **values
        )
    )

    result_path = (
        output_directory
        / str(
            require_key(
                output_cfg,
                "filename",
                "audit_config.output",
            )
        ).format(
            **values
        )
    )

    partial_result_path = Path(
        str(
            result_path
        )
        + ".partial"
    )

    level_name = str(
        require_key(
            logging_cfg,
            "level",
            "audit_config.logging",
        )
    ).upper()

    if not hasattr(
        logging,
        level_name,
    ):

        raise ValueError(
            f"Unknown logging level: {level_name!r}"
        )

    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(
        getattr(
            logging,
            level_name,
        )
    )

    handler = logging.FileHandler(
        log_path,
        mode="x",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    formatter.converter = (
        time.gmtime
    )

    handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        handler
    )

    return (
        LOGGER,
        log_path,
        result_path,
        partial_result_path,
    )


# ======================================================================
# Frozen full-dev synthetic data
# ======================================================================

def build_targets(
    *,
    bonafide_count: int,
    attack_count: int,
) -> torch.Tensor:

    return torch.cat(
        [
            torch.zeros(
                bonafide_count,
                dtype=torch.int64,
            ),
            torch.ones(
                attack_count,
                dtype=torch.int64,
            ),
        ],
        dim=0,
    )


def logits_from_margins(
    margins: torch.Tensor,
) -> torch.Tensor:
    """
    Construct logits where:

        z0 = 0
        z1 = margin

    so attack ranking score is exactly z1-z0.
    """

    if margins.dtype != torch.float32:

        margins = margins.to(
            dtype=torch.float32
        )

    return torch.stack(
        [
            torch.zeros_like(
                margins
            ),
            margins,
        ],
        dim=1,
    ).contiguous()


# ======================================================================
# Independent rank-sum AUROC reference
# ======================================================================

def independent_rank_sum_auroc(
    *,
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Independent AUROC implementation using average ranks.

    This intentionally does not call the production pairwise function.
    """

    values = [
        (
            float(
                scores[
                    index
                ].item()
            ),
            int(
                targets[
                    index
                ].item()
            ),
        )
        for index
        in range(
            int(
                scores.numel()
            )
        )
    ]

    values.sort(
        key=lambda pair: pair[
            0
        ]
    )

    ranks = [
        0.0
        for _
        in values
    ]

    start = 0

    while start < len(
        values
    ):

        end = (
            start
            + 1
        )

        while (
            end
            < len(
                values
            )
            and values[
                end
            ][
                0
            ]
            == values[
                start
            ][
                0
            ]
        ):

            end += 1

        # Ranks are 1-indexed.
        first_rank = (
            start
            + 1
        )

        last_rank = (
            end
        )

        average_rank = (
            (
                first_rank
                + last_rank
            )
            / 2.0
        )

        for index in range(
            start,
            end,
        ):

            ranks[
                index
            ] = average_rank

        start = end

    positive_rank_sum = sum(
        rank
        for (
            rank,
            (_, target),
        )
        in zip(
            ranks,
            values,
        )
        if target == 1
    )

    positive_count = sum(
        1
        for (
            _,
            target,
        )
        in values
        if target == 1
    )

    negative_count = (
        len(
            values
        )
        - positive_count
    )

    if (
        positive_count <= 0
        or negative_count <= 0
    ):

        raise ValueError(
            "Independent AUROC requires both classes."
        )

    numerator = (
        positive_rank_sum
        -
        (
            positive_count
            * (
                positive_count
                + 1
            )
            / 2.0
        )
    )

    return (
        numerator
        /
        (
            positive_count
            * negative_count
        )
    )


# ======================================================================
# Frozen contract audit
# ======================================================================

def audit_contract(
    *,
    experiment_cfg: dict[str, Any],
    expected_cfg: dict[str, Any],
) -> dict[str, Any]:

    contract = load_dev_auroc_contract(
        experiment_cfg=experiment_cfg,
    )

    expected_values = {
        "positive_class_name":
            str(
                require_key(
                    expected_cfg,
                    "positive_class_name",
                    "audit.expected_contract",
                )
            ),

        "positive_class_index":
            int(
                require_key(
                    expected_cfg,
                    "positive_class_index",
                    "audit.expected_contract",
                )
            ),

        "diagnostic_enabled":
            bool(
                require_key(
                    expected_cfg,
                    "diagnostic_enabled",
                    "audit.expected_contract",
                )
            ),

        "participates_in_checkpoint_selection":
            bool(
                require_key(
                    expected_cfg,
                    "participates_in_checkpoint_selection",
                    "audit.expected_contract",
                )
            ),

        "stage_b_disagreement_guard_enabled":
            bool(
                require_key(
                    expected_cfg,
                    "stage_b_disagreement_guard_enabled",
                    "audit.expected_contract",
                )
            ),

        "stage_b_disagreement_threshold":
            float(
                require_key(
                    expected_cfg,
                    "stage_b_disagreement_threshold",
                    "audit.expected_contract",
                )
            ),

        "automatically_switch_checkpoint":
            bool(
                require_key(
                    expected_cfg,
                    "automatically_switch_checkpoint",
                    "audit.expected_contract",
                )
            ),
    }

    actual_values = {
        "positive_class_name":
            contract.positive_class_name,

        "positive_class_index":
            contract.positive_class_index,

        "diagnostic_enabled":
            contract.diagnostic_enabled,

        "participates_in_checkpoint_selection":
            contract.participates_in_checkpoint_selection,

        "stage_b_disagreement_guard_enabled":
            contract.stage_b_disagreement_guard_enabled,

        "stage_b_disagreement_threshold":
            contract.stage_b_disagreement_threshold,

        "automatically_switch_checkpoint":
            contract.automatically_switch_checkpoint,
    }

    if actual_values != expected_values:

        raise RuntimeError(
            "Development-AUROC contract mismatch:\n"
            f"  expected={expected_values}\n"
            f"  actual={actual_values}"
        )

    guard_definition = str(
        experiment_cfg[
            "transfer_learning"
        ][
            "stage_b"
        ][
            "auroc_disagreement_guard"
        ][
            "definition"
        ]
    )

    expected_definition = str(
        require_key(
            expected_cfg,
            "guard_definition",
            "audit.expected_contract",
        )
    )

    if (
        guard_definition
        != expected_definition
    ):

        raise RuntimeError(
            "Stage-B AUROC disagreement definition changed:\n"
            f"  expected={expected_definition!r}\n"
            f"  actual={guard_definition!r}"
        )

    threshold = (
        contract
        .stage_b_disagreement_threshold
    )

    below = math.nextafter(
        threshold,
        -math.inf,
    )

    above = math.nextafter(
        threshold,
        math.inf,
    )

    guard_boundary = {
        "below_threshold":
            {
                "difference":
                    below,

                "flag":
                    (
                        below
                        > threshold
                    ),
            },

        "exact_threshold":
            {
                "difference":
                    threshold,

                "flag":
                    (
                        threshold
                        > threshold
                    ),
            },

        "above_threshold":
            {
                "difference":
                    above,

                "flag":
                    (
                        above
                        > threshold
                    ),
            },
    }

    if guard_boundary[
        "below_threshold"
    ][
        "flag"
    ] is not False:

        raise RuntimeError(
            "Below-threshold AUROC disagreement unexpectedly flags."
        )

    if guard_boundary[
        "exact_threshold"
    ][
        "flag"
    ] is not False:

        raise RuntimeError(
            "Exact 0.01 AUROC disagreement must NOT flag."
        )

    if guard_boundary[
        "above_threshold"
    ][
        "flag"
    ] is not True:

        raise RuntimeError(
            "AUROC disagreement strictly above 0.01 must flag."
        )

    LOGGER.info(
        "[PASS] attack positive-class contract"
    )

    LOGGER.info(
        "[PASS] dev AUROC is diagnostic-only"
    )

    LOGGER.info(
        "[PASS] AUROC cannot select checkpoint"
    )

    LOGGER.info(
        "[PASS] Stage-B disagreement definition"
    )

    LOGGER.info(
        "[PASS] exact 0.01 disagreement does not flag"
    )

    LOGGER.info(
        "[PASS] value strictly above 0.01 flags"
    )

    return {
        "contract":
            actual_values,

        "guard_definition":
            guard_definition,

        "guard_boundary":
            guard_boundary,
    }


# ======================================================================
# Full-dev AUROC cases
# ======================================================================

def audit_full_dev_case(
    *,
    experiment_cfg: dict[str, Any],
    case_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    expected_cfg: dict[str, Any],
    expected_pair_count: int,
) -> dict[str, Any]:

    logits_before = logits.clone()

    result = compute_dev_auroc(
        experiment_cfg=experiment_cfg,
        logits=logits,
        targets=targets,
    )

    if not torch.equal(
        logits,
        logits_before,
    ):

        raise RuntimeError(
            f"{case_name}: AUROC computation mutated logits."
        )

    expected_auroc = float(
        require_key(
            expected_cfg,
            "auroc",
            f"audit.expected_cases.{case_name}",
        )
    )

    expected_wins = int(
        require_key(
            expected_cfg,
            "strict_attack_wins",
            f"audit.expected_cases.{case_name}",
        )
    )

    expected_ties = int(
        require_key(
            expected_cfg,
            "tied_pairs",
            f"audit.expected_cases.{case_name}",
        )
    )

    if result.auroc != expected_auroc:

        raise RuntimeError(
            f"{case_name}: AUROC mismatch:\n"
            f"  expected={expected_auroc}\n"
            f"  actual={result.auroc}"
        )

    if (
        result.positive_negative_pair_count
        != expected_pair_count
    ):

        raise RuntimeError(
            f"{case_name}: pair-count mismatch."
        )

    if (
        result.strict_attack_wins
        != expected_wins
    ):

        raise RuntimeError(
            f"{case_name}: strict-win count mismatch."
        )

    if (
        result.tied_attack_bonafide_pairs
        != expected_ties
    ):

        raise RuntimeError(
            f"{case_name}: tied-pair count mismatch."
        )

    if (
        result.score_definition
        != "attack_logit_minus_bonafide_logit"
    ):

        raise RuntimeError(
            f"{case_name}: unexpected score definition."
        )

    scores = attack_ranking_score_from_logits(
        logits
    )

    reference_auroc = (
        independent_rank_sum_auroc(
            scores=scores,
            targets=targets,
        )
    )

    if not math.isclose(
        result.auroc,
        reference_auroc,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):

        raise RuntimeError(
            f"{case_name}: production pairwise AUROC disagrees "
            "with independent rank-sum reference:\n"
            f"  production={result.auroc}\n"
            f"  reference={reference_auroc}"
        )

    LOGGER.info(
        "[PASS] case=%s | auroc=%.12f | wins=%d | ties=%d",
        case_name,
        result.auroc,
        result.strict_attack_wins,
        result.tied_attack_bonafide_pairs,
    )

    return {
        "auroc":
            result.auroc,

        "independent_rank_sum_auroc":
            reference_auroc,

        "sample_count":
            result.sample_count,

        "bonafide_count":
            result.bonafide_count,

        "attack_count":
            result.attack_count,

        "positive_negative_pair_count":
            result.positive_negative_pair_count,

        "strict_attack_wins":
            result.strict_attack_wins,

        "tied_attack_bonafide_pairs":
            result.tied_attack_bonafide_pairs,

        "score_definition":
            result.score_definition,

        "input_logits_unchanged":
            True,
    }


# ======================================================================
# Margin / softmax ranking equivalence
# ======================================================================

def audit_margin_softmax_equivalence(
    *,
    experiment_cfg: dict[str, Any],
    targets: torch.Tensor,
) -> dict[str, Any]:

    sample_count = int(
        targets.numel()
    )

    margins = torch.linspace(
        -3.0,
        3.0,
        steps=sample_count,
        dtype=torch.float32,
    )

    logits = logits_from_margins(
        margins
    )

    margin_scores = (
        attack_ranking_score_from_logits(
            logits
        )
    )

    attack_probabilities = (
        torch.softmax(
            logits.to(
                dtype=torch.float64
            ),
            dim=1,
        )[
            :,
            1
        ]
        .contiguous()
    )

    if (
        torch.unique(
            margin_scores
        ).numel()
        != sample_count
    ):

        raise RuntimeError(
            "Softmax-equivalence probe margins are not unique."
        )

    if (
        torch.unique(
            attack_probabilities
        ).numel()
        != sample_count
    ):

        raise RuntimeError(
            "Softmax-equivalence probe probabilities are not unique."
        )

    margin_order = torch.argsort(
        margin_scores
    )

    probability_order = torch.argsort(
        attack_probabilities
    )

    if not torch.equal(
        margin_order,
        probability_order,
    ):

        raise RuntimeError(
            "Attack-logit margin and attack softmax probability "
            "do not produce identical ranking."
        )

    (
        margin_auroc,
        margin_pairs,
        margin_wins,
        margin_ties,
    ) = binary_auroc_pairwise(
        scores=margin_scores,
        targets=targets,
        positive_class_index=1,
    )

    (
        probability_auroc,
        probability_pairs,
        probability_wins,
        probability_ties,
    ) = binary_auroc_pairwise(
        scores=attack_probabilities,
        targets=targets,
        positive_class_index=1,
    )

    if (
        margin_pairs
        != probability_pairs
    ):

        raise RuntimeError(
            "Margin/probability pair counts differ."
        )

    if (
        margin_wins
        != probability_wins
    ):

        raise RuntimeError(
            "Margin/probability strict-win counts differ."
        )

    if (
        margin_ties
        != probability_ties
    ):

        raise RuntimeError(
            "Margin/probability tie counts differ."
        )

    if margin_auroc != probability_auroc:

        raise RuntimeError(
            "Margin/probability AUROC values differ."
        )

    full_result = compute_dev_auroc(
        experiment_cfg=experiment_cfg,
        logits=logits,
        targets=targets,
    )

    if full_result.auroc != margin_auroc:

        raise RuntimeError(
            "Complete dev AUROC differs from direct margin AUROC."
        )

    LOGGER.info(
        "[PASS] attack margin and attack softmax probability "
        "have identical ranking"
    )

    LOGGER.info(
        "[PASS] attack margin and attack softmax probability "
        "have identical AUROC"
    )

    return {
        "sample_count":
            sample_count,

        "margin_auroc":
            margin_auroc,

        "attack_probability_auroc":
            probability_auroc,

        "identical_ranking":
            True,

        "strict_wins":
            margin_wins,

        "ties":
            margin_ties,
    }


# ======================================================================
# Input guards
# ======================================================================

def audit_input_guards(
    *,
    experiment_cfg: dict[str, Any],
    targets: torch.Tensor,
) -> dict[str, Any]:

    logits = logits_from_margins(
        torch.linspace(
            -1.0,
            1.0,
            steps=int(
                targets.numel()
            ),
            dtype=torch.float32,
        )
    )

    results: dict[
        str,
        bool,
    ] = {}

    results[
        "partial_dev_rejected"
    ] = expect_exception(
        label="partial dev input rejected",
        function=lambda: compute_dev_auroc(
            experiment_cfg=experiment_cfg,
            logits=logits[
                :-1
            ],
            targets=targets[
                :-1
            ],
        ),
        exception_types=(
            ValueError,
        ),
    )

    wrong_counts = targets.clone()

    first_attack_index = int(
        (
            targets
            == 1
        )
        .nonzero(
            as_tuple=False
        )[
            0
        ]
        .item()
    )

    wrong_counts[
        first_attack_index
    ] = 0

    results[
        "wrong_class_counts_rejected"
    ] = expect_exception(
        label="wrong complete-dev class counts rejected",
        function=lambda: compute_dev_auroc(
            experiment_cfg=experiment_cfg,
            logits=logits,
            targets=wrong_counts,
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "float64_logits_rejected"
    ] = expect_exception(
        label="non-float32 dev logits rejected",
        function=lambda: compute_dev_auroc(
            experiment_cfg=experiment_cfg,
            logits=logits.to(
                dtype=torch.float64
            ),
            targets=targets,
        ),
        exception_types=(
            TypeError,
        ),
    )

    results[
        "int32_targets_rejected"
    ] = expect_exception(
        label="non-int64 targets rejected",
        function=lambda: compute_dev_auroc(
            experiment_cfg=experiment_cfg,
            logits=logits,
            targets=targets.to(
                dtype=torch.int32
            ),
        ),
        exception_types=(
            TypeError,
        ),
    )

    nan_logits = logits.clone()

    nan_logits[
        0,
        0
    ] = float(
        "nan"
    )

    results[
        "nan_logits_rejected"
    ] = expect_exception(
        label="NaN logits rejected",
        function=lambda: compute_dev_auroc(
            experiment_cfg=experiment_cfg,
            logits=nan_logits,
            targets=targets,
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "wrong_positive_index_rejected"
    ] = expect_exception(
        label="non-attack positive index rejected",
        function=lambda: binary_auroc_pairwise(
            scores=attack_ranking_score_from_logits(
                logits
            ),
            targets=targets,
            positive_class_index=0,
        ),
        exception_types=(
            ValueError,
        ),
    )

    return results


# ======================================================================
# Contract negative controls
# ======================================================================

def audit_contract_negative_controls(
    *,
    experiment_cfg: dict[str, Any],
) -> dict[str, bool]:

    results: dict[
        str,
        bool,
    ] = {}

    def mutated_config(
        mutation: Callable[
            [
                dict[
                    str,
                    Any,
                ]
            ],
            None,
        ],
    ) -> dict[str, Any]:

        value = copy.deepcopy(
            experiment_cfg
        )

        mutation(
            value
        )

        return value

    results[
        "positive_class_name_drift_rejected"
    ] = expect_exception(
        label="positive-class name drift rejected",
        function=lambda: load_dev_auroc_contract(
            experiment_cfg=mutated_config(
                lambda cfg:
                    cfg[
                        "class_contract"
                    ].__setitem__(
                        "positive_class",
                        "bonafide",
                    )
            )
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "positive_class_index_drift_rejected"
    ] = expect_exception(
        label="positive-class index drift rejected",
        function=lambda: load_dev_auroc_contract(
            experiment_cfg=mutated_config(
                lambda cfg:
                    cfg[
                        "class_contract"
                    ].__setitem__(
                        "positive_class_index",
                        0,
                    )
            )
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "diagnostic_disable_rejected"
    ] = expect_exception(
        label="dev AUROC disable drift rejected",
        function=lambda: load_dev_auroc_contract(
            experiment_cfg=mutated_config(
                lambda cfg:
                    cfg[
                        "evaluation"
                    ][
                        "diagnostic_metrics"
                    ][
                        "dev_auroc"
                    ].__setitem__(
                        "enabled",
                        False,
                    )
            )
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "checkpoint_selection_participation_rejected"
    ] = expect_exception(
        label="AUROC checkpoint-selection participation rejected",
        function=lambda: load_dev_auroc_contract(
            experiment_cfg=mutated_config(
                lambda cfg:
                    cfg[
                        "evaluation"
                    ][
                        "diagnostic_metrics"
                    ][
                        "dev_auroc"
                    ].__setitem__(
                        "participates_in_checkpoint_selection",
                        True,
                    )
            )
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "guard_threshold_drift_rejected"
    ] = expect_exception(
        label="AUROC guard threshold drift rejected",
        function=lambda: load_dev_auroc_contract(
            experiment_cfg=mutated_config(
                lambda cfg:
                    cfg[
                        "transfer_learning"
                    ][
                        "stage_b"
                    ][
                        "auroc_disagreement_guard"
                    ].__setitem__(
                        "flag_if_strictly_greater_than",
                        0.02,
                    )
            )
        ),
        exception_types=(
            ValueError,
        ),
    )

    results[
        "automatic_checkpoint_switch_rejected"
    ] = expect_exception(
        label="automatic checkpoint switching rejected",
        function=lambda: load_dev_auroc_contract(
            experiment_cfg=mutated_config(
                lambda cfg:
                    cfg[
                        "transfer_learning"
                    ][
                        "stage_b"
                    ][
                        "auroc_disagreement_guard"
                    ][
                        "action"
                    ].__setitem__(
                        "automatically_switch_checkpoint",
                        True,
                    )
            )
        ),
        exception_types=(
            ValueError,
        ),
    )

    return results


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit Tech-2 development-only diagnostic AUROC."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/experiments/"
            "resnet18_gradcam.yaml"
        ),
    )

    parser.add_argument(
        "--machine-config",
        default="configs/local.yaml",
    )

    parser.add_argument(
        "--audit-config",
        default=(
            "tools/"
            "audit_resnet18_development_metrics_config.yaml"
        ),
    )

    args = parser.parse_args()

    audit_config_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        audit_config_path
    )

    if (
        tool_cfg.get(
            "schema_version"
        )
        != 1
    ):

        raise ValueError(
            "Development-metrics audit schema_version must equal 1."
        )

    commit_sha = require_clean_git()

    experiment_cfg, experiment_path = (
        load_experiment_config(
            args.config
        )
    )

    machine_cfg, machine_path = (
        load_machine_config(
            args.machine_config,
            required=True,
        )
    )

    (
        validator_log_path,
        validator_log_sha,
    ) = run_validator(
        experiment_path=experiment_path,
        machine_path=machine_path,
    )

    machine_cfg_section = require_mapping(
        require_key(
            machine_cfg,
            "machine",
            "machine_config",
        ),
        "machine_config.machine",
    )

    machine_id = str(
        require_key(
            machine_cfg_section,
            "id",
            "machine_config.machine",
        )
    )

    (
        logger,
        log_path,
        result_path,
        partial_result_path,
    ) = configure_outputs(
        tool_cfg=tool_cfg,
        machine_id=machine_id,
    )

    try:

        audit_cfg = require_mapping(
            require_key(
                tool_cfg,
                "audit",
                "audit_config",
            ),
            "audit_config.audit",
        )

        expected_experiment_sha = str(
            require_key(
                audit_cfg,
                "expected_experiment_config_sha256",
                "audit_config.audit",
            )
        )

        actual_experiment_sha = sha256_file(
            experiment_path
        )

        if (
            actual_experiment_sha
            != expected_experiment_sha
        ):

            raise RuntimeError(
                "Experiment config SHA-256 mismatch:\n"
                f"  expected={expected_experiment_sha}\n"
                f"  actual={actual_experiment_sha}"
            )

        expected_contract_cfg = require_mapping(
            require_key(
                audit_cfg,
                "expected_contract",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_contract",
        )

        expected_dev_cfg = require_mapping(
            require_key(
                audit_cfg,
                "expected_dev",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_dev",
        )

        expected_cases_cfg = require_mapping(
            require_key(
                audit_cfg,
                "expected_cases",
                "audit_config.audit",
            ),
            "audit_config.audit.expected_cases",
        )

        bonafide_count = int(
            require_key(
                expected_dev_cfg,
                "bonafide",
                "audit.expected_dev",
            )
        )

        attack_count = int(
            require_key(
                expected_dev_cfg,
                "attack",
                "audit.expected_dev",
            )
        )

        total_count = int(
            require_key(
                expected_dev_cfg,
                "total",
                "audit.expected_dev",
            )
        )

        expected_pair_count = int(
            require_key(
                expected_dev_cfg,
                "positive_negative_pairs",
                "audit.expected_dev",
            )
        )

        if (
            bonafide_count,
            attack_count,
            total_count,
        ) != (
            153,
            306,
            459,
        ):

            raise ValueError(
                "Audit frozen dev counts changed."
            )

        if (
            expected_pair_count
            != (
                bonafide_count
                * attack_count
            )
        ):

            raise ValueError(
                "Expected AUROC pair count does not reconcile."
            )

        # ==============================================================
        # Provenance
        # ==============================================================

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 DEVELOPMENT AUROC AUDIT"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Git commit: %s",
            commit_sha,
        )

        logger.info(
            "Git working tree clean before evidence generation: True"
        )

        logger.info(
            "Machine ID: %s",
            machine_id,
        )

        logger.info(
            "Experiment config SHA-256: %s",
            actual_experiment_sha,
        )

        logger.info(
            "Validator log: %s",
            validator_log_path,
        )

        logger.info(
            "Validator log SHA-256: %s",
            validator_log_sha,
        )

        logger.info(
            "Audit script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "Audit config SHA-256: %s",
            sha256_file(
                audit_config_path
            ),
        )

        logger.info(
            "src/development_metrics.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "development_metrics.py"
            ),
        )

        logger.info(
            "Execution device: CPU only"
        )

        logger.info(
            "FantasyID image access: NONE"
        )

        logger.info(
            "Scientific training: NOT PERFORMED"
        )

        logger.info(
            "FPR10 thresholding: NOT IMPLEMENTED / NOT PERFORMED"
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # ==============================================================
        # Contract
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Frozen AUROC contract ---"
        )

        contract_result = audit_contract(
            experiment_cfg=experiment_cfg,
            expected_cfg=expected_contract_cfg,
        )

        # ==============================================================
        # Full-dev targets
        # ==============================================================

        targets = build_targets(
            bonafide_count=bonafide_count,
            attack_count=attack_count,
        )

        if int(
            targets.numel()
        ) != total_count:

            raise RuntimeError(
                "Synthetic full-dev target count mismatch."
            )

        # ==============================================================
        # Perfect
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Perfect ranking ---"
        )

        perfect_margins = torch.cat(
            [
                torch.full(
                    (
                        bonafide_count,
                    ),
                    -1.0,
                    dtype=torch.float32,
                ),
                torch.full(
                    (
                        attack_count,
                    ),
                    1.0,
                    dtype=torch.float32,
                ),
            ]
        )

        perfect_result = audit_full_dev_case(
            experiment_cfg=experiment_cfg,
            case_name="perfect",
            logits=logits_from_margins(
                perfect_margins
            ),
            targets=targets,
            expected_cfg=require_mapping(
                expected_cases_cfg[
                    "perfect"
                ],
                "audit.expected_cases.perfect",
            ),
            expected_pair_count=expected_pair_count,
        )

        # ==============================================================
        # Reversed
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Reversed ranking ---"
        )

        reversed_margins = (
            -perfect_margins
        )

        reversed_result = audit_full_dev_case(
            experiment_cfg=experiment_cfg,
            case_name="reversed",
            logits=logits_from_margins(
                reversed_margins
            ),
            targets=targets,
            expected_cfg=require_mapping(
                expected_cases_cfg[
                    "reversed"
                ],
                "audit.expected_cases.reversed",
            ),
            expected_pair_count=expected_pair_count,
        )

        # ==============================================================
        # All ties
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- All ties ---"
        )

        tied_margins = torch.zeros(
            total_count,
            dtype=torch.float32,
        )

        tied_result = audit_full_dev_case(
            experiment_cfg=experiment_cfg,
            case_name="all_tied",
            logits=logits_from_margins(
                tied_margins
            ),
            targets=targets,
            expected_cfg=require_mapping(
                expected_cases_cfg[
                    "all_tied"
                ],
                "audit.expected_cases.all_tied",
            ),
            expected_pair_count=expected_pair_count,
        )

        # ==============================================================
        # Mixed wins / ties / losses
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Mixed wins / ties / losses ---"
        )

        if (
            attack_count
            % 3
            != 0
        ):

            raise RuntimeError(
                "Mixed-tie audit requires attack count divisible by 3."
            )

        attack_third = (
            attack_count
            // 3
        )

        mixed_margins = torch.cat(
            [
                # All bonafides have margin 0.
                torch.zeros(
                    bonafide_count,
                    dtype=torch.float32,
                ),

                # One third of attacks beat all bonafides.
                torch.ones(
                    attack_third,
                    dtype=torch.float32,
                ),

                # One third tie every bonafide.
                torch.zeros(
                    attack_third,
                    dtype=torch.float32,
                ),

                # One third lose to every bonafide.
                -torch.ones(
                    attack_third,
                    dtype=torch.float32,
                ),
            ]
        )

        mixed_result = audit_full_dev_case(
            experiment_cfg=experiment_cfg,
            case_name="mixed_ties",
            logits=logits_from_margins(
                mixed_margins
            ),
            targets=targets,
            expected_cfg=require_mapping(
                expected_cases_cfg[
                    "mixed_ties"
                ],
                "audit.expected_cases.mixed_ties",
            ),
            expected_pair_count=expected_pair_count,
        )

        # ==============================================================
        # Margin / softmax equivalence
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Margin / softmax ranking equivalence ---"
        )

        softmax_result = (
            audit_margin_softmax_equivalence(
                experiment_cfg=experiment_cfg,
                targets=targets,
            )
        )

        # ==============================================================
        # Input guards
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Input guards ---"
        )

        input_guard_result = (
            audit_input_guards(
                experiment_cfg=experiment_cfg,
                targets=targets,
            )
        )

        # ==============================================================
        # Contract negative controls
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "--- Contract drift negative controls ---"
        )

        negative_control_result = (
            audit_contract_negative_controls(
                experiment_cfg=experiment_cfg,
            )
        )

        # ==============================================================
        # Result artifact
        # ==============================================================

        result = {
            "schema_version":
                1,

            "status":
                "PASS",

            "machine":
                {
                    "id":
                        machine_id,
                },

            "provenance":
                {
                    "git_commit":
                        commit_sha,

                    "experiment_config_sha256":
                        actual_experiment_sha,

                    "development_metrics_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "development_metrics.py"
                        ),

                    "audit_script_sha256":
                        sha256_file(
                            Path(
                                __file__
                            ).resolve()
                        ),

                    "audit_config_sha256":
                        sha256_file(
                            audit_config_path
                        ),

                    "validator_log":
                        str(
                            validator_log_path
                        ),

                    "validator_log_sha256":
                        validator_log_sha,
                },

            "scope":
                {
                    "execution_device":
                        "cpu",

                    "synthetic_only":
                        True,

                    "fantasyid_images_accessed":
                        False,

                    "scientific_training_performed":
                        False,

                    "fpr10_thresholding_performed":
                        False,

                    "held_out_test_accessed":
                        False,
                },

            "contract":
                contract_result,

            "full_dev_cases":
                {
                    "perfect":
                        perfect_result,

                    "reversed":
                        reversed_result,

                    "all_tied":
                        tied_result,

                    "mixed_ties":
                        mixed_result,
                },

            "margin_softmax_equivalence":
                softmax_result,

            "input_guards":
                input_guard_result,

            "contract_negative_controls":
                negative_control_result,

            "interpretation":
                (
                    "PASS establishes deterministic binary dev AUROC "
                    "with attack=1 polarity, explicit pairwise tie "
                    "handling, complete frozen-dev enforcement, "
                    "agreement with an independent rank-sum reference, "
                    "ranking equivalence between attack-logit margin "
                    "and two-class attack softmax probability, and the "
                    "frozen diagnostic-only Stage-B >0.01 disagreement "
                    "contract. FPR10 threshold semantics remain "
                    "deliberately deferred."
                ),
        }

        with partial_result_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                result,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        partial_result_path.replace(
            result_path
        )

        result_sha = sha256_file(
            result_path
        )

        # ==============================================================
        # Summary
        # ==============================================================

        logger.info(
            ""
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "AUDIT SUMMARY"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "[PASS] attack = positive class = 1"
        )

        logger.info(
            "[PASS] perfect ranking AUROC = 1"
        )

        logger.info(
            "[PASS] reversed ranking AUROC = 0"
        )

        logger.info(
            "[PASS] all ties AUROC = 0.5"
        )

        logger.info(
            "[PASS] mixed ties receive 0.5 pair credit"
        )

        logger.info(
            "[PASS] production pairwise AUROC agrees with "
            "independent rank-sum reference"
        )

        logger.info(
            "[PASS] full frozen dev count/class contract enforced"
        )

        logger.info(
            "[PASS] logit margin and attack softmax ranking equivalent"
        )

        logger.info(
            "[PASS] AUROC remains diagnostic-only"
        )

        logger.info(
            "[PASS] exact guard difference 0.01 does not flag"
        )

        logger.info(
            "[PASS] guard difference strictly >0.01 flags"
        )

        logger.info(
            "[PASS] FPR10 remains deferred"
        )

        logger.info(
            "[PASS] held-out test NOT ACCESSED"
        )

        logger.info(
            "Audit result: %s",
            result_path,
        )

        logger.info(
            "Audit result SHA-256: %s",
            result_sha,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 DEVELOPMENT AUROC AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 DEVELOPMENT AUROC AUDIT: FAIL"
        )

        if partial_result_path.exists():

            partial_result_path.unlink()

        return 1

    finally:

        for handler in list(
            logger.handlers
        ):

            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )