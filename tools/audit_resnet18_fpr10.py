#!/usr/bin/env python3
"""
Synthetic audit of the frozen Tech-2 FPR10 implementation.

Validated here
--------------
- canonical experiment config still passes its validator;
- frozen resolution-selection artifact provenance;
- r512 / LR 1e-4 is the frozen selected configuration;
- bonafide = 0, attack = 1;
- p_attack is computed from stored FP32 logits in float64;
- score conversion does not mutate logits;
- fixed 0.5 equality predicts attack;
- derive_fpr10_threshold() structurally accepts no attack scores;
- N_bonafide = 153;
- K = floor(.10 * 153) = 15;
- first disallowed rank = 16;
- threshold = nextafter(b16, +infinity);
- no-tie cohort produces exactly 15 false positives;
- boundary ties are completely excluded;
- all-tied bona-fide scores produce zero false positives;
- p_attack=1 boundary may legitimately produce threshold >1;
- NaN / Inf inputs fail.

Not performed
-------------
- FantasyID image decoding;
- checkpoint loading;
- model inference;
- training;
- real dev threshold derivation;
- held-out test access.

No print() is used.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

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

from src.fpr10 import (
    FPR10Contract,
    apply_attack_threshold,
    attack_probability_from_logits,
    derive_fpr10_threshold,
    load_fpr10_contract,
)


LOGGER = logging.getLogger(
    "audit_resnet18_fpr10"
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
) -> Mapping[str, Any]:

    if not isinstance(
        value,
        Mapping,
    ):

        raise TypeError(
            f"{label} must be a mapping."
        )

    return value


def require_clean_git() -> str:

    commit = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status = subprocess.run(
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
    ).stdout.strip()

    if status:

        raise RuntimeError(
            "Git working tree must be clean before FPR10 audit.\n\n"
            f"{status}"
        )

    if len(
        commit
    ) != 40:

        raise RuntimeError(
            "Git HEAD is not a full 40-character SHA."
        )

    return commit


def expect_exception(
    *,
    label: str,
    function: Callable[[], Any],
    exception_types: tuple[
        type[BaseException],
        ...,
    ],
) -> None:

    try:

        function()

    except exception_types:

        LOGGER.info(
            "[PASS] %s",
            label,
        )

        return

    raise RuntimeError(
        f"Expected exception was not raised: {label}"
    )


# ======================================================================
# Canonical experiment validator
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
            "Experiment validator did not return PASS."
        )

    artifact_path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = match.group(
        "sha256"
    )

    if not artifact_path.is_file():

        raise FileNotFoundError(
            artifact_path
        )

    actual_sha = sha256_file(
        artifact_path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Validator artifact SHA-256 mismatch."
        )

    return (
        artifact_path,
        expected_sha,
    )


# ======================================================================
# Outputs
# ======================================================================

def configure_outputs(
    *,
    audit_cfg: Mapping[str, Any],
    machine_id: str,
) -> tuple[
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
        audit_cfg[
            "logging"
        ],
        "logging",
    )

    output_cfg = require_mapping(
        audit_cfg[
            "output"
        ],
        "output",
    )

    log_directory = resolve_repo_path(
        logging_cfg[
            "directory"
        ]
    )

    output_directory = resolve_repo_path(
        output_cfg[
            "directory"
        ]
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
            logging_cfg[
                "filename"
            ]
        ).format(
            **values
        )
    )

    output_path = (
        output_directory
        / str(
            output_cfg[
                "filename"
            ]
        ).format(
            **values
        )
    )

    LOGGER.handlers.clear()
    LOGGER.propagate = False

    level_name = str(
        logging_cfg[
            "level"
        ]
    ).upper()

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
        log_path,
        output_path,
    )


# ======================================================================
# Synthetic helpers
# ======================================================================

def logits_from_attack_margins(
    margins: torch.Tensor,
) -> torch.Tensor:

    if margins.ndim != 1:

        raise RuntimeError(
            "Synthetic margins must have shape [N]."
        )

    if margins.dtype != torch.float32:

        raise RuntimeError(
            "Synthetic margins must be float32."
        )

    return torch.stack(
        (
            torch.zeros_like(
                margins
            ),
            margins,
        ),
        dim=1,
    ).contiguous()


def independent_expected_threshold(
    *,
    bonafide_scores: torch.Tensor,
    target_fpr: float,
) -> tuple[
    int,
    int,
    float,
    float,
]:

    count = int(
        bonafide_scores.numel()
    )

    allowed = math.floor(
        target_fpr
        * count
    )

    sorted_scores = sorted(
        [
            float(
                value
            )

            for value
            in bonafide_scores.tolist()
        ],
        reverse=True,
    )

    boundary_rank = (
        allowed
        + 1
    )

    boundary = sorted_scores[
        allowed
    ]

    threshold = math.nextafter(
        boundary,
        math.inf,
    )

    return (
        allowed,
        boundary_rank,
        boundary,
        threshold,
    )


def audit_threshold_case(
    *,
    label: str,
    bonafide_logits: torch.Tensor,
    contract: FPR10Contract,
    expected_false_positives: int,
) -> dict[str, Any]:

    scores = attack_probability_from_logits(
        bonafide_logits
    )

    (
        expected_allowed,
        expected_rank,
        expected_boundary,
        expected_threshold,
    ) = independent_expected_threshold(
        bonafide_scores=scores,
        target_fpr=(
            contract.target_false_positive_rate
        ),
    )

    result = derive_fpr10_threshold(
        bonafide_scores=scores,
        contract=contract,
    )

    if (
        result.allowed_false_positive_count
        != expected_allowed
    ):

        raise RuntimeError(
            f"{label}: allowed FP count mismatch."
        )

    if (
        result.boundary_rank_one_based
        != expected_rank
    ):

        raise RuntimeError(
            f"{label}: boundary rank mismatch."
        )

    if (
        result.boundary_score
        != expected_boundary
    ):

        raise RuntimeError(
            f"{label}: boundary score mismatch."
        )

    if (
        result.threshold
        != expected_threshold
    ):

        raise RuntimeError(
            f"{label}: Python nextafter threshold mismatch."
        )

    # Independent binary64 nextafter reference from torch.
    boundary_tensor = torch.tensor(
        expected_boundary,
        dtype=torch.float64,
    )

    infinity_tensor = torch.tensor(
        math.inf,
        dtype=torch.float64,
    )

    torch_threshold = float(
        torch.nextafter(
            boundary_tensor,
            infinity_tensor,
        ).item()
    )

    if result.threshold != torch_threshold:

        raise RuntimeError(
            f"{label}: torch/Python nextafter disagreement."
        )

    if (
        result.achieved_false_positive_count
        != expected_false_positives
    ):

        raise RuntimeError(
            f"{label}: unexpected achieved FP count:\n"
            f"  expected={expected_false_positives}\n"
            f"  actual="
            f"{result.achieved_false_positive_count}"
        )

    if (
        result.achieved_false_positive_count
        > result.allowed_false_positive_count
    ):

        raise RuntimeError(
            f"{label}: empirical FPR cap violated."
        )

    LOGGER.info(
        "[PASS] %s | K=%d | boundary_rank=%d | "
        "boundary=%.17g | threshold=%.17g | "
        "FP=%d/%d | FPR=%.12f | boundary_ties=%d",
        label,
        result.allowed_false_positive_count,
        result.boundary_rank_one_based,
        result.boundary_score,
        result.threshold,
        result.achieved_false_positive_count,
        result.bonafide_count,
        result.achieved_false_positive_rate,
        result.boundary_tie_count,
    )

    return {
        "boundary_score":
            result.boundary_score,

        "threshold":
            result.threshold,

        "boundary_rank_one_based":
            result.boundary_rank_one_based,

        "allowed_false_positive_count":
            result.allowed_false_positive_count,

        "achieved_false_positive_count":
            result.achieved_false_positive_count,

        "achieved_false_positive_rate":
            result.achieved_false_positive_rate,

        "boundary_tie_count":
            result.boundary_tie_count,

        "threshold_above_one":
            result.threshold_above_one,
    }


# ======================================================================
# Main audit
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen ResNet-18 FPR10 threshold contract."
        )
    )

    parser.add_argument(
        "--audit-config",
        default=(
            "tools/"
            "audit_resnet18_fpr10_config.yaml"
        ),
    )

    args = parser.parse_args()

    git_commit = (
        require_clean_git()
    )

    audit_cfg_path = resolve_repo_path(
        args.audit_config
    )

    audit_cfg = load_yaml(
        audit_cfg_path
    )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            audit_cfg[
                "experiment"
            ][
                "config"
            ]
        )
    )

    machine_cfg, machine_path = (
        load_machine_config(
            audit_cfg[
                "experiment"
            ][
                "machine_config"
            ],
            required=True,
        )
    )

    machine_cfg_section = require_mapping(
        machine_cfg[
            "machine"
        ],
        "machine",
    )

    machine_id = str(
        machine_cfg_section[
            "id"
        ]
    )

    (
        validator_path,
        validator_sha,
    ) = run_validator(
        experiment_path=(
            experiment_path
        ),
        machine_path=(
            machine_path
        ),
    )

    (
        log_path,
        output_path,
    ) = configure_outputs(
        audit_cfg=(
            audit_cfg
        ),
        machine_id=(
            machine_id
        ),
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 FPR10 PRIMITIVE AUDIT"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            git_commit,
        )

        LOGGER.info(
            "Machine: %s",
            machine_id,
        )

        LOGGER.info(
            "Validator artifact: %s",
            validator_path,
        )

        LOGGER.info(
            "Validator artifact SHA-256: %s",
            validator_sha,
        )

        # ==============================================================
        # Evaluation protocol
        # ==============================================================

        protocol_path = resolve_repo_path(
            audit_cfg[
                "evaluation"
            ][
                "protocol"
            ]
        )

        protocol_cfg = load_yaml(
            protocol_path
        )

        contract = load_fpr10_contract(
            evaluation_cfg=(
                protocol_cfg
            )
        )

        LOGGER.info(
            "[PASS] final-detection protocol parsed"
        )

        # ==============================================================
        # Frozen upstream SHA identities
        # ==============================================================

        upstream_cfg = require_mapping(
            protocol_cfg[
                "upstream"
            ],
            "upstream",
        )

        for upstream_name in (
            "training_config",
            "resolution_selection",
        ):

            upstream_item = require_mapping(
                upstream_cfg[
                    upstream_name
                ],
                f"upstream.{upstream_name}",
            )

            upstream_path = resolve_repo_path(
                upstream_item[
                    "path"
                ]
            )

            expected_sha = str(
                upstream_item[
                    "sha256"
                ]
            )

            actual_sha = sha256_file(
                upstream_path
            )

            if actual_sha != expected_sha:

                raise RuntimeError(
                    "Upstream SHA-256 mismatch:\n"
                    f"  source={upstream_name}\n"
                    f"  expected={expected_sha}\n"
                    f"  actual={actual_sha}"
                )

            LOGGER.info(
                "[PASS] upstream %s SHA-256 = %s",
                upstream_name,
                actual_sha,
            )

        # ==============================================================
        # Cross-check native class contract against training config
        # ==============================================================

        training_class_cfg = require_mapping(
            experiment_cfg[
                "class_contract"
            ],
            "experiment.class_contract",
        )

        training_index_by_name = require_mapping(
            training_class_cfg[
                "index_by_name"
            ],
            "experiment.class_contract.index_by_name",
        )

        if dict(
            training_index_by_name
        ) != {
            "bonafide":
                0,

            "attack":
                1,
        }:

            raise RuntimeError(
                "Experiment class mapping is not bonafide=0/attack=1."
            )

        if (
            training_class_cfg[
                "positive_class"
            ]
            != "attack"
        ):

            raise RuntimeError(
                "Experiment positive class is not attack."
            )

        LOGGER.info(
            "[PASS] ground truth: bonafide=0, attack=1, positive=attack"
        )

        # ==============================================================
        # Cross-check selected configuration
        # ==============================================================

        selection_path = resolve_repo_path(
            upstream_cfg[
                "resolution_selection"
            ][
                "path"
            ]
        )

        selection_cfg = load_yaml(
            selection_path
        )

        if selection_cfg[
            "status"
        ] != "FROZEN":

            raise RuntimeError(
                "Resolution-selection artifact is not FROZEN."
            )

        resolution_cfg = require_mapping(
            selection_cfg[
                "resolution_selection"
            ],
            "resolution_selection",
        )

        if (
            resolution_cfg[
                "selected_resolution"
            ]
            != "r512"
        ):

            raise RuntimeError(
                "Frozen selected resolution is not r512."
            )

        if float(
            resolution_cfg[
                "selected_backbone_lr"
            ]
        ) != 0.0001:

            raise RuntimeError(
                "Frozen selected backbone LR is not 1e-4."
            )

        LOGGER.info(
            "[PASS] selected configuration = r512 / backbone LR 1e-4"
        )

        # ==============================================================
        # Structural attack-score independence
        # ==============================================================

        signature = inspect.signature(
            derive_fpr10_threshold
        )

        parameter_names = tuple(
            signature.parameters
        )

        if parameter_names != (
            "bonafide_scores",
            "contract",
        ):

            raise RuntimeError(
                "derive_fpr10_threshold has unexpected API:\n"
                f"  parameters={parameter_names}"
            )

        LOGGER.info(
            "[PASS] threshold API accepts bona-fide scores only"
        )

        LOGGER.info(
            "[PASS] attack scores cannot enter threshold derivation API"
        )

        # ==============================================================
        # Score conversion + non-mutation
        # ==============================================================

        probe_logits = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    -1.0,
                ],
                [
                    -1.0,
                    1.0,
                ],
            ],
            dtype=torch.float32,
        )

        probe_before = (
            probe_logits
            .clone()
        )

        probe_scores = attack_probability_from_logits(
            probe_logits
        )

        if not torch.equal(
            probe_logits,
            probe_before,
        ):

            raise RuntimeError(
                "Score conversion mutated source logits."
            )

        independent_scores = torch.softmax(
            probe_logits.to(
                dtype=torch.float64
            ),
            dim=1,
        )[
            :,
            1,
        ]

        if not torch.equal(
            probe_scores,
            independent_scores,
        ):

            raise RuntimeError(
                "p_attack differs from explicit float64 softmax."
            )

        if probe_scores.dtype != torch.float64:

            raise RuntimeError(
                "p_attack is not float64."
            )

        LOGGER.info(
            "[PASS] FP32 logits -> float64 softmax p_attack"
        )

        LOGGER.info(
            "[PASS] score conversion does not mutate stored logits"
        )

        # ==============================================================
        # Fixed 0.5 equality contract
        # ==============================================================

        exactly_half_logits = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                ],
            ],
            dtype=torch.float32,
        )

        exactly_half_score = (
            attack_probability_from_logits(
                exactly_half_logits
            )
        )

        if float(
            exactly_half_score[
                0
            ].item()
        ) != 0.5:

            raise RuntimeError(
                "Equal logits did not produce exact p_attack=0.5."
            )

        fixed_prediction = apply_attack_threshold(
            scores=(
                exactly_half_score
            ),
            threshold=(
                contract.fixed_threshold
            ),
        )

        if int(
            fixed_prediction[
                0
            ].item()
        ) != 1:

            raise RuntimeError(
                "p_attack=0.5 must classify as attack."
            )

        LOGGER.info(
            "[PASS] fixed threshold 0.5 uses >= and ties classify attack"
        )

        # ==============================================================
        # Exact N=153 / K=15 arithmetic
        # ==============================================================

        if (
            contract.expected_dev_bonafide_count
            != 153
        ):

            raise RuntimeError(
                "Frozen dev bona-fide count is not 153."
            )

        calculated_k = math.floor(
            contract.target_false_positive_rate
            * contract.expected_dev_bonafide_count
        )

        if calculated_k != 15:

            raise RuntimeError(
                "Expected K=floor(.10*153)=15."
            )

        LOGGER.info(
            "[PASS] K = floor(0.10 * 153) = 15"
        )

        LOGGER.info(
            "[PASS] first disallowed rank = K+1 = 16"
        )

        # ==============================================================
        # No-tie order-statistic case
        # ==============================================================

        no_tie_margins = torch.linspace(
            -6.0,
            -1.0,
            steps=153,
            dtype=torch.float32,
        )

        no_tie_logits = logits_from_attack_margins(
            no_tie_margins
        )

        no_tie_scores = attack_probability_from_logits(
            no_tie_logits
        )

        if int(
            torch.unique(
                no_tie_scores
            ).numel()
        ) != 153:

            raise RuntimeError(
                "No-tie synthetic cohort unexpectedly contains ties."
            )

        no_tie_result = audit_threshold_case(
            label="no-tie b16 nextafter",
            bonafide_logits=(
                no_tie_logits
            ),
            contract=(
                contract
            ),
            expected_false_positives=15,
        )

        # ==============================================================
        # Boundary tie
        # ==============================================================

        # 13 unique scores above boundary.
        # Ranks 14..18 share exactly the same boundary score.
        #
        # b16 is therefore inside this tie group.
        #
        # nextafter(b16,+inf) excludes all five tied samples,
        # leaving only the 13 strictly larger scores as false positives.
        tie_top = torch.linspace(
            2.0,
            0.8,
            steps=13,
            dtype=torch.float32,
        )

        tie_group = torch.full(
            (
                5,
            ),
            0.5,
            dtype=torch.float32,
        )

        tie_lower = torch.linspace(
            0.4,
            -6.0,
            steps=135,
            dtype=torch.float32,
        )

        tie_margins = torch.cat(
            (
                tie_top,
                tie_group,
                tie_lower,
            ),
            dim=0,
        )

        if int(
            tie_margins.numel()
        ) != 153:

            raise RuntimeError(
                "Boundary-tie cohort does not contain 153 negatives."
            )

        tie_result = audit_threshold_case(
            label="boundary-tie exclusion",
            bonafide_logits=(
                logits_from_attack_margins(
                    tie_margins
                )
            ),
            contract=(
                contract
            ),
            expected_false_positives=13,
        )

        if (
            tie_result[
                "boundary_tie_count"
            ]
            != 5
        ):

            raise RuntimeError(
                "Boundary tie-group size should equal 5."
            )

        LOGGER.info(
            "[PASS] complete boundary tie group excluded: "
            "13 FP rather than 18"
        )

        # ==============================================================
        # All scores tied
        # ==============================================================

        all_tied_margins = torch.full(
            (
                153,
            ),
            0.25,
            dtype=torch.float32,
        )

        all_tied_result = audit_threshold_case(
            label="all-bonafide-tied",
            bonafide_logits=(
                logits_from_attack_margins(
                    all_tied_margins
                )
            ),
            contract=(
                contract
            ),
            expected_false_positives=0,
        )

        if (
            all_tied_result[
                "boundary_tie_count"
            ]
            != 153
        ):

            raise RuntimeError(
                "All-tied cohort should have boundary tie count 153."
            )

        LOGGER.info(
            "[PASS] all-tied bona-fide cohort yields empirical FPR=0"
        )

        # ==============================================================
        # Boundary exactly p_attack = 1.0
        # ==============================================================

        # Float64 softmax of [0,1000] underflows the first class to zero
        # and yields an exact attack probability of 1.0.
        #
        # The frozen contract intentionally permits:
        #
        #     nextafter(1.0,+inf) > 1.0
        #
        # rather than clamping back to 1.0 and re-admitting the tie.
        probability_one_logits = torch.tensor(
            [
                [
                    0.0,
                    1000.0,
                ],
            ],
            dtype=torch.float32,
        ).repeat(
            153,
            1,
        )

        probability_one_scores = (
            attack_probability_from_logits(
                probability_one_logits
            )
        )

        if not bool(
            (
                probability_one_scores
                == 1.0
            ).all()
        ):

            raise RuntimeError(
                "Pathological score probe did not produce exact 1.0."
            )

        probability_one_result = audit_threshold_case(
            label="probability-one-no-clamp",
            bonafide_logits=(
                probability_one_logits
            ),
            contract=(
                contract
            ),
            expected_false_positives=0,
        )

        if (
            probability_one_result[
                "threshold_above_one"
            ]
            is not True
        ):

            raise RuntimeError(
                "nextafter(1,+inf) was incorrectly clamped."
            )

        LOGGER.info(
            "[PASS] threshold may exceed 1.0; no probability clamping"
        )

        # ==============================================================
        # Fatal numerical-input guards
        # ==============================================================

        nan_logits = torch.tensor(
            [
                [
                    0.0,
                    float(
                        "nan"
                    ),
                ],
            ],
            dtype=torch.float32,
        )

        expect_exception(
            label="NaN logits are fatal",
            function=lambda:
                attack_probability_from_logits(
                    nan_logits
                ),
            exception_types=(
                RuntimeError,
            ),
        )

        inf_logits = torch.tensor(
            [
                [
                    0.0,
                    float(
                        "inf"
                    ),
                ],
            ],
            dtype=torch.float32,
        )

        expect_exception(
            label="infinite logits are fatal",
            function=lambda:
                attack_probability_from_logits(
                    inf_logits
                ),
            exception_types=(
                RuntimeError,
            ),
        )

        wrong_dtype_logits = torch.zeros(
            (
                2,
                2,
            ),
            dtype=torch.float64,
        )

        expect_exception(
            label="non-FP32 stored logits are rejected",
            function=lambda:
                attack_probability_from_logits(
                    wrong_dtype_logits
                ),
            exception_types=(
                RuntimeError,
            ),
        )

        wrong_bonafide_count = torch.full(
            (
                152,
            ),
            0.2,
            dtype=torch.float64,
        )

        expect_exception(
            label="incomplete dev bona-fide population is rejected",
            function=lambda:
                derive_fpr10_threshold(
                    bonafide_scores=(
                        wrong_bonafide_count
                    ),
                    contract=(
                        contract
                    ),
                ),
            exception_types=(
                RuntimeError,
            ),
        )

        # ==============================================================
        # Audit artifact
        # ==============================================================

        artifact = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_fpr10_primitive_audit",

            "status":
                "PASS",

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_commit":
                git_commit,

            "machine_id":
                machine_id,

            "validator":
                {
                    "path":
                        str(
                            validator_path
                        ),

                    "sha256":
                        validator_sha,
                },

            "source_sha256":
                {
                    "src/fpr10.py":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "fpr10.py"
                        ),

                    "evaluation_protocol":
                        sha256_file(
                            protocol_path
                        ),

                    "audit_script":
                        sha256_file(
                            Path(
                                __file__
                            ).resolve()
                        ),

                    "audit_config":
                        sha256_file(
                            audit_cfg_path
                        ),
                },

            "contract":
                {
                    "ground_truth":
                        {
                            "bonafide":
                                contract.bonafide_index,

                            "attack":
                                contract.attack_index,

                            "positive_class":
                                contract.positive_class,
                        },

                    "threshold_score":
                        {
                            "representation":
                                (
                                    contract
                                    .threshold_score_representation
                                ),

                            "stored_logit_dtype":
                                contract.stored_logit_dtype,

                            "score_dtype":
                                contract.threshold_score_dtype,
                        },

                    "fixed_threshold":
                        contract.fixed_threshold,

                    "fpr10":
                        {
                            "population":
                                (
                                    contract
                                    .threshold_selection_population
                                ),

                            "expected_dev_bonafide_count":
                                (
                                    contract
                                    .expected_dev_bonafide_count
                                ),

                            "target_fpr":
                                (
                                    contract
                                    .target_false_positive_rate
                                ),

                            "allowed_false_positive_count":
                                (
                                    contract
                                    .expected_allowed_false_positive_count
                                ),

                            "boundary_rank":
                                16,

                            "threshold_rule":
                                contract.threshold_rule,

                            "attack_scores_participate":
                                (
                                    contract
                                    .attack_scores_participate
                                ),

                            "interpolation":
                                contract.interpolation,

                            "randomized_tie_handling":
                                (
                                    contract
                                    .randomized_tie_handling
                                ),

                            "threshold_clamping":
                                contract.threshold_clamping,
                        },
                },

            "synthetic_cases":
                {
                    "no_tie":
                        no_tie_result,

                    "boundary_tie":
                        tie_result,

                    "all_tied":
                        all_tied_result,

                    "probability_one":
                        probability_one_result,
                },

            "boundaries":
                {
                    "fantasyid_images_decoded":
                        False,

                    "checkpoint_loaded":
                        False,

                    "model_inference_performed":
                        False,

                    "training_performed":
                        False,

                    "real_dev_threshold_derived":
                        False,

                    "held_out_test_accessed":
                        False,
                },
        }

        with output_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                artifact,
                file,
                sort_keys=False,
            )

        output_sha = sha256_file(
            output_path
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] FantasyID image decoding: NONE"
        )

        LOGGER.info(
            "[PASS] checkpoint/model inference: NONE"
        )

        LOGGER.info(
            "[PASS] real dev threshold derivation: NONE"
        )

        LOGGER.info(
            "[PASS] held-out test: NOT ACCESSED"
        )

        LOGGER.info(
            "Audit artifact: %s",
            output_path,
        )

        LOGGER.info(
            "Audit artifact SHA-256: %s",
            output_sha,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 FPR10 PRIMITIVE AUDIT: PASS"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "RESNET-18 FPR10 PRIMITIVE AUDIT: FAIL"
        )

        return 1

    finally:

        for handler in list(
            LOGGER.handlers
        ):

            handler.flush()
            handler.close()

        LOGGER.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )