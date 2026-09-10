#!/usr/bin/env python3
"""
Finalize the frozen Tech-2 ResNet-18 resolution decision.

This tool reads ONLY committed development-run evidence:

    r256 seeds 8, 9, 10
    r512 seeds 8, 9, 10

and applies the already-frozen practical non-inferiority rule:

    d_s = L512(s) - L256(s)

    mean_d = mean(d8, d9, d10)

    margin = 0.05 * mean(L256)

    choose r512 if mean_d <= margin
    otherwise choose r256

It also determines the representative Grad-CAM seed using the frozen
median-dev-loss rule.

No image is decoded.
No model/checkpoint binary is loaded.
No threshold is derived.
No held-out test is accessed.
No counterfactual result participates in selection.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


# ======================================================================
# Repository
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
)


LOGGER = logging.getLogger(
    "finalize_resnet18_resolution_selection"
)


# ======================================================================
# Helpers
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
            f"YAML root must be a mapping: {path}"
        )

    return value


def write_yaml_exclusive(
    *,
    path: Path,
    value: Mapping[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "x",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            dict(
                value
            ),
            file,
            sort_keys=False,
        )


# ======================================================================
# Git gate
# ======================================================================

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
            "Git working tree must be clean before "
            "resolution finalization:\n"
            f"{status}"
        )

    if len(
        commit
    ) != 40:

        raise RuntimeError(
            "Git HEAD must be a full 40-character SHA."
        )

    return commit


# ======================================================================
# Logging / outputs
# ======================================================================

def configure_outputs(
    *,
    tool_cfg: Mapping[str, Any],
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

    log_cfg = require_mapping(
        tool_cfg[
            "logging"
        ],
        "logging",
    )

    output_cfg = require_mapping(
        tool_cfg[
            "output"
        ],
        "output",
    )

    log_dir = resolve_repo_path(
        str(
            log_cfg[
                "directory"
            ]
        )
    )

    output_dir = resolve_repo_path(
        str(
            output_cfg[
                "directory"
            ]
        )
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / str(
            log_cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    output_path = (
        output_dir
        / str(
            output_cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    LOGGER.handlers.clear()
    LOGGER.propagate = False

    LOGGER.setLevel(
        getattr(
            logging,
            str(
                log_cfg[
                    "level"
                ]
            ).upper(),
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
# Frozen selection contract
# ======================================================================

def load_selection_contract(
    *,
    experiment_cfg: Mapping[str, Any],
) -> dict[str, Any]:

    development = require_mapping(
        experiment_cfg[
            "development_selection"
        ],
        "development_selection",
    )

    stability = require_mapping(
        development[
            "multi_seed_confirmation"
        ],
        "development_selection.multi_seed_confirmation",
    )

    resolution_cfg = require_mapping(
        development[
            "resolution_selection"
        ],
        "development_selection.resolution_selection",
    )

    post_selection = require_mapping(
        experiment_cfg[
            "post_selection"
        ],
        "post_selection",
    )

    evaluation = require_mapping(
        experiment_cfg[
            "evaluation"
        ],
        "evaluation",
    )

    resolutions = tuple(
        str(
            value
        )
        for value
        in stability[
            "resolutions"
        ]
    )

    seeds = tuple(
        int(
            value
        )
        for value
        in stability[
            "seeds"
        ]
    )

    if resolutions != (
        "r256",
        "r512",
    ):

        raise RuntimeError(
            "Frozen resolution set changed."
        )

    if seeds != (
        8,
        9,
        10,
    ):

        raise RuntimeError(
            "Frozen stability seed set changed."
        )

    metric = require_mapping(
        resolution_cfg[
            "comparison_metric"
        ],
        "resolution_selection.comparison_metric",
    )

    if (
        metric[
            "name"
        ]
        != "class_weighted_dev_cross_entropy"
    ):

        raise RuntimeError(
            "Frozen resolution comparison metric changed."
        )

    if (
        metric[
            "value_per_seed"
        ]
        != "raw_minimum_for_that_resolution_selected_lr"
    ):

        raise RuntimeError(
            "Frozen per-seed loss source changed."
        )

    paired_difference = require_mapping(
        resolution_cfg[
            "paired_seed_difference"
        ],
        "resolution_selection.paired_seed_difference",
    )

    if (
        paired_difference[
            "definition"
        ]
        != "d_s = L_512(s) - L_256(s)"
    ):

        raise RuntimeError(
            "Frozen paired-difference definition changed."
        )

    margin_cfg = require_mapping(
        resolution_cfg[
            "noninferiority_margin"
        ],
        "resolution_selection.noninferiority_margin",
    )

    margin_fraction = float(
        margin_cfg[
            "relative_fraction"
        ]
    )

    if margin_fraction != 0.05:

        raise RuntimeError(
            "Frozen non-inferiority margin changed."
        )

    decision_rule = require_mapping(
        resolution_cfg[
            "decision_rule"
        ],
        "resolution_selection.decision_rule",
    )

    if (
        decision_rule[
            "select_r512_if"
        ]
        != "mean_d <= margin"
    ):

        raise RuntimeError(
            "Frozen r512 decision rule changed."
        )

    if (
        decision_rule[
            "select_r256_if"
        ]
        != "mean_d > margin"
    ):

        raise RuntimeError(
            "Frozen r256 decision rule changed."
        )

    if (
        resolution_cfg[
            "masked_dev_diagnostics_influence_selection"
        ]
        is not False
    ):

        raise RuntimeError(
            "Counterfactual/masked diagnostics must not "
            "influence resolution selection."
        )

    selected_checkpoint_cfg = require_mapping(
        post_selection[
            "selected_resolution_seed_checkpoints"
        ],
        "post_selection.selected_resolution_seed_checkpoints",
    )

    final_detection = require_mapping(
        selected_checkpoint_cfg[
            "final_detection_test"
        ],
        (
            "post_selection.selected_resolution_seed_checkpoints."
            "final_detection_test"
        ),
    )

    if (
        final_detection[
            "evaluate_all_three"
        ]
        is not True
    ):

        raise RuntimeError(
            "Frozen protocol must evaluate all three selected "
            "resolution checkpoints."
        )

    threshold_cfg = require_mapping(
        evaluation[
            "threshold_selection"
        ],
        "evaluation.threshold_selection",
    )

    if (
        threshold_cfg[
            "source"
        ]
        != "dev_val"
    ):

        raise RuntimeError(
            "Threshold source must remain dev_val."
        )

    if (
        threshold_cfg[
            "threshold_per_seed_checkpoint"
        ]
        is not True
    ):

        raise RuntimeError(
            "Each seed checkpoint must derive its own threshold."
        )

    if float(
        threshold_cfg[
            "target_false_positive_rate"
        ]
    ) != 0.10:

        raise RuntimeError(
            "Frozen target FPR changed."
        )

    return {
        "resolutions":
            resolutions,

        "seeds":
            seeds,

        "margin_fraction":
            margin_fraction,

        "target_fpr":
            0.10,
    }


# ======================================================================
# Source-run validation
# ======================================================================

def validate_source_run(
    *,
    path: Path,
    expected_resolution: str,
    expected_seed: int,
    expected_experiment_sha256: str,
) -> dict[str, Any]:

    payload = load_yaml(
        path
    )

    run = require_mapping(
        payload[
            "run"
        ],
        "run",
    )

    scientific_config = require_mapping(
        payload[
            "scientific_config"
        ],
        "scientific_config",
    )

    frozen_data = require_mapping(
        payload[
            "frozen_data"
        ],
        "frozen_data",
    )

    artifacts = require_mapping(
        payload[
            "artifacts"
        ],
        "artifacts",
    )

    result = require_mapping(
        payload[
            "result"
        ],
        "result",
    )

    if run[
        "status"
    ] != "completed":

        raise RuntimeError(
            f"Source run not completed: {path}"
        )

    resolution_name = str(
        run[
            "resolution_name"
        ]
    )

    run_seed = int(
        run[
            "run_seed"
        ]
    )

    if (
        resolution_name
        != expected_resolution
    ):

        raise RuntimeError(
            "Source-run resolution mismatch:\n"
            f"  expected={expected_resolution}\n"
            f"  actual={resolution_name}"
        )

    if run_seed != expected_seed:

        raise RuntimeError(
            "Source-run seed mismatch:\n"
            f"  expected={expected_seed}\n"
            f"  actual={run_seed}"
        )

    workflow = str(
        run[
            "workflow"
        ]
    )

    if expected_seed == 8:

        expected_workflow = (
            "resolution_lr_screening"
        )

    else:

        expected_workflow = (
            "selected_lr_multi_seed_stability"
        )

    if workflow != expected_workflow:

        raise RuntimeError(
            "Unexpected source workflow:\n"
            f"  expected={expected_workflow}\n"
            f"  actual={workflow}"
        )

    config_sha = str(
        scientific_config[
            "sha256"
        ]
    )

    if (
        config_sha
        != expected_experiment_sha256
    ):

        raise RuntimeError(
            "Source run used unexpected scientific config SHA."
        )

    if (
        frozen_data[
            "held_out_test_accessed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Source development run reports held-out-test access."
        )

    checkpoint = require_mapping(
        artifacts[
            "selected_checkpoint"
        ],
        "artifacts.selected_checkpoint",
    )

    backbone_lr = float(
        checkpoint[
            "backbone_lr"
        ]
    )

    expected_lr = {
        "r256":
            0.0003,

        "r512":
            0.0001,
    }[
        expected_resolution
    ]

    if backbone_lr != expected_lr:

        raise RuntimeError(
            "Source run selected wrong frozen LR:\n"
            f"  resolution={expected_resolution}\n"
            f"  expected={expected_lr}\n"
            f"  actual={backbone_lr}"
        )

    weighted_dev_loss = float(
        checkpoint[
            "weighted_dev_loss"
        ]
    )

    if weighted_dev_loss < 0.0:

        raise RuntimeError(
            "Invalid source weighted dev loss."
        )

    checkpoint_file_sha = str(
        checkpoint[
            "file_sha256"
        ]
    )

    checkpoint_model_sha = str(
        checkpoint[
            "model_state_sha256"
        ]
    )

    if len(
        checkpoint_file_sha
    ) != 64:

        raise RuntimeError(
            "Malformed checkpoint file SHA-256."
        )

    if len(
        checkpoint_model_sha
    ) != 64:

        raise RuntimeError(
            "Malformed checkpoint model-state SHA-256."
        )

    if result.get(
        "protocol_review_required"
    ) is not False:

        raise RuntimeError(
            "Source run requires unresolved protocol review."
        )

    if expected_seed == 8:

        result_loss = float(
            result[
                "selected_raw_best_weighted_dev_loss"
            ]
        )

        result_lr = float(
            result[
                "selected_backbone_lr"
            ]
        )

    else:

        result_loss = float(
            result[
                "raw_best_weighted_dev_loss"
            ]
        )

        result_lr = float(
            result[
                "backbone_lr"
            ]
        )

    if result_loss != weighted_dev_loss:

        raise RuntimeError(
            "Source run checkpoint loss and result loss disagree."
        )

    if result_lr != backbone_lr:

        raise RuntimeError(
            "Source run checkpoint LR and result LR disagree."
        )

    return {
        "run_yaml":
            path.relative_to(
                REPO_ROOT
            ).as_posix(),

        "run_yaml_sha256":
            sha256_file(
                path
            ),

        "run_id":
            str(
                run[
                    "run_id"
                ]
            ),

        "workflow":
            workflow,

        "training_machine_id":
            str(
                payload[
                    "machine"
                ][
                    "id"
                ]
            ),

        "resolution_name":
            resolution_name,

        "run_seed":
            run_seed,

        "backbone_lr":
            backbone_lr,

        "raw_best_weighted_dev_loss":
            weighted_dev_loss,

        "checkpoint":
            {
                "path":
                    str(
                        checkpoint[
                            "path"
                        ]
                    ),

                "file_sha256":
                    checkpoint_file_sha,

                "model_state_sha256":
                    checkpoint_model_sha,

                "epoch":
                    int(
                        checkpoint[
                            "checkpoint_epoch"
                        ]
                    ),
            },
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Finalize frozen ResNet-18 r256/r512 "
            "resolution selection."
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
        "--selection-config",
        default=(
            "tools/"
            "finalize_resnet18_resolution_selection_config.yaml"
        ),
    )

    args = parser.parse_args()

    git_commit = (
        require_clean_git()
    )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            args.config
        )
    )

    experiment_sha = sha256_file(
        experiment_path
    )

    tool_cfg_path = resolve_repo_path(
        args.selection_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    contract = load_selection_contract(
        experiment_cfg=(
            experiment_cfg
        )
    )

    (
        log_path,
        output_path,
    ) = configure_outputs(
        tool_cfg=(
            tool_cfg
        )
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 RESOLUTION SELECTION FINALIZATION"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            git_commit,
        )

        LOGGER.info(
            "Experiment config SHA-256: %s",
            experiment_sha,
        )

        LOGGER.info(
            "Decision rule: select r512 if mean_d <= margin"
        )

        LOGGER.info(
            "Margin fraction: %.6f",
            contract[
                "margin_fraction"
            ],
        )

        LOGGER.info(
            "Counterfactual diagnostics influence selection: False"
        )

        LOGGER.info(
            "Held-out test: NOT ACCESSED"
        )

        sources_cfg = require_mapping(
            tool_cfg[
                "sources"
            ],
            "sources",
        )

        evidence: dict[
            str,
            dict[
                int,
                dict[str, Any],
            ],
        ] = {
            "r256":
                {},

            "r512":
                {},
        }

        for resolution_name in (
            contract[
                "resolutions"
            ]
        ):

            resolution_sources = require_mapping(
                sources_cfg[
                    resolution_name
                ],
                f"sources.{resolution_name}",
            )

            for seed in (
                contract[
                    "seeds"
                ]
            ):

                seed_cfg = (
                    resolution_sources.get(
                        seed
                    )
                    or
                    resolution_sources.get(
                        str(
                            seed
                        )
                    )
                )

                seed_cfg = require_mapping(
                    seed_cfg,
                    (
                        f"sources."
                        f"{resolution_name}."
                        f"{seed}"
                    ),
                )

                run_yaml = resolve_repo_path(
                    seed_cfg[
                        "run_yaml"
                    ]
                )

                source = validate_source_run(
                    path=(
                        run_yaml
                    ),
                    expected_resolution=(
                        resolution_name
                    ),
                    expected_seed=(
                        seed
                    ),
                    expected_experiment_sha256=(
                        experiment_sha
                    ),
                )

                evidence[
                    resolution_name
                ][
                    seed
                ] = source

                LOGGER.info(
                    "Source validated | resolution=%s | "
                    "seed=%d | loss=%.12f | LR=%.8g | "
                    "machine=%s",
                    resolution_name,
                    seed,
                    source[
                        "raw_best_weighted_dev_loss"
                    ],
                    source[
                        "backbone_lr"
                    ],
                    source[
                        "training_machine_id"
                    ],
                )

        # ==============================================================
        # Frozen practical NI calculation
        # ==============================================================

        losses_256 = {
            seed:
                evidence[
                    "r256"
                ][
                    seed
                ][
                    "raw_best_weighted_dev_loss"
                ]

            for seed
            in contract[
                "seeds"
            ]
        }

        losses_512 = {
            seed:
                evidence[
                    "r512"
                ][
                    seed
                ][
                    "raw_best_weighted_dev_loss"
                ]

            for seed
            in contract[
                "seeds"
            ]
        }

        differences = {
            seed:
                (
                    losses_512[
                        seed
                    ]
                    -
                    losses_256[
                        seed
                    ]
                )

            for seed
            in contract[
                "seeds"
            ]
        }

        mean_256 = statistics.fmean(
            losses_256.values()
        )

        mean_512 = statistics.fmean(
            losses_512.values()
        )

        mean_d = statistics.fmean(
            differences.values()
        )

        margin = (
            contract[
                "margin_fraction"
            ]
            * mean_256
        )

        if mean_d <= margin:

            selected_resolution = (
                "r512"
            )

            losing_resolution = (
                "r256"
            )

        else:

            selected_resolution = (
                "r256"
            )

            losing_resolution = (
                "r512"
            )

        selected_lr = {
            "r256":
                0.0003,

            "r512":
                0.0001,
        }[
            selected_resolution
        ]

        sign_pattern = {
            seed:
                (
                    "-"
                    if differences[
                        seed
                    ] < 0.0

                    else (
                        "+"
                        if differences[
                            seed
                        ] > 0.0

                        else "0"
                    )
                )

            for seed
            in contract[
                "seeds"
            ]
        }

        # ==============================================================
        # Representative checkpoint
        # ==============================================================

        selected_losses = (
            losses_512
            if selected_resolution == "r512"
            else losses_256
        )

        median_loss = statistics.median(
            selected_losses.values()
        )

        representative_candidates = [
            seed

            for (
                seed,
                loss,
            ) in selected_losses.items()

            if loss == median_loss
        ]

        representative_seed = min(
            representative_candidates
        )

        representative_checkpoint = (
            evidence[
                selected_resolution
            ][
                representative_seed
            ][
                "checkpoint"
            ]
        )

        # ==============================================================
        # Canonical freeze artifact
        # ==============================================================

        artifact = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_resolution_selection",

            "status":
                "FROZEN",

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_commit":
                git_commit,

            "scientific_config":
                {
                    "path":
                        experiment_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),

                    "sha256":
                        experiment_sha,
                },

            "development_evidence":
                {
                    resolution_name:
                        {
                            seed:
                                evidence[
                                    resolution_name
                                ][
                                    seed
                                ]

                            for seed
                            in contract[
                                "seeds"
                            ]
                        }

                    for resolution_name
                    in contract[
                        "resolutions"
                    ]
                },

            "resolution_selection":
                {
                    "metric":
                        "class_weighted_dev_cross_entropy",

                    "metric_source":
                        (
                            "raw_minimum_for_that_resolution_"
                            "selected_lr"
                        ),

                    "paired_difference":
                        "d_s = L512(s) - L256(s)",

                    "losses":
                        {
                            "r256":
                                losses_256,

                            "r512":
                                losses_512,
                        },

                    "differences":
                        differences,

                    "sign_pattern":
                        sign_pattern,

                    "mean_r256_loss":
                        mean_256,

                    "mean_r512_loss":
                        mean_512,

                    "mean_d":
                        mean_d,

                    "margin_fraction":
                        contract[
                            "margin_fraction"
                        ],

                    "margin":
                        margin,

                    "decision_rule":
                        (
                            "select r512 if mean_d <= margin; "
                            "otherwise select r256"
                        ),

                    "selected_resolution":
                        selected_resolution,

                    "selected_backbone_lr":
                        selected_lr,

                    "losing_resolution":
                        losing_resolution,

                    "counterfactual_diagnostics_used_for_selection":
                        False,

                    "localisation_metrics_used_for_selection":
                        False,
                },

            "final_detection_plan":
                {
                    "resolution":
                        selected_resolution,

                    "backbone_lr":
                        selected_lr,

                    "seeds":
                        list(
                            contract[
                                "seeds"
                            ]
                        ),

                    "evaluate_all_three_checkpoints":
                        True,

                    "checkpoint_sources":
                        {
                            seed:
                                {
                                    "training_machine_id":
                                        evidence[
                                            selected_resolution
                                        ][
                                            seed
                                        ][
                                            "training_machine_id"
                                        ],

                                    "run_id":
                                        evidence[
                                            selected_resolution
                                        ][
                                            seed
                                        ][
                                            "run_id"
                                        ],

                                    "checkpoint":
                                        evidence[
                                            selected_resolution
                                        ][
                                            seed
                                        ][
                                            "checkpoint"
                                        ],
                                }

                            for seed
                            in contract[
                                "seeds"
                            ]
                        },

                    "threshold_policy":
                        {
                            "source":
                                "own_dev_val_scores",

                            "one_threshold_per_seed_checkpoint":
                                True,

                            "target_false_positive_rate":
                                contract[
                                    "target_fpr"
                                ],

                            "test_threshold_tuning":
                                False,
                        },

                    "held_out_test_accessed_during_selection":
                        False,

                    "best_test_seed_selection_permitted":
                        False,
                },

            "localisation_representative_checkpoint":
                {
                    "rule":
                        "median_raw_best_weighted_dev_loss",

                    "exact_tie_breaker":
                        "lowest_seed_number",

                    "median_loss":
                        median_loss,

                    "seed":
                        representative_seed,

                    "training_machine_id":
                        evidence[
                            selected_resolution
                        ][
                            representative_seed
                        ][
                            "training_machine_id"
                        ],

                    "checkpoint":
                        representative_checkpoint,
                },

            "next_gate":
                {
                    "fpr10_threshold_derivation":
                        "READY",

                    "held_out_test":
                        (
                            "NOT_YET_OPENED__"
                            "THRESHOLD_MACHINERY_MUST_BE_VALIDATED_FIRST"
                        ),
                },
        }

        write_yaml_exclusive(
            path=(
                output_path
            ),
            value=(
                artifact
            ),
        )

        output_sha = sha256_file(
            output_path
        )

        # ==============================================================
        # Evidence log
        # ==============================================================

        LOGGER.info(
            ""
        )

        for seed in (
            contract[
                "seeds"
            ]
        ):

            LOGGER.info(
                "Seed %d | L256=%.12f | L512=%.12f | "
                "d=%.12f | sign=%s",
                seed,
                losses_256[
                    seed
                ],
                losses_512[
                    seed
                ],
                differences[
                    seed
                ],
                sign_pattern[
                    seed
                ],
            )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "Mean r256 loss: %.12f",
            mean_256,
        )

        LOGGER.info(
            "Mean r512 loss: %.12f",
            mean_512,
        )

        LOGGER.info(
            "Mean paired difference: %.12f",
            mean_d,
        )

        LOGGER.info(
            "5%% r256-relative margin: %.12f",
            margin,
        )

        LOGGER.info(
            "Decision comparison: %.12f <= %.12f -> %s",
            mean_d,
            margin,
            (
                mean_d
                <= margin
            ),
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] selected resolution: %s",
            selected_resolution,
        )

        LOGGER.info(
            "[PASS] selected backbone LR: %.8g",
            selected_lr,
        )

        LOGGER.info(
            "[PASS] representative Grad-CAM seed: %d",
            representative_seed,
        )

        LOGGER.info(
            "[PASS] all three selected-resolution checkpoints "
            "assigned to final detection"
        )

        LOGGER.info(
            "[PASS] threshold source remains per-checkpoint dev_val"
        )

        LOGGER.info(
            "[PASS] counterfactual diagnostic did not "
            "participate in selection"
        )

        LOGGER.info(
            "[PASS] held-out test was NOT ACCESSED"
        )

        LOGGER.info(
            "Selection artifact: %s",
            output_path,
        )

        LOGGER.info(
            "Selection artifact SHA-256: %s",
            output_sha,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 RESOLUTION SELECTION: FROZEN"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "RESNET-18 RESOLUTION SELECTION FINALIZATION: FAIL"
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