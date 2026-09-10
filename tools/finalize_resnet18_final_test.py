#!/usr/bin/env python3
"""
Freeze the canonical three-seed ResNet-18 held-out-test result.

This tool performs NO:
- image access;
- model loading;
- checkpoint loading;
- inference;
- threshold derivation;
- threshold modification;
- seed selection.

It consumes only already-committed:
- final test result YAMLs;
- final test prediction CSVs;
- frozen protocol/evidence artifacts.

For every seed it independently reconstructs FP32 logits from the
prediction CSV and recomputes the frozen metrics. This verifies that the
persisted result YAML agrees with the per-image evidence.

The resulting summary reports:
- all three individual seed results;
- mean / minimum / maximum across seeds;
- no "best seed".

No print() is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.final_evaluation import (
    compute_final_binary_evaluation,
    load_final_evaluation_contract,
)

from src.fpr10 import (
    apply_attack_threshold,
    attack_probability_from_logits,
)


LOGGER = logging.getLogger(
    "finalize_resnet18_final_test"
)


SEEDS = (
    8,
    9,
    10,
)


PREDICTION_COLUMNS = (
    "row_index",
    "source_workbook_row",
    "image_path",
    "image_sha256",
    "verified_image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "face_db",
    "face_id",
    "gender",
    "target",
    "bonafide_logit_fp32",
    "attack_logit_fp32",
    "attack_margin_float64",
    "p_attack_float64",
    "prediction_fixed_0_5",
    "prediction_fpr10",
)


IDENTITY_COLUMNS = (
    "row_index",
    "source_workbook_row",
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "face_db",
    "face_id",
    "gender",
    "target",
)


SUMMARY_METRICS = (
    "auroc",
    "accuracy",
    "support_weighted_f1",
    "attack_f1",
    "balanced_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "half_total_error_rate",
)


# ======================================================================
# Helpers
# ======================================================================

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


def relative_repo_path(
    path: Path,
) -> str:

    return (
        path.resolve()
        .relative_to(
            REPO_ROOT
        )
        .as_posix()
    )


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
) -> Mapping[Any, Any]:

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{label} must be a mapping."
        )

    return value


def seed_value(
    mapping: Mapping[Any, Any],
    seed: int,
) -> Any:

    if seed in mapping:
        return mapping[
            seed
        ]

    if str(seed) in mapping:
        return mapping[
            str(seed)
        ]

    raise KeyError(
        f"Missing seed {seed}."
    )


def require_file_sha(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
) -> str:

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch:\n"
            f"  expected={expected_sha256}\n"
            f"  actual={actual}"
        )

    return actual


def require_clean_git() -> tuple[str, str]:

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

    branch = subprocess.run(
        [
            "git",
            "rev-parse",
            "--abbrev-ref",
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
            "Git working tree must be clean before final-test "
            "aggregation.\n\n"
            f"{status}"
        )

    return (
        commit,
        branch,
    )


def assert_close(
    *,
    label: str,
    actual: float,
    expected: float,
    tolerance: float = 1.0e-14,
) -> None:

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"{label} mismatch:\n"
            f"  expected={expected:.17g}\n"
            f"  actual={actual:.17g}"
        )


# ======================================================================
# Logging
# ======================================================================

def configure_logging(
    *,
    config: Mapping[Any, Any],
    timestamp: str,
) -> Path:

    logging_cfg = require_mapping(
        config[
            "logging"
        ],
        "logging",
    )

    directory = resolve_repo_path(
        logging_cfg[
            "directory"
        ]
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / str(
            logging_cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
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
        path,
        mode="x",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    formatter.converter = time.gmtime

    handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        handler
    )

    return path


# ======================================================================
# Prediction loading / validation
# ======================================================================

def load_prediction_csv(
    *,
    path: Path,
) -> tuple[
    tuple[dict[str, str], ...],
    torch.Tensor,
    torch.Tensor,
]:

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                "Prediction CSV has no header."
            )

        if tuple(
            reader.fieldnames
        ) != PREDICTION_COLUMNS:
            raise RuntimeError(
                "Prediction CSV column contract changed."
            )

        rows = tuple(
            reader
        )

    if len(
        rows
    ) != 1385:
        raise RuntimeError(
            "Prediction CSV must contain exactly 1385 rows."
        )

    logits: list[
        tuple[
            float,
            float,
        ]
    ] = []

    targets: list[
        int
    ] = []

    for expected_index, row in enumerate(
        rows
    ):

        if int(
            row[
                "row_index"
            ]
        ) != expected_index:
            raise RuntimeError(
                "Prediction row_index/order changed."
            )

        traffic_type = row[
            "traffic_type"
        ]

        target = int(
            row[
                "target"
            ]
        )

        if traffic_type == "bonafide":
            expected_target = 0

        elif traffic_type == "attack":
            expected_target = 1

        else:
            raise RuntimeError(
                "Unexpected traffic_type."
            )

        if target != expected_target:
            raise RuntimeError(
                "Prediction traffic_type/target mismatch."
            )

        if (
            row[
                "image_sha256"
            ]
            != row[
                "verified_image_sha256"
            ]
        ):
            raise RuntimeError(
                "Persisted file-integrity evidence mismatch."
            )

        logits.append(
            (
                float(
                    row[
                        "bonafide_logit_fp32"
                    ]
                ),
                float(
                    row[
                        "attack_logit_fp32"
                    ]
                ),
            )
        )

        targets.append(
            target
        )

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
        device="cpu",
    )

    targets_tensor = torch.tensor(
        targets,
        dtype=torch.int64,
        device="cpu",
    )

    if tuple(
        logits_tensor.shape
    ) != (
        1385,
        2,
    ):
        raise RuntimeError(
            "Reconstructed logits have unexpected shape."
        )

    return (
        rows,
        logits_tensor,
        targets_tensor,
    )


def verify_prediction_derived_columns(
    *,
    rows: tuple[dict[str, str], ...],
    logits: torch.Tensor,
    fpr10_threshold: float,
) -> None:

    probabilities = attack_probability_from_logits(
        logits
    )

    margins = (
        logits[
            :,
            1,
        ].to(
            dtype=torch.float64
        )
        -
        logits[
            :,
            0,
        ].to(
            dtype=torch.float64
        )
    )

    fixed_predictions = apply_attack_threshold(
        scores=probabilities,
        threshold=0.5,
    )

    fpr10_predictions = apply_attack_threshold(
        scores=probabilities,
        threshold=fpr10_threshold,
    )

    for index, row in enumerate(
        rows
    ):

        assert_close(
            label=(
                f"row {index} attack margin"
            ),
            actual=float(
                row[
                    "attack_margin_float64"
                ]
            ),
            expected=float(
                margins[
                    index
                ].item()
            ),
        )

        assert_close(
            label=(
                f"row {index} p_attack"
            ),
            actual=float(
                row[
                    "p_attack_float64"
                ]
            ),
            expected=float(
                probabilities[
                    index
                ].item()
            ),
        )

        if int(
            row[
                "prediction_fixed_0_5"
            ]
        ) != int(
            fixed_predictions[
                index
            ].item()
        ):
            raise RuntimeError(
                "Stored fixed-0.5 prediction mismatch."
            )

        if int(
            row[
                "prediction_fpr10"
            ]
        ) != int(
            fpr10_predictions[
                index
            ].item()
        ):
            raise RuntimeError(
                "Stored FPR10 prediction mismatch."
            )


# ======================================================================
# Result metric validation
# ======================================================================

def scalar_metrics_from_evaluation(
    evaluation: Any,
) -> dict[str, dict[str, float]]:

    fixed = evaluation.fixed_0_5
    controlled = evaluation.controlled_fpr10

    return {
        "overall":
            {
                "auroc":
                    float(
                        evaluation.auroc
                    ),
            },

        "fixed_0_5":
            {
                "accuracy":
                    float(
                        fixed.accuracy
                    ),

                "support_weighted_f1":
                    float(
                        fixed.support_weighted_f1
                    ),

                "attack_f1":
                    float(
                        fixed.attack.f1
                    ),

                "balanced_accuracy":
                    float(
                        fixed.balanced_accuracy
                    ),

                "false_positive_rate":
                    float(
                        fixed.false_positive_rate
                    ),

                "false_negative_rate":
                    float(
                        fixed.false_negative_rate
                    ),

                "half_total_error_rate":
                    float(
                        fixed.half_total_error_rate
                    ),
            },

        "controlled_fpr10":
            {
                "accuracy":
                    float(
                        controlled.accuracy
                    ),

                "support_weighted_f1":
                    float(
                        controlled.support_weighted_f1
                    ),

                "attack_f1":
                    float(
                        controlled.attack.f1
                    ),

                "balanced_accuracy":
                    float(
                        controlled.balanced_accuracy
                    ),

                "false_positive_rate":
                    float(
                        controlled.false_positive_rate
                    ),

                "false_negative_rate":
                    float(
                        controlled.false_negative_rate
                    ),

                "half_total_error_rate":
                    float(
                        controlled.half_total_error_rate
                    ),
            },
    }


def verify_result_metrics(
    *,
    result: Mapping[str, Any],
    recomputed: Any,
) -> None:

    stored = require_mapping(
        result[
            "metrics"
        ],
        "result.metrics",
    )

    if int(
        stored[
            "sample_count"
        ]
    ) != 1385:
        raise RuntimeError(
            "Stored sample count changed."
        )

    if (
        int(
            stored[
                "bonafide_count"
            ]
        ),
        int(
            stored[
                "attack_count"
            ]
        ),
    ) != (
        300,
        1085,
    ):
        raise RuntimeError(
            "Stored class counts changed."
        )

    assert_close(
        label="AUROC",
        actual=float(
            stored[
                "auroc"
            ]
        ),
        expected=float(
            recomputed.auroc
        ),
    )

    for operating_name, recomputed_metrics in (
        (
            "fixed_0_5",
            recomputed.fixed_0_5,
        ),
        (
            "controlled_fpr10",
            recomputed.controlled_fpr10,
        ),
    ):

        stored_metrics = require_mapping(
            stored[
                operating_name
            ],
            f"metrics.{operating_name}",
        )

        integer_fields = {
            "true_positive":
                recomputed_metrics.true_positive,

            "false_positive":
                recomputed_metrics.false_positive,

            "true_negative":
                recomputed_metrics.true_negative,

            "false_negative":
                recomputed_metrics.false_negative,
        }

        for field, expected in (
            integer_fields.items()
        ):

            if int(
                stored_metrics[
                    field
                ]
            ) != expected:
                raise RuntimeError(
                    f"{operating_name}.{field} mismatch."
                )

        float_fields = {
            "accuracy":
                recomputed_metrics.accuracy,

            "support_weighted_f1":
                recomputed_metrics.support_weighted_f1,

            "balanced_accuracy":
                recomputed_metrics.balanced_accuracy,

            "false_positive_rate":
                recomputed_metrics.false_positive_rate,

            "false_negative_rate":
                recomputed_metrics.false_negative_rate,

            "half_total_error_rate":
                recomputed_metrics.half_total_error_rate,

            "apcer":
                recomputed_metrics.apcer,

            "bpcer":
                recomputed_metrics.bpcer,
        }

        for field, expected in (
            float_fields.items()
        ):

            assert_close(
                label=(
                    f"{operating_name}.{field}"
                ),
                actual=float(
                    stored_metrics[
                        field
                    ]
                ),
                expected=float(
                    expected
                ),
            )

        assert_close(
            label=(
                f"{operating_name}.attack.f1"
            ),
            actual=float(
                stored_metrics[
                    "attack"
                ][
                    "f1"
                ]
            ),
            expected=float(
                recomputed_metrics.attack.f1
            ),
        )


# ======================================================================
# Aggregate summaries
# ======================================================================

def mean_min_max(
    values: list[float],
) -> dict[str, float]:

    if len(
        values
    ) != 3:
        raise RuntimeError(
            "Three-seed aggregation requires exactly three values."
        )

    return {
        "mean":
            sum(
                values
            )
            / len(
                values
            ),

        "min":
            min(
                values
            ),

        "max":
            max(
                values
            ),
    }


def build_aggregate(
    per_seed_metrics: Mapping[
        int,
        Mapping[str, Mapping[str, float]],
    ],
) -> dict[str, Any]:

    aggregate: dict[
        str,
        Any,
    ] = {
        "overall":
            {},

        "fixed_0_5":
            {},

        "controlled_fpr10":
            {},
    }

    aggregate[
        "overall"
    ][
        "auroc"
    ] = mean_min_max(
        [
            per_seed_metrics[
                seed
            ][
                "overall"
            ][
                "auroc"
            ]

            for seed
            in SEEDS
        ]
    )

    for operating_name in (
        "fixed_0_5",
        "controlled_fpr10",
    ):

        for metric_name in (
            "accuracy",
            "support_weighted_f1",
            "attack_f1",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "half_total_error_rate",
        ):

            aggregate[
                operating_name
            ][
                metric_name
            ] = mean_min_max(
                [
                    per_seed_metrics[
                        seed
                    ][
                        operating_name
                    ][
                        metric_name
                    ]

                    for seed
                    in SEEDS
                ]
            )

    return aggregate


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Freeze canonical three-seed ResNet-18 final test summary."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "tools/"
            "finalize_resnet18_final_test_config.yaml"
        ),
    )

    args = parser.parse_args()

    git_commit, git_branch = (
        require_clean_git()
    )

    config_path = resolve_repo_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
    )

    # --------------------------------------------------------------
    # Validate frozen common sources before creating this run's log.
    # --------------------------------------------------------------

    scientific_cfg = require_mapping(
        config[
            "scientific_config"
        ],
        "scientific_config",
    )

    scientific_path = resolve_repo_path(
        scientific_cfg[
            "path"
        ]
    )

    scientific_sha = require_file_sha(
        path=scientific_path,
        expected_sha256=str(
            scientific_cfg[
                "sha256"
            ]
        ),
        label="Scientific config",
    )

    protocol_cfg = require_mapping(
        config[
            "evaluation_protocol"
        ],
        "evaluation_protocol",
    )

    protocol_path = resolve_repo_path(
        protocol_cfg[
            "path"
        ]
    )

    protocol_sha = require_file_sha(
        path=protocol_path,
        expected_sha256=str(
            protocol_cfg[
                "sha256"
            ]
        ),
        label="Evaluation protocol",
    )

    protocol = load_yaml(
        protocol_path
    )

    contract = load_final_evaluation_contract(
        evaluation_cfg=protocol
    )

    audit_cfg = require_mapping(
        config[
            "final_evaluation_audit"
        ],
        "final_evaluation_audit",
    )

    audit_path = resolve_repo_path(
        audit_cfg[
            "path"
        ]
    )

    audit_sha = require_file_sha(
        path=audit_path,
        expected_sha256=str(
            audit_cfg[
                "sha256"
            ]
        ),
        label="Final evaluation audit",
    )

    audit = load_yaml(
        audit_path
    )

    if audit.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Final evaluation primitive audit is not PASS."
        )

    if (
        audit[
            "source_sha256"
        ][
            "src/final_evaluation.py"
        ]
        != sha256_file(
            REPO_ROOT
            / "src"
            / "final_evaluation.py"
        )
    ):
        raise RuntimeError(
            "Current final_evaluation.py differs from audited source."
        )

    manifest_cfg = require_mapping(
        config[
            "canonical_test_manifest"
        ],
        "canonical_test_manifest",
    )

    metadata_cfg = require_mapping(
        manifest_cfg[
            "metadata"
        ],
        "canonical_test_manifest.metadata",
    )

    csv_cfg = require_mapping(
        manifest_cfg[
            "csv"
        ],
        "canonical_test_manifest.csv",
    )

    metadata_path = resolve_repo_path(
        metadata_cfg[
            "path"
        ]
    )

    metadata_sha = require_file_sha(
        path=metadata_path,
        expected_sha256=str(
            metadata_cfg[
                "sha256"
            ]
        ),
        label="Canonical test metadata",
    )

    manifest_path = resolve_repo_path(
        csv_cfg[
            "path"
        ]
    )

    manifest_sha = require_file_sha(
        path=manifest_path,
        expected_sha256=str(
            csv_cfg[
                "sha256"
            ]
        ),
        label="Canonical test manifest",
    )

    log_path = configure_logging(
        config=config,
        timestamp=timestamp,
    )

    output_cfg = require_mapping(
        config[
            "output"
        ],
        "output",
    )

    output_directory = resolve_repo_path(
        output_cfg[
            "directory"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = list(
        output_directory.glob(
            "resnet18_r512_three_seed_final_test_*.yaml"
        )
    )

    if existing:
        raise RuntimeError(
            "A canonical three-seed final-test summary already exists:\n"
            + "\n".join(
                str(
                    path
                )
                for path
                in existing
            )
        )

    output_path = (
        output_directory
        / str(
            output_cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 THREE-SEED FINAL TEST FINALIZATION"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            git_commit,
        )

        LOGGER.info(
            "[PASS] scientific config SHA-256 = %s",
            scientific_sha,
        )

        LOGGER.info(
            "[PASS] evaluation protocol SHA-256 = %s",
            protocol_sha,
        )

        LOGGER.info(
            "[PASS] final evaluation audit SHA-256 = %s",
            audit_sha,
        )

        LOGGER.info(
            "[PASS] canonical test manifest SHA-256 = %s",
            manifest_sha,
        )

        seeds_cfg = require_mapping(
            config[
                "seeds"
            ],
            "seeds",
        )

        per_seed_output: dict[
            int,
            dict[str, Any],
        ] = {}

        per_seed_metrics: dict[
            int,
            dict[str, dict[str, float]],
        ] = {}

        canonical_identity: tuple[
            tuple[str, ...],
            ...,
        ] | None = None

        for seed in SEEDS:

            seed_cfg = require_mapping(
                seed_value(
                    seeds_cfg,
                    seed,
                ),
                f"seeds.{seed}",
            )

            result_cfg = require_mapping(
                seed_cfg[
                    "result"
                ],
                f"seeds.{seed}.result",
            )

            predictions_cfg = require_mapping(
                seed_cfg[
                    "predictions"
                ],
                f"seeds.{seed}.predictions",
            )

            result_path = resolve_repo_path(
                result_cfg[
                    "path"
                ]
            )

            result_sha = require_file_sha(
                path=result_path,
                expected_sha256=str(
                    result_cfg[
                        "sha256"
                    ]
                ),
                label=(
                    f"Seed {seed} result"
                ),
            )

            predictions_path = resolve_repo_path(
                predictions_cfg[
                    "path"
                ]
            )

            predictions_sha = require_file_sha(
                path=predictions_path,
                expected_sha256=str(
                    predictions_cfg[
                        "sha256"
                    ]
                ),
                label=(
                    f"Seed {seed} predictions"
                ),
            )

            result = load_yaml(
                result_path
            )

            if result.get(
                "artifact_type"
            ) != "resnet18_final_held_out_test_result":
                raise RuntimeError(
                    f"Seed {seed} result artifact type changed."
                )

            if result.get(
                "status"
            ) != "FINAL_HELD_OUT_TEST_EVALUATION_COMPLETE":
                raise RuntimeError(
                    f"Seed {seed} final test is not complete."
                )

            if int(
                result[
                    "checkpoint"
                ][
                    "seed"
                ]
            ) != seed:
                raise RuntimeError(
                    f"Seed {seed} checkpoint identity mismatch."
                )

            if (
                result[
                    "machine"
                ][
                    "id"
                ]
                != seed_cfg[
                    "expected_machine"
                ]
            ):
                raise RuntimeError(
                    f"Seed {seed} origin machine mismatch."
                )

            if result[
                "scientific_config"
            ][
                "sha256"
            ] != scientific_sha:
                raise RuntimeError(
                    f"Seed {seed} scientific config mismatch."
                )

            if result[
                "evaluation_protocol"
            ][
                "sha256"
            ] != protocol_sha:
                raise RuntimeError(
                    f"Seed {seed} evaluation protocol mismatch."
                )

            if result[
                "final_evaluation_primitive_audit"
            ][
                "sha256"
            ] != audit_sha:
                raise RuntimeError(
                    f"Seed {seed} audit evidence mismatch."
                )

            held_out = require_mapping(
                result[
                    "held_out_test"
                ],
                f"seed{seed}.held_out_test",
            )

            if (
                int(
                    held_out[
                        "sample_count"
                    ]
                ),
                int(
                    held_out[
                        "bonafide_count"
                    ]
                ),
                int(
                    held_out[
                        "attack_count"
                    ]
                ),
                held_out[
                    "resolution"
                ],
            ) != (
                1385,
                300,
                1085,
                "r512",
            ):
                raise RuntimeError(
                    f"Seed {seed} test population/resolution changed."
                )

            if held_out[
                "manifest_sha256"
            ] != manifest_sha:
                raise RuntimeError(
                    f"Seed {seed} manifest SHA mismatch."
                )

            if held_out[
                "manifest_metadata_sha256"
            ] != metadata_sha:
                raise RuntimeError(
                    f"Seed {seed} metadata SHA mismatch."
                )

            prediction_evidence = require_mapping(
                result[
                    "predictions"
                ],
                f"seed{seed}.predictions",
            )

            if prediction_evidence[
                "sha256"
            ] != predictions_sha:
                raise RuntimeError(
                    f"Seed {seed} result/prediction SHA mismatch."
                )

            if int(
                prediction_evidence[
                    "rows"
                ]
            ) != 1385:
                raise RuntimeError(
                    f"Seed {seed} prediction row count changed."
                )

            if prediction_evidence[
                "all_file_sha256_verified"
            ] is not True:
                raise RuntimeError(
                    f"Seed {seed} did not verify all image hashes."
                )

            boundaries = require_mapping(
                result[
                    "scientific_boundaries"
                ],
                f"seed{seed}.scientific_boundaries",
            )

            if boundaries[
                "held_out_test_accessed"
            ] is not True:
                raise RuntimeError(
                    f"Seed {seed} does not report held-out access."
                )

            forbidden = (
                "training_performed",
                "checkpoint_selection_performed",
                "learning_rate_selection_performed",
                "resolution_selection_performed",
                "threshold_derivation_from_test",
                "threshold_modification_on_test",
                "best_test_seed_selection_performed",
            )

            for key in forbidden:

                if boundaries[
                    key
                ] is not False:
                    raise RuntimeError(
                        f"Seed {seed} violates frozen boundary: {key}"
                    )

            fixed_threshold = float(
                result[
                    "thresholds"
                ][
                    "fixed_0_5"
                ][
                    "threshold"
                ]
            )

            if fixed_threshold != 0.5:
                raise RuntimeError(
                    "Fixed threshold changed."
                )

            fpr10_info = require_mapping(
                result[
                    "thresholds"
                ][
                    "controlled_fpr10"
                ],
                f"seed{seed}.controlled_fpr10",
            )

            if fpr10_info[
                "derived_from_test"
            ] is not False:
                raise RuntimeError(
                    f"Seed {seed} threshold was test-derived."
                )

            if fpr10_info[
                "modified_on_test"
            ] is not False:
                raise RuntimeError(
                    f"Seed {seed} threshold changed on test."
                )

            fpr10_threshold = float(
                fpr10_info[
                    "threshold"
                ]
            )

            threshold_artifact_path = resolve_repo_path(
                fpr10_info[
                    "threshold_artifact_path"
                ]
            )

            threshold_artifact_sha = require_file_sha(
                path=threshold_artifact_path,
                expected_sha256=str(
                    fpr10_info[
                        "threshold_artifact_sha256"
                    ]
                ),
                label=(
                    f"Seed {seed} threshold artifact"
                ),
            )

            threshold_artifact = load_yaml(
                threshold_artifact_path
            )

            if int(
                threshold_artifact[
                    "checkpoint"
                ][
                    "seed"
                ]
            ) != seed:
                raise RuntimeError(
                    f"Seed {seed} threshold source seed mismatch."
                )

            assert_close(
                label=(
                    f"Seed {seed} frozen threshold"
                ),
                actual=fpr10_threshold,
                expected=float(
                    threshold_artifact[
                        "controlled_fpr10"
                    ][
                        "threshold"
                    ]
                ),
            )

            (
                rows,
                logits,
                targets,
            ) = load_prediction_csv(
                path=predictions_path
            )

            identity = tuple(
                tuple(
                    row[
                        column
                    ]
                    for column
                    in IDENTITY_COLUMNS
                )
                for row
                in rows
            )

            if canonical_identity is None:
                canonical_identity = identity

            elif identity != canonical_identity:
                raise RuntimeError(
                    "Three seed prediction files do not describe the "
                    "same test samples in the same order."
                )

            verify_prediction_derived_columns(
                rows=rows,
                logits=logits,
                fpr10_threshold=fpr10_threshold,
            )

            recomputed = compute_final_binary_evaluation(
                logits=logits,
                targets=targets,
                fpr10_threshold=fpr10_threshold,
                contract=contract,
            )

            verify_result_metrics(
                result=result,
                recomputed=recomputed,
            )

            seed_metrics = scalar_metrics_from_evaluation(
                recomputed
            )

            per_seed_metrics[
                seed
            ] = seed_metrics

            per_seed_output[
                seed
            ] = {
                "source":
                    {
                        "result_path":
                            relative_repo_path(
                                result_path
                            ),

                        "result_sha256":
                            result_sha,

                        "predictions_path":
                            relative_repo_path(
                                predictions_path
                            ),

                        "predictions_sha256":
                            predictions_sha,

                        "threshold_artifact_path":
                            relative_repo_path(
                                threshold_artifact_path
                            ),

                        "threshold_artifact_sha256":
                            threshold_artifact_sha,
                    },

                "machine":
                    result[
                        "machine"
                    ][
                        "id"
                    ],

                "checkpoint":
                    result[
                        "checkpoint"
                    ],

                "thresholds":
                    {
                        "fixed_0_5":
                            0.5,

                        "controlled_fpr10":
                            fpr10_threshold,
                    },

                "metrics":
                    asdict(
                        recomputed
                    ),
            }

            LOGGER.info(
                "[PASS] seed %d | AUROC=%.12f | "
                "fixed HTER=%.12f | FPR10 HTER=%.12f",
                seed,
                recomputed.auroc,
                recomputed
                .fixed_0_5
                .half_total_error_rate,
                recomputed
                .controlled_fpr10
                .half_total_error_rate,
            )

        if canonical_identity is None:
            raise RuntimeError(
                "No prediction identities loaded."
            )

        aggregate = build_aggregate(
            per_seed_metrics
        )

        summary = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_three_seed_final_held_out_test_summary",

            "status":
                "FROZEN_FINAL_BASELINE_TEST_SUMMARY",

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git":
                {
                    "commit_sha":
                        git_commit,

                    "branch":
                        git_branch,
                },

            "scientific_config":
                {
                    "path":
                        relative_repo_path(
                            scientific_path
                        ),

                    "sha256":
                        scientific_sha,
                },

            "evaluation_protocol":
                {
                    "path":
                        relative_repo_path(
                            protocol_path
                        ),

                    "sha256":
                        protocol_sha,
                },

            "final_evaluation_audit":
                {
                    "path":
                        relative_repo_path(
                            audit_path
                        ),

                    "sha256":
                        audit_sha,

                    "status":
                        "PASS",
                },

            "held_out_test":
                {
                    "metadata_path":
                        relative_repo_path(
                            metadata_path
                        ),

                    "metadata_sha256":
                        metadata_sha,

                    "manifest_path":
                        relative_repo_path(
                            manifest_path
                        ),

                    "manifest_sha256":
                        manifest_sha,

                    "sample_count":
                        1385,

                    "bonafide_count":
                        300,

                    "attack_count":
                        1085,

                    "resolution":
                        "r512",
                },

            "seed_policy":
                {
                    "prespecified_seeds":
                        list(
                            SEEDS
                        ),

                    "all_prespecified_seeds_reported":
                        True,

                    "best_test_seed_selected":
                        False,

                    "test_results_used_for_model_selection":
                        False,
                },

            "per_seed":
                per_seed_output,

            "aggregate_mean_min_max":
                aggregate,

            "verification":
                {
                    "prediction_csvs_recomputed":
                        True,

                    "persisted_metrics_reproduced":
                        True,

                    "identical_test_identity_and_order_across_seeds":
                        True,

                    "all_test_image_hashes_had_been_verified_during_runs":
                        True,

                    "images_opened_by_finalizer":
                        False,

                    "models_loaded_by_finalizer":
                        False,

                    "checkpoints_loaded_by_finalizer":
                        False,

                    "inference_performed_by_finalizer":
                        False,

                    "threshold_derived_by_finalizer":
                        False,

                    "threshold_modified_by_finalizer":
                        False,

                    "best_seed_selected":
                        False,
                },
        }

        with output_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                summary,
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
            "Three-seed AUROC mean/min/max = "
            "%.12f / %.12f / %.12f",
            aggregate[
                "overall"
            ][
                "auroc"
            ][
                "mean"
            ],
            aggregate[
                "overall"
            ][
                "auroc"
            ][
                "min"
            ],
            aggregate[
                "overall"
            ][
                "auroc"
            ][
                "max"
            ],
        )

        LOGGER.info(
            "Fixed-0.5 HTER mean/min/max = "
            "%.12f / %.12f / %.12f",
            aggregate[
                "fixed_0_5"
            ][
                "half_total_error_rate"
            ][
                "mean"
            ],
            aggregate[
                "fixed_0_5"
            ][
                "half_total_error_rate"
            ][
                "min"
            ],
            aggregate[
                "fixed_0_5"
            ][
                "half_total_error_rate"
            ][
                "max"
            ],
        )

        LOGGER.info(
            "FPR10 HTER mean/min/max = "
            "%.12f / %.12f / %.12f",
            aggregate[
                "controlled_fpr10"
            ][
                "half_total_error_rate"
            ][
                "mean"
            ],
            aggregate[
                "controlled_fpr10"
            ][
                "half_total_error_rate"
            ][
                "min"
            ],
            aggregate[
                "controlled_fpr10"
            ][
                "half_total_error_rate"
            ][
                "max"
            ],
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] no best test seed selected"
        )

        LOGGER.info(
            "[PASS] no threshold derivation/modification"
        )

        LOGGER.info(
            "[PASS] no image/model/checkpoint access"
        )

        LOGGER.info(
            "Canonical summary: %s",
            output_path,
        )

        LOGGER.info(
            "Canonical summary SHA-256: %s",
            output_sha,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "THREE-SEED FINAL BASELINE SUMMARY: FROZEN"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "THREE-SEED FINAL BASELINE SUMMARY: FAIL"
        )

        if output_path.exists():
            output_path.unlink()

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