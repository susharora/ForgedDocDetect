#!/usr/bin/env python3
"""
Post-hoc diagnostic of the frozen ResNet-18 FantasyID final test.

IMPORTANT
---------
This is NOT model selection.

This tool:
- opens no images;
- loads no model;
- loads no checkpoint;
- performs no inference;
- derives no threshold;
- modifies no threshold;
- selects no seed.

It consumes only the already-frozen three-seed prediction CSVs.

Diagnostics
-----------
1. Overall test reproduction.

2. Attack-family vs all bona-fide:
       digital_3
       facedancer
       textdiffuserft_bfei

   For each family we compute:
       family-vs-bona AUROC
       fixed-0.5 TPR/FNR/FPR/HTER
       frozen-FPR10 TPR/FNR/FPR/HTER
       score summaries

3. Hardware:
       all attacks vs bona-fide within each hardware source.

4. Attack-family x hardware:
       each attack family vs hardware-matched bona-fide.

5. Attack stem exposure:
       project_train_seen_stem
       dev_val_only_stem
       unseen_official_train_stem

   This is especially informative for digital_3 because it distinguishes
   manipulation-method transfer from simple card/stem familiarity.

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "analyze_resnet18_test_subgroups"
)


SEEDS = (
    8,
    9,
    10,
)


ATTACK_VARIANTS = (
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
)


HARDWARE_ORDER = (
    "huawei",
    "iphone15",
    "iphone15pro",
    "scan",
)


STEM_EXPOSURES = (
    "project_train_seen_stem",
    "dev_val_only_stem",
    "unseen_official_train_stem",
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


CSV_COLUMNS = (
    "cohort_type",
    "cohort_id",
    "seed",
    "variant",
    "hardware",
    "stem_exposure",
    "n_attack",
    "n_bonafide",
    "auroc",
    "fixed_tpr",
    "fixed_fnr",
    "fixed_fpr",
    "fixed_hter",
    "fpr10_tpr",
    "fpr10_fnr",
    "fpr10_fpr",
    "fpr10_hter",
    "attack_p_mean",
    "attack_p_q10",
    "attack_p_median",
    "attack_p_q90",
    "bonafide_p_mean",
    "bonafide_p_q10",
    "bonafide_p_median",
    "bonafide_p_q90",
    "attack_margin_median",
    "bonafide_margin_median",
)


# ======================================================================
# Generic helpers
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

    text = str(
        seed
    )

    if text in mapping:
        return mapping[
            text
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
            "Git tree must be clean before post-hoc diagnostic.\n\n"
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
    tolerance: float = 1.0e-13,
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

    LOGGER.setLevel(
        getattr(
            logging,
            str(
                logging_cfg[
                    "level"
                ]
            ).upper(),
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
# Frozen development stem sets
# ======================================================================

def load_manifest_stems(
    *,
    path: Path,
    expected_sha256: str,
    expected_rows: int,
    label: str,
) -> set[str]:

    require_file_sha(
        path=path,
        expected_sha256=expected_sha256,
        label=label,
    )

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
                f"{label} has no header."
            )

        if "file_stem" not in reader.fieldnames:
            raise RuntimeError(
                f"{label} lacks file_stem."
            )

        rows = tuple(
            reader
        )

    if len(
        rows
    ) != expected_rows:
        raise RuntimeError(
            f"{label} row count changed."
        )

    return {
        str(
            row[
                "file_stem"
            ]
        )
        for row
        in rows
    }


def classify_stem_exposure(
    *,
    file_stem: str,
    project_train_stems: set[str],
    dev_val_stems: set[str],
) -> str:

    in_train = (
        file_stem
        in project_train_stems
    )

    in_dev = (
        file_stem
        in dev_val_stems
    )

    if in_train and in_dev:
        raise RuntimeError(
            "Frozen train/dev stem sets overlap."
        )

    if in_train:
        return (
            "project_train_seen_stem"
        )

    if in_dev:
        return (
            "dev_val_only_stem"
        )

    return (
        "unseen_official_train_stem"
    )


# ======================================================================
# Canonical test manifest
# ======================================================================

def load_test_manifest(
    *,
    path: Path,
    expected_sha256: str,
) -> tuple[
    dict[str, str],
    ...,
]:

    require_file_sha(
        path=path,
        expected_sha256=expected_sha256,
        label="Canonical test manifest",
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = tuple(
            reader
        )

    if len(
        rows
    ) != 1385:
        raise RuntimeError(
            "Canonical test manifest must contain 1385 rows."
        )

    return rows


# ======================================================================
# Prediction loading
# ======================================================================

def load_prediction_csv(
    *,
    path: Path,
    expected_sha256: str,
    canonical_test_rows: Sequence[
        Mapping[str, str]
    ],
    fpr10_threshold: float,
) -> tuple[
    tuple[
        dict[str, str],
        ...,
    ],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    require_file_sha(
        path=path,
        expected_sha256=expected_sha256,
        label="Prediction CSV",
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if tuple(
            reader.fieldnames
            or ()
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
            "Prediction CSV must contain 1385 rows."
        )

    logits_values: list[
        tuple[
            float,
            float,
        ]
    ] = []

    targets_values: list[
        int
    ] = []

    for index, (
        row,
        manifest_row,
    ) in enumerate(
        zip(
            rows,
            canonical_test_rows,
            strict=True,
        )
    ):

        if int(
            row[
                "row_index"
            ]
        ) != index:
            raise RuntimeError(
                "Prediction row_index changed."
            )

        identity_pairs = (
            (
                "source_workbook_row",
                "source_workbook_row",
            ),
            (
                "image_path",
                "image_path",
            ),
            (
                "image_sha256",
                "image_sha256",
            ),
            (
                "file_stem",
                "file_stem",
            ),
            (
                "traffic_type",
                "traffic_type",
            ),
            (
                "variant",
                "variant",
            ),
            (
                "hardware_source",
                "hardware_source",
            ),
            (
                "face_db",
                "face_db",
            ),
            (
                "face_id",
                "face_id",
            ),
            (
                "gender",
                "gender",
            ),
        )

        for (
            prediction_column,
            manifest_column,
        ) in identity_pairs:

            if (
                row[
                    prediction_column
                ]
                != manifest_row[
                    manifest_column
                ]
            ):
                raise RuntimeError(
                    "Prediction/test-manifest identity mismatch:\n"
                    f"  row={index}\n"
                    f"  column={prediction_column}"
                )

        target = int(
            row[
                "target"
            ]
        )

        manifest_target = int(
            manifest_row[
                "label"
            ]
        )

        if target != manifest_target:
            raise RuntimeError(
                "Prediction target differs from canonical manifest."
            )

        traffic_type = row[
            "traffic_type"
        ]

        expected_target = (
            0
            if traffic_type == "bonafide"
            else 1
            if traffic_type == "attack"
            else None
        )

        if expected_target is None:
            raise RuntimeError(
                "Unexpected traffic_type."
            )

        if target != expected_target:
            raise RuntimeError(
                "traffic_type/target polarity mismatch."
            )

        if (
            row[
                "verified_image_sha256"
            ]
            != row[
                "image_sha256"
            ]
        ):
            raise RuntimeError(
                "Persisted image-SHA verification mismatch."
            )

        logits_values.append(
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

        targets_values.append(
            target
        )

    logits = torch.tensor(
        logits_values,
        dtype=torch.float32,
        device="cpu",
    )

    targets = torch.tensor(
        targets_values,
        dtype=torch.int64,
        device="cpu",
    )

    probabilities = (
        attack_probability_from_logits(
            logits
        )
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

    fixed_predictions = (
        apply_attack_threshold(
            scores=probabilities,
            threshold=0.5,
        )
    )

    fpr10_predictions = (
        apply_attack_threshold(
            scores=probabilities,
            threshold=fpr10_threshold,
        )
    )

    for index, row in enumerate(
        rows
    ):

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

    return (
        rows,
        logits,
        targets,
        probabilities,
        margins,
    )


# ======================================================================
# Score summaries
# ======================================================================

def score_summary(
    values: torch.Tensor,
) -> dict[str, float]:

    if values.ndim != 1:
        raise RuntimeError(
            "Score summary expects rank-1 tensor."
        )

    if values.numel() == 0:
        raise RuntimeError(
            "Cannot summarize empty score cohort."
        )

    values = values.to(
        dtype=torch.float64,
        device="cpu",
    )

    quantiles = torch.quantile(
        values,
        torch.tensor(
            [
                0.10,
                0.50,
                0.90,
            ],
            dtype=torch.float64,
        ),
    )

    return {
        "mean":
            float(
                values.mean().item()
            ),

        "q10":
            float(
                quantiles[
                    0
                ].item()
            ),

        "median":
            float(
                quantiles[
                    1
                ].item()
            ),

        "q90":
            float(
                quantiles[
                    2
                ].item()
            ),
    }


# ======================================================================
# Cohort computations
# ======================================================================

def binary_cohort_record(
    *,
    cohort_type: str,
    cohort_id: str,
    seed: int,
    variant: str,
    hardware: str,
    stem_exposure: str,
    indices: Sequence[int],
    logits: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    margins: torch.Tensor,
    fpr10_threshold: float,
    contract: Any,
) -> dict[str, Any]:

    index_tensor = torch.tensor(
        list(
            indices
        ),
        dtype=torch.int64,
    )

    cohort_logits = logits[
        index_tensor
    ]

    cohort_targets = targets[
        index_tensor
    ]

    cohort_probabilities = probabilities[
        index_tensor
    ]

    cohort_margins = margins[
        index_tensor
    ]

    n_attack = int(
        (
            cohort_targets
            == 1
        )
        .sum()
        .item()
    )

    n_bonafide = int(
        (
            cohort_targets
            == 0
        )
        .sum()
        .item()
    )

    if (
        n_attack <= 0
        or n_bonafide <= 0
    ):
        raise RuntimeError(
            f"Binary cohort {cohort_id} lacks one class."
        )

    evaluation = (
        compute_final_binary_evaluation(
            logits=cohort_logits,
            targets=cohort_targets,
            fpr10_threshold=fpr10_threshold,
            contract=contract,
        )
    )

    attack_mask = (
        cohort_targets
        == 1
    )

    bonafide_mask = (
        cohort_targets
        == 0
    )

    attack_p = score_summary(
        cohort_probabilities[
            attack_mask
        ]
    )

    bonafide_p = score_summary(
        cohort_probabilities[
            bonafide_mask
        ]
    )

    attack_margin = score_summary(
        cohort_margins[
            attack_mask
        ]
    )

    bonafide_margin = score_summary(
        cohort_margins[
            bonafide_mask
        ]
    )

    fixed = (
        evaluation
        .fixed_0_5
    )

    controlled = (
        evaluation
        .controlled_fpr10
    )

    return {
        "cohort_type":
            cohort_type,

        "cohort_id":
            cohort_id,

        "seed":
            seed,

        "variant":
            variant,

        "hardware":
            hardware,

        "stem_exposure":
            stem_exposure,

        "n_attack":
            n_attack,

        "n_bonafide":
            n_bonafide,

        "auroc":
            float(
                evaluation.auroc
            ),

        "fixed_tpr":
            float(
                fixed.true_positive_rate
            ),

        "fixed_fnr":
            float(
                fixed.false_negative_rate
            ),

        "fixed_fpr":
            float(
                fixed.false_positive_rate
            ),

        "fixed_hter":
            float(
                fixed.half_total_error_rate
            ),

        "fpr10_tpr":
            float(
                controlled.true_positive_rate
            ),

        "fpr10_fnr":
            float(
                controlled.false_negative_rate
            ),

        "fpr10_fpr":
            float(
                controlled.false_positive_rate
            ),

        "fpr10_hter":
            float(
                controlled.half_total_error_rate
            ),

        "attack_p_mean":
            attack_p[
                "mean"
            ],

        "attack_p_q10":
            attack_p[
                "q10"
            ],

        "attack_p_median":
            attack_p[
                "median"
            ],

        "attack_p_q90":
            attack_p[
                "q90"
            ],

        "bonafide_p_mean":
            bonafide_p[
                "mean"
            ],

        "bonafide_p_q10":
            bonafide_p[
                "q10"
            ],

        "bonafide_p_median":
            bonafide_p[
                "median"
            ],

        "bonafide_p_q90":
            bonafide_p[
                "q90"
            ],

        "attack_margin_median":
            attack_margin[
                "median"
            ],

        "bonafide_margin_median":
            bonafide_margin[
                "median"
            ],
    }


def attack_only_cohort_record(
    *,
    cohort_type: str,
    cohort_id: str,
    seed: int,
    variant: str,
    hardware: str,
    stem_exposure: str,
    indices: Sequence[int],
    probabilities: torch.Tensor,
    margins: torch.Tensor,
    fpr10_threshold: float,
) -> dict[str, Any]:

    index_tensor = torch.tensor(
        list(
            indices
        ),
        dtype=torch.int64,
    )

    cohort_probabilities = probabilities[
        index_tensor
    ]

    cohort_margins = margins[
        index_tensor
    ]

    if cohort_probabilities.numel() == 0:
        raise RuntimeError(
            f"Attack-only cohort {cohort_id} is empty."
        )

    fixed_predictions = (
        apply_attack_threshold(
            scores=cohort_probabilities,
            threshold=0.5,
        )
    )

    controlled_predictions = (
        apply_attack_threshold(
            scores=cohort_probabilities,
            threshold=fpr10_threshold,
        )
    )

    n_attack = int(
        cohort_probabilities.numel()
    )

    fixed_tpr = float(
        fixed_predictions
        .to(
            dtype=torch.float64
        )
        .mean()
        .item()
    )

    fpr10_tpr = float(
        controlled_predictions
        .to(
            dtype=torch.float64
        )
        .mean()
        .item()
    )

    p_summary = score_summary(
        cohort_probabilities
    )

    margin_summary = score_summary(
        cohort_margins
    )

    return {
        "cohort_type":
            cohort_type,

        "cohort_id":
            cohort_id,

        "seed":
            seed,

        "variant":
            variant,

        "hardware":
            hardware,

        "stem_exposure":
            stem_exposure,

        "n_attack":
            n_attack,

        "n_bonafide":
            0,

        "auroc":
            None,

        "fixed_tpr":
            fixed_tpr,

        "fixed_fnr":
            (
                1.0
                - fixed_tpr
            ),

        "fixed_fpr":
            None,

        "fixed_hter":
            None,

        "fpr10_tpr":
            fpr10_tpr,

        "fpr10_fnr":
            (
                1.0
                - fpr10_tpr
            ),

        "fpr10_fpr":
            None,

        "fpr10_hter":
            None,

        "attack_p_mean":
            p_summary[
                "mean"
            ],

        "attack_p_q10":
            p_summary[
                "q10"
            ],

        "attack_p_median":
            p_summary[
                "median"
            ],

        "attack_p_q90":
            p_summary[
                "q90"
            ],

        "bonafide_p_mean":
            None,

        "bonafide_p_q10":
            None,

        "bonafide_p_median":
            None,

        "bonafide_p_q90":
            None,

        "attack_margin_median":
            margin_summary[
                "median"
            ],

        "bonafide_margin_median":
            None,
    }


# ======================================================================
# Aggregate mean/min/max
# ======================================================================

AGGREGATE_FIELDS = (
    "auroc",
    "fixed_tpr",
    "fixed_fnr",
    "fixed_fpr",
    "fixed_hter",
    "fpr10_tpr",
    "fpr10_fnr",
    "fpr10_fpr",
    "fpr10_hter",
    "attack_p_mean",
    "attack_p_median",
    "bonafide_p_mean",
    "bonafide_p_median",
    "attack_margin_median",
    "bonafide_margin_median",
)


def mean_min_max(
    values: Sequence[float],
) -> dict[str, float]:

    if len(
        values
    ) != 3:
        raise RuntimeError(
            "Aggregate requires one value from each of three seeds."
        )

    return {
        "mean":
            sum(
                values
            )
            / 3.0,

        "min":
            min(
                values
            ),

        "max":
            max(
                values
            ),
    }


def aggregate_records(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> list[
    dict[str, Any]
]:

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[
            Mapping[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for record in records:

        grouped[
            (
                str(
                    record[
                        "cohort_type"
                    ]
                ),
                str(
                    record[
                        "cohort_id"
                    ]
                ),
            )
        ].append(
            record
        )

    output: list[
        dict[str, Any]
    ] = []

    for (
        cohort_type,
        cohort_id,
    ), cohort_records in sorted(
        grouped.items()
    ):

        if len(
            cohort_records
        ) != 3:
            raise RuntimeError(
                "Every cohort must appear for all three seeds:\n"
                f"  {cohort_type}/{cohort_id}"
            )

        if {
            int(
                record[
                    "seed"
                ]
            )
            for record
            in cohort_records
        } != set(
            SEEDS
        ):
            raise RuntimeError(
                "Cohort seed set changed."
            )

        n_attack_values = {
            int(
                record[
                    "n_attack"
                ]
            )
            for record
            in cohort_records
        }

        n_bonafide_values = {
            int(
                record[
                    "n_bonafide"
                ]
            )
            for record
            in cohort_records
        }

        if (
            len(
                n_attack_values
            ) != 1
            or len(
                n_bonafide_values
            ) != 1
        ):
            raise RuntimeError(
                "Cohort support changed across seeds."
            )

        template = cohort_records[
            0
        ]

        aggregate = {
            "cohort_type":
                cohort_type,

            "cohort_id":
                cohort_id,

            "variant":
                template[
                    "variant"
                ],

            "hardware":
                template[
                    "hardware"
                ],

            "stem_exposure":
                template[
                    "stem_exposure"
                ],

            "n_attack":
                next(
                    iter(
                        n_attack_values
                    )
                ),

            "n_bonafide":
                next(
                    iter(
                        n_bonafide_values
                    )
                ),

            "metrics":
                {},
        }

        for field in AGGREGATE_FIELDS:

            values = [
                record[
                    field
                ]
                for record
                in cohort_records
            ]

            if all(
                value is None
                for value
                in values
            ):
                aggregate[
                    "metrics"
                ][
                    field
                ] = None

                continue

            if any(
                value is None
                for value
                in values
            ):
                raise RuntimeError(
                    "Metric availability differs across seeds:\n"
                    f"  cohort={cohort_id}\n"
                    f"  metric={field}"
                )

            aggregate[
                "metrics"
            ][
                field
            ] = mean_min_max(
                [
                    float(
                        value
                    )
                    for value
                    in values
                ]
            )

        output.append(
            aggregate
        )

    return output


# ======================================================================
# CSV persistence
# ======================================================================

def csv_value(
    value: Any,
) -> Any:

    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        return format(
            value,
            ".17g",
        )

    return value


def write_records_csv(
    *,
    path: Path,
    records: Sequence[
        Mapping[str, Any]
    ],
) -> str:

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )

        writer.writeheader()

        for record in records:

            writer.writerow(
                {
                    column:
                        csv_value(
                            record[
                                column
                            ]
                        )
                    for column
                    in CSV_COLUMNS
                }
            )

    return sha256_file(
        path
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc ResNet-18 final-test subgroup diagnostic."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "tools/"
            "analyze_resnet18_test_subgroups_config.yaml"
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

    # ==============================================================
    # Frozen final summary
    # ==============================================================

    summary_cfg = require_mapping(
        config[
            "frozen_final_summary"
        ],
        "frozen_final_summary",
    )

    summary_path = resolve_repo_path(
        summary_cfg[
            "path"
        ]
    )

    summary_sha = require_file_sha(
        path=summary_path,
        expected_sha256=str(
            summary_cfg[
                "sha256"
            ]
        ),
        label="Frozen three-seed final summary",
    )

    summary = load_yaml(
        summary_path
    )

    if summary.get(
        "status"
    ) != "FROZEN_FINAL_BASELINE_TEST_SUMMARY":
        raise RuntimeError(
            "Three-seed final baseline summary is not frozen."
        )

    seed_policy = require_mapping(
        summary[
            "seed_policy"
        ],
        "summary.seed_policy",
    )

    if seed_policy[
        "prespecified_seeds"
    ] != [
        8,
        9,
        10,
    ]:
        raise RuntimeError(
            "Frozen final seed set changed."
        )

    if seed_policy[
        "best_test_seed_selected"
    ] is not False:
        raise RuntimeError(
            "Frozen summary reports best-test-seed selection."
        )

    # ==============================================================
    # Frozen scientific config + development stem exposure
    # ==============================================================

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

    scientific = load_yaml(
        scientific_path
    )

    frozen_split = require_mapping(
        scientific[
            "data"
        ][
            "frozen_split"
        ],
        "scientific.data.frozen_split",
    )

    project_train_cfg = require_mapping(
        frozen_split[
            "project_train"
        ],
        "frozen_split.project_train",
    )

    dev_val_cfg = require_mapping(
        frozen_split[
            "dev_val"
        ],
        "frozen_split.dev_val",
    )

    project_train_path = resolve_repo_path(
        project_train_cfg[
            "path"
        ]
    )

    dev_val_path = resolve_repo_path(
        dev_val_cfg[
            "path"
        ]
    )

    project_train_stems = load_manifest_stems(
        path=project_train_path,
        expected_sha256=str(
            project_train_cfg[
                "sha256"
            ]
        ),
        expected_rows=int(
            project_train_cfg[
                "images"
            ]
        ),
        label="project_train manifest",
    )

    dev_val_stems = load_manifest_stems(
        path=dev_val_path,
        expected_sha256=str(
            dev_val_cfg[
                "sha256"
            ]
        ),
        expected_rows=int(
            dev_val_cfg[
                "images"
            ]
        ),
        label="dev_val manifest",
    )

    if len(
        project_train_stems
    ) != 160:
        raise RuntimeError(
            "Expected 160 unique project_train file stems."
        )

    if len(
        dev_val_stems
    ) != 51:
        raise RuntimeError(
            "Expected 51 unique dev_val file stems."
        )

    if (
        project_train_stems
        & dev_val_stems
    ):
        raise RuntimeError(
            "project_train/dev_val file-stem overlap detected."
        )

    if len(
        project_train_stems
        | dev_val_stems
    ) != 211:
        raise RuntimeError(
            "Frozen development stem union must contain 211 stems."
        )

    # ==============================================================
    # Evaluation protocol from frozen summary
    # ==============================================================

    protocol_info = require_mapping(
        summary[
            "evaluation_protocol"
        ],
        "summary.evaluation_protocol",
    )

    protocol_path = resolve_repo_path(
        protocol_info[
            "path"
        ]
    )

    require_file_sha(
        path=protocol_path,
        expected_sha256=str(
            protocol_info[
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

    # ==============================================================
    # Canonical test manifest metadata
    # ==============================================================

    held_out_info = require_mapping(
        summary[
            "held_out_test"
        ],
        "summary.held_out_test",
    )

    if (
        int(
            held_out_info[
                "sample_count"
            ]
        ),
        int(
            held_out_info[
                "bonafide_count"
            ]
        ),
        int(
            held_out_info[
                "attack_count"
            ]
        ),
    ) != (
        1385,
        300,
        1085,
    ):
        raise RuntimeError(
            "Frozen held-out population changed."
        )

    test_manifest_path = resolve_repo_path(
        held_out_info[
            "manifest_path"
        ]
    )

    canonical_test_rows = load_test_manifest(
        path=test_manifest_path,
        expected_sha256=str(
            held_out_info[
                "manifest_sha256"
            ]
        ),
    )

    # ==============================================================
    # Add frozen stem-exposure labels once.
    # ==============================================================

    metadata: list[
        dict[str, str]
    ] = []

    exposure_counts: dict[
        str,
        dict[
            str,
            int,
        ],
    ] = defaultdict(
        lambda: defaultdict(
            int
        )
    )

    for row in canonical_test_rows:

        prepared = dict(
            row
        )

        exposure = classify_stem_exposure(
            file_stem=str(
                row[
                    "file_stem"
                ]
            ),
            project_train_stems=project_train_stems,
            dev_val_stems=dev_val_stems,
        )

        prepared[
            "stem_exposure"
        ] = exposure

        metadata.append(
            prepared
        )

        if row[
            "traffic_type"
        ] == "attack":

            exposure_counts[
                row[
                    "variant"
                ]
            ][
                exposure
            ] += 1

    observed_variants = {
        row[
            "variant"
        ]
        for row
        in metadata
        if row[
            "traffic_type"
        ] == "attack"
    }

    if observed_variants != set(
        ATTACK_VARIANTS
    ):
        raise RuntimeError(
            "Frozen attack-family set changed."
        )

    observed_hardware = {
        row[
            "hardware_source"
        ]
        for row
        in metadata
    }

    if observed_hardware != set(
        HARDWARE_ORDER
    ):
        raise RuntimeError(
            "Frozen hardware set changed."
        )

    # ==============================================================
    # Only now create diagnostic log/output paths.
    # ==============================================================

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

    yaml_path = (
        output_directory
        / str(
            output_cfg[
                "yaml_filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    csv_path = (
        output_directory
        / str(
            output_cfg[
                "csv_filename"
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
            "RESNET-18 POST-HOC FINAL-TEST SUBGROUP DIAGNOSTIC"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            git_commit,
        )

        LOGGER.info(
            "[PASS] frozen final baseline summary SHA-256 = %s",
            summary_sha,
        )

        LOGGER.info(
            "[PASS] scientific config SHA-256 = %s",
            scientific_sha,
        )

        LOGGER.info(
            "[PASS] development stems = "
            "160 project_train + 51 dev_val = 211"
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "--- Attack stem-exposure counts ---"
        )

        for variant in ATTACK_VARIANTS:

            LOGGER.info(
                "%s | train_seen=%d | dev_only=%d | unseen=%d",
                variant,
                exposure_counts[
                    variant
                ][
                    "project_train_seen_stem"
                ],
                exposure_counts[
                    variant
                ][
                    "dev_val_only_stem"
                ],
                exposure_counts[
                    variant
                ][
                    "unseen_official_train_stem"
                ],
            )

        # ==========================================================
        # Three-seed analysis
        # ==========================================================

        all_records: list[
            dict[str, Any]
        ] = []

        family_reconciliation: dict[
            int,
            dict[str, float]
        ] = {}

        per_seed_summary = require_mapping(
            summary[
                "per_seed"
            ],
            "summary.per_seed",
        )

        canonical_prediction_identity: tuple[
            tuple[
                str,
                ...,
            ],
            ...,
        ] | None = None

        for seed in SEEDS:

            seed_summary = require_mapping(
                seed_value(
                    per_seed_summary,
                    seed,
                ),
                f"summary.per_seed.{seed}",
            )

            source_info = require_mapping(
                seed_summary[
                    "source"
                ],
                f"seed{seed}.source",
            )

            result_path = resolve_repo_path(
                source_info[
                    "result_path"
                ]
            )

            require_file_sha(
                path=result_path,
                expected_sha256=str(
                    source_info[
                        "result_sha256"
                    ]
                ),
                label=(
                    f"Seed {seed} result"
                ),
            )

            result = load_yaml(
                result_path
            )

            if result.get(
                "status"
            ) != "FINAL_HELD_OUT_TEST_EVALUATION_COMPLETE":
                raise RuntimeError(
                    f"Seed {seed} result is not final."
                )

            predictions_path = resolve_repo_path(
                source_info[
                    "predictions_path"
                ]
            )

            threshold = float(
                seed_summary[
                    "thresholds"
                ][
                    "controlled_fpr10"
                ]
            )

            (
                rows,
                logits,
                targets,
                probabilities,
                margins,
            ) = load_prediction_csv(
                path=predictions_path,
                expected_sha256=str(
                    source_info[
                        "predictions_sha256"
                    ]
                ),
                canonical_test_rows=canonical_test_rows,
                fpr10_threshold=threshold,
            )

            identity = tuple(
                (
                    row[
                        "image_path"
                    ],
                    row[
                        "image_sha256"
                    ],
                    row[
                        "traffic_type"
                    ],
                    row[
                        "variant"
                    ],
                    row[
                        "hardware_source"
                    ],
                    row[
                        "target"
                    ],
                )
                for row
                in rows
            )

            if canonical_prediction_identity is None:
                canonical_prediction_identity = identity

            elif identity != canonical_prediction_identity:
                raise RuntimeError(
                    "Prediction identities differ between seeds."
                )

            # ------------------------------------------------------
            # Full-test reproduction.
            # ------------------------------------------------------

            full_evaluation = (
                compute_final_binary_evaluation(
                    logits=logits,
                    targets=targets,
                    fpr10_threshold=threshold,
                    contract=contract,
                )
            )

            assert_close(
                label=(
                    f"seed {seed} frozen overall AUROC"
                ),
                actual=float(
                    full_evaluation.auroc
                ),
                expected=float(
                    seed_summary[
                        "metrics"
                    ][
                        "auroc"
                    ]
                ),
            )

            all_indices = list(
                range(
                    1385
                )
            )

            all_records.append(
                binary_cohort_record(
                    cohort_type="overall",
                    cohort_id="all_attacks_vs_all_bonafide",
                    seed=seed,
                    variant="ALL_ATTACK",
                    hardware="ALL",
                    stem_exposure="ALL",
                    indices=all_indices,
                    logits=logits,
                    targets=targets,
                    probabilities=probabilities,
                    margins=margins,
                    fpr10_threshold=threshold,
                    contract=contract,
                )
            )

            LOGGER.info(
                ""
            )

            LOGGER.info(
                "Seed %d overall | AUROC=%.12f | "
                "fixed_TPR=%.12f | FPR10_TPR=%.12f",
                seed,
                full_evaluation.auroc,
                full_evaluation
                .fixed_0_5
                .true_positive_rate,
                full_evaluation
                .controlled_fpr10
                .true_positive_rate,
            )

            # ------------------------------------------------------
            # Family vs all bona-fide.
            #
            # Because every family is compared with the SAME 300
            # bona-fide records, overall AUROC is exactly the
            # attack-count-weighted average of family AUROCs.
            # ------------------------------------------------------

            bona_indices = [
                index
                for index, row
                in enumerate(
                    metadata
                )
                if row[
                    "traffic_type"
                ] == "bonafide"
            ]

            family_records_for_seed: list[
                dict[str, Any]
            ] = []

            for variant in ATTACK_VARIANTS:

                attack_indices = [
                    index
                    for index, row
                    in enumerate(
                        metadata
                    )
                    if (
                        row[
                            "traffic_type"
                        ] == "attack"
                        and row[
                            "variant"
                        ] == variant
                    )
                ]

                cohort_indices = sorted(
                    attack_indices
                    + bona_indices
                )

                record = binary_cohort_record(
                    cohort_type="attack_family_vs_all_bonafide",
                    cohort_id=(
                        f"{variant}_vs_all_bonafide"
                    ),
                    seed=seed,
                    variant=variant,
                    hardware="ALL",
                    stem_exposure="ALL",
                    indices=cohort_indices,
                    logits=logits,
                    targets=targets,
                    probabilities=probabilities,
                    margins=margins,
                    fpr10_threshold=threshold,
                    contract=contract,
                )

                family_records_for_seed.append(
                    record
                )

                all_records.append(
                    record
                )

                LOGGER.info(
                    "Seed %d family=%s | n=%d | AUROC=%.12f | "
                    "fixed_TPR=%.12f | FPR10_TPR=%.12f | "
                    "median_p_attack=%.12f",
                    seed,
                    variant,
                    record[
                        "n_attack"
                    ],
                    record[
                        "auroc"
                    ],
                    record[
                        "fixed_tpr"
                    ],
                    record[
                        "fpr10_tpr"
                    ],
                    record[
                        "attack_p_median"
                    ],
                )

            weighted_family_auroc = (
                sum(
                    float(
                        record[
                            "n_attack"
                        ]
                    )
                    * float(
                        record[
                            "auroc"
                        ]
                    )
                    for record
                    in family_records_for_seed
                )
                / 1085.0
            )

            weighted_fixed_tpr = (
                sum(
                    float(
                        record[
                            "n_attack"
                        ]
                    )
                    * float(
                        record[
                            "fixed_tpr"
                        ]
                    )
                    for record
                    in family_records_for_seed
                )
                / 1085.0
            )

            weighted_fpr10_tpr = (
                sum(
                    float(
                        record[
                            "n_attack"
                        ]
                    )
                    * float(
                        record[
                            "fpr10_tpr"
                        ]
                    )
                    for record
                    in family_records_for_seed
                )
                / 1085.0
            )

            assert_close(
                label=(
                    f"seed {seed} family-weighted AUROC reconstruction"
                ),
                actual=weighted_family_auroc,
                expected=float(
                    full_evaluation.auroc
                ),
            )

            assert_close(
                label=(
                    f"seed {seed} family-weighted fixed TPR"
                ),
                actual=weighted_fixed_tpr,
                expected=float(
                    full_evaluation
                    .fixed_0_5
                    .true_positive_rate
                ),
            )

            assert_close(
                label=(
                    f"seed {seed} family-weighted FPR10 TPR"
                ),
                actual=weighted_fpr10_tpr,
                expected=float(
                    full_evaluation
                    .controlled_fpr10
                    .true_positive_rate
                ),
            )

            family_reconciliation[
                seed
            ] = {
                "overall_auroc":
                    float(
                        full_evaluation.auroc
                    ),

                "family_weighted_auroc":
                    weighted_family_auroc,

                "overall_fixed_tpr":
                    float(
                        full_evaluation
                        .fixed_0_5
                        .true_positive_rate
                    ),

                "family_weighted_fixed_tpr":
                    weighted_fixed_tpr,

                "overall_fpr10_tpr":
                    float(
                        full_evaluation
                        .controlled_fpr10
                        .true_positive_rate
                    ),

                "family_weighted_fpr10_tpr":
                    weighted_fpr10_tpr,
            }

            # ------------------------------------------------------
            # Hardware-specific binary performance.
            # ------------------------------------------------------

            for hardware in HARDWARE_ORDER:

                hardware_indices = [
                    index
                    for index, row
                    in enumerate(
                        metadata
                    )
                    if row[
                        "hardware_source"
                    ] == hardware
                ]

                hardware_targets = targets[
                    torch.tensor(
                        hardware_indices,
                        dtype=torch.int64,
                    )
                ]

                if (
                    bool(
                        (
                            hardware_targets
                            == 0
                        ).any()
                    )
                    and bool(
                        (
                            hardware_targets
                            == 1
                        ).any()
                    )
                ):

                    all_records.append(
                        binary_cohort_record(
                            cohort_type="hardware",
                            cohort_id=(
                                f"all_attacks_vs_bonafide"
                                f"|{hardware}"
                            ),
                            seed=seed,
                            variant="ALL_ATTACK",
                            hardware=hardware,
                            stem_exposure="ALL",
                            indices=hardware_indices,
                            logits=logits,
                            targets=targets,
                            probabilities=probabilities,
                            margins=margins,
                            fpr10_threshold=threshold,
                            contract=contract,
                        )
                    )

            # ------------------------------------------------------
            # Family x hardware, compared with hardware-matched bona.
            # ------------------------------------------------------

            for variant in ATTACK_VARIANTS:

                for hardware in HARDWARE_ORDER:

                    attack_indices = [
                        index
                        for index, row
                        in enumerate(
                            metadata
                        )
                        if (
                            row[
                                "traffic_type"
                            ] == "attack"
                            and row[
                                "variant"
                            ] == variant
                            and row[
                                "hardware_source"
                            ] == hardware
                        )
                    ]

                    bona_hardware_indices = [
                        index
                        for index, row
                        in enumerate(
                            metadata
                        )
                        if (
                            row[
                                "traffic_type"
                            ] == "bonafide"
                            and row[
                                "hardware_source"
                            ] == hardware
                        )
                    ]

                    if (
                        attack_indices
                        and bona_hardware_indices
                    ):

                        all_records.append(
                            binary_cohort_record(
                                cohort_type="attack_family_x_hardware",
                                cohort_id=(
                                    f"{variant}"
                                    f"|{hardware}"
                                ),
                                seed=seed,
                                variant=variant,
                                hardware=hardware,
                                stem_exposure="ALL",
                                indices=sorted(
                                    attack_indices
                                    + bona_hardware_indices
                                ),
                                logits=logits,
                                targets=targets,
                                probabilities=probabilities,
                                margins=margins,
                                fpr10_threshold=threshold,
                                contract=contract,
                            )
                        )

            # ------------------------------------------------------
            # Attack-only stem exposure.
            #
            # No AUROC is defined here because these cohorts contain
            # attacks only. We report detection rate and score summary.
            # ------------------------------------------------------

            for variant in (
                "ALL_ATTACK",
                *ATTACK_VARIANTS,
            ):

                for exposure in STEM_EXPOSURES:

                    indices = [
                        index
                        for index, row
                        in enumerate(
                            metadata
                        )
                        if (
                            row[
                                "traffic_type"
                            ] == "attack"
                            and row[
                                "stem_exposure"
                            ] == exposure
                            and (
                                variant == "ALL_ATTACK"
                                or row[
                                    "variant"
                                ] == variant
                            )
                        )
                    ]

                    if not indices:
                        continue

                    all_records.append(
                        attack_only_cohort_record(
                            cohort_type="attack_stem_exposure",
                            cohort_id=(
                                f"{variant}"
                                f"|{exposure}"
                            ),
                            seed=seed,
                            variant=variant,
                            hardware="ALL",
                            stem_exposure=exposure,
                            indices=indices,
                            probabilities=probabilities,
                            margins=margins,
                            fpr10_threshold=threshold,
                        )
                    )

        # ==========================================================
        # Aggregate across seeds without selecting any seed.
        # ==========================================================

        aggregate = aggregate_records(
            all_records
        )

        csv_sha = write_records_csv(
            path=csv_path,
            records=all_records,
        )

        diagnostic = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_final_test_posthoc_subgroup_diagnostic",

            "status":
                "POST_HOC_DIAGNOSTIC_NOT_MODEL_SELECTION",

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

            "frozen_baseline":
                {
                    "path":
                        relative_repo_path(
                            summary_path
                        ),

                    "sha256":
                        summary_sha,

                    "status":
                        "FROZEN_FINAL_BASELINE_TEST_SUMMARY",
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

            "development_stem_reference":
                {
                    "project_train_manifest":
                        {
                            "path":
                                relative_repo_path(
                                    project_train_path
                                ),

                            "sha256":
                                project_train_cfg[
                                    "sha256"
                                ],

                            "unique_file_stems":
                                160,
                        },

                    "dev_val_manifest":
                        {
                            "path":
                                relative_repo_path(
                                    dev_val_path
                                ),

                            "sha256":
                                dev_val_cfg[
                                    "sha256"
                                ],

                            "unique_file_stems":
                                51,
                        },

                    "overlap":
                        0,

                    "union":
                        211,
                },

            "attack_stem_exposure_counts":
                {
                    variant:
                        {
                            exposure:
                                int(
                                    exposure_counts[variant][exposure]
                                )
                            for exposure
                            in STEM_EXPOSURES
                        }
                    for variant
                    in ATTACK_VARIANTS
                },

            "family_partition_reconciliation":
                family_reconciliation,

            "per_seed_cohorts":
                all_records,

            "aggregate_mean_min_max":
                aggregate,

            "table":
                {
                    "path":
                        relative_repo_path(
                            csv_path
                        ),

                    "sha256":
                        csv_sha,

                    "rows":
                        len(
                            all_records
                        ),
                },

            "scientific_boundaries":
                {
                    "post_hoc":
                        True,

                    "used_for_model_selection":
                        False,

                    "used_for_checkpoint_selection":
                        False,

                    "used_for_threshold_selection":
                        False,

                    "best_seed_selected":
                        False,

                    "threshold_derived":
                        False,

                    "threshold_modified":
                        False,

                    "images_opened":
                        False,

                    "model_loaded":
                        False,

                    "checkpoint_loaded":
                        False,

                    "inference_performed":
                        False,

                    "prediction_files_only":
                        True,
                },
        }

        with yaml_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                diagnostic,
                file,
                sort_keys=False,
            )

        yaml_sha = sha256_file(
            yaml_path
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] family AUROC decomposition exactly "
            "reconstructs overall AUROC for all three seeds"
        )

        LOGGER.info(
            "[PASS] family attack-recall decomposition exactly "
            "reconstructs overall attack recall"
        )

        LOGGER.info(
            "[PASS] diagnostic uses frozen prediction files only"
        )

        LOGGER.info(
            "[PASS] images/models/checkpoints opened: FALSE"
        )

        LOGGER.info(
            "[PASS] threshold derivation/modification: FALSE"
        )

        LOGGER.info(
            "[PASS] model/seed selection: FALSE"
        )

        LOGGER.info(
            "Diagnostic CSV: %s",
            csv_path,
        )

        LOGGER.info(
            "Diagnostic CSV SHA-256: %s",
            csv_sha,
        )

        LOGGER.info(
            "Diagnostic YAML: %s",
            yaml_path,
        )

        LOGGER.info(
            "Diagnostic YAML SHA-256: %s",
            yaml_sha,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "POST-HOC SUBGROUP DIAGNOSTIC: COMPLETE"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "POST-HOC SUBGROUP DIAGNOSTIC: FAIL"
        )

        for path in (
            csv_path,
            yaml_path,
        ):

            if path.exists():
                path.unlink()

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