#!/usr/bin/env python3
"""
Audit the frozen project_train / dev_val card composition by
Tech-1 derived stem_family.

Purpose
-------
Before using dev_val for transfer-learning model selection, verify that:

1. the frozen card manifest is unchanged;
2. it contains exactly 211 unique cards;
3. project_train / dev_val contain exactly 160 / 51 cards;
4. the expected ten Tech-1 stem families are present;
5. every family occurs in both project_train and dev_val;
6. the 51-card dev_val allocation exactly matches the proportional
   largest-remainder family quota implied by the full 211-card pool.

Important terminology
---------------------
This audit intentionally uses `stem_family`.

Tech-1 defined it conservatively as the file_stem prefix before the
first "-" and explicitly did not claim that this field is necessarily
a formal document-template identifier.

This tool is read-only.

No image data are decoded.
No dev labels are used for model training.
Held-out test is not accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from src.config import (
    load_experiment_config,
    load_machine_config,
)


REQUIRED_CARD_COLUMNS = {
    "file_stem",
    "stem_family",
    "source_image_count",
    "project_role",
}


# ======================================================================
# Generic utilities
# ======================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
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
    ) as f:

        value = (
            yaml.safe_load(
                f
            )
            or {}
        )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Top level of YAML must be a mapping: "
            f"{path}"
        )

    return value


def git_commit_sha() -> str:

    result = subprocess.run(
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

    return result.stdout.strip()


def require_clean_git() -> str:
    """
    Check cleanliness before this audit creates its own log.
    """

    commit = git_commit_sha()

    result = subprocess.run(
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
        result.stdout
        .strip()
    )

    if status:
        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit or stash outstanding changes before "
            "running this audit.\n\n"
            f"{status}"
        )

    return commit


# ======================================================================
# Logging
# ======================================================================

def configure_outputs(
    tool_cfg: dict[str, Any],
) -> tuple[
    logging.Logger,
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

    logging_cfg = tool_cfg[
        "logging"
    ]

    output_cfg = tool_cfg[
        "output"
    ]

    log_dir = resolve_repo_path(
        logging_cfg[
            "directory"
        ]
    )

    output_dir = resolve_repo_path(
        output_cfg[
            "directory"
        ]
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
        / logging_cfg[
            "filename"
        ].format(
            timestamp=timestamp
        )
    )

    output_path = (
        output_dir
        / output_cfg[
            "filename"
        ].format(
            timestamp=timestamp
        )
    )

    level = getattr(
        logging,
        str(
            logging_cfg[
                "level"
            ]
        ).upper(),
    )

    logger = logging.getLogger(
        "audit_split_stem_family"
    )

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(
        level
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

    logger.addHandler(
        handler
    )

    return (
        logger,
        log_path,
        output_path,
    )


# ======================================================================
# Frozen Tech-1 provenance
# ======================================================================

def run_frozen_provenance_gate(
    *,
    experiment_path: Path,
    machine_path: Path,
    logger: logging.Logger,
) -> None:

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

    for line in (
        result.stdout
        .splitlines()
    ):
        logger.info(
            "PROVENANCE_GATE | %s",
            line,
        )

    for line in (
        result.stderr
        .splitlines()
    ):
        logger.error(
            "PROVENANCE_GATE_STDERR | %s",
            line,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen Tech-1 provenance gate failed."
        )


# ======================================================================
# Largest-remainder proportional quota
# ======================================================================

def largest_remainder_quota(
    counts: dict[str, int],
    target_total: int,
) -> dict[
    str,
    dict[str, float | int],
]:
    """
    Allocate target_total proportionally using Hamilton /
    largest-remainder allocation.

    For family f:

        ideal_f =
            source_count_f
            * target_total
            / total_source_cards

        base_f = floor(ideal_f)

    Remaining slots are assigned to the largest fractional remainders.

    Ties are resolved deterministically by family name.
    """

    if target_total <= 0:
        raise ValueError(
            "target_total must be positive."
        )

    source_total = sum(
        counts.values()
    )

    if source_total <= 0:
        raise ValueError(
            "Source family counts are empty."
        )

    rows: dict[
        str,
        dict[str, float | int],
    ] = {}

    allocated = 0

    for family in sorted(
        counts
    ):

        source_count = int(
            counts[
                family
            ]
        )

        ideal = (
            source_count
            * target_total
            / source_total
        )

        base = int(
            ideal
            // 1
        )

        remainder = (
            ideal
            - base
        )

        rows[
            family
        ] = {
            "source_count":
                source_count,

            "ideal_dev_count":
                ideal,

            "base_dev_count":
                base,

            "fractional_remainder":
                remainder,

            "target_dev_count":
                base,
        }

        allocated += base

    remaining = (
        target_total
        - allocated
    )

    if remaining < 0:
        raise RuntimeError(
            "Quota base allocation exceeds target."
        )

    ranked = sorted(
        rows,
        key=lambda family: (
            -float(
                rows[
                    family
                ][
                    "fractional_remainder"
                ]
            ),
            family,
        ),
    )

    for family in ranked[
        :remaining
    ]:

        rows[
            family
        ][
            "target_dev_count"
        ] = (
            int(
                rows[
                    family
                ][
                    "target_dev_count"
                ]
            )
            + 1
        )

    target_sum = sum(
        int(
            row[
                "target_dev_count"
            ]
        )
        for row
        in rows.values()
    )

    if target_sum != target_total:
        raise RuntimeError(
            "Largest-remainder quota failed to reconcile:\n"
            f"  expected={target_total}\n"
            f"  actual={target_sum}"
        )

    return rows


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen project_train/dev_val "
            "stem-family stratification."
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
            "audit_split_stem_family_config.yaml"
        ),
    )

    args = parser.parse_args()

    tool_cfg_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    # Must occur before logger creates an untracked file.
    commit_sha = require_clean_git()

    (
        logger,
        log_path,
        output_path,
    ) = configure_outputs(
        tool_cfg
    )

    try:

        experiment_cfg, experiment_path = (
            load_experiment_config(
                args.config
            )
        )

        _, machine_path = (
            load_machine_config(
                args.machine_config,
                required=True,
            )
        )

        audit_cfg = tool_cfg[
            "audit"
        ]

        expected_total = int(
            audit_cfg[
                "expected_cards_total"
            ]
        )

        expected_train = int(
            audit_cfg[
                "expected_project_train_cards"
            ]
        )

        expected_dev = int(
            audit_cfg[
                "expected_dev_val_cards"
            ]
        )

        expected_images_per_card = int(
            audit_cfg[
                "expected_images_per_card"
            ]
        )

        expected_families = {
            str(
                value
            ).casefold()
            for value
            in audit_cfg[
                "expected_stem_families"
            ]
        }

        if (
            audit_cfg[
                "quota_method"
            ]
            != "largest_remainder"
        ):
            raise ValueError(
                "This audit currently supports only "
                "quota_method=largest_remainder."
            )

        logger.info(
            "=" * 72
        )

        logger.info(
            "FROZEN TRAIN/DEV STEM-FAMILY AUDIT"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Git commit: %s",
            commit_sha,
        )

        logger.info(
            "Script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "Audit config SHA-256: %s",
            sha256_file(
                tool_cfg_path
            ),
        )

        logger.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        logger.info(
            "Terminology: auditing Tech-1 "
            "'stem_family', not asserting formal template identity."
        )

        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        # ----------------------------------------------------------
        # Frozen upstream validation.
        # ----------------------------------------------------------

        run_frozen_provenance_gate(
            experiment_path=experiment_path,
            machine_path=machine_path,
            logger=logger,
        )

        logger.info(
            "[PASS] Frozen Tech-1 provenance gate"
        )

        cards_cfg = (
            experiment_cfg[
                "data"
            ][
                "frozen_split"
            ][
                "cards"
            ]
        )

        cards_path = resolve_repo_path(
            cards_cfg[
                "path"
            ]
        )

        expected_cards_sha = cards_cfg[
            "sha256"
        ]

        actual_cards_sha = sha256_file(
            cards_path
        )

        if (
            actual_cards_sha
            != expected_cards_sha
        ):
            raise RuntimeError(
                "Frozen cards manifest SHA-256 mismatch:\n"
                f"  expected={expected_cards_sha}\n"
                f"  actual={actual_cards_sha}"
            )

        cards_df = pd.read_csv(
            cards_path
        )

        missing_columns = (
            REQUIRED_CARD_COLUMNS
            - set(
                cards_df.columns
            )
        )

        if missing_columns:
            raise RuntimeError(
                "Frozen card manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        logger.info(
            "Frozen cards manifest: %s",
            cards_path,
        )

        logger.info(
            "Frozen cards SHA-256: %s",
            actual_cards_sha,
        )

        # ----------------------------------------------------------
        # Basic card contracts.
        # ----------------------------------------------------------

        if len(
            cards_df
        ) != expected_total:
            raise RuntimeError(
                "Unexpected frozen card-row count:\n"
                f"  expected={expected_total}\n"
                f"  actual={len(cards_df)}"
            )

        unique_cards = (
            cards_df[
                "file_stem"
            ]
            .nunique()
        )

        if unique_cards != expected_total:
            raise RuntimeError(
                "file_stem is not unique at card level:\n"
                f"  expected={expected_total}\n"
                f"  actual={unique_cards}"
            )

        if cards_df[
            "file_stem"
        ].duplicated().any():
            raise RuntimeError(
                "Duplicate file_stem rows exist in cards manifest."
            )

        bad_image_counts = cards_df[
            cards_df[
                "source_image_count"
            ]
            != expected_images_per_card
        ]

        if not bad_image_counts.empty:
            raise RuntimeError(
                "One or more cards do not contain the expected "
                f"{expected_images_per_card} source images."
            )

        roles = set(
            cards_df[
                "project_role"
            ]
            .astype(str)
            .unique()
        )

        expected_roles = {
            "project_train",
            "dev_val",
        }

        if roles != expected_roles:
            raise RuntimeError(
                "Unexpected project_role values:\n"
                f"  expected={sorted(expected_roles)}\n"
                f"  actual={sorted(roles)}"
            )

        role_counts = (
            cards_df[
                "project_role"
            ]
            .value_counts()
            .to_dict()
        )

        if int(
            role_counts.get(
                "project_train",
                0,
            )
        ) != expected_train:
            raise RuntimeError(
                "project_train card count mismatch."
            )

        if int(
            role_counts.get(
                "dev_val",
                0,
            )
        ) != expected_dev:
            raise RuntimeError(
                "dev_val card count mismatch."
            )

        logger.info(
            "[PASS] Card counts: "
            "total=%d project_train=%d dev_val=%d",
            expected_total,
            expected_train,
            expected_dev,
        )

        # ----------------------------------------------------------
        # Stem-family normalization and frozen family contract.
        # ----------------------------------------------------------

        cards_df[
            "stem_family"
        ] = (
            cards_df[
                "stem_family"
            ]
            .astype(str)
            .str.strip()
            .str.casefold()
        )

        actual_families = set(
            cards_df[
                "stem_family"
            ]
            .unique()
        )

        if (
            actual_families
            != expected_families
        ):
            raise RuntimeError(
                "Frozen stem-family set mismatch:\n"
                f"  expected={sorted(expected_families)}\n"
                f"  actual={sorted(actual_families)}"
            )

        logger.info(
            "[PASS] Expected stem-family set: %d families",
            len(
                actual_families
            ),
        )

        # ----------------------------------------------------------
        # Combined source-family distribution.
        # ----------------------------------------------------------

        source_counts_series = (
            cards_df[
                "stem_family"
            ]
            .value_counts()
            .sort_index()
        )

        source_counts = {
            family:
                int(count)
            for (
                family,
                count,
            )
            in source_counts_series.items()
        }

        quota = largest_remainder_quota(
            source_counts,
            expected_dev,
        )

        # ----------------------------------------------------------
        # Actual frozen project_train/dev distributions.
        # ----------------------------------------------------------

        dev_df = cards_df[
            cards_df[
                "project_role"
            ]
            == "dev_val"
        ]

        train_df = cards_df[
            cards_df[
                "project_role"
            ]
            == "project_train"
        ]

        dev_counts = (
            dev_df[
                "stem_family"
            ]
            .value_counts()
            .to_dict()
        )

        train_counts = (
            train_df[
                "stem_family"
            ]
            .value_counts()
            .to_dict()
        )

        missing_from_dev = (
            actual_families
            - set(
                dev_counts
            )
        )

        missing_from_train = (
            actual_families
            - set(
                train_counts
            )
        )

        if missing_from_dev:
            raise RuntimeError(
                "One or more stem families are absent from dev_val: "
                f"{sorted(missing_from_dev)}"
            )

        if missing_from_train:
            raise RuntimeError(
                "One or more stem families are absent "
                "from project_train: "
                f"{sorted(missing_from_train)}"
            )

        # ----------------------------------------------------------
        # Build auditable summary table.
        # ----------------------------------------------------------

        summary_rows = []

        quota_failures = []

        for family in sorted(
            actual_families
        ):

            source_count = int(
                source_counts[
                    family
                ]
            )

            train_count = int(
                train_counts.get(
                    family,
                    0,
                )
            )

            dev_count = int(
                dev_counts.get(
                    family,
                    0,
                )
            )

            expected_dev_count = int(
                quota[
                    family
                ][
                    "target_dev_count"
                ]
            )

            ideal_dev_count = float(
                quota[
                    family
                ][
                    "ideal_dev_count"
                ]
            )

            quota_delta = (
                dev_count
                - expected_dev_count
            )

            if quota_delta != 0:
                quota_failures.append(
                    (
                        family,
                        expected_dev_count,
                        dev_count,
                    )
                )

            source_prop = (
                source_count
                / expected_total
            )

            train_prop = (
                train_count
                / expected_train
            )

            dev_prop = (
                dev_count
                / expected_dev
            )

            summary_rows.append(
                {
                    "stem_family":
                        family,

                    "source_cards":
                        source_count,

                    "source_proportion":
                        source_prop,

                    "ideal_dev_cards":
                        ideal_dev_count,

                    "target_dev_cards":
                        expected_dev_count,

                    "actual_dev_cards":
                        dev_count,

                    "dev_quota_delta":
                        quota_delta,

                    "actual_project_train_cards":
                        train_count,

                    "dev_proportion":
                        dev_prop,

                    "project_train_proportion":
                        train_prop,

                    "dev_minus_source_proportion":
                        (
                            dev_prop
                            - source_prop
                        ),

                    "dev_minus_source_percentage_points":
                        100.0
                        * (
                            dev_prop
                            - source_prop
                        ),

                    "present_in_dev":
                        dev_count > 0,

                    "present_in_project_train":
                        train_count > 0,
                }
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

        if quota_failures:
            raise RuntimeError(
                "Frozen dev_val stem-family allocation does not "
                "match reconstructed proportional quotas:\n"
                f"{quota_failures}"
            )

        if int(
            summary_df[
                "actual_dev_cards"
            ].sum()
        ) != expected_dev:
            raise RuntimeError(
                "dev_val family counts do not reconcile to 51."
            )

        if int(
            summary_df[
                "actual_project_train_cards"
            ].sum()
        ) != expected_train:
            raise RuntimeError(
                "project_train family counts do not reconcile to 160."
            )

        # ----------------------------------------------------------
        # Persist evidence.
        # ----------------------------------------------------------

        with output_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as f:

            summary_df.to_csv(
                f,
                index=False,
            )

        # ----------------------------------------------------------
        # Human-readable audit log.
        # ----------------------------------------------------------

        logger.info(
            "-" * 72
        )

        logger.info(
            "STEM-FAMILY DISTRIBUTION"
        )

        for row in (
            summary_df
            .sort_values(
                "stem_family"
            )
            .itertuples(
                index=False
            )
        ):

            logger.info(
                "  %-10s "
                "source=%3d "
                "train=%3d "
                "dev=%2d "
                "target=%2d "
                "delta=%+d "
                "dev-source=%+.3f pp",
                row.stem_family,
                row.source_cards,
                row.actual_project_train_cards,
                row.actual_dev_cards,
                row.target_dev_cards,
                row.dev_quota_delta,
                row.dev_minus_source_percentage_points,
            )

        max_abs_pp = float(
            summary_df[
                "dev_minus_source_percentage_points"
            ]
            .abs()
            .max()
        )

        logger.info(
            "-" * 72
        )

        logger.info(
            "Maximum absolute dev-vs-source "
            "stem-family proportion difference: %.3f percentage points",
            max_abs_pp,
        )

        logger.info(
            "[PASS] All %d stem families occur in project_train.",
            len(
                actual_families
            ),
        )

        logger.info(
            "[PASS] All %d stem families occur in dev_val.",
            len(
                actual_families
            ),
        )

        logger.info(
            "[PASS] dev_val exactly matches reconstructed "
            "largest-remainder stem-family quotas."
        )

        logger.info(
            "Audit CSV: %s",
            output_path,
        )

        logger.info(
            "Audit CSV SHA-256: %s",
            sha256_file(
                output_path
            ),
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "FROZEN TRAIN/DEV STEM-FAMILY AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "FROZEN TRAIN/DEV STEM-FAMILY AUDIT: FAIL"
        )

        return 1

    finally:

        for handler in (
            logger.handlers
        ):
            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )