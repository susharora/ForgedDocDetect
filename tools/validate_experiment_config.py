#!/usr/bin/env python3
"""
Validate the Tech-2 ResNet-18 scientific experiment configuration.

This validator is a read-only scientific protocol gate.

It verifies:

1. machine-local configuration;
2. reproducibility / seed contract;
3. frozen Tech-1 provenance;
4. frozen train/dev composition;
5. committed preflight evidence;
6. ResNet-18 model contract;
7. preprocessing candidates;
8. DataLoader contract;
9. class weighting and weighted-dev-loss semantics;
10. AdamW optimizer contract;
11. Stage-A head-only transfer-learning contract;
12. Stage-B full-backbone transfer-learning contract;
13. early stopping versus raw-argmin checkpoint semantics;
14. LR screening and multi-seed development plan;
15. asymmetric resolution-selection rule;
16. post-selection seed/checkpoint policy;
17. validation threshold policy;
18. deferred-ablation boundaries;
19. Grad-CAM deferral contract;
20. held-out test protection.

Evidence policy
---------------
Detailed validation evidence is written to one timestamped canonical
log under ./logs/.

Normal stdout contains only one final handoff record:

    VALIDATION_ARTIFACT | status=... | path=... | sha256=...

This allows tools that invoke the validator as a subprocess to record
the canonical validation artifact without duplicating hundreds of
validation lines.

No scientific data are modified.
No held-out test data are accessed.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# ======================================================================
# Repository / imports
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


LOGGER = logging.getLogger(
    "validate_experiment_config"
)


EXPECTED_RESNET18_IMAGENET1K_V1_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


# ======================================================================
# Generic file / YAML helpers
# ======================================================================

def sha256_file(
    path: Path,
) -> str:
    """
    Return SHA-256 without loading the whole file into memory.
    """

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


def resolve_path(
    path_value: str | Path,
) -> Path:
    """
    Resolve repo-relative paths while preserving absolute paths.
    """

    path = Path(
        path_value
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
) -> dict[str, Any]:
    """
    Require a mapping and return it.
    """

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
    """
    Return a required mapping key with a useful failure message.
    """

    if key not in mapping:

        raise KeyError(
            f"Missing required key: "
            f"{label}.{key}"
        )

    return mapping[
        key
    ]


def load_yaml_file(
    path: Path,
) -> dict[str, Any]:
    """
    Load YAML as a mapping.
    """

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = (
            yaml.safe_load(
                f
            )
            or {}
        )

    return require_mapping(
        data,
        str(
            path
        ),
    )


def count_csv_rows(
    path: Path,
) -> int:
    """
    Count CSV data rows excluding header.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.reader(
            f
        )

        try:

            next(
                reader
            )

        except StopIteration:

            return 0

        return sum(
            1
            for _
            in reader
        )


def load_csv_dict_rows(
    path: Path,
) -> list[
    dict[str, str]
]:
    """
    Load a CSV into dictionaries without pandas.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(
                f
            )
        )


def parse_csv_bool(
    value: str,
) -> bool:

    normalized = (
        str(
            value
        )
        .strip()
        .casefold()
    )

    if normalized == "true":

        return True

    if normalized == "false":

        return False

    raise ValueError(
        "Expected CSV boolean value, "
        f"got {value!r}"
    )


# ======================================================================
# Nested configuration helpers
# ======================================================================

def config_value(
    config: dict[str, Any],
    dotted_path: str,
) -> Any:
    """
    Read a nested config value using dotted notation.

    Integer-looking components support YAML integer keys.

    Example:

        class_contract.name_by_index.0
    """

    current: Any = config

    for component in (
        dotted_path.split(
            "."
        )
    ):

        if not isinstance(
            current,
            dict,
        ):

            raise TypeError(
                "Configuration path traversed "
                "a non-mapping value:\n"
                f"  path={dotted_path}\n"
                f"  component={component}\n"
                f"  current={current!r}"
            )

        key: Any = component

        if key not in current:

            if component.isdigit():

                integer_key = int(
                    component
                )

                if integer_key in current:

                    key = (
                        integer_key
                    )

        if key not in current:

            raise KeyError(
                "Missing scientific "
                "configuration path:\n"
                f"  {dotted_path}"
            )

        current = current[
            key
        ]

    return current


# ======================================================================
# Logging
# ======================================================================

class HandoffOnlyFilter(
    logging.Filter
):
    """
    File handler receives all evidence.

    stdout receives only records explicitly marked handoff=True.
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        return bool(
            getattr(
                record,
                "handoff",
                False,
            )
        )


def configure_logging(
    tool_config_path: Path,
) -> tuple[
    logging.Logger,
    Path,
    logging.Handler,
    logging.Handler,
]:

    tool_config = load_yaml_file(
        tool_config_path
    )

    if (
        tool_config.get(
            "schema_version"
        )
        != 1
    ):

        raise ValueError(
            "Validator tool config "
            "schema_version must currently equal 1."
        )

    logging_cfg = require_mapping(
        require_key(
            tool_config,
            "logging",
            "validator_tool_config",
        ),
        "validator_tool_config.logging",
    )

    log_directory = resolve_path(
        require_key(
            logging_cfg,
            "directory",
            "validator_tool_config.logging",
        )
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
    )

    filename_template = str(
        require_key(
            logging_cfg,
            "filename",
            "validator_tool_config.logging",
        )
    )

    log_path = (
        log_directory
        / filename_template.format(
            timestamp=timestamp
        )
    )

    level_name = str(
        require_key(
            logging_cfg,
            "level",
            "validator_tool_config.logging",
        )
    ).upper()

    if not hasattr(
        logging,
        level_name,
    ):

        raise ValueError(
            "Unknown logging level: "
            f"{level_name!r}"
        )

    level = getattr(
        logging,
        level_name,
    )

    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(
        level
    )

    # ------------------------------------------------------------------
    # Canonical detailed evidence log.
    # ------------------------------------------------------------------

    file_handler = logging.FileHandler(
        log_path,
        mode="x",
        encoding="utf-8",
    )

    file_formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_formatter.converter = (
        time.gmtime
    )

    file_handler.setFormatter(
        file_formatter
    )

    # ------------------------------------------------------------------
    # stdout receives only the final artifact pointer.
    # ------------------------------------------------------------------

    stream_handler = logging.StreamHandler(
        sys.stdout
    )

    stream_handler.setLevel(
        logging.INFO
    )

    stream_handler.addFilter(
        HandoffOnlyFilter()
    )

    stream_handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    LOGGER.addHandler(
        file_handler
    )

    LOGGER.addHandler(
        stream_handler
    )

    return (
        LOGGER,
        log_path,
        file_handler,
        stream_handler,
    )


def log_pass(
    label: str,
    detail: Any | None = None,
) -> None:

    LOGGER.info(
        "[PASS] %s",
        label,
    )

    if detail is not None:

        LOGGER.info(
            "       %s",
            detail,
        )


# ======================================================================
# Git provenance
# ======================================================================

def git_state() -> tuple[
    str,
    str,
]:
    """
    Return commit SHA and porcelain status.

    This must be called BEFORE configure_logging(), because the new
    validation log itself may otherwise appear as an untracked file.
    """

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

    return (
        commit_result.stdout.strip(),
        status_result.stdout.strip(),
    )


# ======================================================================
# Generic validation helpers
# ======================================================================

def check_equal(
    label: str,
    actual: Any,
    expected: Any,
) -> None:

    if actual != expected:

        raise ValueError(
            f"{label} mismatch:\n"
            f"  expected: {expected!r}\n"
            f"  actual:   {actual!r}"
        )

    log_pass(
        label
    )


def check_close(
    label: str,
    actual: Any,
    expected: float,
    *,
    atol: float = 1.0e-12,
) -> None:

    if (
        isinstance(
            actual,
            bool,
        )
        or
        not isinstance(
            actual,
            (
                int,
                float,
            ),
        )
    ):

        raise TypeError(
            f"{label} must be numeric, "
            f"got {actual!r}"
        )

    if not math.isclose(
        float(
            actual
        ),
        float(
            expected
        ),
        rel_tol=0.0,
        abs_tol=atol,
    ):

        raise ValueError(
            f"{label} mismatch:\n"
            f"  expected: {expected!r}\n"
            f"  actual:   {actual!r}"
        )

    log_pass(
        label
    )


def check_config_value(
    config: dict[str, Any],
    dotted_path: str,
    expected: Any,
) -> None:

    actual = config_value(
        config,
        dotted_path,
    )

    check_equal(
        dotted_path,
        actual,
        expected,
    )


def check_sha256(
    label: str,
    path: Path,
    expected: str,
) -> None:

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: "
            f"{path}"
        )

    actual = sha256_file(
        path
    )

    if (
        actual.lower()
        != str(
            expected
        ).lower()
    ):

        raise ValueError(
            f"{label} SHA-256 mismatch:\n"
            f"  path:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    log_pass(
        f"{label} SHA-256",
        actual,
    )


def check_csv_rows(
    label: str,
    path: Path,
    expected: int,
) -> None:

    actual = count_csv_rows(
        path
    )

    if actual != expected:

        raise ValueError(
            f"{label} row-count mismatch:\n"
            f"  path:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    log_pass(
        f"{label} rows = {actual}"
    )


def require_sha256_string(
    label: str,
    value: Any,
) -> str:

    if not isinstance(
        value,
        str,
    ):

        raise TypeError(
            f"{label} must be a scalar SHA-256 string, "
            f"got {type(value).__name__}: {value!r}"
        )

    normalized = (
        value
        .strip()
        .lower()
    )

    if (
        len(
            normalized
        )
        != 64
        or
        any(
            character
            not in "0123456789abcdef"
            for character
            in normalized
        )
    ):

        raise ValueError(
            f"{label} is not a valid "
            f"SHA-256 string: {value!r}"
        )

    log_pass(
        f"{label} is a valid SHA-256 scalar"
    )

    return normalized


# ======================================================================
# Machine config
# ======================================================================

def validate_machine_config(
    machine_cfg: dict[str, Any],
) -> None:

    paths = require_mapping(
        require_key(
            machine_cfg,
            "paths",
            "machine_config",
        ),
        "machine_config.paths",
    )

    dataset_root = resolve_path(
        require_key(
            paths,
            "dataset_root",
            "machine_config.paths",
        )
    )

    runs_root = resolve_path(
        require_key(
            paths,
            "runs_root",
            "machine_config.paths",
        )
    )

    if not dataset_root.is_dir():

        raise FileNotFoundError(
            "Configured dataset_root does not exist "
            "or is not a directory: "
            f"{dataset_root}"
        )

    log_pass(
        "dataset_root exists",
        dataset_root,
    )

    LOGGER.info(
        "[INFO] runs_root"
    )

    LOGGER.info(
        "       %s",
        runs_root,
    )

    runtime = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
    )

    device = require_key(
        runtime,
        "device",
        "machine_config.runtime",
    )

    if (
        not isinstance(
            device,
            str,
        )
        or
        not device
    ):

        raise TypeError(
            "machine_config.runtime.device must be "
            f"a non-empty string, got {device!r}"
        )

    log_pass(
        "machine_config.runtime.device",
        device,
    )

    num_workers = require_key(
        runtime,
        "num_workers",
        "machine_config.runtime",
    )

    if (
        not isinstance(
            num_workers,
            int,
        )
        or
        isinstance(
            num_workers,
            bool,
        )
        or
        num_workers < 0
    ):

        raise TypeError(
            "machine_config.runtime.num_workers must be "
            "a non-negative integer, "
            f"got {num_workers!r}"
        )

    log_pass(
        "machine_config.runtime.num_workers",
        num_workers,
    )

    machine = require_mapping(
        require_key(
            machine_cfg,
            "machine",
            "machine_config",
        ),
        "machine_config.machine",
    )

    machine_id = require_key(
        machine,
        "id",
        "machine_config.machine",
    )

    if (
        not isinstance(
            machine_id,
            str,
        )
        or
        not machine_id
    ):

        raise TypeError(
            "machine_config.machine.id must be "
            f"a non-empty string, got {machine_id!r}"
        )

    log_pass(
        "machine_config.machine.id",
        machine_id,
    )


# ========================
# validate branch init
# ========================

def validate_stage_b_branch_initialisation(
    experiment_cfg: dict[str, Any],
) -> None:
    """
    Validate the frozen Stage-A -> Stage-B branch-reset contract.

    Scientific intent:
    - carry only the Stage-A raw-best MODEL state;
    - reset mutable RNG/DataLoader state from run_seed;
    - create a fresh Stage-B optimizer;
    - keep corresponding Stage-B batch ordering identical across
      backbone-LR candidates.
    """

    expected_values = {
        (
            "transfer_learning.stage_b."
            "branch_initialisation.policy"
        ):
            (
                "reset_from_run_seed_before_each_stage_b_branch"
            ),

        (
            "transfer_learning.stage_b."
            "branch_initialisation.ordered_steps"
        ):
            [
                "configure_run_reproducibility_from_run_seed",
                "build_fresh_development_dataloaders_from_run_seed",
                "build_fresh_resnet18_classifier",
                "restore_exact_stage_a_raw_argmin_model_state",
                "apply_stage_b_train_contract",
                "construct_fresh_stage_b_optimizer",
            ],

        (
            "transfer_learning.stage_b."
            "branch_initialisation.model_checkpoint.source"
        ):
            "exact_stage_a_raw_argmin_checkpoint",

        (
            "transfer_learning.stage_b."
            "branch_initialisation.model_checkpoint."
            "contents_carried.model_parameters_and_buffers"
        ):
            True,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.model_checkpoint."
            "contents_carried.optimizer_state"
        ):
            False,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.model_checkpoint."
            "contents_carried.global_rng_state"
        ):
            False,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.model_checkpoint."
            "contents_carried.dataloader_generator_state"
        ):
            False,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.global_rng.policy"
        ):
            "reset_from_run_seed",

        (
            "transfer_learning.stage_b."
            "branch_initialisation.global_rng.carry_stage_a_state"
        ):
            False,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.dataloader_rng.policy"
        ):
            "fresh_generators_seeded_from_run_seed",

        (
            "transfer_learning.stage_b."
            "branch_initialisation.dataloader_rng."
            "project_train_generator"
        ):
            "run_seed",

        (
            "transfer_learning.stage_b."
            "branch_initialisation.dataloader_rng."
            "dev_val_generator"
        ):
            "run_seed",

        (
            "transfer_learning.stage_b."
            "branch_initialisation.dataloader_rng."
            "carry_stage_a_project_train_generator_state"
        ):
            False,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.dataloader_rng."
            "carry_stage_a_dev_val_generator_state"
        ):
            False,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.lr_branch_comparability."
            "all_lr_candidates_same_stage_a_model_checkpoint"
        ):
            True,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.lr_branch_comparability."
            "all_lr_candidates_same_initial_global_rng_state"
        ):
            True,

        (
            "transfer_learning.stage_b."
            "branch_initialisation.lr_branch_comparability."
            "same_project_train_order_for_corresponding_stage_b_epochs"
        ):
            True,
    }

    for (
        dotted_path,
        expected,
    ) in expected_values.items():

        check_config_value(
            experiment_cfg,
            dotted_path,
            expected,
        )

    # --------------------------------------------------------------
    # Cross-check against the already-frozen Stage-B model source.
    # --------------------------------------------------------------

    existing_source = config_value(
        experiment_cfg,
        (
            "transfer_learning."
            "stage_b."
            "initial_checkpoint."
            "source"
        ),
    )

    branch_source = config_value(
        experiment_cfg,
        (
            "transfer_learning."
            "stage_b."
            "branch_initialisation."
            "model_checkpoint."
            "source"
        ),
    )

    check_equal(
        "Stage-B initial checkpoint vs branch model source",
        branch_source,
        existing_source,
    )

    # --------------------------------------------------------------
    # Cross-check against the already-frozen fresh-optimizer rule.
    # --------------------------------------------------------------

    check_equal(
        "Stage-B branch reset requires fresh optimizer",
        config_value(
            experiment_cfg,
            (
                "transfer_learning."
                "stage_b."
                "optimizer."
                "fresh_optimizer_instance"
            ),
        ),
        True,
    )

    # --------------------------------------------------------------
    # Cross-check against seed binding.
    # --------------------------------------------------------------

    check_equal(
        "Stage-B train-generator seed vs run-seed binding",
        config_value(
            experiment_cfg,
            (
                "reproducibility."
                "seed_plan."
                "seed_binding."
                "dataloader_generator"
            ),
        ),
        "run_seed",
    )

# ======================================================================
# Seed contract
# ======================================================================

def validate_seeds(
    config: dict[str, Any],
) -> None:

    reproducibility = require_mapping(
        require_key(
            config,
            "reproducibility",
            "config",
        ),
        "reproducibility",
    )

    seeds = require_mapping(
        require_key(
            reproducibility,
            "seeds",
            "reproducibility",
        ),
        "reproducibility.seeds",
    )

    required_seeds = (
        "python",
        "numpy",
        "torch",
        "dataloader",
    )

    for name in required_seeds:

        value = require_key(
            seeds,
            name,
            "reproducibility.seeds",
        )

        if (
            not isinstance(
                value,
                int,
            )
            or
            isinstance(
                value,
                bool,
            )
        ):

            raise TypeError(
                f"reproducibility.seeds.{name} "
                "must be an integer, "
                f"got {value!r}"
            )

    log_pass(
        "required random seeds are explicitly defined"
    )


# ======================================================================
# Frozen Tech-1 provenance
# ======================================================================

def validate_frozen_data(
    config: dict[str, Any],
) -> None:

    data_cfg = require_mapping(
        require_key(
            config,
            "data",
            "config",
        ),
        "data",
    )

    # ------------------------------------------------------------------
    # Frozen discovery workbook
    # ------------------------------------------------------------------

    source_cfg = require_mapping(
        require_key(
            data_cfg,
            "source_discovery",
            "data",
        ),
        "data.source_discovery",
    )

    workbook_path = resolve_path(
        require_key(
            source_cfg,
            "workbook",
            "data.source_discovery",
        )
    )

    workbook_sha = require_key(
        source_cfg,
        "sha256",
        "data.source_discovery",
    )

    check_sha256(
        "source discovery workbook",
        workbook_path,
        workbook_sha,
    )

    # ------------------------------------------------------------------
    # Frozen split bundle
    # ------------------------------------------------------------------

    split_cfg = require_mapping(
        require_key(
            data_cfg,
            "frozen_split",
            "data",
        ),
        "data.frozen_split",
    )

    bundle_path = resolve_path(
        require_key(
            split_cfg,
            "bundle",
            "data.frozen_split",
        )
    )

    bundle_sha = require_key(
        split_cfg,
        "bundle_sha256",
        "data.frozen_split",
    )

    check_sha256(
        "frozen split bundle",
        bundle_path,
        bundle_sha,
    )

    bundle = load_yaml_file(
        bundle_path
    )

    bundle_source = require_mapping(
        require_key(
            bundle,
            "source_discovery",
            "bundle",
        ),
        "bundle.source_discovery",
    )

    bundle_counts = require_mapping(
        require_key(
            bundle,
            "counts",
            "bundle",
        ),
        "bundle.counts",
    )

    bundle_artifacts = require_mapping(
        require_key(
            bundle,
            "artifacts",
            "bundle",
        ),
        "bundle.artifacts",
    )

    check_equal(
        "bundle source workbook filename",
        workbook_path.name,
        require_key(
            bundle_source,
            "workbook",
            "bundle.source_discovery",
        ),
    )

    check_equal(
        "bundle source workbook SHA-256",
        workbook_sha,
        require_key(
            bundle_source,
            "workbook_sha256",
            "bundle.source_discovery",
        ),
    )

    # ------------------------------------------------------------------
    # Frozen manifests
    # ------------------------------------------------------------------

    manifest_specs = (
        (
            "cards",
            "rows",
        ),
        (
            "project_train",
            "images",
        ),
        (
            "dev_val",
            "images",
        ),
    )

    for (
        manifest_name,
        config_row_key,
    ) in manifest_specs:

        manifest_cfg = require_mapping(
            require_key(
                split_cfg,
                manifest_name,
                "data.frozen_split",
            ),
            f"data.frozen_split.{manifest_name}",
        )

        bundle_manifest = require_mapping(
            require_key(
                bundle_artifacts,
                manifest_name,
                "bundle.artifacts",
            ),
            f"bundle.artifacts.{manifest_name}",
        )

        manifest_path = resolve_path(
            require_key(
                manifest_cfg,
                "path",
                f"data.frozen_split.{manifest_name}",
            )
        )

        manifest_sha = require_key(
            manifest_cfg,
            "sha256",
            f"data.frozen_split.{manifest_name}",
        )

        expected_rows = require_key(
            manifest_cfg,
            config_row_key,
            f"data.frozen_split.{manifest_name}",
        )

        check_equal(
            f"{manifest_name} filename agrees with bundle",
            manifest_path.name,
            require_key(
                bundle_manifest,
                "filename",
                f"bundle.artifacts.{manifest_name}",
            ),
        )

        check_equal(
            f"{manifest_name} SHA-256 agrees with bundle",
            manifest_sha,
            require_key(
                bundle_manifest,
                "sha256",
                f"bundle.artifacts.{manifest_name}",
            ),
        )

        check_equal(
            f"{manifest_name} expected rows agrees with bundle",
            expected_rows,
            require_key(
                bundle_manifest,
                "rows",
                f"bundle.artifacts.{manifest_name}",
            ),
        )

        check_sha256(
            f"{manifest_name} manifest",
            manifest_path,
            manifest_sha,
        )

        check_csv_rows(
            f"{manifest_name} manifest",
            manifest_path,
            expected_rows,
        )

    # ------------------------------------------------------------------
    # Card/image counts
    # ------------------------------------------------------------------

    project_train_cfg = require_mapping(
        split_cfg[
            "project_train"
        ],
        "data.frozen_split.project_train",
    )

    dev_val_cfg = require_mapping(
        split_cfg[
            "dev_val"
        ],
        "data.frozen_split.dev_val",
    )

    check_equal(
        "project_train card count",
        require_key(
            project_train_cfg,
            "cards",
            "data.frozen_split.project_train",
        ),
        require_key(
            bundle_counts,
            "project_train_cards",
            "bundle.counts",
        ),
    )

    check_equal(
        "project_train image count",
        require_key(
            project_train_cfg,
            "images",
            "data.frozen_split.project_train",
        ),
        require_key(
            bundle_counts,
            "project_train_images",
            "bundle.counts",
        ),
    )

    check_equal(
        "dev_val card count",
        require_key(
            dev_val_cfg,
            "cards",
            "data.frozen_split.dev_val",
        ),
        require_key(
            bundle_counts,
            "dev_val_cards",
            "bundle.counts",
        ),
    )

    check_equal(
        "dev_val image count",
        require_key(
            dev_val_cfg,
            "images",
            "data.frozen_split.dev_val",
        ),
        require_key(
            bundle_counts,
            "dev_val_images",
            "bundle.counts",
        ),
    )


# ======================================================================
# Frozen manifest class-count audit
# ======================================================================

def manifest_traffic_counts(
    path: Path,
) -> dict[
    str,
    int,
]:

    rows = load_csv_dict_rows(
        path
    )

    counts = {
        "bonafide": 0,
        "attack": 0,
    }

    for row in rows:

        traffic = row.get(
            "traffic_type"
        )

        if traffic not in counts:

            raise ValueError(
                "Unexpected traffic_type in frozen manifest:\n"
                f"  path={path}\n"
                f"  traffic_type={traffic!r}"
            )

        counts[
            traffic
        ] += 1

    return counts


# ======================================================================
# Preflight evidence
# ======================================================================

def validate_protocol_preflight_evidence(
    config: dict[str, Any],
) -> None:
    """
    Verify the empirical checks that enabled the development protocol.

    - frozen train/dev stem-family audit;
    - FP32 512x864 batch-32 full Stage-B training-memory probes
      on both machines.
    """

    preflight = require_mapping(
        require_key(
            config,
            "preflight_evidence",
            "config",
        ),
        "preflight_evidence",
    )

    # ------------------------------------------------------------------
    # Stem-family audit
    # ------------------------------------------------------------------

    stem = require_mapping(
        require_key(
            preflight,
            "stem_family_audit",
            "preflight_evidence",
        ),
        "preflight_evidence.stem_family_audit",
    )

    check_equal(
        "stem-family preflight status",
        require_key(
            stem,
            "status",
            "preflight_evidence.stem_family_audit",
        ),
        "PASS",
    )

    stem_csv_cfg = require_mapping(
        require_key(
            stem,
            "csv",
            "preflight_evidence.stem_family_audit",
        ),
        "preflight_evidence.stem_family_audit.csv",
    )

    stem_csv_path = resolve_path(
        require_key(
            stem_csv_cfg,
            "path",
            "preflight_evidence.stem_family_audit.csv",
        )
    )

    stem_csv_sha = require_key(
        stem_csv_cfg,
        "sha256",
        "preflight_evidence.stem_family_audit.csv",
    )

    check_sha256(
        "stem-family audit CSV",
        stem_csv_path,
        stem_csv_sha,
    )

    stem_rows = load_csv_dict_rows(
        stem_csv_path
    )

    if len(
        stem_rows
    ) != 10:

        raise ValueError(
            "Stem-family preflight CSV must contain "
            f"exactly 10 family rows, got {len(stem_rows)}"
        )

    source_total = 0
    train_total = 0
    dev_total = 0

    for row in stem_rows:

        if int(
            row[
                "dev_quota_delta"
            ]
        ) != 0:

            raise ValueError(
                "Stem-family audit contains a "
                "non-zero dev quota delta:\n"
                f"  {row}"
            )

        if not parse_csv_bool(
            row[
                "present_in_dev"
            ]
        ):

            raise ValueError(
                "Stem-family absent from dev_val:\n"
                f"  {row['stem_family']}"
            )

        if not parse_csv_bool(
            row[
                "present_in_project_train"
            ]
        ):

            raise ValueError(
                "Stem-family absent from project_train:\n"
                f"  {row['stem_family']}"
            )

        source_total += int(
            row[
                "source_cards"
            ]
        )

        train_total += int(
            row[
                "actual_project_train_cards"
            ]
        )

        dev_total += int(
            row[
                "actual_dev_cards"
            ]
        )

    check_equal(
        "stem-family source-card reconciliation",
        source_total,
        211,
    )

    check_equal(
        "stem-family project_train reconciliation",
        train_total,
        160,
    )

    check_equal(
        "stem-family dev_val reconciliation",
        dev_total,
        51,
    )

    # ------------------------------------------------------------------
    # FP32 memory probes
    # ------------------------------------------------------------------

    memory = require_mapping(
        require_key(
            preflight,
            "fp32_training_memory",
            "preflight_evidence",
        ),
        "preflight_evidence.fp32_training_memory",
    )

    check_equal(
        "FP32 memory-preflight status",
        require_key(
            memory,
            "status",
            "preflight_evidence.fp32_training_memory",
        ),
        "PASS",
    )

    tested = require_mapping(
        require_key(
            memory,
            "tested_configuration",
            "preflight_evidence.fp32_training_memory",
        ),
        (
            "preflight_evidence."
            "fp32_training_memory.tested_configuration"
        ),
    )

    expected_batch = int(
        config_value(
            config,
            "dataloader.batch_size",
        )
    )

    r512_height = int(
        config_value(
            config,
            (
                "preprocessing."
                "candidate_canvases."
                "r512.canvas_height"
            ),
        )
    )

    r512_width = int(
        config_value(
            config,
            (
                "preprocessing."
                "candidate_canvases."
                "r512.canvas_width"
            ),
        )
    )

    check_equal(
        "memory preflight architecture",
        tested[
            "architecture"
        ],
        "resnet18",
    )

    check_equal(
        "memory preflight full-backbone state",
        tested[
            "full_backbone_trainable"
        ],
        True,
    )

    check_equal(
        "memory preflight batch size",
        tested[
            "batch_size"
        ],
        expected_batch,
    )

    check_equal(
        "memory preflight tensor shape",
        tested[
            "input_nchw"
        ],
        [
            expected_batch,
            3,
            r512_height,
            r512_width,
        ],
    )

    check_equal(
        "memory preflight dtype",
        tested[
            "dtype"
        ],
        "float32",
    )

    check_equal(
        "memory preflight AMP",
        tested[
            "amp"
        ],
        False,
    )

    check_equal(
        "memory preflight optimizer",
        tested[
            "optimizer"
        ],
        "AdamW",
    )

    for machine_key in (
        "home",
        "lab",
    ):

        machine_entry = require_mapping(
            require_key(
                memory,
                machine_key,
                "preflight_evidence.fp32_training_memory",
            ),
            (
                "preflight_evidence."
                f"fp32_training_memory.{machine_key}"
            ),
        )

        result_cfg = require_mapping(
            require_key(
                machine_entry,
                "result",
                (
                    "preflight_evidence."
                    f"fp32_training_memory.{machine_key}"
                ),
            ),
            (
                "preflight_evidence."
                f"fp32_training_memory."
                f"{machine_key}.result"
            ),
        )

        result_path = resolve_path(
            require_key(
                result_cfg,
                "path",
                (
                    "preflight_evidence."
                    "fp32_training_memory."
                    f"{machine_key}.result"
                ),
            )
        )

        result_sha = require_key(
            result_cfg,
            "sha256",
            (
                "preflight_evidence."
                "fp32_training_memory."
                f"{machine_key}.result"
            ),
        )

        check_sha256(
            f"{machine_key} memory-probe result",
            result_path,
            result_sha,
        )

        result = load_yaml_file(
            result_path
        )

        check_equal(
            f"{machine_key} memory-probe schema",
            require_key(
                result,
                "schema_version",
                str(
                    result_path
                ),
            ),
            1,
        )

        check_equal(
            f"{machine_key} memory-probe status",
            require_key(
                result,
                "status",
                str(
                    result_path
                ),
            ),
            "PASS",
        )

        result_machine = require_mapping(
            require_key(
                result,
                "machine",
                str(
                    result_path
                ),
            ),
            f"{result_path}.machine",
        )

        check_equal(
            f"{machine_key} memory-probe machine ID",
            result_machine[
                "id"
            ],
            machine_entry[
                "machine_id"
            ],
        )

        check_equal(
            f"{machine_key} memory-probe GPU",
            result_machine[
                "gpu"
            ],
            machine_entry[
                "gpu"
            ],
        )

        result_probe = require_mapping(
            require_key(
                result,
                "probe",
                str(
                    result_path
                ),
            ),
            f"{result_path}.probe",
        )

        check_equal(
            f"{machine_key} probe architecture",
            result_probe[
                "architecture"
            ],
            "resnet18",
        )

        check_equal(
            f"{machine_key} probe weights",
            result_probe[
                "weights"
            ],
            "IMAGENET1K_V1",
        )

        check_equal(
            f"{machine_key} probe batch size",
            result_probe[
                "batch_size"
            ],
            expected_batch,
        )

        check_equal(
            f"{machine_key} probe tensor shape",
            result_probe[
                "input_shape_nchw"
            ],
            [
                expected_batch,
                3,
                r512_height,
                r512_width,
            ],
        )

        check_equal(
            f"{machine_key} probe dtype",
            result_probe[
                "dtype"
            ],
            "float32",
        )

        check_equal(
            f"{machine_key} probe AMP",
            result_probe[
                "amp"
            ],
            False,
        )

        check_equal(
            f"{machine_key} probe full-backbone state",
            result_probe[
                "full_backbone_trainable"
            ],
            True,
        )

        check_equal(
            f"{machine_key} probe model mode",
            result_probe[
                "model_mode"
            ],
            "train",
        )

        check_equal(
            f"{machine_key} probe optimizer",
            result_probe[
                "optimizer"
            ],
            "AdamW",
        )

        check_equal(
            f"{machine_key} probe AdamW foreach",
            result_probe[
                "optimizer_foreach"
            ],
            False,
        )

        check_equal(
            f"{machine_key} probe AdamW fused",
            result_probe[
                "optimizer_fused"
            ],
            False,
        )

        result_memory = require_mapping(
            require_key(
                result,
                "memory",
                str(
                    result_path
                ),
            ),
            f"{result_path}.memory",
        )

        peak_fraction = float(
            result_memory[
                "peak_reserved_fraction"
            ]
        )

        allowed_fraction = float(
            result_memory[
                "max_allowed_peak_reserved_fraction"
            ]
        )

        if (
            peak_fraction
            > allowed_fraction
        ):

            raise ValueError(
                f"{machine_key} committed memory probe "
                "does not satisfy its own capacity guard:\n"
                f"  peak_reserved_fraction="
                f"{peak_fraction}\n"
                f"  allowed={allowed_fraction}"
            )

        log_pass(
            f"{machine_key} memory-probe capacity guard",
            (
                f"peak_reserved_fraction="
                f"{peak_fraction:.6f}"
            ),
        )


# ======================================================================
# Full schema-v2 scientific protocol
# ======================================================================

def validate_scientific_protocol(
    config: dict[str, Any],
    machine_cfg: dict[str, Any],
) -> None:
    """
    Validate the encoded ResNet-18 development protocol.

    Both fixed scientific values and duplicated internal contracts are
    checked so silent config drift cannot occur.
    """

    # ------------------------------------------------------------------
    # Fixed scientific contract
    # ------------------------------------------------------------------

    fixed_values = {
        # --------------------------------------------------------------
        # Experiment identity
        # --------------------------------------------------------------

        "schema_version":
            3,

        "experiment.name":
            "resnet18_gradcam_fantasyid",

        "experiment.pipeline":
            "resnet18_gradcam",

        "experiment.task":
            "binary_classification",

        "experiment.protocol_stage":
            "transfer_learning_development",

        "experiment.primary_model_scope.supervision":
            "image_level_labels_only",

        "experiment.primary_model_scope.localisation_supervision":
            False,

        "experiment.primary_model_scope.targeted_region_dropout":
            False,

        "experiment.primary_model_scope.patch_training":
            False,

        "experiment.primary_model_scope.segmentation_head":
            False,

        # --------------------------------------------------------------
        # Class contract
        # --------------------------------------------------------------

        "class_contract.index_by_name.bonafide":
            0,

        "class_contract.index_by_name.attack":
            1,

        "class_contract.name_by_index.0":
            "bonafide",

        "class_contract.name_by_index.1":
            "attack",

        "class_contract.positive_class":
            "attack",

        "class_contract.positive_class_index":
            1,

        # --------------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------------

        "reproducibility.seed_plan.screening_seed":
            8,

        "reproducibility.seed_plan.confirmation_seeds":
            [
                8,
                9,
                10,
            ],

        (
            "reproducibility."
            "seed_plan."
            "reuse_screening_seed_run_during_confirmation"
        ):
            True,

        "reproducibility.seed_plan.seed_binding.python_random":
            "run_seed",

        "reproducibility.seed_plan.seed_binding.numpy":
            "run_seed",

        "reproducibility.seed_plan.seed_binding.torch_cpu":
            "run_seed",

        "reproducibility.seed_plan.seed_binding.torch_cuda_all":
            "run_seed",

        (
            "reproducibility."
            "seed_plan."
            "seed_binding."
            "dataloader_generator"
        ):
            "run_seed",

        (
            "reproducibility."
            "seed_plan."
            "seed_binding."
            "dataloader_workers"
        ):
            "derived_from_run_seed",

        (
            "reproducibility."
            "seed_plan."
            "seed_binding."
            "classifier_initialisation"
        ):
            "run_seed",

        "reproducibility.determinism.cudnn_benchmark":
            False,

        "reproducibility.determinism.cudnn_deterministic":
            True,

        "reproducibility.determinism.deterministic_algorithms":
            True,

        "reproducibility.determinism.cublas_workspace_config":
            ":4096:8",

        "reproducibility.determinism.allow_tf32_matmul":
            False,

        "reproducibility.determinism.allow_tf32_cudnn":
            False,

        (
            "reproducibility."
            "determinism."
            "failure_policy."
            "nondeterministic_required_operation"
        ):
            "fail",

        # --------------------------------------------------------------
        # Development data policy
        # --------------------------------------------------------------

        "data.development_policy.training_source":
            "project_train",

        "data.development_policy.model_selection_source":
            "dev_val",

        (
            "data."
            "development_policy."
            "project_train_dev_card_overlap_required"
        ):
            0,

        (
            "data."
            "development_policy."
            "stem_family."
            "terminology"
        ):
            "stem_family_not_formal_template_identity",

        (
            "data."
            "development_policy."
            "stem_family."
            "expected_family_count"
        ):
            10,

        (
            "data."
            "development_policy."
            "stem_family."
            "every_family_required_in_project_train"
        ):
            True,

        (
            "data."
            "development_policy."
            "stem_family."
            "every_family_required_in_dev_val"
        ):
            True,

        (
            "data."
            "development_policy."
            "stem_family."
            "dev_allocation."
            "policy"
        ):
            "frozen_largest_remainder_quota",

        (
            "data."
            "development_policy."
            "stem_family."
            "dev_allocation."
            "status"
        ):
            "validated",

        # --------------------------------------------------------------
        # Model
        # --------------------------------------------------------------

        "model.architecture":
            "resnet18",

        "model.pretrained_weights.source":
            "torchvision",

        "model.pretrained_weights.enum":
            "IMAGENET1K_V1",

        "model.classifier.replace_original_fc":
            True,

        "model.classifier.type":
            "Linear",

        "model.classifier.in_features":
            512,

        "model.classifier.out_features":
            2,

        "model.classifier.initialisation.policy":
            "pytorch_nn_linear_default",

        "model.classifier.initialisation.controlled_by":
            "run_seed",

        "model.dtype":
            "float32",

        "model.amp":
            False,

        # --------------------------------------------------------------
        # Decode / preprocessing
        # --------------------------------------------------------------

        "preprocessing.decode.decoder":
            "Pillow",

        "preprocessing.decode.force_rgb":
            True,

        "preprocessing.decode.exif_orientation_policy.action":
            "none",

        "preprocessing.candidate_canvases.r256.content_height":
            256,

        "preprocessing.candidate_canvases.r256.canvas_height":
            256,

        "preprocessing.candidate_canvases.r256.canvas_width":
            432,

        "preprocessing.candidate_canvases.r512.content_height":
            512,

        "preprocessing.candidate_canvases.r512.canvas_height":
            512,

        "preprocessing.candidate_canvases.r512.canvas_width":
            864,

        "preprocessing.resize.preserve_aspect_ratio":
            True,

        "preprocessing.resize.crop":
            "none",

        "preprocessing.resize.implementation.library":
            "torchvision",

        "preprocessing.resize.implementation.operation":
            "transforms.functional.resize",

        "preprocessing.resize.size_argument":
            "content_height_as_integer_short_side",

        "preprocessing.resize.interpolation":
            "bilinear",

        "preprocessing.resize.antialias":
            True,

        "preprocessing.resize.width_overflow_policy":
            "fail",

        "preprocessing.padding.required_dimension":
            "horizontal_only",

        "preprocessing.padding.alignment":
            "center",

        "preprocessing.padding.left_pixels.rule":
            "total_horizontal_padding_floor_div_2",

        "preprocessing.padding.right_pixels.rule":
            "total_horizontal_padding_minus_left",

        "preprocessing.padding.top_pixels":
            0,

        "preprocessing.padding.bottom_pixels":
            0,

        (
            "preprocessing."
            "baseline_transform_description."
            "deterministic_class_independent_resampling"
        ):
            True,

        (
            "preprocessing."
            "baseline_transform_description."
            "stochastic_resampling"
        ):
            False,

        # --------------------------------------------------------------
        # DataLoader
        # --------------------------------------------------------------

        "dataloader.batch_size":
            32,

        "dataloader.num_workers_source":
            "machine_config.runtime.num_workers",

        "dataloader.project_train.shuffle":
            True,

        "dataloader.project_train.drop_last":
            False,

        "dataloader.dev_val.shuffle":
            False,

        "dataloader.dev_val.drop_last":
            False,

        # --------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------

        "loss.type":
            "CrossEntropyLoss",

        "loss.class_weighting.enabled":
            True,

        "loss.class_weighting.source":
            "frozen_project_train_only",

        "loss.class_weighting.formula":
            "w_c = N / (2 * n_c)",

        (
            "loss."
            "class_weighting."
            "runtime_assert_against_class_contract"
        ):
            True,

        "loss.training_loss.weighted":
            True,

        "loss.dev_selection_loss.weighted":
            True,

        (
            "loss."
            "dev_selection_loss."
            "reuse_project_train_weights"
        ):
            True,

        "loss.epoch_reduction.definition":
            "weighted_cross_entropy_over_entire_dataset",

        "loss.epoch_reduction.numerator":
            "sum_sample_weight_times_per_sample_ce",

        "loss.epoch_reduction.denominator":
            "sum_sample_weights",

        "loss.label_smoothing":
            0.0,

        # --------------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------------

        "optimizer.name":
            "AdamW",

        "optimizer.betas":
            [
                0.9,
                0.999,
            ],

        "optimizer.eps":
            1.0e-8,

        "optimizer.amsgrad":
            False,

        "optimizer.weight_decay":
            1.0e-4,

        (
            "optimizer."
            "weight_decay_exclusions."
            "all_bias_parameters"
        ):
            True,

        (
            "optimizer."
            "weight_decay_exclusions."
            "batchnorm_affine_parameters"
        ):
            True,

        "optimizer.foreach":
            False,

        "optimizer.fused":
            False,

        "optimizer.learning_rate_schedule.type":
            "constant",

        # --------------------------------------------------------------
        # Global stopping semantics
        # --------------------------------------------------------------

        (
            "stopping_definition."
            "meaningful_relative_improvement_fraction"
        ):
            0.005,

        (
            "stopping_definition."
            "patience_logic."
            "comparison"
        ):
            (
                "current_dev_loss < "
                "patience_anchor_loss * (1 - 0.005)"
            ),

        (
            "stopping_definition."
            "patience_logic."
            "on_meaningful_improvement."
            "reset_patience"
        ):
            True,

        (
            "stopping_definition."
            "patience_logic."
            "on_meaningful_improvement."
            "update_patience_anchor"
        ):
            True,

        (
            "stopping_definition."
            "patience_logic."
            "otherwise."
            "increment_patience"
        ):
            True,

        (
            "stopping_definition."
            "checkpoint_logic."
            "metric"
        ):
            "class_weighted_dev_cross_entropy",

        (
            "stopping_definition."
            "checkpoint_logic."
            "policy"
        ):
            "raw_argmin",

        (
            "stopping_definition."
            "checkpoint_logic."
            "minimum_improvement_required_for_checkpoint"
        ):
            False,

        # --------------------------------------------------------------
        # Stage A
        # --------------------------------------------------------------

        "transfer_learning.stage_a.name":
            "head_only_adaptation",

        (
            "transfer_learning."
            "stage_a."
            "backbone."
            "requires_grad"
        ):
            False,

        (
            "transfer_learning."
            "stage_a."
            "backbone."
            "module_mode"
        ):
            "eval",

        (
            "transfer_learning."
            "stage_a."
            "backbone."
            "batchnorm."
            "affine_requires_grad"
        ):
            False,

        (
            "transfer_learning."
            "stage_a."
            "backbone."
            "batchnorm."
            "running_statistics_update"
        ):
            False,

        (
            "transfer_learning."
            "stage_a."
            "classifier."
            "requires_grad"
        ):
            True,

        (
            "transfer_learning."
            "stage_a."
            "classifier."
            "module_mode"
        ):
            "train",

        (
            "transfer_learning."
            "stage_a."
            "optimizer."
            "scope"
        ):
            "classifier_only",

        (
            "transfer_learning."
            "stage_a."
            "optimizer."
            "fresh_optimizer_instance"
        ):
            True,

        (
            "transfer_learning."
            "stage_a."
            "optimizer."
            "classifier_lr"
        ):
            0.001,

        (
            "transfer_learning."
            "stage_a."
            "stopping."
            "metric"
        ):
            "class_weighted_dev_cross_entropy",

        (
            "transfer_learning."
            "stage_a."
            "stopping."
            "patience_epochs"
        ):
            2,

        (
            "transfer_learning."
            "stage_a."
            "stopping."
            "maximum_epochs"
        ):
            10,

        (
            "transfer_learning."
            "stage_a."
            "stopping."
            "meaningful_relative_improvement_fraction"
        ):
            0.005,

        (
            "transfer_learning."
            "stage_a."
            "checkpoint."
            "policy"
        ):
            "raw_argmin_class_weighted_dev_cross_entropy",

        (
            "transfer_learning."
            "stage_a."
            "execution_scope."
            "once_per_resolution_and_seed"
        ):
            True,

        (
            "transfer_learning."
            "stage_a."
            "execution_scope."
            "reuse_checkpoint_for_all_stage_b_lr_candidates"
        ):
            True,

        # --------------------------------------------------------------
        # Stage B
        # --------------------------------------------------------------

        "transfer_learning.stage_b.name":
            "full_backbone_adaptation",

        (
            "transfer_learning."
            "stage_b."
            "initial_checkpoint."
            "source"
        ):
            "exact_stage_a_raw_argmin_checkpoint",

        (
            "transfer_learning."
            "stage_b."
            "optimizer."
            "fresh_optimizer_instance"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "optimizer."
            "backbone_lr_candidates"
        ):
            [
                0.00003,
                0.0001,
                0.0003,
            ],

        (
            "transfer_learning."
            "stage_b."
            "optimizer."
            "classifier_lr"
        ):
            0.001,

        (
            "transfer_learning."
            "stage_b."
            "backbone."
            "requires_grad"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "backbone."
            "module_mode"
        ):
            "train",

        (
            "transfer_learning."
            "stage_b."
            "classifier."
            "requires_grad"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "classifier."
            "module_mode"
        ):
            "train",

        (
            "transfer_learning."
            "stage_b."
            "batchnorm."
            "affine_requires_grad"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "batchnorm."
            "running_statistics_update"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "epoch_1_transition_note."
            "required_in_history"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "stopping."
            "metric"
        ):
            "class_weighted_dev_cross_entropy",

        (
            "transfer_learning."
            "stage_b."
            "stopping."
            "patience_epochs"
        ):
            5,

        (
            "transfer_learning."
            "stage_b."
            "stopping."
            "maximum_epochs"
        ):
            30,

        (
            "transfer_learning."
            "stage_b."
            "stopping."
            "meaningful_relative_improvement_fraction"
        ):
            0.005,

        (
            "transfer_learning."
            "stage_b."
            "checkpoint."
            "policy"
        ):
            "raw_argmin_class_weighted_dev_cross_entropy",

        (
            "transfer_learning."
            "stage_b."
            "auroc_disagreement_guard."
            "enabled"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "auroc_disagreement_guard."
            "flag_if_strictly_greater_than"
        ):
            0.01,

        (
            "transfer_learning."
            "stage_b."
            "auroc_disagreement_guard."
            "action."
            "flag_for_protocol_review"
        ):
            True,

        (
            "transfer_learning."
            "stage_b."
            "auroc_disagreement_guard."
            "action."
            "automatically_switch_checkpoint"
        ):
            False,

        # --------------------------------------------------------------
        # LR screening
        # --------------------------------------------------------------

        "development_selection.lr_screening.seed":
            8,

        (
            "development_selection."
            "lr_screening."
            "candidate_resolutions"
        ):
            [
                "r256",
                "r512",
            ],

        (
            "development_selection."
            "lr_screening."
            "backbone_lr_candidates"
        ):
            [
                0.00003,
                0.0001,
                0.0003,
            ],

        (
            "development_selection."
            "lr_screening."
            "classifier_lr_fixed"
        ):
            0.001,

        (
            "development_selection."
            "lr_screening."
            "stage_a."
            "run_once_per_resolution"
        ):
            True,

        (
            "development_selection."
            "lr_screening."
            "stage_b."
            "branch_all_lr_candidates_from_same_stage_a_checkpoint"
        ):
            True,

        (
            "development_selection."
            "lr_screening."
            "lr_selection_metric."
            "name"
        ):
            "class_weighted_dev_cross_entropy",

        (
            "development_selection."
            "lr_screening."
            "lr_selection_metric."
            "source"
        ):
            "raw_argmin_checkpoint",

        (
            "development_selection."
            "lr_screening."
            "select."
            "rule"
        ):
            "lowest_best_weighted_dev_loss",

        (
            "development_selection."
            "lr_screening."
            "select."
            "exact_tie_breaker."
            "rule"
        ):
            "lower_backbone_learning_rate",

        (
            "development_selection."
            "lr_screening."
            "auroc_guard."
            "does_not_change_selection_rule"
        ):
            True,

        # --------------------------------------------------------------
        # Multi-seed confirmation
        # --------------------------------------------------------------

        (
            "development_selection."
            "multi_seed_confirmation."
            "resolutions"
        ):
            [
                "r256",
                "r512",
            ],

        (
            "development_selection."
            "multi_seed_confirmation."
            "seeds"
        ):
            [
                8,
                9,
                10,
            ],

        (
            "development_selection."
            "multi_seed_confirmation."
            "learning_rate_per_resolution."
            "source"
        ):
            "winner_of_lr_screening_for_that_resolution",

        (
            "development_selection."
            "multi_seed_confirmation."
            "reuse_existing_seed_8_screening_run"
        ):
            True,

        (
            "development_selection."
            "multi_seed_confirmation."
            "terminology"
        ):
            (
                "multi_seed_stability_analysis_"
                "not_independent_holdout_confirmation"
            ),

        # --------------------------------------------------------------
        # Resolution selection
        # --------------------------------------------------------------

        (
            "development_selection."
            "resolution_selection."
            "comparison_metric."
            "name"
        ):
            "class_weighted_dev_cross_entropy",

        (
            "development_selection."
            "resolution_selection."
            "comparison_metric."
            "value_per_seed"
        ):
            "raw_minimum_for_that_resolution_selected_lr",

        (
            "development_selection."
            "resolution_selection."
            "paired_seed_difference."
            "definition"
        ):
            "d_s = L_512(s) - L_256(s)",

        (
            "development_selection."
            "resolution_selection."
            "decision_statistic."
            "definition"
        ):
            "mean_d = mean(d_8, d_9, d_10)",

        (
            "development_selection."
            "resolution_selection."
            "noninferiority_margin."
            "type"
        ):
            "relative_to_r256_mean_loss",

        (
            "development_selection."
            "resolution_selection."
            "noninferiority_margin."
            "relative_fraction"
        ):
            0.05,

        (
            "development_selection."
            "resolution_selection."
            "decision_rule."
            "select_r512_if"
        ):
            "mean_d <= margin",

        (
            "development_selection."
            "resolution_selection."
            "decision_rule."
            "select_r256_if"
        ):
            "mean_d > margin",

        (
            "development_selection."
            "resolution_selection."
            "robustness_annotation."
            "report_sign_of_each_seed_difference"
        ):
            True,

        (
            "development_selection."
            "resolution_selection."
            "robustness_annotation."
            "sign_pattern_changes_decision"
        ):
            False,

        (
            "development_selection."
            "resolution_selection."
            "masked_dev_diagnostics_influence_selection"
        ):
            False,

        (
            "development_selection."
            "resolution_selection."
            "localisation_metrics_influence_selection"
        ):
            False,

        # --------------------------------------------------------------
        # Post-selection
        # --------------------------------------------------------------

        (
            "post_selection."
            "selected_resolution_seed_checkpoints."
            "seeds"
        ):
            [
                8,
                9,
                10,
            ],

        (
            "post_selection."
            "selected_resolution_seed_checkpoints."
            "final_detection_test."
            "evaluate_all_three"
        ):
            True,

        (
            "post_selection."
            "localisation_representative_checkpoint."
            "rule"
        ):
            "seed_with_median_raw_best_weighted_dev_loss",

        (
            "post_selection."
            "localisation_representative_checkpoint."
            "exact_loss_tie_breaker"
        ):
            "lowest_seed_number",

        (
            "post_selection."
            "losing_resolution."
            "role"
        ):
            "development_resolution_ablation",

        (
            "post_selection."
            "losing_resolution."
            "held_out_test_access"
        ):
            False,

        # --------------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------------

        (
            "evaluation."
            "checkpoint_selection."
            "threshold_independent"
        ):
            True,

        (
            "evaluation."
            "checkpoint_selection."
            "primary_metric"
        ):
            "class_weighted_dev_cross_entropy",

        (
            "evaluation."
            "diagnostic_metrics."
            "dev_auroc."
            "enabled"
        ):
            True,

        (
            "evaluation."
            "diagnostic_metrics."
            "dev_auroc."
            "participates_in_checkpoint_selection"
        ):
            False,

        (
            "evaluation."
            "threshold_selection."
            "source"
        ):
            "dev_val",

        (
            "evaluation."
            "threshold_selection."
            "threshold_per_seed_checkpoint"
        ):
            True,

        (
            "evaluation."
            "threshold_selection."
            "target_false_positive_rate"
        ):
            0.10,

        (
            "evaluation."
            "threshold_selection."
            "implementation"
        ):
            "reuse_frozen_tech1_threshold_and_metric_conventions_exactly",

        (
            "evaluation."
            "threshold_selection."
            "positive_class"
        ):
            "attack",

        # --------------------------------------------------------------
        # Deferred work
        # --------------------------------------------------------------

        (
            "deferred_work."
            "masked_dev_occlusion_diagnostics."
            "status"
        ):
            "deferred",

        (
            "deferred_work."
            "masked_dev_occlusion_diagnostics."
            "selection_metric"
        ):
            False,

        (
            "deferred_work."
            "targeted_region_dropout."
            "status"
        ):
            "deferred_optional_ablation",

        (
            "deferred_work."
            "targeted_region_dropout."
            "part_of_primary_model"
        ):
            False,

        (
            "deferred_work."
            "jpeg_blur_noise_augmentation."
            "part_of_primary_model"
        ):
            False,

        (
            "deferred_work."
            "high_pass_or_srm_input."
            "part_of_primary_model"
        ):
            False,

        (
            "deferred_work."
            "patch_training."
            "part_of_primary_model"
        ):
            False,

        (
            "deferred_work."
            "from_scratch_resnet18."
            "part_of_primary_model"
        ):
            False,

        # --------------------------------------------------------------
        # Grad-CAM
        # --------------------------------------------------------------

        "gradcam.status":
            "deferred_until_classifier_validated",

        "gradcam.classifier_training_uses_gradcam_supervision":
            False,

        "gradcam.target_layer.status":
            "not_yet_frozen",

        "gradcam.canonical_raw_map_contract.dtype":
            "float32",

        "gradcam.canonical_raw_map_contract.shape":
            "HxW",

        (
            "gradcam."
            "rendered_visualisation_separate_from_raw_map"
        ):
            True,

        # --------------------------------------------------------------
        # Held-out test
        # --------------------------------------------------------------

        "held_out_test.policy":
            "untouched_until_final_evaluation",

        (
            "held_out_test."
            "final_detection_access."
            "only_after_primary_configuration_is_frozen"
        ):
            True,

        (
            "held_out_test."
            "final_detection_access."
            "checkpoints"
        ):
            "all_three_selected_resolution_seed_checkpoints",

        (
            "held_out_test."
            "final_detection_access."
            "thresholds"
        ):
            "each_checkpoint_uses_its_own_dev_derived_threshold",
    }

    for (
        dotted_path,
        expected,
    ) in fixed_values.items():

        check_config_value(
            config,
            dotted_path,
            expected,
        )

    # ------------------------------------------------------------------
    # Stage-A -> Stage-B branch/RNG reset contract
    # ------------------------------------------------------------------

    validate_stage_b_branch_initialisation(
        config
    )

    # ------------------------------------------------------------------
    # Protocol status can advance from validation-ready to frozen later.
    # ------------------------------------------------------------------

    protocol_status = config_value(
        config,
        "experiment.protocol_status",
    )

    if protocol_status not in {
        "ready_for_validation",
        "frozen",
    }:

        raise ValueError(
            "experiment.protocol_status must be either "
            "'ready_for_validation' or 'frozen', "
            f"got {protocol_status!r}"
        )

    log_pass(
        "experiment.protocol_status",
        protocol_status,
    )

    # ------------------------------------------------------------------
    # Pretrained checkpoint identity
    # ------------------------------------------------------------------

    checkpoint_sha = require_sha256_string(
        "model.pretrained_weights.checkpoint_sha256",
        config_value(
            config,
            "model.pretrained_weights.checkpoint_sha256",
        ),
    )

    check_equal(
        "ResNet-18 IMAGENET1K_V1 checkpoint SHA-256",
        checkpoint_sha,
        EXPECTED_RESNET18_IMAGENET1K_V1_SHA256,
    )

    # ------------------------------------------------------------------
    # Concrete screening seeds must agree.
    # ------------------------------------------------------------------

    screening_seed = int(
        config_value(
            config,
            "reproducibility.seed_plan.screening_seed",
        )
    )

    for seed_name in (
        "python",
        "numpy",
        "torch",
        "dataloader",
    ):

        check_equal(
            f"screening seed binding: {seed_name}",
            config_value(
                config,
                f"reproducibility.seeds.{seed_name}",
            ),
            screening_seed,
        )

    check_equal(
        "LR-screening seed matches reproducibility plan",
        config_value(
            config,
            "development_selection.lr_screening.seed",
        ),
        screening_seed,
    )

    confirmation_seeds = config_value(
        config,
        "reproducibility.seed_plan.confirmation_seeds",
    )

    check_equal(
        "multi-seed plan matches reproducibility plan",
        config_value(
            config,
            (
                "development_selection."
                "multi_seed_confirmation."
                "seeds"
            ),
        ),
        confirmation_seeds,
    )

    check_equal(
        "post-selection seeds match confirmation plan",
        config_value(
            config,
            (
                "post_selection."
                "selected_resolution_seed_checkpoints."
                "seeds"
            ),
        ),
        confirmation_seeds,
    )

    # ------------------------------------------------------------------
    # Runtime machine contract referenced by scientific config.
    # ------------------------------------------------------------------

    runtime = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
    )

    num_workers = require_key(
        runtime,
        "num_workers",
        "machine_config.runtime",
    )

    if (
        not isinstance(
            num_workers,
            int,
        )
        or
        isinstance(
            num_workers,
            bool,
        )
        or
        num_workers < 0
    ):

        raise TypeError(
            "machine_config.runtime.num_workers must be "
            f"a non-negative integer, got {num_workers!r}"
        )

    log_pass(
        "machine runtime num_workers scientific reference",
        num_workers,
    )

    # ------------------------------------------------------------------
    # Class counts from actual frozen manifests
    # ------------------------------------------------------------------

    train_path = resolve_path(
        config_value(
            config,
            "data.frozen_split.project_train.path",
        )
    )

    dev_path = resolve_path(
        config_value(
            config,
            "data.frozen_split.dev_val.path",
        )
    )

    train_counts = manifest_traffic_counts(
        train_path
    )

    dev_counts = manifest_traffic_counts(
        dev_path
    )

    expected_train_counts = {
        "bonafide": 480,
        "attack": 960,
    }

    expected_dev_counts = {
        "bonafide": 153,
        "attack": 306,
    }

    check_equal(
        "actual frozen project_train class counts",
        train_counts,
        expected_train_counts,
    )

    check_equal(
        "actual frozen dev_val class counts",
        dev_counts,
        expected_dev_counts,
    )

    check_equal(
        "class-contract project_train bona-fide count",
        config_value(
            config,
            (
                "class_contract."
                "project_train_counts."
                "bonafide"
            ),
        ),
        train_counts[
            "bonafide"
        ],
    )

    check_equal(
        "class-contract project_train attack count",
        config_value(
            config,
            (
                "class_contract."
                "project_train_counts."
                "attack"
            ),
        ),
        train_counts[
            "attack"
        ],
    )

    check_equal(
        "class-contract project_train total",
        config_value(
            config,
            "class_contract.project_train_counts.total",
        ),
        sum(
            train_counts.values()
        ),
    )

    check_equal(
        "class-contract dev_val bona-fide count",
        config_value(
            config,
            "class_contract.dev_val_counts.bonafide",
        ),
        dev_counts[
            "bonafide"
        ],
    )

    check_equal(
        "class-contract dev_val attack count",
        config_value(
            config,
            "class_contract.dev_val_counts.attack",
        ),
        dev_counts[
            "attack"
        ],
    )

    check_equal(
        "class-contract dev_val total",
        config_value(
            config,
            "class_contract.dev_val_counts.total",
        ),
        sum(
            dev_counts.values()
        ),
    )

    # ------------------------------------------------------------------
    # Independently derive class weights.
    # ------------------------------------------------------------------

    total_train = sum(
        train_counts.values()
    )

    derived_weights = {
        class_name:
            (
                total_train
                /
                (
                    2.0
                    * class_count
                )
            )
        for (
            class_name,
            class_count,
        ) in train_counts.items()
    }

    for class_name in (
        "bonafide",
        "attack",
    ):

        check_close(
            (
                "derived class weight vs "
                "configured class-name weight: "
                f"{class_name}"
            ),
            config_value(
                config,
                (
                    "loss."
                    "class_weighting."
                    "expected_by_class_name."
                    f"{class_name}"
                ),
            ),
            derived_weights[
                class_name
            ],
        )

        class_index = int(
            config_value(
                config,
                (
                    "class_contract."
                    "index_by_name."
                    f"{class_name}"
                ),
            )
        )

        check_close(
            (
                "derived class weight vs "
                "configured class-index weight: "
                f"{class_index}"
            ),
            config_value(
                config,
                (
                    "loss."
                    "class_weighting."
                    "expected_by_class_index."
                    f"{class_index}"
                ),
            ),
            derived_weights[
                class_name
            ],
        )

    # ------------------------------------------------------------------
    # Neutral padding / ImageNet normalization
    # ------------------------------------------------------------------

    imagenet_mean = [
        0.485,
        0.456,
        0.406,
    ]

    imagenet_std = [
        0.229,
        0.224,
        0.225,
    ]

    check_equal(
        "padding fill equals ImageNet mean",
        config_value(
            config,
            "preprocessing.padding.fill_rgb_0_to_1",
        ),
        imagenet_mean,
    )

    check_equal(
        "normalization mean",
        config_value(
            config,
            "preprocessing.normalization.mean",
        ),
        imagenet_mean,
    )

    check_equal(
        "normalization std",
        config_value(
            config,
            "preprocessing.normalization.std",
        ),
        imagenet_std,
    )

    # ------------------------------------------------------------------
    # Baseline augmentation must remain entirely disabled.
    # ------------------------------------------------------------------

    stochastic = require_mapping(
        config_value(
            config,
            "preprocessing.stochastic_augmentation",
        ),
        "preprocessing.stochastic_augmentation",
    )

    for (
        name,
        value,
    ) in stochastic.items():

        if value is not False:

            raise ValueError(
                "Primary baseline stochastic augmentation "
                "must remain disabled:\n"
                f"  {name}={value!r}"
            )

    log_pass(
        "all stochastic baseline augmentations disabled"
    )

    # ------------------------------------------------------------------
    # Stopping semantics duplicated across stages must agree.
    # ------------------------------------------------------------------

    global_relative_improvement = float(
        config_value(
            config,
            (
                "stopping_definition."
                "meaningful_relative_improvement_fraction"
            ),
        )
    )

    for stage in (
        "stage_a",
        "stage_b",
    ):

        check_close(
            (
                f"transfer_learning.{stage} stopping "
                "relative-improvement criterion"
            ),
            config_value(
                config,
                (
                    f"transfer_learning.{stage}."
                    "stopping."
                    "meaningful_relative_improvement_fraction"
                ),
            ),
            global_relative_improvement,
        )

    # ------------------------------------------------------------------
    # Stage-B LR screen consistency
    # ------------------------------------------------------------------

    stage_b_lrs = config_value(
        config,
        (
            "transfer_learning."
            "stage_b."
            "optimizer."
            "backbone_lr_candidates"
        ),
    )

    screening_lrs = config_value(
        config,
        (
            "development_selection."
            "lr_screening."
            "backbone_lr_candidates"
        ),
    )

    check_equal(
        "Stage-B LR candidates equal screening candidates",
        stage_b_lrs,
        screening_lrs,
    )

    stage_a_fc_lr = float(
        config_value(
            config,
            (
                "transfer_learning."
                "stage_a."
                "optimizer."
                "classifier_lr"
            ),
        )
    )

    stage_b_fc_lr = float(
        config_value(
            config,
            (
                "transfer_learning."
                "stage_b."
                "optimizer."
                "classifier_lr"
            ),
        )
    )

    screening_fc_lr = float(
        config_value(
            config,
            (
                "development_selection."
                "lr_screening."
                "classifier_lr_fixed"
            ),
        )
    )

    check_close(
        "Stage-A classifier LR vs Stage-B classifier LR",
        stage_a_fc_lr,
        stage_b_fc_lr,
    )

    check_close(
        "Stage-B classifier LR vs screening classifier LR",
        stage_b_fc_lr,
        screening_fc_lr,
    )

    # ------------------------------------------------------------------
    # Candidate resolution lists must reconcile.
    # ------------------------------------------------------------------

    candidate_resolutions = list(
        config_value(
            config,
            "preprocessing.candidate_canvases",
        ).keys()
    )

    check_equal(
        "LR-screening resolutions match candidate canvases",
        config_value(
            config,
            (
                "development_selection."
                "lr_screening."
                "candidate_resolutions"
            ),
        ),
        candidate_resolutions,
    )

    check_equal(
        "multi-seed resolutions match candidate canvases",
        config_value(
            config,
            (
                "development_selection."
                "multi_seed_confirmation."
                "resolutions"
            ),
        ),
        candidate_resolutions,
    )

    # ------------------------------------------------------------------
    # Resolution rule
    # ------------------------------------------------------------------

    check_close(
        "resolution non-inferiority margin",
        config_value(
            config,
            (
                "development_selection."
                "resolution_selection."
                "noninferiority_margin."
                "relative_fraction"
            ),
        ),
        0.05,
    )

    check_equal(
        "resolution r512 decision rule",
        config_value(
            config,
            (
                "development_selection."
                "resolution_selection."
                "decision_rule."
                "select_r512_if"
            ),
        ),
        "mean_d <= margin",
    )

    check_equal(
        "resolution r256 decision rule",
        config_value(
            config,
            (
                "development_selection."
                "resolution_selection."
                "decision_rule."
                "select_r256_if"
            ),
        ),
        "mean_d > margin",
    )

    # ------------------------------------------------------------------
    # Threshold contract
    # ------------------------------------------------------------------

    check_close(
        "dev threshold target false-positive rate",
        config_value(
            config,
            (
                "evaluation."
                "threshold_selection."
                "target_false_positive_rate"
            ),
        ),
        0.10,
    )

    check_equal(
        "threshold positive class agrees with class contract",
        config_value(
            config,
            (
                "evaluation."
                "threshold_selection."
                "positive_class"
            ),
        ),
        config_value(
            config,
            "class_contract.positive_class",
        ),
    )

    # ------------------------------------------------------------------
    # Held-out test forbidden stages
    # ------------------------------------------------------------------

    forbidden = set(
        config_value(
            config,
            "held_out_test.forbidden_during",
        )
    )

    required_forbidden = {
        "preprocessing_selection",
        "learning_rate_screening",
        "multi_seed_stability_analysis",
        "resolution_selection",
        "checkpoint_selection",
        "augmentation_selection",
        "gradcam_target_layer_selection",
        "threshold_design",
    }

    check_equal(
        "held-out test forbidden development stages",
        forbidden,
        required_forbidden,
    )

    # ------------------------------------------------------------------
    # Grad-CAM raw-map range
    # ------------------------------------------------------------------

    check_close(
        "Grad-CAM raw-map minimum",
        config_value(
            config,
            (
                "gradcam."
                "canonical_raw_map_contract."
                "range.minimum"
            ),
        ),
        0.0,
    )

    check_close(
        "Grad-CAM raw-map maximum",
        config_value(
            config,
            (
                "gradcam."
                "canonical_raw_map_contract."
                "range.maximum"
            ),
        ),
        1.0,
    )

    # ------------------------------------------------------------------
    # Empirical support artifacts
    # ------------------------------------------------------------------

    validate_protocol_preflight_evidence(
        config
    )

    log_pass(
        "ResNet-18 transfer-learning scientific protocol"
    )


# ======================================================================
# Held-out test policy
# ======================================================================

def validate_test_policy(
    config: dict[str, Any],
) -> None:

    test_cfg = require_mapping(
        require_key(
            config,
            "held_out_test",
            "config",
        ),
        "held_out_test",
    )

    check_equal(
        "held-out test policy",
        require_key(
            test_cfg,
            "policy",
            "held_out_test",
        ),
        "untouched_until_final_evaluation",
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Validate Tech-2 scientific experiment "
            "configuration and frozen provenance."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/experiments/"
            "resnet18_gradcam.yaml"
        ),
        help=(
            "Committed scientific experiment YAML."
        ),
    )

    parser.add_argument(
        "--machine-config",
        default="configs/local.yaml",
        help=(
            "Machine-local YAML."
        ),
    )

    parser.add_argument(
        "--validator-config",
        default=(
            "tools/"
            "validate_experiment_config_config.yaml"
        ),
        help=(
            "Validator logging configuration."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Capture Git state BEFORE creating the new log.
    # ------------------------------------------------------------------

    commit_sha, git_porcelain = (
        git_state()
    )

    tool_config_path = resolve_path(
        args.validator_config
    )

    (
        logger,
        log_path,
        file_handler,
        stream_handler,
    ) = configure_logging(
        tool_config_path
    )

    status = "FAIL"
    exit_code = 1

    try:

        logger.info(
            "=" * 72
        )

        logger.info(
            "TECH-2 EXPERIMENT CONFIGURATION VALIDATION"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Git commit: %s",
            commit_sha,
        )

        logger.info(
            "Git working tree clean before validation log creation: %s",
            not bool(
                git_porcelain
            ),
        )

        if git_porcelain:

            logger.warning(
                "Git working tree contains uncommitted or "
                "untracked changes. Validation is diagnostic "
                "until the scientific code/config is committed."
            )

            for line in (
                git_porcelain.splitlines()
            ):

                logger.warning(
                    "GIT_STATUS | %s",
                    line,
                )

        logger.info(
            "Validator script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "Validator config: %s",
            tool_config_path,
        )

        logger.info(
            "Validator config SHA-256: %s",
            sha256_file(
                tool_config_path
            ),
        )

        # --------------------------------------------------------------
        # Load experiment and machine configs.
        # --------------------------------------------------------------

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

        machine = require_mapping(
            require_key(
                machine_cfg,
                "machine",
                "machine_config",
            ),
            "machine_config.machine",
        )

        machine_id = require_key(
            machine,
            "id",
            "machine_config.machine",
        )

        logger.info(
            "Experiment config: %s",
            experiment_path,
        )

        logger.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        logger.info(
            "Machine config: %s",
            machine_path,
        )

        logger.info(
            "Machine ID: %s",
            machine_id,
        )

        logger.info(
            "Machine-config SHA-256 intentionally not "
            "treated as scientific provenance."
        )

        # --------------------------------------------------------------
        # Machine contract
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Machine contract ---"
        )

        validate_machine_config(
            machine_cfg
        )

        # --------------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Reproducibility contract ---"
        )

        validate_seeds(
            experiment_cfg
        )

        # --------------------------------------------------------------
        # Frozen upstream provenance
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Frozen Tech-1 provenance ---"
        )

        validate_frozen_data(
            experiment_cfg
        )

        # --------------------------------------------------------------
        # Schema-v2 scientific protocol
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- ResNet-18 scientific development protocol ---"
        )

        validate_scientific_protocol(
            experiment_cfg,
            machine_cfg,
        )

        # --------------------------------------------------------------
        # Held-out test policy
        # --------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Held-out test contract ---"
        )

        validate_test_policy(
            experiment_cfg
        )

        logger.info(
            ""
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "EXPERIMENT CONFIGURATION VALIDATION: PASS"
        )

        logger.info(
            "=" * 72
        )

        status = "PASS"
        exit_code = 0

    except Exception:

        logger.exception(
            "EXPERIMENT CONFIGURATION VALIDATION: FAIL"
        )

        status = "FAIL"
        exit_code = 1

    finally:

        logger.info(
            "Canonical validation log: %s",
            log_path,
        )

        logger.info(
            "Final status: %s",
            status,
        )

        # --------------------------------------------------------------
        # Finalise canonical log before computing its digest.
        # --------------------------------------------------------------

        file_handler.flush()

        logger.removeHandler(
            file_handler
        )

        file_handler.close()

        validation_log_sha = sha256_file(
            log_path
        )

        # --------------------------------------------------------------
        # The ONLY routine stdout record.
        #
        # Existing audit tools can capture this one line and therefore
        # reference the canonical validation evidence without copying
        # the entire transcript into their own logs.
        # --------------------------------------------------------------

        logger.info(
            (
                "VALIDATION_ARTIFACT"
                " | status=%s"
                " | path=%s"
                " | sha256=%s"
            ),
            status,
            log_path,
            validation_log_sha,
            extra={
                "handoff": True,
            },
        )

        stream_handler.flush()

        logger.removeHandler(
            stream_handler
        )

        stream_handler.close()

        logger.handlers.clear()

    return exit_code


if __name__ == "__main__":

    raise SystemExit(
        main()
    )