#!/usr/bin/env python3
"""
Derive one final ResNet-18 checkpoint's FPR10 threshold from dev_val.

Scientific role
---------------
This is post-selection evaluation setup.

For one already-frozen selected checkpoint:

    checkpoint
        ->
    complete frozen dev_val inference
        ->
    extract ONLY the 153 bona-fide logits
        ->
    float64 p_attack
        ->
    b16
        ->
    nextafter(b16, +infinity)
        ->
    seed-specific FPR10 threshold

After the threshold has been derived, complete dev predictions are
persisted for later metric calculation.

Important boundaries
--------------------
This tool:

- does NOT train;
- does NOT perform checkpoint selection;
- does NOT perform LR or resolution selection;
- does NOT use attack scores to select FPR10;
- does NOT access held-out test;
- supports dev_val only through the already-frozen development Dataset;
- requires execution on the physical machine holding the canonical
  checkpoint binary.

The first real invocation should be seed 9 on Home. That run also acts
as the real integration validation before seeds 8 and 10 are processed
on Lab.

No print() is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import DataLoader


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

from src.data import (
    build_fantasyid_dataset,
)

from src.development_metrics import (
    attack_ranking_score_from_logits,
    compute_dev_auroc,
)

from src.engine import (
    evaluate_dev_one_epoch,
)

from src.fpr10 import (
    apply_attack_threshold,
    attack_probability_from_logits,
    derive_fpr10_threshold,
    load_fpr10_contract,
)

from src.modeling import (
    build_resnet18_classifier,
)

from src.objective import (
    WeightedCrossEntropyObjective,
)

from src.reproducibility import (
    configure_run_reproducibility,
    establish_pre_cuda_environment,
    make_dataloader_generator,
    seed_dataloader_worker,
)

from src.screening_artifacts import (
    verify_checkpoint_artifact,
)

from src.training_control import (
    state_dict_sha256,
)


LOGGER = logging.getLogger(
    "derive_resnet18_dev_threshold"
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


def seed_value(
    mapping: Mapping[Any, Any],
    seed: int,
    label: str,
) -> Any:

    if seed in mapping:

        return mapping[
            seed
        ]

    text_seed = str(
        seed
    )

    if text_seed in mapping:

        return mapping[
            text_seed
        ]

    raise KeyError(
        f"Missing {label} for seed {seed}."
    )


def require_clean_git() -> tuple[
    str,
    str,
]:

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
            "Git working tree must be clean before real dev threshold "
            "derivation.\n\n"
            f"{status}"
        )

    if len(
        commit
    ) != 40:

        raise RuntimeError(
            "Git HEAD is not a full 40-character SHA."
        )

    return (
        commit,
        branch,
    )


def require_git_tracked(
    path: Path,
) -> None:

    relative = relative_repo_path(
        path
    )

    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            relative,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Required scientific evidence is not Git-tracked:\n"
            f"  {relative}"
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
            "Frozen experiment validator did not return PASS."
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
            "Validator artifact SHA mismatch."
        )

    # ------------------------------------------------------------------
    # The canonical validator must itself have observed a clean Git tree.
    #
    # The threshold runner therefore has to invoke the validator BEFORE
    # creating its own log/output artifacts.
    # ------------------------------------------------------------------

    validator_text = artifact_path.read_text(
        encoding="utf-8"
    )

    required_clean_marker = (
        "Git working tree clean before validation log creation: True"
    )

    if required_clean_marker not in validator_text:

        raise RuntimeError(
            "Canonical experiment validator did not observe a clean "
            "Git working tree."
        )

    diagnostic_warning = (
        "Validation is diagnostic until the scientific code/config "
        "is committed."
    )

    if diagnostic_warning in validator_text:

        raise RuntimeError(
            "Canonical experiment validator marked validation as "
            "diagnostic rather than clean."
        )



    return (
        artifact_path,
        expected_sha,
    )


# ======================================================================
# Logging
# ======================================================================

def configure_logging(
    *,
    tool_cfg: Mapping[str, Any],
    machine_id: str,
    seed: int,
) -> Path:

    logging_cfg = require_mapping(
        tool_cfg[
            "logging"
        ],
        "logging",
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
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
            machine_id=machine_id,
            seed=seed,
            timestamp=timestamp,
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

    formatter.converter = (
        time.gmtime
    )

    handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        handler
    )

    return path


# ======================================================================
# FPR10-audit prerequisite
# ======================================================================

def validate_fpr10_audit(
    *,
    tool_cfg: Mapping[str, Any],
    protocol_path: Path,
) -> tuple[
    Path,
    str,
]:

    prerequisites = require_mapping(
        tool_cfg[
            "prerequisites"
        ],
        "prerequisites",
    )

    audit_cfg = require_mapping(
        prerequisites[
            "fpr10_primitive_audit"
        ],
        "prerequisites.fpr10_primitive_audit",
    )

    audit_path = resolve_repo_path(
        audit_cfg[
            "path"
        ]
    )

    require_git_tracked(
        audit_path
    )

    expected_sha = str(
        audit_cfg[
            "sha256"
        ]
    )

    actual_sha = sha256_file(
        audit_path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "FPR10 primitive audit SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    audit = load_yaml(
        audit_path
    )

    if audit.get(
        "status"
    ) != audit_cfg[
        "required_status"
    ]:

        raise RuntimeError(
            "FPR10 primitive audit did not record PASS."
        )

    boundaries = require_mapping(
        audit[
            "boundaries"
        ],
        "fpr10_audit.boundaries",
    )

    if boundaries[
        "held_out_test_accessed"
    ] is not False:

        raise RuntimeError(
            "FPR10 primitive audit reports held-out-test access."
        )

    source_hashes = require_mapping(
        audit[
            "source_sha256"
        ],
        "fpr10_audit.source_sha256",
    )

    current_fpr10_sha = sha256_file(
        REPO_ROOT
        / "src"
        / "fpr10.py"
    )

    if (
        source_hashes[
            "src/fpr10.py"
        ]
        != current_fpr10_sha
    ):

        raise RuntimeError(
            "Current src/fpr10.py differs from the audited primitive."
        )

    current_protocol_sha = sha256_file(
        protocol_path
    )

    if (
        source_hashes[
            "evaluation_protocol"
        ]
        != current_protocol_sha
    ):

        raise RuntimeError(
            "Current final-detection protocol differs from the "
            "audited protocol."
        )

    return (
        audit_path,
        actual_sha,
    )


# ======================================================================
# Selected checkpoint evidence
# ======================================================================

def validate_selection_and_source(
    *,
    protocol_cfg: Mapping[str, Any],
    experiment_path: Path,
    machine_cfg: Mapping[str, Any],
    machine_id: str,
    seed: int,
) -> dict[str, Any]:

    upstream = require_mapping(
        protocol_cfg[
            "upstream"
        ],
        "evaluation_protocol.upstream",
    )

    training_item = require_mapping(
        upstream[
            "training_config"
        ],
        "upstream.training_config",
    )

    training_path = resolve_repo_path(
        training_item[
            "path"
        ]
    )

    if training_path != experiment_path:

        raise RuntimeError(
            "Evaluation protocol points to a different training config."
        )

    training_sha = sha256_file(
        training_path
    )

    if (
        training_sha
        != training_item[
            "sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen training-config SHA mismatch."
        )

    selection_item = require_mapping(
        upstream[
            "resolution_selection"
        ],
        "upstream.resolution_selection",
    )

    selection_path = resolve_repo_path(
        selection_item[
            "path"
        ]
    )

    require_git_tracked(
        selection_path
    )

    selection_sha = sha256_file(
        selection_path
    )

    if (
        selection_sha
        != selection_item[
            "sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen resolution-selection SHA mismatch."
        )

    selection = load_yaml(
        selection_path
    )

    if selection.get(
        "status"
    ) != "FROZEN":

        raise RuntimeError(
            "Resolution-selection artifact is not FROZEN."
        )

    plan = require_mapping(
        selection[
            "final_detection_plan"
        ],
        "final_detection_plan",
    )

    if plan[
        "resolution"
    ] != "r512":

        raise RuntimeError(
            "Final detection resolution must be r512."
        )

    if float(
        plan[
            "backbone_lr"
        ]
    ) != 0.0001:

        raise RuntimeError(
            "Final detection backbone LR must be 1e-4."
        )

    allowed_seeds = tuple(
        int(
            value
        )
        for value
        in plan[
            "seeds"
        ]
    )

    if allowed_seeds != (
        8,
        9,
        10,
    ):

        raise RuntimeError(
            "Frozen final seed set changed."
        )

    if seed not in allowed_seeds:

        raise RuntimeError(
            f"Seed {seed} is not a frozen final checkpoint."
        )

    checkpoint_sources = require_mapping(
        plan[
            "checkpoint_sources"
        ],
        "final_detection_plan.checkpoint_sources",
    )

    source = require_mapping(
        seed_value(
            checkpoint_sources,
            seed,
            "checkpoint source",
        ),
        f"checkpoint_sources.{seed}",
    )

    source_machine = str(
        source[
            "training_machine_id"
        ]
    )

    if source_machine != machine_id:

        raise RuntimeError(
            "Canonical checkpoint must be evaluated on its origin "
            "machine for this stage:\n"
            f"  seed={seed}\n"
            f"  checkpoint_machine={source_machine}\n"
            f"  current_machine={machine_id}"
        )

    run_id = str(
        source[
            "run_id"
        ]
    )

    checkpoint_cfg = require_mapping(
        source[
            "checkpoint"
        ],
        f"checkpoint_sources.{seed}.checkpoint",
    )

    development = require_mapping(
        selection[
            "development_evidence"
        ],
        "development_evidence",
    )

    r512_evidence = require_mapping(
        development[
            "r512"
        ],
        "development_evidence.r512",
    )

    dev_source = require_mapping(
        seed_value(
            r512_evidence,
            seed,
            "r512 development source",
        ),
        f"development_evidence.r512.{seed}",
    )

    if dev_source[
        "run_id"
    ] != run_id:

        raise RuntimeError(
            "Development evidence and final checkpoint source disagree."
        )

    dev_checkpoint = require_mapping(
        dev_source[
            "checkpoint"
        ],
        f"development_evidence.r512.{seed}.checkpoint",
    )

    for key in (
        "path",
        "file_sha256",
        "model_state_sha256",
        "epoch",
    ):

        if (
            dev_checkpoint[
                key
            ]
            != checkpoint_cfg[
                key
            ]
        ):

            raise RuntimeError(
                "Final checkpoint metadata differs from development "
                f"evidence: {key}"
            )

    run_yaml_path = resolve_repo_path(
        dev_source[
            "run_yaml"
        ]
    )

    require_git_tracked(
        run_yaml_path
    )

    actual_run_yaml_sha = sha256_file(
        run_yaml_path
    )

    if (
        actual_run_yaml_sha
        != dev_source[
            "run_yaml_sha256"
        ]
    ):

        raise RuntimeError(
            "Source run.yaml SHA mismatch."
        )

    source_run = load_yaml(
        run_yaml_path
    )

    if (
        source_run[
            "run"
        ][
            "run_id"
        ]
        != run_id
    ):

        raise RuntimeError(
            "Source run_id mismatch."
        )

    if (
        source_run[
            "machine"
        ][
            "id"
        ]
        != source_machine
    ):

        raise RuntimeError(
            "Source run machine identity mismatch."
        )

    if (
        source_run[
            "frozen_data"
        ][
            "held_out_test_accessed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Training source run reports held-out-test access."
        )

    # ------------------------------------------------------------------
    # Require the actual inference/preprocessing scientific source files
    # to be byte-identical to those used when this checkpoint was
    # trained and selected.
    # ------------------------------------------------------------------

    historical_source_hashes = require_mapping(
        source_run[
            "production_source_sha256"
        ],
        "source_run.production_source_sha256",
    )

    inference_source_files = (
        "src/data.py",
        "src/reproducibility.py",
        "src/modeling.py",
        "src/objective.py",
        "src/engine.py",
        "src/development_metrics.py",
        "src/screening_artifacts.py",
    )

    current_source_hashes: dict[
        str,
        str,
    ] = {}

    for relative_path in inference_source_files:

        if relative_path not in historical_source_hashes:

            raise RuntimeError(
                "Source run does not record required source SHA:\n"
                f"  {relative_path}"
            )

        current_sha = sha256_file(
            REPO_ROOT
            / relative_path
        )

        current_source_hashes[
            relative_path
        ] = current_sha

        if (
            current_sha
            != historical_source_hashes[
                relative_path
            ]
        ):

            raise RuntimeError(
                "Current inference source differs from checkpoint "
                "training-time source:\n"
                f"  file={relative_path}\n"
                f"  historical="
                f"{historical_source_hashes[relative_path]}\n"
                f"  current={current_sha}"
            )

    machine_paths = require_mapping(
        machine_cfg[
            "paths"
        ],
        "machine_config.paths",
    )

    runs_root = Path(
        machine_paths[
            "runs_root"
        ]
    ).expanduser()

    if not runs_root.is_absolute():

        runs_root = (
            REPO_ROOT
            / runs_root
        )

    runs_root = runs_root.resolve()

    checkpoint_path = (
        runs_root
        / run_id
        / str(
            checkpoint_cfg[
                "path"
            ]
        )
    ).resolve()

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            "Canonical checkpoint binary missing on origin machine:\n"
            f"  {checkpoint_path}"
        )

    checkpoint_file_sha = sha256_file(
        checkpoint_path
    )

    if (
        checkpoint_file_sha
        != checkpoint_cfg[
            "file_sha256"
        ]
    ):

        raise RuntimeError(
            "Checkpoint binary SHA mismatch."
        )

    payload = verify_checkpoint_artifact(
        checkpoint_path
    )

    required_checkpoint_values = {
        "run_id":
            run_id,

        "resolution_name":
            "r512",

        "run_seed":
            seed,

        "stage":
            "stage_b",

        "epoch":
            int(
                checkpoint_cfg[
                    "epoch"
                ]
            ),

        "model_state_sha256":
            str(
                checkpoint_cfg[
                    "model_state_sha256"
                ]
            ),
    }

    for (
        key,
        expected,
    ) in required_checkpoint_values.items():

        if payload[
            key
        ] != expected:

            raise RuntimeError(
                "Checkpoint payload metadata mismatch:\n"
                f"  field={key}\n"
                f"  expected={expected!r}\n"
                f"  actual={payload[key]!r}"
            )

    if float(
        payload[
            "backbone_lr"
        ]
    ) != 0.0001:

        raise RuntimeError(
            "Checkpoint backbone LR is not 1e-4."
        )

    historical_dev_loss = float(
        dev_source[
            "raw_best_weighted_dev_loss"
        ]
    )

    if float(
        payload[
            "weighted_dev_loss"
        ]
    ) != historical_dev_loss:

        raise RuntimeError(
            "Checkpoint loss metadata differs from frozen "
            "development evidence."
        )

    source_result = require_mapping(
        source_run[
            "result"
        ],
        "source_run.result",
    )

    historical_dev_auroc = (
        source_result.get(
            "dev_auroc_at_raw_best_checkpoint"
        )
    )

    return {
        "selection_path":
            selection_path,

        "selection_sha256":
            selection_sha,

        "run_id":
            run_id,

        "run_yaml_path":
            run_yaml_path,

        "run_yaml_sha256":
            actual_run_yaml_sha,

        "source_machine":
            source_machine,

        "checkpoint_path":
            checkpoint_path,

        "checkpoint_file_sha256":
            checkpoint_file_sha,

        "checkpoint_model_state_sha256":
            checkpoint_cfg[
                "model_state_sha256"
            ],

        "checkpoint_epoch":
            int(
                checkpoint_cfg[
                    "epoch"
                ]
            ),

        "historical_dev_loss":
            historical_dev_loss,

        "historical_dev_auroc":
            (
                None
                if historical_dev_auroc is None
                else float(
                    historical_dev_auroc
                )
            ),

        "checkpoint_payload":
            payload,

        "current_inference_source_sha256":
            current_source_hashes,
    }


# ======================================================================
# Dev-only DataLoader
# ======================================================================

def build_dev_loader(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    seed: int,
) -> tuple[
    Any,
    DataLoader,
]:

    dataloader_cfg = require_mapping(
        experiment_cfg[
            "dataloader"
        ],
        "dataloader",
    )

    batch_size = int(
        dataloader_cfg[
            "batch_size"
        ]
    )

    if batch_size != 32:

        raise RuntimeError(
            "Frozen batch size must remain 32."
        )

    dev_cfg = require_mapping(
        dataloader_cfg[
            "dev_val"
        ],
        "dataloader.dev_val",
    )

    if dev_cfg[
        "shuffle"
    ] is not False:

        raise RuntimeError(
            "Frozen dev DataLoader requires shuffle=false."
        )

    if dev_cfg[
        "drop_last"
    ] is not False:

        raise RuntimeError(
            "Frozen dev DataLoader requires drop_last=false."
        )

    runtime_cfg = require_mapping(
        machine_cfg[
            "runtime"
        ],
        "machine_config.runtime",
    )

    num_workers = runtime_cfg[
        "num_workers"
    ]

    if (
        not isinstance(
            num_workers,
            int,
        )
        or isinstance(
            num_workers,
            bool,
        )
        or num_workers < 0
    ):

        raise RuntimeError(
            "Machine num_workers must be a non-negative integer."
        )

    dataset = build_fantasyid_dataset(
        experiment_cfg=(
            experiment_cfg
        ),
        machine_cfg=(
            machine_cfg
        ),
        repo_root=(
            REPO_ROOT
        ),
        split_name="dev_val",
        resolution_name="r512",
    )

    generator = make_dataloader_generator(
        seed
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )

    return (
        dataset,
        loader,
    )


# ======================================================================
# Prediction artifact
# ======================================================================

PREDICTION_COLUMNS = (
    "row_index",
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "target",
    "bonafide_logit_fp32",
    "attack_logit_fp32",
    "attack_margin_float64",
    "p_attack_float64",
    "prediction_fixed_0_5",
    "prediction_fpr10",
)


def write_prediction_csv(
    *,
    path: Path,
    dataset: Any,
    logits: torch.Tensor,
    targets: torch.Tensor,
    margins: torch.Tensor,
    scores: torch.Tensor,
    fixed_predictions: torch.Tensor,
    fpr10_predictions: torch.Tensor,
) -> None:

    row_count = len(
        dataset
    )

    tensors = (
        logits,
        targets,
        margins,
        scores,
        fixed_predictions,
        fpr10_predictions,
    )

    for tensor in tensors:

        if int(
            tensor.shape[
                0
            ]
        ) != row_count:

            raise RuntimeError(
                "Prediction artifact tensor row-count mismatch."
            )

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=PREDICTION_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )

        writer.writeheader()

        for index in range(
            row_count
        ):

            row = dataset.rows[
                index
            ]

            expected_target = int(
                row[
                    "label"
                ]
            )

            actual_target = int(
                targets[
                    index
                ].item()
            )

            if (
                actual_target
                != expected_target
            ):

                raise RuntimeError(
                    "Prediction target differs from manifest target."
                )

            writer.writerow(
                {
                    "row_index":
                        index,

                    "image_path":
                        row[
                            "image_path_relative"
                        ],

                    "image_sha256":
                        row[
                            "image_sha256"
                        ],

                    "file_stem":
                        row[
                            "file_stem"
                        ],

                    "traffic_type":
                        row[
                            "traffic_type"
                        ],

                    "variant":
                        row[
                            "variant"
                        ],

                    "hardware_source":
                        row[
                            "hardware_source"
                        ],

                    "target":
                        actual_target,

                    "bonafide_logit_fp32":
                        format(
                            float(
                                logits[
                                    index,
                                    0,
                                ].item()
                            ),
                            ".17g",
                        ),

                    "attack_logit_fp32":
                        format(
                            float(
                                logits[
                                    index,
                                    1,
                                ].item()
                            ),
                            ".17g",
                        ),

                    "attack_margin_float64":
                        format(
                            float(
                                margins[
                                    index
                                ].item()
                            ),
                            ".17g",
                        ),

                    "p_attack_float64":
                        format(
                            float(
                                scores[
                                    index
                                ].item()
                            ),
                            ".17g",
                        ),

                    "prediction_fixed_0_5":
                        int(
                            fixed_predictions[
                                index
                            ].item()
                        ),

                    "prediction_fpr10":
                        int(
                            fpr10_predictions[
                                index
                            ].item()
                        ),
                }
            )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Derive one selected ResNet-18 checkpoint's "
            "dev-only FPR10 threshold."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=(
            8,
            9,
            10,
        ),
    )

    parser.add_argument(
        "--tool-config",
        default=(
            "tools/"
            "derive_resnet18_dev_threshold_config.yaml"
        ),
    )

    args = parser.parse_args()

    seed = int(
        args.seed
    )

    git_commit, git_branch = (
        require_clean_git()
    )

    tool_cfg_path = resolve_repo_path(
        args.tool_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            tool_cfg[
                "experiment"
            ][
                "config"
            ]
        )
    )

    machine_cfg, machine_path = (
        load_machine_config(
            tool_cfg[
                "experiment"
            ][
                "machine_config"
            ],
            required=True,
        )
    )

    machine_id = str(
        machine_cfg[
            "machine"
        ][
            "id"
        ]
    )

    # ==============================================================
    # PRE-OUTPUT PREFLIGHT
    #
    # Nothing belonging to this threshold-derivation run has been
    # created yet. Therefore the canonical validator sees the same
    # clean Git tree already established by require_clean_git().
    # ==============================================================

    workspace = establish_pre_cuda_environment(
        experiment_cfg=(
            experiment_cfg
        )
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



    log_path = configure_logging(
        tool_cfg=(
            tool_cfg
        ),
        machine_id=(
            machine_id
        ),
        seed=seed,
    )

    output_cfg = require_mapping(
        tool_cfg[
            "output"
        ],
        "output",
    )

    threshold_directory = resolve_repo_path(
        output_cfg[
            "threshold_directory"
        ]
    )

    predictions_directory = resolve_repo_path(
        output_cfg[
            "predictions_directory"
        ]
    )

    threshold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_path = (
        threshold_directory
        / str(
            output_cfg[
                "threshold_filename"
            ]
        ).format(
            seed=seed
        )
    )

    predictions_path = (
        predictions_directory
        / str(
            output_cfg[
                "predictions_filename"
            ]
        ).format(
            seed=seed
        )
    )

    threshold_partial = Path(
        str(
            threshold_path
        )
        + ".partial"
    )

    predictions_partial = Path(
        str(
            predictions_path
        )
        + ".partial"
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 REAL DEV THRESHOLD DERIVATION"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            git_commit,
        )

        LOGGER.info(
            "Git branch: %s",
            git_branch,
        )

        LOGGER.info(
            "Machine: %s",
            machine_id,
        )

        LOGGER.info(
            "Seed: %d",
            seed,
        )

        if threshold_path.exists():

            raise FileExistsError(
                "Canonical threshold artifact already exists:\n"
                f"  {threshold_path}"
            )

        if predictions_path.exists():

            raise FileExistsError(
                "Canonical dev-prediction artifact already exists:\n"
                f"  {predictions_path}"
            )

        if threshold_partial.exists():

            raise FileExistsError(
                threshold_partial
            )

        if predictions_partial.exists():

            raise FileExistsError(
                predictions_partial
            )

        

        LOGGER.info(
            "[PASS] pre-CUDA CUBLAS_WORKSPACE_CONFIG = %s",
            workspace,
        )
   
        LOGGER.info(
            "[PASS] canonical experiment validator"
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
        # Evaluation protocol and audited FPR primitive
        # ==============================================================

        protocol_path = resolve_repo_path(
            tool_cfg[
                "evaluation"
            ][
                "protocol"
            ]
        )

        require_git_tracked(
            protocol_path
        )

        protocol_cfg = load_yaml(
            protocol_path
        )

        contract = load_fpr10_contract(
            evaluation_cfg=(
                protocol_cfg
            )
        )

        protocol_sha = sha256_file(
            protocol_path
        )

        (
            fpr_audit_path,
            fpr_audit_sha,
        ) = validate_fpr10_audit(
            tool_cfg=(
                tool_cfg
            ),
            protocol_path=(
                protocol_path
            ),
        )

        LOGGER.info(
            "[PASS] FPR10 primitive audit prerequisite"
        )

        LOGGER.info(
            "FPR10 audit: %s",
            fpr_audit_path,
        )

        LOGGER.info(
            "FPR10 audit SHA-256: %s",
            fpr_audit_sha,
        )

        # ==============================================================
        # Frozen selection + canonical checkpoint
        # ==============================================================

        source = validate_selection_and_source(
            protocol_cfg=(
                protocol_cfg
            ),
            experiment_path=(
                experiment_path
            ),
            machine_cfg=(
                machine_cfg
            ),
            machine_id=(
                machine_id
            ),
            seed=seed,
        )

        LOGGER.info(
            "[PASS] frozen final configuration = r512 / LR 1e-4"
        )

        LOGGER.info(
            "[PASS] seed %d canonical checkpoint origin = %s",
            seed,
            source[
                "source_machine"
            ],
        )

        LOGGER.info(
            "[PASS] checkpoint file SHA-256 = %s",
            source[
                "checkpoint_file_sha256"
            ],
        )

        LOGGER.info(
            "[PASS] checkpoint model-state SHA-256 = %s",
            source[
                "checkpoint_model_state_sha256"
            ],
        )

        LOGGER.info(
            "[PASS] current inference source matches "
            "checkpoint training-time source"
        )

        # ==============================================================
        # Reproducibility / device
        # ==============================================================

        reproducibility_state = (
            configure_run_reproducibility(
                experiment_cfg=(
                    experiment_cfg
                ),
                run_seed=seed,
            )
        )

        runtime_cfg = require_mapping(
            machine_cfg[
                "runtime"
            ],
            "machine_config.runtime",
        )

        device = torch.device(
            str(
                runtime_cfg[
                    "device"
                ]
            )
        )

        if device.type != "cuda":

            raise RuntimeError(
                "Real final-development inference must use the "
                "configured CUDA device."
            )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is not available."
            )

        LOGGER.info(
            "[PASS] deterministic run contract established for seed %d",
            seed,
        )

        LOGGER.info(
            "Configured device: %s",
            device,
        )

        # ==============================================================
        # Build model and restore exact selected state
        # ==============================================================

        model, model_provenance = (
            build_resnet18_classifier(
                experiment_cfg=(
                    experiment_cfg
                )
            )
        )

        checkpoint_payload = (
            source[
                "checkpoint_payload"
            ]
        )

        load_result = model.load_state_dict(
            checkpoint_payload[
                "model_state_dict"
            ],
            strict=True,
        )

        if load_result.missing_keys:

            raise RuntimeError(
                "Checkpoint load produced missing keys:\n"
                f"  {load_result.missing_keys}"
            )

        if load_result.unexpected_keys:

            raise RuntimeError(
                "Checkpoint load produced unexpected keys:\n"
                f"  {load_result.unexpected_keys}"
            )

        loaded_state_sha = state_dict_sha256(
            model.state_dict()
        )

        if (
            loaded_state_sha
            != source[
                "checkpoint_model_state_sha256"
            ]
        ):

            raise RuntimeError(
                "Loaded model state SHA differs from selected checkpoint."
            )

        LOGGER.info(
            "[PASS] exact selected model state restored"
        )

        model = model.to(
            device
        )

        objective = WeightedCrossEntropyObjective(
            experiment_cfg=(
                experiment_cfg
            ),
            device=(
                device
            ),
        )

        # ==============================================================
        # Dev-only Dataset / DataLoader
        # ==============================================================

        dev_dataset, dev_loader = build_dev_loader(
            experiment_cfg=(
                experiment_cfg
            ),
            machine_cfg=(
                machine_cfg
            ),
            seed=seed,
        )

        if len(
            dev_dataset
        ) != 459:

            raise RuntimeError(
                "Frozen dev Dataset must contain exactly 459 images."
            )

        LOGGER.info(
            "[PASS] dev-only Dataset constructed: 459 images"
        )

        LOGGER.info(
            "[PASS] held-out test route absent from Dataset constructor"
        )

        # ==============================================================
        # Complete real dev inference
        # ==============================================================

        inference_started = time.perf_counter()

        dev_result = evaluate_dev_one_epoch(
            model=(
                model
            ),
            loader=(
                dev_loader
            ),
            objective=(
                objective
            ),
            device=(
                device
            ),
        )

        inference_seconds = (
            time.perf_counter()
            - inference_started
        )

        if dev_result.loss.sample_count != 459:

            raise RuntimeError(
                "Real dev inference did not process exactly 459 images."
            )

        if dev_result.loss.bonafide_count != 153:

            raise RuntimeError(
                "Real dev inference did not contain exactly "
                "153 bona-fide images."
            )

        if dev_result.loss.attack_count != 306:

            raise RuntimeError(
                "Real dev inference did not contain exactly "
                "306 attack images."
            )

        LOGGER.info(
            "[PASS] complete dev inference = 459 = 153 bona-fide + "
            "306 attack"
        )

        LOGGER.info(
            "Dev inference seconds: %.6f",
            inference_seconds,
        )

        # ==============================================================
        # Historical selected-checkpoint reproduction
        # ==============================================================

        historical_loss = float(
            source[
                "historical_dev_loss"
            ]
        )

        observed_loss = float(
            dev_result.loss.weighted_loss
        )

        loss_delta = (
            observed_loss
            - historical_loss
        )

        if not math.isclose(
            observed_loss,
            historical_loss,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):

            raise RuntimeError(
                "Loaded checkpoint did not reproduce historical "
                "weighted dev CE:\n"
                f"  historical={historical_loss:.17g}\n"
                f"  observed={observed_loss:.17g}\n"
                f"  delta={loss_delta:.17g}"
            )

        LOGGER.info(
            "[PASS] weighted dev CE reproduces selected checkpoint"
        )

        LOGGER.info(
            "Historical dev CE: %.17g",
            historical_loss,
        )

        LOGGER.info(
            "Observed dev CE:   %.17g",
            observed_loss,
        )

        LOGGER.info(
            "Dev CE delta:      %.17g",
            loss_delta,
        )

        # ==============================================================
        # IMPORTANT:
        #
        # Derive FPR10 BEFORE computing full-dev p_attack.
        #
        # Only bona-fide logits are converted to probabilities for the
        # threshold-selection call.
        # ==============================================================

        bonafide_mask = (
            dev_result.targets
            == 0
        )

        bonafide_logits = (
            dev_result.logits[
                bonafide_mask
            ]
            .contiguous()
        )

        if tuple(
            bonafide_logits.shape
        ) != (
            153,
            2,
        ):

            raise RuntimeError(
                "Threshold-selection logit population must be [153,2]."
            )

        bonafide_scores = (
            attack_probability_from_logits(
                bonafide_logits
            )
        )

        threshold_result = (
            derive_fpr10_threshold(
                bonafide_scores=(
                    bonafide_scores
                ),
                contract=(
                    contract
                ),
            )
        )

        threshold = float(
            threshold_result.threshold
        )

        LOGGER.info(
            "[PASS] FPR10 derived from bona-fide dev scores ONLY"
        )

        LOGGER.info(
            "FPR10 K: %d",
            threshold_result.allowed_false_positive_count,
        )

        LOGGER.info(
            "FPR10 boundary rank: %d",
            threshold_result.boundary_rank_one_based,
        )

        LOGGER.info(
            "FPR10 boundary score b16: %.17g",
            threshold_result.boundary_score,
        )

        LOGGER.info(
            "FPR10 threshold nextafter(b16,+inf): %.17g",
            threshold,
        )

        LOGGER.info(
            "Boundary tie count: %d",
            threshold_result.boundary_tie_count,
        )

        LOGGER.info(
            "Achieved dev bona-fide FP: %d/%d",
            threshold_result.achieved_false_positive_count,
            threshold_result.bonafide_count,
        )

        LOGGER.info(
            "Achieved empirical dev FPR: %.12f",
            threshold_result.achieved_false_positive_rate,
        )

        # ==============================================================
        # Only AFTER threshold derivation: compute full-dev scores and
        # descriptive predictions for persistence.
        # ==============================================================

        all_scores = attack_probability_from_logits(
            dev_result.logits
        )

        all_margins = (
            attack_ranking_score_from_logits(
                dev_result.logits
            )
        )

        fixed_predictions = (
            apply_attack_threshold(
                scores=(
                    all_scores
                ),
                threshold=(
                    contract.fixed_threshold
                ),
            )
        )

        fpr10_predictions = (
            apply_attack_threshold(
                scores=(
                    all_scores
                ),
                threshold=(
                    threshold
                ),
            )
        )

        observed_fpr10_fp = int(
            fpr10_predictions[
                bonafide_mask
            ]
            .sum()
            .item()
        )

        if (
            observed_fpr10_fp
            != threshold_result
            .achieved_false_positive_count
        ):

            raise RuntimeError(
                "Full-dev threshold application disagrees with "
                "bona-fide-only derivation result."
            )

        # ==============================================================
        # AUROC remains threshold-independent / diagnostic.
        # ==============================================================

        auroc_result = compute_dev_auroc(
            experiment_cfg=(
                experiment_cfg
            ),
            logits=(
                dev_result.logits
            ),
            targets=(
                dev_result.targets
            ),
        )

        historical_auroc = source[
            "historical_dev_auroc"
        ]

        if historical_auroc is not None:

            if not math.isclose(
                auroc_result.auroc,
                historical_auroc,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):

                raise RuntimeError(
                    "Loaded checkpoint did not reproduce historical "
                    "dev AUROC:\n"
                    f"  historical={historical_auroc:.17g}\n"
                    f"  observed={auroc_result.auroc:.17g}"
                )

            LOGGER.info(
                "[PASS] dev AUROC reproduces historical raw-best "
                "checkpoint"
            )

        else:

            LOGGER.info(
                "Historical raw-best AUROC not recorded in source "
                "run.yaml; reproduction comparison skipped."
            )

        LOGGER.info(
            "Observed dev AUROC: %.17g",
            auroc_result.auroc,
        )

        # ==============================================================
        # Durable artifacts
        # ==============================================================

        write_prediction_csv(
            path=(
                predictions_partial
            ),
            dataset=(
                dev_dataset
            ),
            logits=(
                dev_result.logits
            ),
            targets=(
                dev_result.targets
            ),
            margins=(
                all_margins
            ),
            scores=(
                all_scores
            ),
            fixed_predictions=(
                fixed_predictions
            ),
            fpr10_predictions=(
                fpr10_predictions
            ),
        )

        predictions_sha = sha256_file(
            predictions_partial
        )

        threshold_artifact = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_dev_derived_fpr10_threshold",

            "status":
                "DERIVED_UNDER_FROZEN_PROTOCOL",

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

            "machine":
                {
                    "id":
                        machine_id,

                    "configured_device":
                        str(
                            device
                        ),
                },

            "scientific_config":
                {
                    "path":
                        relative_repo_path(
                            experiment_path
                        ),

                    "sha256":
                        sha256_file(
                            experiment_path
                        ),
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

            "fpr10_primitive_audit":
                {
                    "path":
                        relative_repo_path(
                            fpr_audit_path
                        ),

                    "sha256":
                        fpr_audit_sha,

                    "status":
                        "PASS",
                },

            "resolution_selection":
                {
                    "path":
                        relative_repo_path(
                            source[
                                "selection_path"
                            ]
                        ),

                    "sha256":
                        source[
                            "selection_sha256"
                        ],

                    "selected_resolution":
                        "r512",

                    "selected_backbone_lr":
                        0.0001,
                },

            "checkpoint":
                {
                    "seed":
                        seed,

                    "source_machine":
                        source[
                            "source_machine"
                        ],

                    "run_id":
                        source[
                            "run_id"
                        ],

                    "source_run_yaml":
                        relative_repo_path(
                            source[
                                "run_yaml_path"
                            ]
                        ),

                    "source_run_yaml_sha256":
                        source[
                            "run_yaml_sha256"
                        ],
                 

                    "file_sha256":
                        source[
                            "checkpoint_file_sha256"
                        ],

                    "model_state_sha256":
                        source[
                            "checkpoint_model_state_sha256"
                        ],

                    "epoch":
                        source[
                            "checkpoint_epoch"
                        ],

                    "historical_weighted_dev_loss":
                        historical_loss,

                    "observed_weighted_dev_loss":
                        observed_loss,

                    "weighted_dev_loss_delta":
                        loss_delta,

                    "historical_dev_auroc":
                        historical_auroc,

                    "observed_dev_auroc":
                        auroc_result.auroc,
                },

            "dev_val":
                {
                    "manifest_path":
                        relative_repo_path(
                            dev_dataset.manifest_path
                        ),

                    "manifest_sha256":
                        dev_dataset.manifest_sha256,

                    "sample_count":
                        459,

                    "bonafide_count":
                        153,

                    "attack_count":
                        306,

                    "resolution":
                        "r512",
                },

            "score":
                {
                    "stored_logits":
                        "float32",

                    "threshold_score":
                        "p_attack",

                    "threshold_score_dtype":
                        "float64",

                    "definition":
                        "softmax(float64([z0,z1]))[1]",
                },

            "fixed_operating_point":
                {
                    "threshold":
                        0.5,

                    "prediction_rule":
                        "attack iff p_attack >= 0.5",
                },

            "controlled_fpr10":
                {
                    "threshold_selection_population":
                        "complete_dev_val_bonafide_subset_only",

                    "target_empirical_fpr":
                        (
                            threshold_result
                            .target_false_positive_rate
                        ),

                    "bonafide_count":
                        threshold_result.bonafide_count,

                    "allowed_false_positive_count":
                        (
                            threshold_result
                            .allowed_false_positive_count
                        ),

                    "boundary_rank_one_based":
                        (
                            threshold_result
                            .boundary_rank_one_based
                        ),

                    "boundary_score_b16":
                        threshold_result.boundary_score,

                    "threshold":
                        threshold,

                    "threshold_dtype":
                        "float64",

                    "threshold_rule":
                        "nextafter(b16,+infinity)",

                    "prediction_rule":
                        "attack iff p_attack >= threshold_FPR10",

                    "boundary_tie_count":
                        threshold_result.boundary_tie_count,

                    "strictly_above_boundary_count":
                        (
                            threshold_result
                            .strictly_above_boundary_count
                        ),

                    "achieved_dev_false_positive_count":
                        (
                            threshold_result
                            .achieved_false_positive_count
                        ),

                    "achieved_dev_false_positive_rate":
                        (
                            threshold_result
                            .achieved_false_positive_rate
                        ),

                    "attack_scores_used_for_threshold_selection":
                        False,

                    "interpolation":
                        False,

                    "randomized_tie_handling":
                        False,

                    "threshold_clamped":
                        False,
                },

            "predictions":
                {
                    "path":
                        relative_repo_path(
                            predictions_path
                        ),

                    "sha256":
                        predictions_sha,

                    "rows":
                        459,

                    "manifest_order_preserved":
                        True,
                },

            "inference_reproducibility":
                {
                    "run_seed":
                        reproducibility_state.run_seed,

                    "cublas_workspace_config":
                        (
                            reproducibility_state
                            .cublas_workspace_config
                        ),

                    "deterministic_algorithms":
                        (
                            reproducibility_state
                            .deterministic_algorithms
                        ),

                    "cudnn_benchmark":
                        (
                            reproducibility_state
                            .cudnn_benchmark
                        ),

                    "cudnn_deterministic":
                        (
                            reproducibility_state
                            .cudnn_deterministic
                        ),

                    "allow_tf32_matmul":
                        (
                            reproducibility_state
                            .allow_tf32_matmul
                        ),

                    "allow_tf32_cudnn":
                        (
                            reproducibility_state
                            .allow_tf32_cudnn
                        ),

                    "inference_seconds":
                        inference_seconds,
                },

            "model_build":
                {
                    "architecture":
                        model_provenance.architecture,

                    "pretrained_weights_enum":
                        (
                            model_provenance
                            .pretrained_weights_enum
                        ),

                    "pretrained_checkpoint_sha256":
                        (
                            model_provenance
                            .pretrained_checkpoint_sha256
                        ),
                },

            "production_inference_source_sha256":
                source[
                    "current_inference_source_sha256"
                ],

            "boundaries":
                {
                    "scientific_training_performed":
                        False,

                    "checkpoint_selection_performed":
                        False,

                    "learning_rate_selection_performed":
                        False,

                    "resolution_selection_performed":
                        False,

                    "test_threshold_tuning_performed":
                        False,

                    "held_out_test_accessed":
                        False,
                },
        }

        with threshold_partial.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                threshold_artifact,
                file,
                sort_keys=False,
            )

        # --------------------------------------------------------------
        # Promote both artifacts only after complete construction.
        # --------------------------------------------------------------

        promoted_predictions = False

        try:

            os.replace(
                predictions_partial,
                predictions_path,
            )

            promoted_predictions = True

            os.replace(
                threshold_partial,
                threshold_path,
            )

        except Exception:

            if (
                promoted_predictions
                and predictions_path.exists()
            ):

                predictions_path.unlink()

            raise

        threshold_sha = sha256_file(
            threshold_path
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] dev predictions persisted: %s",
            predictions_path,
        )

        LOGGER.info(
            "Dev predictions SHA-256: %s",
            predictions_sha,
        )

        LOGGER.info(
            "[PASS] seed-specific FPR10 artifact persisted: %s",
            threshold_path,
        )

        LOGGER.info(
            "Threshold artifact SHA-256: %s",
            threshold_sha,
        )

        LOGGER.info(
            "[PASS] scientific training: NOT PERFORMED"
        )

        LOGGER.info(
            "[PASS] model/checkpoint selection: NOT PERFORMED"
        )

        LOGGER.info(
            "[PASS] held-out test: NOT ACCESSED"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "SEED %d DEV THRESHOLD DERIVATION: COMPLETE",
            seed,
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "SEED %d DEV THRESHOLD DERIVATION: FAIL",
            seed,
        )

        for partial in (
            threshold_partial,
            predictions_partial,
        ):

            if partial.exists():

                partial.unlink()

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