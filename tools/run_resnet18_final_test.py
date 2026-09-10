#!/usr/bin/env python3
"""
Final held-out FantasyID evaluation for one frozen ResNet-18 checkpoint.

THIS TOOL OPENS THE OFFICIAL HELD-OUT TEST SET.

Frozen scientific configuration
--------------------------------
- resolution: r512
- seeds: 8, 9, 10
- bonafide = 0
- attack = 1
- fixed operating threshold = 0.5
- second threshold = checkpoint-specific frozen dev FPR10
- threshold derivation from test is impossible in this runner
- no best-test-seed selection

For one seed:
    frozen checkpoint
        ->
    canonical 1385-image test manifest
        ->
    per-file frozen SHA verification
        ->
    frozen r512 preprocessing
        ->
    one forward pass
        ->
    FP32 stored logits
        ->
    float64 p_attack
        ->
    fixed 0.5 metrics
    frozen dev-FPR10 metrics
    AUROC(z1-z0)

Runtime evidence
----------------
Two timings are recorded:

1. evaluation_loop_seconds:
       file hashing + decode + preprocessing + transfer + inference

2. gpu_forward_seconds:
       CUDA-event time around model forward only

No warm-up pass is performed on held-out data.

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
from dataclasses import asdict
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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.config import (
    load_experiment_config,
    load_machine_config,
)

from src.development_metrics import (
    attack_ranking_score_from_logits,
)

from src.final_evaluation import (
    HELD_OUT_TEST_ACCESS_TOKEN,
    build_held_out_test_dataset,
    compute_final_binary_evaluation,
    load_canonical_held_out_test_manifest,
    load_final_evaluation_contract,
)

from src.fpr10 import (
    apply_attack_threshold,
    attack_probability_from_logits,
)

from src.modeling import (
    build_resnet18_classifier,
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
    "run_resnet18_final_test"
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
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


# ======================================================================
# Basic helpers
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


# ======================================================================
# Git boundary
# ======================================================================

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
            "Git working tree must be clean before held-out-test "
            "evaluation.\n\n"
            f"{status}"
        )

    if len(commit) != 40:
        raise RuntimeError(
            "Git HEAD is not a full SHA."
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
) -> tuple[Path, str]:

    result = subprocess.run(
        [
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
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Canonical experiment validator failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    lines = [
        line.strip()
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise RuntimeError(
            "Expected exactly one validator handoff line."
        )

    match = VALIDATION_HANDOFF_PATTERN.match(
        lines[
            0
        ]
    )

    if match is None:
        raise RuntimeError(
            "Could not parse validator handoff."
        )

    if match.group(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Canonical validator did not return PASS."
        )

    artifact_path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = match.group(
        "sha256"
    )

    if sha256_file(
        artifact_path
    ) != expected_sha:
        raise RuntimeError(
            "Validator artifact SHA mismatch."
        )

    validator_text = artifact_path.read_text(
        encoding="utf-8"
    )

    if (
        "Git working tree clean before validation log creation: True"
        not in validator_text
    ):
        raise RuntimeError(
            "Canonical validator did not observe a clean Git tree."
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
    config: Mapping[Any, Any],
    machine_id: str,
    seed: int,
) -> Path:

    logging_cfg = require_mapping(
        config[
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
# Primitive-audit prerequisite
# ======================================================================

def validate_final_evaluation_audit(
    *,
    config: Mapping[Any, Any],
    protocol_path: Path,
    metadata_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:

    prerequisite_cfg = require_mapping(
        config[
            "prerequisites"
        ][
            "final_evaluation_audit"
        ],
        "prerequisites.final_evaluation_audit",
    )

    audit_path = resolve_repo_path(
        prerequisite_cfg[
            "path"
        ]
    )

    require_git_tracked(
        audit_path
    )

    actual_sha = sha256_file(
        audit_path
    )

    if actual_sha != str(
        prerequisite_cfg[
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Final-evaluation audit SHA mismatch."
        )

    audit = load_yaml(
        audit_path
    )

    if audit.get(
        "status"
    ) != prerequisite_cfg[
        "required_status"
    ]:
        raise RuntimeError(
            "Final-evaluation primitive audit is not PASS."
        )

    source_hashes = require_mapping(
        audit[
            "source_sha256"
        ],
        "final_evaluation_audit.source_sha256",
    )

    current_final_eval_sha = sha256_file(
        REPO_ROOT
        / "src"
        / "final_evaluation.py"
    )

    if (
        source_hashes[
            "src/final_evaluation.py"
        ]
        != current_final_eval_sha
    ):
        raise RuntimeError(
            "src/final_evaluation.py differs from audited source."
        )

    if (
        source_hashes[
            "evaluation_protocol"
        ]
        != sha256_file(
            protocol_path
        )
    ):
        raise RuntimeError(
            "Evaluation protocol differs from audited source."
        )

    if (
        source_hashes[
            "canonical_test_manifest_metadata"
        ]
        != sha256_file(
            metadata_path
        )
    ):
        raise RuntimeError(
            "Canonical test metadata differs from audited source."
        )

    if (
        source_hashes[
            "canonical_test_manifest_csv"
        ]
        != sha256_file(
            manifest_path
        )
    ):
        raise RuntimeError(
            "Canonical test CSV differs from audited source."
        )

    real_access = require_mapping(
        audit[
            "real_held_out_access"
        ],
        "final_evaluation_audit.real_held_out_access",
    )

    forbidden_prior_access = (
        "test_dataset_root_used",
        "test_image_paths_followed",
        "test_image_files_opened",
        "test_image_hashes_recomputed",
        "test_pixels_decoded",
        "model_loaded",
        "checkpoint_loaded",
        "model_inference_performed",
        "test_metrics_computed",
        "threshold_derived_or_modified",
    )

    for key in forbidden_prior_access:
        if real_access[
            key
        ] is not False:
            raise RuntimeError(
                "Primitive audit reports prior real held-out access:\n"
                f"  {key}"
            )

    return {
        "path":
            audit_path,

        "sha256":
            actual_sha,

        "final_evaluation_source_sha256":
            current_final_eval_sha,
    }


# ======================================================================
# Frozen test + threshold evidence
# ======================================================================

def validate_test_and_threshold_evidence(
    *,
    config: Mapping[Any, Any],
    seed: int,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    dict[int, dict[str, Any]],
]:

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

    manifest_path = resolve_repo_path(
        csv_cfg[
            "path"
        ]
    )

    require_git_tracked(
        metadata_path
    )

    require_git_tracked(
        manifest_path
    )

    if sha256_file(
        metadata_path
    ) != str(
        metadata_cfg[
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Canonical test metadata SHA mismatch."
        )

    if sha256_file(
        manifest_path
    ) != str(
        csv_cfg[
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Canonical test CSV SHA mismatch."
        )

    metadata = load_yaml(
        metadata_path
    )

    if metadata.get(
        "status"
    ) != "FROZEN_BEFORE_MODEL_TEST_ACCESS":
        raise RuntimeError(
            "Canonical test manifest is not pre-test frozen."
        )

    threshold_cfg = require_mapping(
        config[
            "dev_thresholds"
        ],
        "dev_thresholds",
    )

    manifest_pretest = require_mapping(
        metadata[
            "pre_test_prerequisites"
        ][
            "dev_thresholds"
        ],
        "test_manifest.pre_test_prerequisites.dev_thresholds",
    )

    thresholds: dict[
        int,
        dict[str, Any],
    ] = {}

    for frozen_seed in (
        8,
        9,
        10,
    ):

        configured = require_mapping(
            seed_value(
                threshold_cfg,
                frozen_seed,
                "dev threshold config",
            ),
            f"dev_thresholds.{frozen_seed}",
        )

        threshold_path = resolve_repo_path(
            configured[
                "path"
            ]
        )

        require_git_tracked(
            threshold_path
        )

        expected_sha = str(
            configured[
                "sha256"
            ]
        )

        actual_sha = sha256_file(
            threshold_path
        )

        if actual_sha != expected_sha:
            raise RuntimeError(
                "Frozen threshold artifact SHA mismatch:\n"
                f"  seed={frozen_seed}"
            )

        manifest_item = require_mapping(
            seed_value(
                manifest_pretest,
                frozen_seed,
                "test-manifest threshold evidence",
            ),
            f"test_manifest.dev_thresholds.{frozen_seed}",
        )

        if manifest_item[
            "sha256"
        ] != actual_sha:
            raise RuntimeError(
                "Test manifest was frozen against a different threshold:\n"
                f"  seed={frozen_seed}"
            )

        threshold = load_yaml(
            threshold_path
        )

        if threshold.get(
            "artifact_type"
        ) != "resnet18_dev_derived_fpr10_threshold":
            raise RuntimeError(
                "Unexpected threshold artifact type."
            )

        if threshold.get(
            "status"
        ) != "DERIVED_UNDER_FROZEN_PROTOCOL":
            raise RuntimeError(
                "Threshold artifact is not frozen."
            )

        if int(
            threshold[
                "checkpoint"
            ][
                "seed"
            ]
        ) != frozen_seed:
            raise RuntimeError(
                "Threshold seed identity mismatch."
            )

        if threshold[
            "boundaries"
        ][
            "held_out_test_accessed"
        ] is not False:
            raise RuntimeError(
                "Threshold artifact was not produced pre-test."
            )

        threshold_value = float(
            threshold[
                "controlled_fpr10"
            ][
                "threshold"
            ]
        )

        if threshold_value != float(
            manifest_item[
                "threshold"
            ]
        ):
            raise RuntimeError(
                "Frozen threshold numeric value differs from "
                "pre-test manifest."
            )

        thresholds[
            frozen_seed
        ] = {
            "path":
                threshold_path,

            "sha256":
                actual_sha,

            "threshold":
                threshold_value,

            "artifact":
                threshold,
        }

    if seed not in thresholds:
        raise RuntimeError(
            "Requested seed is not in frozen final seed set."
        )

    return (
        metadata_path,
        manifest_path,
        metadata,
        thresholds,
    )


# ======================================================================
# Exact checkpoint resolution
# ======================================================================

def validate_and_resolve_checkpoint(
    *,
    threshold_record: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    machine_id: str,
    seed: int,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
]:

    threshold = require_mapping(
        threshold_record[
            "artifact"
        ],
        "threshold artifact",
    )

    checkpoint_evidence = require_mapping(
        threshold[
            "checkpoint"
        ],
        "threshold.checkpoint",
    )

    source_machine = str(
        checkpoint_evidence[
            "source_machine"
        ]
    )

    if source_machine != machine_id:
        raise RuntimeError(
            "Final test must run checkpoint on its canonical origin "
            "machine:\n"
            f"  seed={seed}\n"
            f"  checkpoint_machine={source_machine}\n"
            f"  current_machine={machine_id}"
        )

    # ------------------------------------------------------------------
    # Revalidate the production inference stack against what was used
    # when the dev threshold itself was produced.
    # ------------------------------------------------------------------

    source_hashes = require_mapping(
        threshold[
            "production_inference_source_sha256"
        ],
        "threshold.production_inference_source_sha256",
    )

    for relative_path, expected_sha in (
        source_hashes.items()
    ):

        current_sha = sha256_file(
            REPO_ROOT
            / str(
                relative_path
            )
        )

        if current_sha != expected_sha:
            raise RuntimeError(
                "Current inference source differs from pre-test "
                "threshold evidence:\n"
                f"  file={relative_path}\n"
                f"  expected={expected_sha}\n"
                f"  current={current_sha}"
            )

    # ------------------------------------------------------------------
    # Revalidate src/fpr10.py against its original primitive audit.
    # ------------------------------------------------------------------

    fpr_audit_cfg = require_mapping(
        threshold[
            "fpr10_primitive_audit"
        ],
        "threshold.fpr10_primitive_audit",
    )

    fpr_audit_path = resolve_repo_path(
        fpr_audit_cfg[
            "path"
        ]
    )

    if sha256_file(
        fpr_audit_path
    ) != fpr_audit_cfg[
        "sha256"
    ]:
        raise RuntimeError(
            "FPR10 primitive audit artifact changed."
        )

    fpr_audit = load_yaml(
        fpr_audit_path
    )

    current_fpr10_sha = sha256_file(
        REPO_ROOT
        / "src"
        / "fpr10.py"
    )

    if (
        fpr_audit[
            "source_sha256"
        ][
            "src/fpr10.py"
        ]
        != current_fpr10_sha
    ):
        raise RuntimeError(
            "src/fpr10.py differs from audited threshold primitive."
        )

    # ------------------------------------------------------------------
    # Resolution-selection artifact gives canonical checkpoint path.
    # ------------------------------------------------------------------

    selection_cfg = require_mapping(
        threshold[
            "resolution_selection"
        ],
        "threshold.resolution_selection",
    )

    selection_path = resolve_repo_path(
        selection_cfg[
            "path"
        ]
    )

    require_git_tracked(
        selection_path
    )

    if sha256_file(
        selection_path
    ) != selection_cfg[
        "sha256"
    ]:
        raise RuntimeError(
            "Resolution-selection artifact changed."
        )

    selection = load_yaml(
        selection_path
    )

    if selection.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "Resolution selection is no longer FROZEN."
        )

    plan = require_mapping(
        selection[
            "final_detection_plan"
        ],
        "selection.final_detection_plan",
    )

    if plan[
        "resolution"
    ] != "r512":
        raise RuntimeError(
            "Final resolution must remain r512."
        )

    if float(
        plan[
            "backbone_lr"
        ]
    ) != 0.0001:
        raise RuntimeError(
            "Final backbone LR must remain 1e-4."
        )

    checkpoint_sources = require_mapping(
        plan[
            "checkpoint_sources"
        ],
        "selection.final_detection_plan.checkpoint_sources",
    )

    source = require_mapping(
        seed_value(
            checkpoint_sources,
            seed,
            "checkpoint source",
        ),
        f"checkpoint_sources.{seed}",
    )

    if source[
        "training_machine_id"
    ] != machine_id:
        raise RuntimeError(
            "Selection artifact checkpoint-machine mismatch."
        )

    if source[
        "run_id"
    ] != checkpoint_evidence[
        "run_id"
    ]:
        raise RuntimeError(
            "Threshold and selection run_id disagree."
        )

    checkpoint_cfg = require_mapping(
        source[
            "checkpoint"
        ],
        f"checkpoint_sources.{seed}.checkpoint",
    )

    if (
        checkpoint_cfg[
            "file_sha256"
        ]
        != checkpoint_evidence[
            "file_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint file SHA differs between frozen artifacts."
        )

    if (
        checkpoint_cfg[
            "model_state_sha256"
        ]
        != checkpoint_evidence[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint model-state SHA differs between artifacts."
        )

    paths_cfg = require_mapping(
        machine_cfg[
            "paths"
        ],
        "machine_config.paths",
    )

    runs_root = Path(
        paths_cfg[
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
        / str(
            source[
                "run_id"
            ]
        )
        / str(
            checkpoint_cfg[
                "path"
            ]
        )
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Canonical checkpoint binary is missing:\n"
            f"  {checkpoint_path}"
        )

    actual_checkpoint_sha = sha256_file(
        checkpoint_path
    )

    if (
        actual_checkpoint_sha
        != checkpoint_evidence[
            "file_sha256"
        ]
    ):
        raise RuntimeError(
            "Canonical checkpoint file SHA mismatch."
        )

    payload = verify_checkpoint_artifact(
        checkpoint_path
    )

    expected_payload = {
        "run_id":
            checkpoint_evidence[
                "run_id"
            ],

        "resolution_name":
            "r512",

        "run_seed":
            seed,

        "stage":
            "stage_b",

        "epoch":
            int(
                checkpoint_evidence[
                    "epoch"
                ]
            ),

        "model_state_sha256":
            checkpoint_evidence[
                "model_state_sha256"
            ],
    }

    for key, expected in (
        expected_payload.items()
    ):

        if payload[
            key
        ] != expected:
            raise RuntimeError(
                "Checkpoint payload mismatch:\n"
                f"  key={key}\n"
                f"  expected={expected!r}\n"
                f"  actual={payload[key]!r}"
            )

    return (
        checkpoint_path,
        payload,
        {
            "file_sha256":
                actual_checkpoint_sha,

            "model_state_sha256":
                checkpoint_evidence[
                    "model_state_sha256"
                ],

            "run_id":
                checkpoint_evidence[
                    "run_id"
                ],

            "epoch":
                int(
                    checkpoint_evidence[
                        "epoch"
                    ]
                ),

            "source_machine":
                source_machine,

            "selection_path":
                selection_path,

            "selection_sha256":
                selection_cfg[
                    "sha256"
                ],
        },
    )


# ======================================================================
# Test DataLoader
# ======================================================================

def build_test_loader(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    metadata_path: Path,
    metadata_sha256: str,
    manifest_sha256: str,
    seed: int,
):

    (
        manifest_contract,
        dataset,
    ) = build_held_out_test_dataset(
        experiment_cfg=experiment_cfg,
        machine_cfg=machine_cfg,
        repo_root=REPO_ROOT,
        metadata_path=metadata_path,
        expected_metadata_sha256=metadata_sha256,
        expected_manifest_sha256=manifest_sha256,
        resolution_name="r512",
        access_token=HELD_OUT_TEST_ACCESS_TOKEN,
    )

    if len(
        dataset
    ) != 1385:
        raise RuntimeError(
            "Final held-out Dataset must contain 1385 images."
        )

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

    num_workers = int(
        machine_cfg[
            "runtime"
        ][
            "num_workers"
        ]
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
        persistent_workers=False,
    )

    return (
        manifest_contract,
        dataset,
        loader,
        batch_size,
        num_workers,
    )


# ======================================================================
# Actual held-out inference
# ======================================================================

def run_test_inference(
    *,
    model: torch.nn.Module,
    dataset: Any,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:

    model.eval()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    logits_batches: list[
        torch.Tensor
    ] = []

    target_batches: list[
        torch.Tensor
    ] = []

    verified_hashes: list[
        str
    ] = []

    event_pairs: list[
        tuple[
            torch.cuda.Event,
            torch.cuda.Event,
        ]
    ] = []

    cursor = 0

    first_access_utc = datetime.now(
        timezone.utc
    ).isoformat()

    LOGGER.warning(
        "=" * 72
    )

    LOGGER.warning(
        "HELD-OUT TEST ACCESS BEGINS NOW"
    )

    LOGGER.warning(
        "Official test JPEG files may now be opened."
    )

    LOGGER.warning(
        "All classifier, checkpoint and threshold decisions are frozen."
    )

    LOGGER.warning(
        "=" * 72
    )

    wall_started = time.perf_counter()

    with torch.inference_mode():

        for batch in loader:

            images = batch[
                "image"
            ]

            targets = batch[
                "label"
            ]

            batch_paths = list(
                batch[
                    "image_path"
                ]
            )

            batch_verified = list(
                batch[
                    "verified_image_sha256"
                ]
            )

            batch_size = int(
                images.shape[
                    0
                ]
            )

            if tuple(
                images.shape[
                    1:
                ]
            ) != (
                3,
                512,
                864,
            ):
                raise RuntimeError(
                    "Held-out preprocessing shape changed."
                )

            if images.dtype != torch.float32:
                raise RuntimeError(
                    "Held-out input tensor must be float32."
                )

            if targets.dtype != torch.int64:
                raise RuntimeError(
                    "Held-out targets must be int64."
                )

            for offset in range(
                batch_size
            ):

                row = dataset.rows[
                    cursor
                    + offset
                ]

                if (
                    batch_paths[
                        offset
                    ]
                    != row[
                        "image_path"
                    ]
                ):
                    raise RuntimeError(
                        "Held-out DataLoader order changed."
                    )

                if (
                    batch_verified[
                        offset
                    ]
                    != row[
                        "image_sha256"
                    ]
                ):
                    raise RuntimeError(
                        "Held-out file SHA verification mismatch."
                    )

            verified_hashes.extend(
                str(
                    value
                )
                for value
                in batch_verified
            )

            images_device = images.to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )

            start_event = torch.cuda.Event(
                enable_timing=True
            )

            end_event = torch.cuda.Event(
                enable_timing=True
            )

            start_event.record()

            logits_device = model(
                images_device
            )

            end_event.record()

            if (
                logits_device.ndim != 2
                or logits_device.shape[
                    1
                ] != 2
            ):
                raise RuntimeError(
                    "Final model output must have shape [N,2]."
                )

            if logits_device.dtype != torch.float32:
                raise RuntimeError(
                    "Final stored logits must originate as float32."
                )

            if not bool(
                torch.isfinite(
                    logits_device
                ).all()
            ):
                raise RuntimeError(
                    "Held-out logits contain NaN or Inf."
                )

            logits_batches.append(
                logits_device
                .detach()
                .cpu()
                .contiguous()
            )

            target_batches.append(
                targets
                .detach()
                .cpu()
                .contiguous()
            )

            event_pairs.append(
                (
                    start_event,
                    end_event,
                )
            )

            cursor += batch_size

    torch.cuda.synchronize(
        device
    )

    wall_seconds = (
        time.perf_counter()
        - wall_started
    )

    gpu_forward_ms = sum(
        float(
            start.elapsed_time(
                end
            )
        )
        for start, end
        in event_pairs
    )

    gpu_forward_seconds = (
        gpu_forward_ms
        / 1000.0
    )

    logits = torch.cat(
        logits_batches,
        dim=0,
    )

    targets = torch.cat(
        target_batches,
        dim=0,
    )

    if cursor != 1385:
        raise RuntimeError(
            "Held-out inference did not process exactly 1385 images."
        )

    if tuple(
        logits.shape
    ) != (
        1385,
        2,
    ):
        raise RuntimeError(
            "Final held-out logits must have shape [1385,2]."
        )

    if tuple(
        targets.shape
    ) != (
        1385,
    ):
        raise RuntimeError(
            "Final held-out targets must have shape [1385]."
        )

    bonafide_count = int(
        (
            targets
            == 0
        )
        .sum()
        .item()
    )

    attack_count = int(
        (
            targets
            == 1
        )
        .sum()
        .item()
    )

    if (
        bonafide_count,
        attack_count,
    ) != (
        300,
        1085,
    ):
        raise RuntimeError(
            "Held-out class counts changed."
        )

    if len(
        verified_hashes
    ) != 1385:
        raise RuntimeError(
            "Not every held-out file SHA was verified."
        )

    peak_allocated = int(
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_reserved = int(
        torch.cuda.max_memory_reserved(
            device
        )
    )

    return {
        "first_access_utc":
            first_access_utc,

        "logits":
            logits,

        "targets":
            targets,

        "verified_hashes":
            tuple(
                verified_hashes
            ),

        "evaluation_loop_seconds":
            wall_seconds,

        "gpu_forward_seconds":
            gpu_forward_seconds,

        "gpu_forward_batches":
            len(
                event_pairs
            ),

        "peak_cuda_memory_allocated_bytes":
            peak_allocated,

        "peak_cuda_memory_reserved_bytes":
            peak_reserved,
    }


# ======================================================================
# Prediction persistence
# ======================================================================

def write_predictions(
    *,
    path: Path,
    dataset: Any,
    logits: torch.Tensor,
    targets: torch.Tensor,
    verified_hashes: tuple[str, ...],
    fpr10_threshold: float,
) -> str:

    scores = attack_probability_from_logits(
        logits
    )

    margins = attack_ranking_score_from_logits(
        logits
    )

    fixed_predictions = apply_attack_threshold(
        scores=scores,
        threshold=0.5,
    )

    fpr10_predictions = apply_attack_threshold(
        scores=scores,
        threshold=fpr10_threshold,
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

        for index, row in enumerate(
            dataset.rows
        ):

            target = int(
                targets[
                    index
                ].item()
            )

            if target != int(
                row[
                    "label"
                ]
            ):
                raise RuntimeError(
                    "Persisted test target differs from manifest."
                )

            if (
                verified_hashes[
                    index
                ]
                != row[
                    "image_sha256"
                ]
            ):
                raise RuntimeError(
                    "Persisted verified SHA differs from manifest."
                )

            writer.writerow(
                {
                    "row_index":
                        index,

                    "source_workbook_row":
                        int(
                            row[
                                "source_workbook_row"
                            ]
                        ),

                    "image_path":
                        row[
                            "image_path"
                        ],

                    "image_sha256":
                        row[
                            "image_sha256"
                        ],

                    "verified_image_sha256":
                        verified_hashes[
                            index
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

                    "face_db":
                        row[
                            "face_db"
                        ],

                    "face_id":
                        row[
                            "face_id"
                        ],

                    "gender":
                        row[
                            "gender"
                        ],

                    "target":
                        target,

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

    return sha256_file(
        path
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen ResNet-18 checkpoint on the official "
            "FantasyID held-out test set."
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
        "--config",
        default=(
            "tools/"
            "run_resnet18_final_test_config.yaml"
        ),
    )

    args = parser.parse_args()

    seed = int(
        args.seed
    )

    git_commit, git_branch = (
        require_clean_git()
    )

    config_path = resolve_repo_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            config[
                "experiment"
            ][
                "config"
            ]
        )
    )

    machine_cfg, machine_path = (
        load_machine_config(
            config[
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

    # ------------------------------------------------------------------
    # Clean-tree validator happens BEFORE this runner creates its log.
    # ------------------------------------------------------------------

    workspace = establish_pre_cuda_environment(
        experiment_cfg=experiment_cfg
    )

    validator_path, validator_sha = (
        run_validator(
            experiment_path=experiment_path,
            machine_path=machine_path,
        )
    )

    log_path = configure_logging(
        config=config,
        machine_id=machine_id,
        seed=seed,
    )

    output_cfg = require_mapping(
        config[
            "output"
        ],
        "output",
    )

    predictions_directory = resolve_repo_path(
        output_cfg[
            "predictions_directory"
        ]
    )

    results_directory = resolve_repo_path(
        output_cfg[
            "results_directory"
        ]
    )

    predictions_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
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

    results_path = (
        results_directory
        / str(
            output_cfg[
                "results_filename"
            ]
        ).format(
            seed=seed
        )
    )

    predictions_partial = Path(
        str(
            predictions_path
        )
        + ".partial"
    )

    results_partial = Path(
        str(
            results_path
        )
        + ".partial"
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 FINAL HELD-OUT TEST"
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

        LOGGER.info(
            "[PASS] pre-CUDA CUBLAS_WORKSPACE_CONFIG = %s",
            workspace,
        )

        LOGGER.info(
            "[PASS] canonical experiment validator | sha256=%s",
            validator_sha,
        )

        if any(
            path.exists()
            for path
            in (
                predictions_path,
                results_path,
                predictions_partial,
                results_partial,
            )
        ):
            raise FileExistsError(
                "Canonical output already exists for this seed."
            )

        # ==============================================================
        # Protocol
        # ==============================================================

        protocol_cfg_info = require_mapping(
            config[
                "evaluation_protocol"
            ],
            "evaluation_protocol",
        )

        protocol_path = resolve_repo_path(
            protocol_cfg_info[
                "path"
            ]
        )

        protocol_sha = sha256_file(
            protocol_path
        )

        if protocol_sha != str(
            protocol_cfg_info[
                "sha256"
            ]
        ):
            raise RuntimeError(
                "Frozen evaluation protocol SHA mismatch."
            )

        protocol_cfg = load_yaml(
            protocol_path
        )

        contract = load_final_evaluation_contract(
            evaluation_cfg=protocol_cfg
        )

        LOGGER.info(
            "[PASS] final evaluation contract loaded"
        )

        LOGGER.info(
            "[PASS] native labels = bonafide 0 / attack 1"
        )

        LOGGER.info(
            "[PASS] fixed operating threshold = 0.5"
        )

        # ==============================================================
        # Test manifest + all three pre-test thresholds
        # ==============================================================

        (
            metadata_path,
            manifest_path,
            _manifest_metadata,
            thresholds,
        ) = validate_test_and_threshold_evidence(
            config=config,
            seed=seed,
        )

        LOGGER.info(
            "[PASS] canonical test manifest = 1385 images"
        )

        LOGGER.info(
            "[PASS] all three dev thresholds predate test access"
        )

        for frozen_seed in (
            8,
            9,
            10,
        ):
            LOGGER.info(
                "Frozen threshold seed%d = %.17g",
                frozen_seed,
                thresholds[
                    frozen_seed
                ][
                    "threshold"
                ],
            )

        # ==============================================================
        # Final-evaluation primitive audit
        # ==============================================================

        primitive_audit = (
            validate_final_evaluation_audit(
                config=config,
                protocol_path=protocol_path,
                metadata_path=metadata_path,
                manifest_path=manifest_path,
            )
        )

        LOGGER.info(
            "[PASS] final-evaluation primitive audit | sha256=%s",
            primitive_audit[
                "sha256"
            ],
        )

        # Metadata-only canonical loader once more before any dataset_root
        # access.
        held_out_contract, _ = (
            load_canonical_held_out_test_manifest(
                repo_root=REPO_ROOT,
                metadata_path=metadata_path,
                expected_metadata_sha256=(
                    config[
                        "canonical_test_manifest"
                    ][
                        "metadata"
                    ][
                        "sha256"
                    ]
                ),
                expected_manifest_sha256=(
                    config[
                        "canonical_test_manifest"
                    ][
                        "csv"
                    ][
                        "sha256"
                    ]
                ),
            )
        )

        if (
            held_out_contract.rows,
            held_out_contract.bonafide_count,
            held_out_contract.attack_count,
        ) != (
            1385,
            300,
            1085,
        ):
            raise RuntimeError(
                "Frozen test population changed."
            )

        LOGGER.info(
            "[PASS] PRE-ACCESS population = "
            "1385 = 300 bona-fide + 1085 attack"
        )

        # ==============================================================
        # Seed-specific frozen threshold/checkpoint
        # ==============================================================

        threshold_record = thresholds[
            seed
        ]

        fpr10_threshold = float(
            threshold_record[
                "threshold"
            ]
        )

        (
            checkpoint_path,
            checkpoint_payload,
            checkpoint_identity,
        ) = validate_and_resolve_checkpoint(
            threshold_record=threshold_record,
            machine_cfg=machine_cfg,
            machine_id=machine_id,
            seed=seed,
        )

        LOGGER.info(
            "[PASS] seed %d frozen FPR10 threshold = %.17g",
            seed,
            fpr10_threshold,
        )

        LOGGER.info(
            "[PASS] checkpoint file SHA-256 = %s",
            checkpoint_identity[
                "file_sha256"
            ],
        )

        LOGGER.info(
            "[PASS] checkpoint model-state SHA-256 = %s",
            checkpoint_identity[
                "model_state_sha256"
            ],
        )

        # ==============================================================
        # Determinism / model restoration
        # ==============================================================

        reproducibility = (
            configure_run_reproducibility(
                experiment_cfg=experiment_cfg,
                run_seed=seed,
            )
        )

        device = torch.device(
            str(
                machine_cfg[
                    "runtime"
                ][
                    "device"
                ]
            )
        )

        if device.type != "cuda":
            raise RuntimeError(
                "Final held-out evaluation requires configured CUDA."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable."
            )

        model, model_provenance = (
            build_resnet18_classifier(
                experiment_cfg=experiment_cfg
            )
        )

        load_result = model.load_state_dict(
            checkpoint_payload[
                "model_state_dict"
            ],
            strict=True,
        )

        if (
            load_result.missing_keys
            or load_result.unexpected_keys
        ):
            raise RuntimeError(
                "Strict checkpoint restore failed."
            )

        restored_sha = state_dict_sha256(
            model.state_dict()
        )

        if (
            restored_sha
            != checkpoint_identity[
                "model_state_sha256"
            ]
        ):
            raise RuntimeError(
                "Restored checkpoint model-state SHA mismatch."
            )

        LOGGER.info(
            "[PASS] exact final checkpoint restored"
        )

        model = model.to(
            device
        )

        # ==============================================================
        # Dataset constructor still does NOT open an image.
        #
        # The first official image access occurs only when loader
        # iteration begins in run_test_inference().
        # ==============================================================

        (
            manifest_contract,
            dataset,
            loader,
            batch_size,
            num_workers,
        ) = build_test_loader(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            metadata_path=metadata_path,
            metadata_sha256=(
                config[
                    "canonical_test_manifest"
                ][
                    "metadata"
                ][
                    "sha256"
                ]
            ),
            manifest_sha256=(
                config[
                    "canonical_test_manifest"
                ][
                    "csv"
                ][
                    "sha256"
                ]
            ),
            seed=seed,
        )

        LOGGER.info(
            "[PASS] held-out Dataset authorized but not yet iterated"
        )

        LOGGER.info(
            "Batch size: %d",
            batch_size,
        )

        LOGGER.info(
            "DataLoader workers: %d",
            num_workers,
        )

        # ==============================================================
        # FIRST OFFICIAL HELD-OUT PIXEL ACCESS
        # ==============================================================

        inference = run_test_inference(
            model=model,
            dataset=dataset,
            loader=loader,
            device=device,
        )

        logits = inference[
            "logits"
        ]

        targets = inference[
            "targets"
        ]

        LOGGER.info(
            "[PASS] complete held-out inference = "
            "1385 = 300 bona-fide + 1085 attack"
        )

        LOGGER.info(
            "[PASS] frozen file SHA verified for all 1385 test images"
        )

        # ==============================================================
        # Final metrics
        # ==============================================================

        evaluation = compute_final_binary_evaluation(
            logits=logits,
            targets=targets,
            fpr10_threshold=fpr10_threshold,
            contract=contract,
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "TEST AUROC: %.12f",
            evaluation.auroc,
        )

        LOGGER.info(
            ""
        )

        fixed = evaluation.fixed_0_5

        LOGGER.info(
            "FIXED 0.5 | TP=%d FP=%d TN=%d FN=%d",
            fixed.true_positive,
            fixed.false_positive,
            fixed.true_negative,
            fixed.false_negative,
        )

        LOGGER.info(
            "FIXED 0.5 | weighted_F1=%.12f | attack_F1=%.12f | "
            "balanced_acc=%.12f",
            fixed.support_weighted_f1,
            fixed.attack.f1,
            fixed.balanced_accuracy,
        )

        LOGGER.info(
            "FIXED 0.5 | FPR/BPCER=%.12f | FNR/APCER=%.12f | "
            "HTER=%.12f",
            fixed.false_positive_rate,
            fixed.false_negative_rate,
            fixed.half_total_error_rate,
        )

        controlled = evaluation.controlled_fpr10

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "DEV-FPR10 threshold: %.17g",
            fpr10_threshold,
        )

        LOGGER.info(
            "FPR10 | TP=%d FP=%d TN=%d FN=%d",
            controlled.true_positive,
            controlled.false_positive,
            controlled.true_negative,
            controlled.false_negative,
        )

        LOGGER.info(
            "FPR10 | weighted_F1=%.12f | attack_F1=%.12f | "
            "balanced_acc=%.12f",
            controlled.support_weighted_f1,
            controlled.attack.f1,
            controlled.balanced_accuracy,
        )

        LOGGER.info(
            "FPR10 | FPR/BPCER=%.12f | FNR/APCER=%.12f | "
            "HTER=%.12f",
            controlled.false_positive_rate,
            controlled.false_negative_rate,
            controlled.half_total_error_rate,
        )

        # ==============================================================
        # Per-image durable evidence
        # ==============================================================

        predictions_sha = write_predictions(
            path=predictions_partial,
            dataset=dataset,
            logits=logits,
            targets=targets,
            verified_hashes=(
                inference[
                    "verified_hashes"
                ]
            ),
            fpr10_threshold=fpr10_threshold,
        )

        # ==============================================================
        # Runtime
        # ==============================================================

        loop_seconds = float(
            inference[
                "evaluation_loop_seconds"
            ]
        )

        gpu_seconds = float(
            inference[
                "gpu_forward_seconds"
            ]
        )

        end_to_end_throughput = (
            1385.0
            / loop_seconds
        )

        gpu_forward_throughput = (
            1385.0
            / gpu_seconds
            if gpu_seconds > 0.0
            else None
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "Evaluation loop seconds: %.6f",
            loop_seconds,
        )

        LOGGER.info(
            "GPU forward-only seconds: %.6f",
            gpu_seconds,
        )

        LOGGER.info(
            "End-to-end throughput: %.3f images/s",
            end_to_end_throughput,
        )

        LOGGER.info(
            "GPU forward-only throughput: %.3f images/s",
            gpu_forward_throughput,
        )

        LOGGER.info(
            "Peak CUDA allocated: %.3f GiB",
            (
                inference[
                    "peak_cuda_memory_allocated_bytes"
                ]
                / (
                    1024 ** 3
                )
            ),
        )

        # ==============================================================
        # Result artifact
        # ==============================================================

        result_artifact = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_final_held_out_test_result",

            "status":
                "FINAL_HELD_OUT_TEST_EVALUATION_COMPLETE",

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "first_official_test_file_access_utc":
                inference[
                    "first_access_utc"
                ],

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

                    "device":
                        str(
                            device
                        ),
                },

            "validator":
                {
                    "path":
                        relative_repo_path(
                            validator_path
                        ),

                    "sha256":
                        validator_sha,
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

            "final_evaluation_primitive_audit":
                {
                    "path":
                        relative_repo_path(
                            primitive_audit[
                                "path"
                            ]
                        ),

                    "sha256":
                        primitive_audit[
                            "sha256"
                        ],

                    "status":
                        "PASS",
                },

            "held_out_test":
                {
                    "manifest_metadata_path":
                        relative_repo_path(
                            metadata_path
                        ),

                    "manifest_metadata_sha256":
                        manifest_contract
                        .metadata_sha256,

                    "manifest_path":
                        relative_repo_path(
                            manifest_path
                        ),

                    "manifest_sha256":
                        manifest_contract
                        .manifest_sha256,

                    "sample_count":
                        1385,

                    "bonafide_count":
                        300,

                    "attack_count":
                        1085,

                    "resolution":
                        "r512",
                },

            "checkpoint":
                {
                    "seed":
                        seed,

                    "source_machine":
                        checkpoint_identity[
                            "source_machine"
                        ],

                    "run_id":
                        checkpoint_identity[
                            "run_id"
                        ],

                    "epoch":
                        checkpoint_identity[
                            "epoch"
                        ],

                    "file_sha256":
                        checkpoint_identity[
                            "file_sha256"
                        ],

                    "model_state_sha256":
                        checkpoint_identity[
                            "model_state_sha256"
                        ],
                },

            "thresholds":
                {
                    "fixed_0_5":
                        {
                            "threshold":
                                0.5,

                            "source":
                                "a_priori",

                            "modified_on_test":
                                False,
                        },

                    "controlled_fpr10":
                        {
                            "threshold":
                                fpr10_threshold,

                            "source":
                                "frozen_dev_val_bonafide_only",

                            "threshold_artifact_path":
                                relative_repo_path(
                                    threshold_record[
                                        "path"
                                    ]
                                ),

                            "threshold_artifact_sha256":
                                threshold_record[
                                    "sha256"
                                ],

                            "derived_from_test":
                                False,

                            "modified_on_test":
                                False,
                        },
                },

            "metrics":
                asdict(
                    evaluation
                ),

            "predictions":
                {
                    "path":
                        relative_repo_path(
                            predictions_path
                        ),

                    "sha256":
                        predictions_sha,

                    "rows":
                        1385,

                    "manifest_order_preserved":
                        True,

                    "all_file_sha256_verified":
                        True,
                },

            "runtime":
                {
                    "warmup_batches":
                        0,

                    "batch_size":
                        batch_size,

                    "num_workers":
                        num_workers,

                    "evaluation_loop_seconds":
                        loop_seconds,

                    "gpu_forward_only_seconds":
                        gpu_seconds,

                    "gpu_forward_batch_count":
                        inference[
                            "gpu_forward_batches"
                        ],

                    "end_to_end_images_per_second":
                        end_to_end_throughput,

                    "gpu_forward_only_images_per_second":
                        gpu_forward_throughput,

                    "peak_cuda_memory_allocated_bytes":
                        inference[
                            "peak_cuda_memory_allocated_bytes"
                        ],

                    "peak_cuda_memory_reserved_bytes":
                        inference[
                            "peak_cuda_memory_reserved_bytes"
                        ],
                },

            "reproducibility":
                {
                    "run_seed":
                        reproducibility.run_seed,

                    "cublas_workspace_config":
                        reproducibility
                        .cublas_workspace_config,

                    "deterministic_algorithms":
                        reproducibility
                        .deterministic_algorithms,

                    "cudnn_benchmark":
                        reproducibility
                        .cudnn_benchmark,

                    "cudnn_deterministic":
                        reproducibility
                        .cudnn_deterministic,

                    "allow_tf32_matmul":
                        reproducibility
                        .allow_tf32_matmul,

                    "allow_tf32_cudnn":
                        reproducibility
                        .allow_tf32_cudnn,
                },

            "model":
                {
                    "architecture":
                        model_provenance.architecture,

                    "pretrained_weights_enum":
                        model_provenance
                        .pretrained_weights_enum,

                    "pretrained_checkpoint_sha256":
                        model_provenance
                        .pretrained_checkpoint_sha256,
                },

            "scientific_boundaries":
                {
                    "held_out_test_accessed":
                        True,

                    "all_test_images_accessed":
                        True,

                    "all_test_image_hashes_verified":
                        True,

                    "training_performed":
                        False,

                    "checkpoint_selection_performed":
                        False,

                    "learning_rate_selection_performed":
                        False,

                    "resolution_selection_performed":
                        False,

                    "threshold_derivation_from_test":
                        False,

                    "threshold_modification_on_test":
                        False,

                    "best_test_seed_selection_performed":
                        False,
                },
        }

        with results_partial.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                result_artifact,
                file,
                sort_keys=False,
            )

        # --------------------------------------------------------------
        # Promote both artifacts only after successful complete run.
        # --------------------------------------------------------------

        promoted_predictions = False

        try:

            os.replace(
                predictions_partial,
                predictions_path,
            )

            promoted_predictions = True

            os.replace(
                results_partial,
                results_path,
            )

        except Exception:

            if (
                promoted_predictions
                and predictions_path.exists()
            ):
                predictions_path.unlink()

            raise

        results_sha = sha256_file(
            results_path
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] test predictions: %s",
            predictions_path,
        )

        LOGGER.info(
            "Predictions SHA-256: %s",
            predictions_sha,
        )

        LOGGER.info(
            "[PASS] final test result: %s",
            results_path,
        )

        LOGGER.info(
            "Result SHA-256: %s",
            results_sha,
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "[PASS] threshold derivation from test: FALSE"
        )

        LOGGER.info(
            "[PASS] threshold modification on test: FALSE"
        )

        LOGGER.info(
            "[PASS] checkpoint/model selection on test: FALSE"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "SEED %d FINAL HELD-OUT TEST: COMPLETE",
            seed,
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "SEED %d FINAL HELD-OUT TEST: FAIL",
            seed,
        )

        for partial in (
            predictions_partial,
            results_partial,
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