#!/usr/bin/env python3
"""
Focused synthetic integration audit for the REAL Tech-2 LR-screening
entry point:

    tools/run_resnet18_lr_screening.py

Purpose
-------
The lower scientific components already have dedicated validation.

This audit therefore tests only the final glue:

    clean/provenance inputs
        ->
    run directory creation
        ->
    initialized
        ->
    running
        ->
    screening result handoff
        ->
    persistence handoff
        ->
    completed

and separately:

    initialized
        ->
    running
        ->
    synthetic scientific failure
        ->
    failed run retained

The expensive scientific functions are stubbed.

No FantasyID Dataset is constructed.
No image is decoded.
No model is constructed.
No CUDA computation is performed.
No forward/backward pass is performed.
No optimizer step is performed.
No threshold is derived.
No held-out test is accessed.

The existing stage-runner synthetic audit is rerun separately to prove
that one reported epoch still means exactly:

    one train_one_epoch
        ->
    one evaluate_dev_one_epoch
        ->
    one AUROC calculation

after live timing instrumentation was added.

No print() is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import logging
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

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
    load_machine_config,
)


RUNNER_PATH = (
    REPO_ROOT
    / "tools"
    / "run_resnet18_lr_screening.py"
)


spec = importlib.util.spec_from_file_location(
    "tech2_real_screening_runner_audit_target",
    RUNNER_PATH,
)

if (
    spec is None
    or spec.loader is None
):

    raise RuntimeError(
        "Could not import real LR-screening entry point."
    )


runner = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    runner
)


LOGGER = logging.getLogger(
    "audit_resnet18_real_screening_entrypoint"
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


def synthetic_sha(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


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

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        value = yaml.safe_load(
            file
        )

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            f"{path} must contain a YAML mapping."
        )

    return value


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
        f"Expected exception not raised: {label}"
    )


# ======================================================================
# Clean Git gate for the AUDIT itself
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
            "Git working tree is not clean before audit:\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Audit logging
# ======================================================================

def configure_audit_outputs(
    *,
    tool_cfg: dict[str, Any],
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

    log_cfg = tool_cfg[
        "logging"
    ]

    output_cfg = tool_cfg[
        "output"
    ]

    log_dir = resolve_repo_path(
        log_cfg[
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

    substitutions = {
        "machine_id":
            machine_id,

        "timestamp":
            timestamp,
    }

    log_path = (
        log_dir
        / str(
            log_cfg[
                "filename"
            ]
        ).format(
            **substitutions
        )
    )

    result_path = (
        output_dir
        / str(
            output_cfg[
                "filename"
            ]
        ).format(
            **substitutions
        )
    )

    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(
        logging.INFO
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
        result_path,
    )


# ======================================================================
# Synthetic scientific result
# ======================================================================

def build_synthetic_screening_result() -> Any:

    stage_a = SimpleNamespace(
        epochs_completed=3,
        stop_reason="patience",
        raw_best_epoch=2,
        raw_best_weighted_dev_loss=0.60,
        dev_auroc_at_raw_best_checkpoint=0.74,
        best_dev_auroc=0.75,
    )

    candidate_specs = (
        (
            0.00003,
            0.52,
            0.77,
            0.78,
            0.01,
            False,
        ),
        (
            0.0001,
            0.45,
            0.82,
            0.825,
            0.005,
            False,
        ),
        (
            0.0003,
            0.49,
            0.76,
            0.79,
            0.03,
            True,
        ),
    )

    candidates = []

    for (
        lr,
        loss,
        selected_auc,
        best_auc,
        difference,
        review,
    ) in candidate_specs:

        disagreement = SimpleNamespace(
            difference=difference,
            protocol_review_flag=review,
        )

        stage_b = SimpleNamespace(
            epochs_completed=6,
            stop_reason="patience",
            raw_best_epoch=2,
            raw_best_weighted_dev_loss=loss,
            dev_auroc_at_raw_best_checkpoint=(
                selected_auc
            ),
            best_dev_auroc=best_auc,
            stage_b_auroc_disagreement=(
                disagreement
            ),
        )

        candidates.append(
            SimpleNamespace(
                backbone_lr=lr,
                stage_b_result=stage_b,
            )
        )

    selected_model_sha = synthetic_sha(
        "synthetic-selected-stage-b-model"
    )

    return SimpleNamespace(
        stage_a_result=stage_a,
        stage_b_candidates=tuple(
            candidates
        ),
        selected_backbone_lr=0.0001,
        selected_raw_best_weighted_dev_loss=0.45,
        selected_stage_b_checkpoint_sha256=(
            selected_model_sha
        ),
        exact_loss_tie_encountered=False,
        protocol_review_required=True,
    )


# ======================================================================
# Synthetic persistence handoff
# ======================================================================

def build_fake_persist_function(
    *,
    expected_screening_result: Any,
    calls: dict[str, Any],
) -> Callable[..., Any]:

    def fake_persist(
        *,
        result: Any,
        run_dir: Path,
        run_id: str,
        git_commit: str,
        experiment_config_sha256: str,
    ) -> Any:

        calls[
            "persistence_calls"
        ] += 1

        if result is not expected_screening_result:

            raise RuntimeError(
                "Persistence received wrong screening-result object."
            )

        calls[
            "persistence_run_id"
        ] = run_id

        calls[
            "persistence_git_commit"
        ] = git_commit

        calls[
            "persistence_experiment_sha"
        ] = (
            experiment_config_sha256
        )

        summary_relative = (
            "metrics/lr_screening_summary.yaml"
        )

        manifest_relative = (
            "manifests/lr_screening_artifacts.yaml"
        )

        checkpoint_relative = (
            "checkpoints/"
            "stage_b_lr_1e-04_raw_best.pt"
        )

        summary_path = (
            run_dir
            / summary_relative
        )

        manifest_path = (
            run_dir
            / manifest_relative
        )

        checkpoint_path = (
            run_dir
            / checkpoint_relative
        )

        with summary_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                {
                    "synthetic":
                        True,

                    "selected_backbone_lr":
                        0.0001,
                },
                file,
                sort_keys=False,
            )

        with manifest_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                {
                    "synthetic":
                        True,

                    "artifact_count":
                        3,
                },
                file,
                sort_keys=False,
            )

        checkpoint_path.write_bytes(
            b"synthetic selected checkpoint"
        )

        summary_artifact = SimpleNamespace(
            path=summary_relative,
            sha256=sha256_file(
                summary_path
            ),
            artifact_type=(
                "lr_screening_summary_yaml"
            ),
            stage=None,
            selected=True,
        )

        manifest_artifact = SimpleNamespace(
            path=manifest_relative,
            sha256=sha256_file(
                manifest_path
            ),
            artifact_type=(
                "artifact_manifest_yaml"
            ),
            stage=None,
            selected=True,
        )

        checkpoint_artifact = (
            SimpleNamespace(
                path=checkpoint_relative,
                sha256=sha256_file(
                    checkpoint_path
                ),
                artifact_type=(
                    "raw_argmin_model_checkpoint"
                ),
                stage="stage_b",
                selected=True,
                model_state_sha256=(
                    expected_screening_result
                    .selected_stage_b_checkpoint_sha256
                ),
                backbone_lr=0.0001,
                checkpoint_epoch=2,
                weighted_dev_loss=0.45,
            )
        )

        return SimpleNamespace(
            run_id=run_id,
            summary_path=(
                summary_relative
            ),
            manifest_path=(
                manifest_relative
            ),
            selected_backbone_lr=0.0001,
            artifacts=(
                summary_artifact,
                manifest_artifact,
                checkpoint_artifact,
            ),
        )

    return fake_persist


# ======================================================================
# Common patches
# ======================================================================

def fake_runtime_provenance(
    *,
    configured_device: str,
) -> dict[str, Any]:

    return {
        "synthetic_audit":
            True,

        "configured_device":
            configured_device,

        "cuda_execution_performed":
            False,
    }


def fake_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> dict[str, str]:

    return {
        "path":
            "synthetic_validator_artifact",

        "sha256":
            synthetic_sha(
                "synthetic-validator-artifact"
            ),
    }


# ======================================================================
# Success lifecycle
# ======================================================================

def audit_success_lifecycle(
    *,
    git_commit: str,
    experiment_sha: str,
    resolution: str,
    run_seed: int,
    expected_total_epochs: int,
) -> dict[str, Any]:

    synthetic_result = (
        build_synthetic_screening_result()
    )

    calls: dict[str, Any] = {
        "screening_calls":
            0,

        "persistence_calls":
            0,

        "persistence_run_id":
            None,

        "persistence_git_commit":
            None,

        "persistence_experiment_sha":
            None,
    }

    def fake_screening(
        *,
        experiment_cfg: Any,
        machine_cfg: Any,
        repo_root: Path,
        resolution_name: str,
        run_seed: int,
    ) -> Any:

        calls[
            "screening_calls"
        ] += 1

        if resolution_name != resolution:

            raise RuntimeError(
                "Entry point forwarded wrong resolution."
            )

        if run_seed != 8:

            raise RuntimeError(
                "Entry point forwarded wrong screening seed."
            )

        if (
            Path(
                repo_root
            ).resolve()
            != REPO_ROOT
        ):

            raise RuntimeError(
                "Entry point forwarded wrong repository root."
            )

        return synthetic_result

    fake_persist = (
        build_fake_persist_function(
            expected_screening_result=(
                synthetic_result
            ),
            calls=calls,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="tech2_entrypoint_success_"
    ) as temporary_root:

        runs_root = (
            Path(
                temporary_root
            )
            / "runs"
        )

        statuses: list[
            str | None
        ] = []

        original_write_yaml = (
            runner.write_yaml_atomic
        )

        def tracking_write_yaml(
            *,
            path: Path,
            value: dict[str, Any],
        ) -> None:

            if path.name == "run.yaml":

                statuses.append(
                    value.get(
                        "run",
                        {},
                    ).get(
                        "status"
                    )
                )

            original_write_yaml(
                path=path,
                value=value,
            )

        argv = [
            str(
                RUNNER_PATH
            ),
            "--resolution",
            resolution,
            "--run-seed",
            str(
                run_seed
            ),
        ]

        fake_git = {
            "commit_sha":
                git_commit,

            "branch":
                "main",

            "working_tree_clean_at_run_start":
                True,
        }

        with (
            patch.object(
                runner,
                "require_clean_git",
                return_value=fake_git,
            ),
            patch.object(
                runner,
                "run_canonical_validator",
                side_effect=fake_validator,
            ),
            patch.object(
                runner,
                "runtime_provenance",
                side_effect=(
                    fake_runtime_provenance
                ),
            ),
            patch.object(
                runner,
                "resolve_runs_root",
                return_value=(
                    runs_root
                ),
            ),
            patch.object(
                runner,
                "run_resolution_lr_screening",
                side_effect=(
                    fake_screening
                ),
            ),
            patch.object(
                runner,
                "persist_resolution_lr_screening_result",
                side_effect=(
                    fake_persist
                ),
            ),
            patch.object(
                runner,
                "write_yaml_atomic",
                side_effect=(
                    tracking_write_yaml
                ),
            ),
            patch.object(
                sys,
                "argv",
                argv,
            ),
            redirect_stdout(
                io.StringIO()
            ),
        ):

            return_code = (
                runner.main()
            )

        if return_code != 0:

            raise RuntimeError(
                "Synthetic successful run returned non-zero."
            )

        if statuses != [
            "initialized",
            "running",
            "completed",
        ]:

            raise RuntimeError(
                "Successful run lifecycle mismatch:\n"
                f"  {statuses}"
            )

        run_directories = [
            path
            for path
            in runs_root.iterdir()
            if path.is_dir()
        ]

        if len(
            run_directories
        ) != 1:

            raise RuntimeError(
                "Successful audit did not create exactly one run directory."
            )

        run_dir = (
            run_directories[
                0
            ]
        )

        run_yaml_path = (
            run_dir
            / "run.yaml"
        )

        with run_yaml_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            run_record = yaml.safe_load(
                file
            )

        if (
            run_record[
                "run"
            ][
                "status"
            ]
            != "completed"
        ):

            raise RuntimeError(
                "Successful run.yaml is not completed."
            )

        if (
            run_record[
                "run"
            ][
                "resolution_name"
            ]
            != resolution
        ):

            raise RuntimeError(
                "Successful run.yaml resolution mismatch."
            )

        if (
            run_record[
                "run"
            ][
                "run_seed"
            ]
            != run_seed
        ):

            raise RuntimeError(
                "Successful run.yaml seed mismatch."
            )

        if (
            run_record[
                "frozen_data"
            ][
                "held_out_test_accessed"
            ]
            is not False
        ):

            raise RuntimeError(
                "Successful run does not explicitly preserve "
                "held-out-test non-access."
            )

        result_record = (
            run_record[
                "result"
            ]
        )

        if (
            result_record[
                "selected_backbone_lr"
            ]
            != 0.0001
        ):

            raise RuntimeError(
                "Selected LR was not propagated to run.yaml."
            )

        if (
            result_record[
                "selected_stage_b_checkpoint_model_state_sha256"
            ]
            != synthetic_result
            .selected_stage_b_checkpoint_sha256
        ):

            raise RuntimeError(
                "Selected model-state identity was not propagated."
            )

        timing = (
            run_record[
                "timing"
            ]
        )

        if (
            timing[
                "total_completed_training_epochs"
            ]
            != expected_total_epochs
        ):

            raise RuntimeError(
                "Total completed epoch count mismatch."
            )

        for key in (
            "screening_seconds",
            "persistence_seconds",
            "mean_seconds_per_completed_epoch",
        ):

            if float(
                timing[
                    key
                ]
            ) < 0.0:

                raise RuntimeError(
                    f"Negative timing measurement: {key}"
                )

        if calls[
            "screening_calls"
        ] != 1:

            raise RuntimeError(
                "Entry point invoked screening more than once."
            )

        if calls[
            "persistence_calls"
        ] != 1:

            raise RuntimeError(
                "Entry point invoked persistence more than once."
            )

        if (
            calls[
                "persistence_git_commit"
            ]
            != git_commit
        ):

            raise RuntimeError(
                "Persistence received wrong Git commit."
            )

        if (
            calls[
                "persistence_experiment_sha"
            ]
            != experiment_sha
        ):

            raise RuntimeError(
                "Persistence received wrong experiment-config SHA."
            )

        if not (
            run_dir
            / "experiment_config.yaml"
        ).is_file():

            raise RuntimeError(
                "Scientific config snapshot missing."
            )

        screening_log = (
            run_dir
            / "logs"
            / "lr_screening.log"
        )

        if not screening_log.is_file():

            raise RuntimeError(
                "Scientific screening log missing."
            )

        log_text = screening_log.read_text(
            encoding="utf-8"
        )

        if (
            "REAL LR SCREENING: PASS"
            not in log_text
        ):

            raise RuntimeError(
                "Success marker missing from scientific log."
            )

        if (
            "Held-out test: NOT ACCESSED"
            not in log_text
        ):

            raise RuntimeError(
                "Held-out-test non-access marker missing from log."
            )

        LOGGER.info(
            "[PASS] success lifecycle initialized -> running -> completed"
        )

        LOGGER.info(
            "[PASS] screening called exactly once"
        )

        LOGGER.info(
            "[PASS] persistence called exactly once"
        )

        LOGGER.info(
            "[PASS] result/checkpoint identity propagated into run.yaml"
        )

        LOGGER.info(
            "[PASS] run timing fields populated"
        )

        return {
            "status_sequence":
                statuses,

            "screening_calls":
                calls[
                    "screening_calls"
                ],

            "persistence_calls":
                calls[
                    "persistence_calls"
                ],

            "total_completed_training_epochs":
                timing[
                    "total_completed_training_epochs"
                ],

            "selected_backbone_lr":
                result_record[
                    "selected_backbone_lr"
                ],

            "held_out_test_accessed":
                False,

            "scientific_log_created":
                True,

            "config_snapshot_created":
                True,
        }


# ======================================================================
# Failure lifecycle
# ======================================================================

def audit_failure_lifecycle(
    *,
    git_commit: str,
    resolution: str,
    run_seed: int,
    failure_message: str,
) -> dict[str, Any]:

    with tempfile.TemporaryDirectory(
        prefix="tech2_entrypoint_failure_"
    ) as temporary_root:

        runs_root = (
            Path(
                temporary_root
            )
            / "runs"
        )

        statuses: list[
            str | None
        ] = []

        persistence_calls = {
            "count":
                0,
        }

        original_write_yaml = (
            runner.write_yaml_atomic
        )

        def tracking_write_yaml(
            *,
            path: Path,
            value: dict[str, Any],
        ) -> None:

            if path.name == "run.yaml":

                statuses.append(
                    value.get(
                        "run",
                        {},
                    ).get(
                        "status"
                    )
                )

            original_write_yaml(
                path=path,
                value=value,
            )

        def fail_screening(
            **_: Any,
        ) -> Any:

            raise RuntimeError(
                failure_message
            )

        def forbidden_persistence(
            **_: Any,
        ) -> Any:

            persistence_calls[
                "count"
            ] += 1

            raise RuntimeError(
                "Persistence must not run after screening failure."
            )

        argv = [
            str(
                RUNNER_PATH
            ),
            "--resolution",
            resolution,
            "--run-seed",
            str(
                run_seed
            ),
        ]

        fake_git = {
            "commit_sha":
                git_commit,

            "branch":
                "main",

            "working_tree_clean_at_run_start":
                True,
        }

        with (
            patch.object(
                runner,
                "require_clean_git",
                return_value=fake_git,
            ),
            patch.object(
                runner,
                "run_canonical_validator",
                side_effect=fake_validator,
            ),
            patch.object(
                runner,
                "runtime_provenance",
                side_effect=(
                    fake_runtime_provenance
                ),
            ),
            patch.object(
                runner,
                "resolve_runs_root",
                return_value=(
                    runs_root
                ),
            ),
            patch.object(
                runner,
                "run_resolution_lr_screening",
                side_effect=(
                    fail_screening
                ),
            ),
            patch.object(
                runner,
                "persist_resolution_lr_screening_result",
                side_effect=(
                    forbidden_persistence
                ),
            ),
            patch.object(
                runner,
                "write_yaml_atomic",
                side_effect=(
                    tracking_write_yaml
                ),
            ),
            patch.object(
                sys,
                "argv",
                argv,
            ),
            redirect_stdout(
                io.StringIO()
            ),
        ):

            return_code = (
                runner.main()
            )

        if return_code != 1:

            raise RuntimeError(
                "Synthetic failed run did not return 1."
            )

        if statuses != [
            "initialized",
            "running",
            "failed",
        ]:

            raise RuntimeError(
                "Failed run lifecycle mismatch:\n"
                f"  {statuses}"
            )

        if persistence_calls[
            "count"
        ] != 0:

            raise RuntimeError(
                "Persistence executed despite screening failure."
            )

        run_directories = [
            path
            for path
            in runs_root.iterdir()
            if path.is_dir()
        ]

        if len(
            run_directories
        ) != 1:

            raise RuntimeError(
                "Failed run directory was not retained."
            )

        run_dir = (
            run_directories[
                0
            ]
        )

        with (
            run_dir
            / "run.yaml"
        ).open(
            "r",
            encoding="utf-8",
        ) as file:

            run_record = yaml.safe_load(
                file
            )

        if (
            run_record[
                "run"
            ][
                "status"
            ]
            != "failed"
        ):

            raise RuntimeError(
                "Failed run.yaml does not retain failed state."
            )

        failure = (
            run_record[
                "failure"
            ]
        )

        if (
            failure[
                "exception_type"
            ]
            != "RuntimeError"
        ):

            raise RuntimeError(
                "Failure exception type was not persisted."
            )

        if (
            failure[
                "message"
            ]
            != failure_message
        ):

            raise RuntimeError(
                "Failure message was not persisted exactly."
            )

        if (
            run_record[
                "frozen_data"
            ][
                "held_out_test_accessed"
            ]
            is not False
        ):

            raise RuntimeError(
                "Failed run lost held-out-test non-access state."
            )

        screening_log = (
            run_dir
            / "logs"
            / "lr_screening.log"
        )

        if not screening_log.is_file():

            raise RuntimeError(
                "Failed scientific run did not retain its log."
            )

        if (
            "REAL LR SCREENING: FAIL"
            not in screening_log.read_text(
                encoding="utf-8"
            )
        ):

            raise RuntimeError(
                "Failure marker missing from retained scientific log."
            )

        LOGGER.info(
            "[PASS] failure lifecycle initialized -> running -> failed"
        )

        LOGGER.info(
            "[PASS] failed run directory and scientific log retained"
        )

        LOGGER.info(
            "[PASS] persistence not called after screening failure"
        )

        return {
            "status_sequence":
                statuses,

            "return_code":
                return_code,

            "persistence_calls":
                persistence_calls[
                    "count"
                ],

            "failure_type":
                failure[
                    "exception_type"
                ],

            "failure_message":
                failure[
                    "message"
                ],

            "failed_run_retained":
                True,

            "held_out_test_accessed":
                False,
        }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser()

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
            "audit_resnet18_real_screening_entrypoint_config.yaml"
        ),
    )

    args = parser.parse_args()

    commit_sha = require_clean_git()

    audit_config_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        audit_config_path
    )

    experiment_cfg, experiment_path = (
        load_experiment_config(
            args.config
        )
    )

    machine_cfg, _ = (
        load_machine_config(
            args.machine_config,
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

    (
        log_path,
        result_path,
    ) = configure_audit_outputs(
        tool_cfg=tool_cfg,
        machine_id=machine_id,
    )

    try:

        audit_cfg = (
            tool_cfg[
                "audit"
            ]
        )

        experiment_sha = sha256_file(
            experiment_path
        )

        expected_experiment_sha = str(
            audit_cfg[
                "expected_experiment_config_sha256"
            ]
        )

        if (
            experiment_sha
            != expected_experiment_sha
        ):

            raise RuntimeError(
                "Experiment-config SHA mismatch."
            )

        resolution = str(
            audit_cfg[
                "resolution"
            ]
        )

        run_seed = int(
            audit_cfg[
                "run_seed"
            ]
        )

        expected_total_epochs = int(
            audit_cfg[
                "expected_total_completed_epochs"
            ]
        )

        failure_message = str(
            audit_cfg[
                "failure_message"
            ]
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 REAL SCREENING ENTRY-POINT SYNTHETIC AUDIT"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            commit_sha,
        )

        LOGGER.info(
            "Machine ID: %s",
            machine_id,
        )

        LOGGER.info(
            "Entry-point SHA-256: %s",
            sha256_file(
                RUNNER_PATH
            ),
        )

        LOGGER.info(
            "stage_runner.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "stage_runner.py"
            ),
        )

        LOGGER.info(
            "lr_screening.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "lr_screening.py"
            ),
        )

        LOGGER.info(
            "Experiment config SHA-256: %s",
            experiment_sha,
        )

        LOGGER.info(
            "Scientific training: NONE"
        )

        LOGGER.info(
            "FantasyID data/images: NOT ACCESSED"
        )

        LOGGER.info(
            "CUDA computation: NONE"
        )

        LOGGER.info(
            "Held-out test: NOT ACCESSED"
        )

        success = (
            audit_success_lifecycle(
                git_commit=commit_sha,
                experiment_sha=(
                    experiment_sha
                ),
                resolution=resolution,
                run_seed=run_seed,
                expected_total_epochs=(
                    expected_total_epochs
                ),
            )
        )

        failure = (
            audit_failure_lifecycle(
                git_commit=commit_sha,
                resolution=resolution,
                run_seed=run_seed,
                failure_message=(
                    failure_message
                ),
            )
        )

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
                        experiment_sha,

                    "entrypoint_sha256":
                        sha256_file(
                            RUNNER_PATH
                        ),

                    "stage_runner_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "stage_runner.py"
                        ),

                    "lr_screening_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "lr_screening.py"
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
                },

            "scope":
                {
                    "synthetic_only":
                        True,

                    "fantasyid_dataset_constructed":
                        False,

                    "images_decoded":
                        False,

                    "model_constructed":
                        False,

                    "cuda_computation_performed":
                        False,

                    "scientific_training_performed":
                        False,

                    "fpr10_threshold_derived":
                        False,

                    "held_out_test_accessed":
                        False,
                },

            "success_lifecycle":
                success,

            "failure_lifecycle":
                failure,

            "interpretation":
                (
                    "PASS establishes the real LR-screening command's "
                    "run lifecycle and final handoff glue without "
                    "scientific training. Successful execution records "
                    "initialized -> running -> completed, propagates "
                    "selection/checkpoint/timing evidence and persists "
                    "held-out-test non-access. Synthetic scientific "
                    "failure records initialized -> running -> failed, "
                    "retains the run directory/log and does not invoke "
                    "persistence."
                ),
        }

        temporary_result = Path(
            str(
                result_path
            )
            + ".partial"
        )

        with temporary_result.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file:

            yaml.safe_dump(
                result,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        temporary_result.replace(
            result_path
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "AUDIT SUMMARY"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "[PASS] success lifecycle initialized -> running -> completed"
        )

        LOGGER.info(
            "[PASS] screening invoked once"
        )

        LOGGER.info(
            "[PASS] persistence invoked once"
        )

        LOGGER.info(
            "[PASS] selected LR/checkpoint propagated"
        )

        LOGGER.info(
            "[PASS] timing fields populated"
        )

        LOGGER.info(
            "[PASS] failure lifecycle initialized -> running -> failed"
        )

        LOGGER.info(
            "[PASS] failed run/log retained"
        )

        LOGGER.info(
            "[PASS] persistence skipped after scientific failure"
        )

        LOGGER.info(
            "[PASS] held-out test NOT ACCESSED"
        )

        LOGGER.info(
            "Audit result: %s",
            result_path,
        )

        LOGGER.info(
            "Audit result SHA-256: %s",
            sha256_file(
                result_path
            ),
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "REAL SCREENING ENTRY-POINT SYNTHETIC AUDIT: PASS"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "REAL SCREENING ENTRY-POINT SYNTHETIC AUDIT: FAIL"
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