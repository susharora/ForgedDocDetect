#!/usr/bin/env python3
"""
Synthetic audit for Tech-2 held-out Dataset + final metric primitives.

Canonical held-out evidence read by this audit
----------------------------------------------
Only:

    canonical manifest YAML
    canonical manifest CSV

No dataset_root is supplied while validating those files, so canonical
test image paths cannot be followed.

Pixel-level Dataset tests
-------------------------
A temporary synthetic dataset is created under /tmp. Those generated
JPEGs exercise:

- explicit held-out access token;
- SHA-256 verification before decode;
- frozen r512 preprocessing;
- target/metadata return;
- corruption rejection.

Metric tests
------------
Synthetic CPU logits validate:

- bonafide=0 / attack=1;
- p_attack threshold decisions;
- exact 0.5 equality -> attack;
- confusion semantics;
- support-weighted F1;
- attack F1;
- balanced accuracy;
- FPR/FNR/HTER;
- APCER=FNR;
- BPCER=FPR;
- AUROC from z1-z0;
- permutation invariance;
- NaN/invalid-target rejection.

No real test JPEG is opened.
No model/checkpoint is loaded.
No model inference is performed.
No threshold is derived.

No print() is used.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import yaml
from PIL import Image


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
)

from src.data import (
    build_preprocessor_from_config,
)

from src.final_evaluation import (
    HELD_OUT_TEST_ACCESS_TOKEN,
    HeldOutTestManifestDataset,
    compute_final_binary_evaluation,
    compute_operating_point_metrics,
    load_canonical_held_out_test_manifest,
    load_final_evaluation_contract,
)


LOGGER = logging.getLogger(
    "audit_resnet18_final_evaluation"
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
            "Git tree must be clean before final-evaluation audit.\n\n"
            f"{status}"
        )

    return commit


def expect_exception(
    *,
    label: str,
    function: Callable[[], Any],
    expected: tuple[
        type[BaseException],
        ...,
    ],
) -> None:

    try:

        function()

    except expected:

        LOGGER.info(
            "[PASS] %s",
            label,
        )

        return

    raise RuntimeError(
        f"Expected exception was not raised: {label}"
    )


def assert_close(
    *,
    label: str,
    actual: float,
    expected: float,
    tolerance: float = 1.0e-15,
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

def configure_outputs(
    *,
    config: Mapping[Any, Any],
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

    log_cfg = require_mapping(
        config[
            "logging"
        ],
        "logging",
    )

    output_cfg = require_mapping(
        config[
            "output"
        ],
        "output",
    )

    log_directory = resolve_repo_path(
        log_cfg[
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
            log_cfg[
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
# Synthetic Dataset
# ======================================================================

def synthetic_dataset_audit(
    *,
    experiment_cfg: Mapping[str, Any],
) -> dict[str, Any]:

    with tempfile.TemporaryDirectory(
        prefix="tech2_final_eval_"
    ) as temporary:

        root = Path(
            temporary
        )

        bona_path = (
            root
            / "test"
            / "bonafide"
            / "huawei"
            / "bona.jpg"
        )

        attack_path = (
            root
            / "test"
            / "attack"
            / "digital_3"
            / "huawei"
            / "attack.jpg"
        )

        bona_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        attack_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.new(
            "RGB",
            (
                320,
                200,
            ),
            (
                10,
                20,
                30,
            ),
        ).save(
            bona_path,
            format="JPEG",
            quality=95,
        )

        Image.new(
            "RGB",
            (
                320,
                200,
            ),
            (
                220,
                210,
                200,
            ),
        ).save(
            attack_path,
            format="JPEG",
            quality=95,
        )

        bona_sha = sha256_file(
            bona_path
        )

        attack_sha = sha256_file(
            attack_path
        )

        rows = (
            {
                "source_workbook_row":
                    2,

                "image_path":
                    "test/bonafide/huawei/bona.jpg",

                "image_sha256":
                    bona_sha,

                "file_stem":
                    "bona",

                "traffic_type":
                    "bonafide",

                "label":
                    0,

                "variant":
                    "",

                "hardware_source":
                    "huawei",

                "face_db":
                    "synthetic",

                "face_id":
                    "1",

                "gender":
                    "F",

                "project_role":
                    "held_out_test",
            },

            {
                "source_workbook_row":
                    3,

                "image_path":
                    "test/attack/digital_3/huawei/attack.jpg",

                "image_sha256":
                    attack_sha,

                "file_stem":
                    "attack",

                "traffic_type":
                    "attack",

                "label":
                    1,

                "variant":
                    "digital_3",

                "hardware_source":
                    "huawei",

                "face_db":
                    "synthetic",

                "face_id":
                    "2",

                "gender":
                    "M",

                "project_role":
                    "held_out_test",
            },
        )

        preprocessor = (
            build_preprocessor_from_config(
                experiment_cfg=(
                    experiment_cfg
                ),
                resolution_name="r512",
            )
        )

        expect_exception(
            label=(
                "held-out Dataset rejects missing/incorrect access token"
            ),
            function=lambda:
                HeldOutTestManifestDataset(
                    rows=rows,
                    dataset_root=root,
                    preprocessor=preprocessor,
                    expected_rows=2,
                    expected_class_counts={
                        "bonafide":
                            1,

                        "attack":
                            1,
                    },
                    access_token="NO",
                ),
            expected=(
                PermissionError,
            ),
        )

        dataset = HeldOutTestManifestDataset(
            rows=rows,
            dataset_root=root,
            preprocessor=preprocessor,
            expected_rows=2,
            expected_class_counts={
                "bonafide":
                    1,

                "attack":
                    1,
            },
            access_token=(
                HELD_OUT_TEST_ACCESS_TOKEN
            ),
        )

        if len(
            dataset
        ) != 2:

            raise RuntimeError(
                "Synthetic held-out Dataset length mismatch."
            )

        bona = dataset[
            0
        ]

        attack = dataset[
            1
        ]

        expected_shape = (
            3,
            512,
            864,
        )

        for (
            label,
            sample,
            expected_target,
        ) in (
            (
                "bona",
                bona,
                0,
            ),
            (
                "attack",
                attack,
                1,
            ),
        ):

            if tuple(
                sample[
                    "image"
                ].shape
            ) != expected_shape:

                raise RuntimeError(
                    f"{label}: r512 preprocessing shape changed."
                )

            if (
                sample[
                    "image"
                ].dtype
                != torch.float32
            ):

                raise RuntimeError(
                    f"{label}: preprocessed tensor not float32."
                )

            if int(
                sample[
                    "label"
                ]
            ) != expected_target:

                raise RuntimeError(
                    f"{label}: target mismatch."
                )

            if (
                sample[
                    "verified_image_sha256"
                ]
                != sample[
                    "image_sha256"
                ]
            ):

                raise RuntimeError(
                    f"{label}: file SHA was not verified."
                )

        LOGGER.info(
            "[PASS] synthetic held-out access requires explicit token"
        )

        LOGGER.info(
            "[PASS] synthetic held-out file SHA verified before decode"
        )

        LOGGER.info(
            "[PASS] synthetic r512 decode/preprocess = [3,512,864]"
        )

        LOGGER.info(
            "[PASS] synthetic labels = bonafide 0 / attack 1"
        )

        # --------------------------------------------------------------
        # Corruption after Dataset creation must be detected.
        # --------------------------------------------------------------

        with attack_path.open(
            "ab"
        ) as file:

            file.write(
                b"corruption"
            )

        expect_exception(
            label="held-out file mutation is rejected by SHA-256",
            function=lambda:
                dataset[
                    1
                ],
            expected=(
                RuntimeError,
            ),
        )

        return {
            "synthetic_rows":
                2,

            "r512_shape":
                list(
                    expected_shape
                ),

            "sha_verification":
                True,

            "corruption_rejected":
                True,

            "access_token_required":
                True,
        }


# ======================================================================
# Synthetic metrics
# ======================================================================

def synthetic_metric_audit(
    *,
    contract: Any,
) -> dict[str, Any]:

    # Same four ranking margins in each class.
    #
    # At p_attack >= .5:
    #
    #   TN=2 FP=2 FN=2 TP=2
    #
    # so every headline balanced metric = .5.
    margins = torch.tensor(
        [
            -2.0,
            -0.5,
            0.5,
            2.0,

            -2.0,
            -0.5,
            0.5,
            2.0,
        ],
        dtype=torch.float32,
    )

    logits = torch.stack(
        (
            torch.zeros_like(
                margins
            ),
            margins,
        ),
        dim=1,
    )

    targets = torch.tensor(
        [
            0,
            0,
            0,
            0,

            1,
            1,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    result = compute_final_binary_evaluation(
        logits=logits,
        targets=targets,
        fpr10_threshold=0.2,
        contract=contract,
    )

    fixed = result.fixed_0_5

    expected_confusion = (
        fixed.true_positive,
        fixed.false_positive,
        fixed.true_negative,
        fixed.false_negative,
    )

    if expected_confusion != (
        2,
        2,
        2,
        2,
    ):

        raise RuntimeError(
            "Fixed-0.5 confusion semantics failed:\n"
            f"  actual={expected_confusion}"
        )

    for (
        label,
        actual,
    ) in (
        (
            "fixed accuracy",
            fixed.accuracy,
        ),
        (
            "fixed weighted F1",
            fixed.support_weighted_f1,
        ),
        (
            "fixed attack F1",
            fixed.attack.f1,
        ),
        (
            "fixed balanced accuracy",
            fixed.balanced_accuracy,
        ),
        (
            "fixed FPR",
            fixed.false_positive_rate,
        ),
        (
            "fixed FNR",
            fixed.false_negative_rate,
        ),
        (
            "fixed HTER",
            fixed.half_total_error_rate,
        ),
        (
            "fixed APCER",
            fixed.apcer,
        ),
        (
            "fixed BPCER",
            fixed.bpcer,
        ),
        (
            "AUROC",
            result.auroc,
        ),
    ):

        assert_close(
            label=label,
            actual=actual,
            expected=0.5,
        )

    if fixed.matthews_correlation_coefficient != 0.0:

        raise RuntimeError(
            "Expected MCC=0 for symmetric synthetic case."
        )

    LOGGER.info(
        "[PASS] fixed 0.5 confusion = TP2 FP2 TN2 FN2"
    )

    LOGGER.info(
        "[PASS] fixed support-weighted F1 = 0.5"
    )

    LOGGER.info(
        "[PASS] fixed attack F1 = 0.5"
    )

    LOGGER.info(
        "[PASS] fixed FPR/FNR/HTER = 0.5/0.5/0.5"
    )

    LOGGER.info(
        "[PASS] APCER=FNR and BPCER=FPR"
    )

    LOGGER.info(
        "[PASS] fixed balanced accuracy = 1 - HTER = 0.5"
    )

    LOGGER.info(
        "[PASS] AUROC(z1-z0) = 0.5"
    )

    # --------------------------------------------------------------
    # Synthetic second threshold p_attack >= .2.
    #
    # Per class:
    #     margins -2 < probability .2
    #     margins -.5,.5,2 > probability .2
    #
    # TN=1 FP=3 FN=1 TP=3
    # --------------------------------------------------------------

    controlled = (
        result
        .controlled_fpr10
    )

    if (
        controlled.true_positive,
        controlled.false_positive,
        controlled.true_negative,
        controlled.false_negative,
    ) != (
        3,
        3,
        1,
        1,
    ):

        raise RuntimeError(
            "Controlled-threshold confusion calculation failed."
        )

    assert_close(
        label="controlled FPR",
        actual=(
            controlled
            .false_positive_rate
        ),
        expected=0.75,
    )

    assert_close(
        label="controlled FNR",
        actual=(
            controlled
            .false_negative_rate
        ),
        expected=0.25,
    )

    assert_close(
        label="controlled HTER",
        actual=(
            controlled
            .half_total_error_rate
        ),
        expected=0.5,
    )

    assert_close(
        label="controlled attack F1",
        actual=(
            controlled
            .attack
            .f1
        ),
        expected=0.6,
    )

    assert_close(
        label="controlled bonafide F1",
        actual=(
            controlled
            .bonafide
            .f1
        ),
        expected=(
            1.0
            / 3.0
        ),
    )

    assert_close(
        label="controlled support-weighted F1",
        actual=(
            controlled
            .support_weighted_f1
        ),
        expected=(
            7.0
            / 15.0
        ),
    )

    LOGGER.info(
        "[PASS] arbitrary frozen threshold uses same generic evaluator"
    )

    LOGGER.info(
        "[PASS] support-weighted F1 is true-support weighted"
    )

    # --------------------------------------------------------------
    # Exact p_attack=.5 equality.
    # --------------------------------------------------------------

    tie_logits = torch.tensor(
        [
            [
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
            ],
        ],
        dtype=torch.float32,
    )

    tie_targets = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    tie_result = (
        compute_final_binary_evaluation(
            logits=tie_logits,
            targets=tie_targets,
            fpr10_threshold=0.5,
            contract=contract,
        )
    )

    if (
        tie_result
        .fixed_0_5
        .false_positive
        != 1
    ):

        raise RuntimeError(
            "Exact p_attack=.5 bona-fide did not classify as attack."
        )

    if (
        tie_result
        .fixed_0_5
        .true_positive
        != 1
    ):

        raise RuntimeError(
            "Exact p_attack=.5 attack did not classify as attack."
        )

    LOGGER.info(
        "[PASS] exact p_attack=0.5 equality predicts attack"
    )

    # --------------------------------------------------------------
    # Permutation invariance.
    # --------------------------------------------------------------

    permutation = torch.tensor(
        [
            7,
            0,
            4,
            2,
            6,
            1,
            5,
            3,
        ],
        dtype=torch.int64,
    )

    permuted = compute_final_binary_evaluation(
        logits=logits[
            permutation
        ],
        targets=targets[
            permutation
        ],
        fpr10_threshold=0.2,
        contract=contract,
    )

    if asdict(
        permuted
    ) != asdict(
        result
    ):

        raise RuntimeError(
            "Final metrics changed under sample permutation."
        )

    LOGGER.info(
        "[PASS] final metrics are permutation invariant"
    )

    # --------------------------------------------------------------
    # Imbalanced support-weighting probe.
    # --------------------------------------------------------------

    imbalanced_scores = torch.tensor(
        [
            0.1,
            0.1,
            0.1,
            0.1,
        ],
        dtype=torch.float64,
    )

    imbalanced_targets = torch.tensor(
        [
            0,
            0,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    imbalanced = compute_operating_point_metrics(
        scores=imbalanced_scores,
        targets=imbalanced_targets,
        threshold=0.5,
        name="support_weight_probe",
    )

    expected_bona_f1 = (
        6.0
        / 7.0
    )

    expected_weighted_f1 = (
        3.0
        / 4.0
        * expected_bona_f1
    )

    assert_close(
        label="imbalanced bona F1",
        actual=(
            imbalanced
            .bonafide
            .f1
        ),
        expected=(
            expected_bona_f1
        ),
    )

    assert_close(
        label="support weighted F1 probe",
        actual=(
            imbalanced
            .support_weighted_f1
        ),
        expected=(
            expected_weighted_f1
        ),
    )

    LOGGER.info(
        "[PASS] weighted F1 uses sample support, not training CE weights"
    )

    # --------------------------------------------------------------
    # Fatal numerical / target guards.
    # --------------------------------------------------------------

    bad_logits = (
        logits
        .clone()
    )

    bad_logits[
        0,
        0,
    ] = float(
        "nan"
    )

    expect_exception(
        label="NaN final logits are fatal",
        function=lambda:
            compute_final_binary_evaluation(
                logits=bad_logits,
                targets=targets,
                fpr10_threshold=0.2,
                contract=contract,
            ),
        expected=(
            RuntimeError,
        ),
    )

    bad_targets = (
        targets
        .clone()
    )

    bad_targets[
        0
    ] = 2

    expect_exception(
        label="target outside frozen {0,1} is fatal",
        function=lambda:
            compute_final_binary_evaluation(
                logits=logits,
                targets=bad_targets,
                fpr10_threshold=0.2,
                contract=contract,
            ),
        expected=(
            RuntimeError,
        ),
    )

    return {
        "fixed_0_5":
            asdict(
                fixed
            ),

        "controlled_probe":
            asdict(
                controlled
            ),

        "auroc":
            result.auroc,

        "permutation_invariant":
            True,

        "support_weighting_verified":
            True,

        "exact_half_predicts_attack":
            True,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit final ResNet-18 held-out Dataset and metrics "
            "without opening real test images."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "tools/"
            "audit_resnet18_final_evaluation_config.yaml"
        ),
    )

    args = parser.parse_args()

    git_commit = (
        require_clean_git()
    )

    config_path = resolve_repo_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    machine_id = platform.node()

    (
        log_path,
        output_path,
    ) = configure_outputs(
        config=config,
        machine_id=machine_id,
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "RESNET-18 FINAL EVALUATION PRIMITIVE AUDIT"
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

        # ==============================================================
        # Frozen experiment config
        # ==============================================================

        experiment_cfg_info = require_mapping(
            config[
                "experiment"
            ],
            "experiment",
        )

        experiment_path = resolve_repo_path(
            experiment_cfg_info[
                "config"
            ]
        )

        experiment_sha = sha256_file(
            experiment_path
        )

        if (
            experiment_sha
            != experiment_cfg_info[
                "sha256"
            ]
        ):

            raise RuntimeError(
                "Frozen experiment config SHA mismatch."
            )

        experiment_cfg, loaded_path = (
            load_experiment_config(
                experiment_path
            )
        )

        if (
            loaded_path.resolve()
            != experiment_path
        ):

            raise RuntimeError(
                "Experiment config loader resolved unexpected path."
            )

        LOGGER.info(
            "[PASS] frozen scientific config SHA-256 = %s",
            experiment_sha,
        )

        # ==============================================================
        # Final-detection protocol
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

        if (
            protocol_sha
            != protocol_cfg_info[
                "sha256"
            ]
        ):

            raise RuntimeError(
                "Final detection protocol SHA mismatch."
            )

        protocol_cfg = load_yaml(
            protocol_path
        )

        contract = (
            load_final_evaluation_contract(
                evaluation_cfg=(
                    protocol_cfg
                )
            )
        )

        LOGGER.info(
            "[PASS] final binary contract parsed"
        )

        LOGGER.info(
            "[PASS] class polarity = bonafide 0 / attack 1"
        )

        LOGGER.info(
            "[PASS] fixed operating point = p_attack >= 0.5"
        )

        LOGGER.info(
            "[PASS] controlled threshold applied unchanged from dev"
        )

        LOGGER.info(
            "[PASS] AUROC ranking score = z1-z0"
        )

        # ==============================================================
        # Canonical held-out manifest metadata ONLY
        # ==============================================================

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

        csv_path = resolve_repo_path(
            csv_cfg[
                "path"
            ]
        )

        if (
            sha256_file(
                csv_path
            )
            != csv_cfg[
                "sha256"
            ]
        ):

            raise RuntimeError(
                "Canonical held-out CSV SHA mismatch."
            )

        (
            held_out_contract,
            held_out_rows,
        ) = load_canonical_held_out_test_manifest(
            repo_root=(
                REPO_ROOT
            ),
            metadata_path=(
                metadata_path
            ),
            expected_metadata_sha256=str(
                metadata_cfg[
                    "sha256"
                ]
            ),
            expected_manifest_sha256=str(
                csv_cfg[
                    "sha256"
                ]
            ),
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
                "Canonical held-out population changed."
            )

        if len(
            held_out_rows
        ) != 1385:

            raise RuntimeError(
                "Canonical held-out row materialization failed."
            )

        LOGGER.info(
            "[PASS] canonical test manifest metadata = "
            "1385 = 300 bona-fide + 1085 attack"
        )

        LOGGER.info(
            "[PASS] canonical held-out CSV SHA-256 = %s",
            held_out_contract.manifest_sha256,
        )

        LOGGER.info(
            "[PASS] canonical manifest validation has no dataset_root "
            "and cannot follow test image paths"
        )

        # ==============================================================
        # Synthetic pixel-level Dataset audit
        # ==============================================================

        dataset_audit = synthetic_dataset_audit(
            experiment_cfg=(
                experiment_cfg
            )
        )

        # ==============================================================
        # Synthetic metric audit
        # ==============================================================

        metric_audit = synthetic_metric_audit(
            contract=(
                contract
            )
        )

        # ==============================================================
        # Evidence artifact
        # ==============================================================

        artifact = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_final_evaluation_primitive_audit",

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

            "source_sha256":
                {
                    "src/final_evaluation.py":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "final_evaluation.py"
                        ),

                    "evaluation_protocol":
                        protocol_sha,

                    "canonical_test_manifest_metadata":
                        sha256_file(
                            metadata_path
                        ),

                    "canonical_test_manifest_csv":
                        sha256_file(
                            csv_path
                        ),

                    "audit_script":
                        sha256_file(
                            Path(
                                __file__
                            ).resolve()
                        ),

                    "audit_config":
                        sha256_file(
                            config_path
                        ),
                },

            "canonical_held_out_contract":
                asdict(
                    held_out_contract
                ),

            "dataset_synthetic_audit":
                dataset_audit,

            "metric_synthetic_audit":
                metric_audit,

            "scientific_contract":
                {
                    "bonafide_index":
                        contract.bonafide_index,

                    "attack_index":
                        contract.attack_index,

                    "positive_class":
                        contract.positive_class,

                    "fixed_threshold":
                        contract.fixed_threshold,

                    "threshold_score": 
                        contract.threshold_score,

                    "threshold_score_dtype":
                        contract.threshold_score_dtype,

                    "auroc_score":
                        contract.auroc_score,
                },

            "real_held_out_access":
                {
                    "canonical_manifest_metadata_read":
                        True,

                    "canonical_manifest_csv_read":
                        True,

                    "test_dataset_root_used":
                        False,

                    "test_image_paths_followed":
                        False,

                    "test_image_files_opened":
                        False,

                    "test_image_hashes_recomputed":
                        False,

                    "test_pixels_decoded":
                        False,

                    "test_json_files_opened":
                        False,

                    "model_loaded":
                        False,

                    "checkpoint_loaded":
                        False,

                    "model_inference_performed":
                        False,

                    "test_metrics_computed":
                        False,

                    "threshold_derived_or_modified":
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
            "[PASS] real held-out dataset_root used: FALSE"
        )

        LOGGER.info(
            "[PASS] real test image files opened: FALSE"
        )

        LOGGER.info(
            "[PASS] real test image hashes recomputed: FALSE"
        )

        LOGGER.info(
            "[PASS] real test pixels decoded: FALSE"
        )

        LOGGER.info(
            "[PASS] model/checkpoint loaded: FALSE"
        )

        LOGGER.info(
            "[PASS] test inference performed: FALSE"
        )

        LOGGER.info(
            "[PASS] threshold derivation/modification: FALSE"
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
            "RESNET-18 FINAL EVALUATION PRIMITIVE AUDIT: PASS"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "RESNET-18 FINAL EVALUATION PRIMITIVE AUDIT: FAIL"
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