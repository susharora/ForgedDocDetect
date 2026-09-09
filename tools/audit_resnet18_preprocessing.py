#!/usr/bin/env python3
"""
Exhaustively audit the frozen ResNet-18 Dataset/preprocessing contract.

Scope
-----
project_train:
    1,440 source images

dev_val:
    459 source images

Total unique development images:
    1,899

Each is transformed at:
    r256
    r512

Total preprocessing outputs:
    3,798

The audit verifies:

1. frozen experiment validation passes first;
2. Git was clean before evidence generation;
3. experiment protocol status is frozen;
4. held-out test cannot be constructed through src.data;
5. frozen manifests construct successfully;
6. every source image SHA-256 matches its frozen manifest;
7. no source image/path/hash overlaps across train/dev;
8. every image actually decodes through Dataset.__getitem__;
9. labels and metadata survive Dataset access unchanged;
10. output tensors have the frozen shape/dtype;
11. all values are finite;
12. resize height equals the frozen content height;
13. aspect ratio is preserved within integer-width rounding;
14. no resized document exceeds its canvas;
15. padding geometry reconciles exactly;
16. padding becomes zero after ImageNet normalization;
17. source geometry is identical between r256/r512 processing;
18. every resulting tensor receives a diagnostic SHA-256 fingerprint.

This audit does NOT:
- construct a DataLoader;
- shuffle data;
- train a model;
- access held-out test;
- use augmentation;
- use region annotations;
- perform Grad-CAM.

All detailed evidence is written to ./logs/.
No print() is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from src.data import (
    build_fantasyid_dataset,
)


LOGGER = logging.getLogger(
    "audit_resnet18_preprocessing"
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


CSV_FIELDS = [
    "split",
    "resolution",
    "manifest_index",
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "label",
    "variant",
    "hardware_source",
    "original_width",
    "original_height",
    "resized_width",
    "resized_height",
    "canvas_width",
    "canvas_height",
    "pad_left",
    "pad_right",
    "pad_top",
    "pad_bottom",
    "horizontal_padding",
    "aspect_width_error_pixels",
    "padding_max_abs_normalized",
    "tensor_sha256",
]


# ======================================================================
# Basic utilities
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


def sha256_tensor(
    tensor: torch.Tensor,
) -> str:
    """
    Fingerprint the exact CPU float32 tensor bytes.

    This is diagnostic provenance, not a selection metric.
    """

    if tensor.device.type != "cpu":

        raise RuntimeError(
            "Preprocessing audit expects CPU tensors."
        )

    contiguous = (
        tensor
        .detach()
        .contiguous()
    )

    return hashlib.sha256(
        contiguous
        .numpy()
        .tobytes(
            order="C"
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
) -> dict[str, Any]:

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

    if key not in mapping:

        raise KeyError(
            f"Missing required key: "
            f"{label}.{key}"
        )

    return mapping[
        key
    ]


# ======================================================================
# Git cleanliness
# ======================================================================

def require_clean_git() -> str:
    """
    Must be called before any audit/validator output is created.
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

    status = (
        status_result
        .stdout
        .strip()
    )

    if status:

        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit or remove outstanding files before "
            "running the preprocessing audit.\n\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Frozen protocol gate
# ======================================================================

def run_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> tuple[
    Path,
    str,
]:
    """
    Run the canonical scientific validator before creating this audit log.

    The validator itself writes the detailed evidence and returns only
    its artifact handoff line through stdout.
    """

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

    handoff_lines = [
        line.strip()
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]

    if len(
        handoff_lines
    ) != 1:

        raise RuntimeError(
            "Expected exactly one validator handoff line, got:\n"
            f"{handoff_lines}"
        )

    match = VALIDATION_HANDOFF_PATTERN.match(
        handoff_lines[
            0
        ]
    )

    if match is None:

        raise RuntimeError(
            "Could not parse validator handoff:\n"
            f"{handoff_lines[0]}"
        )

    status = match.group(
        "status"
    )

    if status != "PASS":

        raise RuntimeError(
            "Validator did not return PASS:\n"
            f"{handoff_lines[0]}"
        )

    validation_path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    validation_sha = match.group(
        "sha256"
    )

    if not validation_path.is_file():

        raise FileNotFoundError(
            "Validator reported a missing log artifact:\n"
            f"{validation_path}"
        )

    actual_sha = sha256_file(
        validation_path
    )

    if actual_sha != validation_sha:

        raise RuntimeError(
            "Validator-log SHA mismatch:\n"
            f"  expected={validation_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        validation_path,
        validation_sha,
    )


# ======================================================================
# Logging / output files
# ======================================================================

def configure_outputs(
    *,
    tool_cfg: dict[str, Any],
    machine_id: str,
) -> tuple[
    logging.Logger,
    Path,
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
        require_key(
            tool_cfg,
            "logging",
            "audit_config",
        ),
        "audit_config.logging",
    )

    output_cfg = require_mapping(
        require_key(
            tool_cfg,
            "output",
            "audit_config",
        ),
        "audit_config.output",
    )

    log_dir = resolve_repo_path(
        require_key(
            logging_cfg,
            "directory",
            "audit_config.logging",
        )
    )

    output_dir = resolve_repo_path(
        require_key(
            output_cfg,
            "directory",
            "audit_config.output",
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

    format_values = {
        "timestamp":
            timestamp,

        "machine_id":
            machine_id,
    }

    log_path = (
        log_dir
        / str(
            require_key(
                logging_cfg,
                "filename",
                "audit_config.logging",
            )
        ).format(
            **format_values
        )
    )

    output_path = (
        output_dir
        / str(
            require_key(
                output_cfg,
                "filename",
                "audit_config.output",
            )
        ).format(
            **format_values
        )
    )

    partial_output_path = Path(
        str(
            output_path
        )
        + ".partial"
    )

    level_name = str(
        require_key(
            logging_cfg,
            "level",
            "audit_config.logging",
        )
    ).upper()

    if not hasattr(
        logging,
        level_name,
    ):

        raise ValueError(
            f"Unknown logging level: {level_name!r}"
        )

    LOGGER.handlers.clear()
    LOGGER.propagate = False
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
        LOGGER,
        log_path,
        output_path,
        partial_output_path,
    )


# ======================================================================
# Padding audit
# ======================================================================

def max_abs_padding_value(
    *,
    tensor: torch.Tensor,
    geometry: dict[str, int],
) -> float:

    pieces: list[
        torch.Tensor
    ] = []

    pad_left = int(
        geometry[
            "pad_left"
        ]
    )

    pad_right = int(
        geometry[
            "pad_right"
        ]
    )

    pad_top = int(
        geometry[
            "pad_top"
        ]
    )

    pad_bottom = int(
        geometry[
            "pad_bottom"
        ]
    )

    canvas_height = int(
        geometry[
            "canvas_height"
        ]
    )

    canvas_width = int(
        geometry[
            "canvas_width"
        ]
    )

    if pad_left > 0:

        pieces.append(
            tensor[
                :,
                :,
                :pad_left,
            ]
        )

    if pad_right > 0:

        pieces.append(
            tensor[
                :,
                :,
                (
                    canvas_width
                    - pad_right
                ):
            ]
        )

    if pad_top > 0:

        pieces.append(
            tensor[
                :,
                :pad_top,
                :,
            ]
        )

    if pad_bottom > 0:

        pieces.append(
            tensor[
                :,
                (
                    canvas_height
                    - pad_bottom
                ):,
                :,
            ]
        )

    if not pieces:

        return 0.0

    return max(
        float(
            piece
            .abs()
            .max()
            .item()
        )
        for piece
        in pieces
    )


# ======================================================================
# Test-access structural guard
# ======================================================================

def validate_test_is_unavailable(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
) -> None:
    """
    Confirm the public Dataset constructor rejects test before looking
    for any test manifest.

    No held-out test data are accessed.
    """

    try:

        build_fantasyid_dataset(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=REPO_ROOT,
            split_name="test",
            resolution_name="r256",
        )

    except ValueError:

        return

    raise RuntimeError(
        "src.data unexpectedly allowed construction "
        "of a held-out test Dataset."
    )


# ======================================================================
# Source-image integrity
# ======================================================================

def audit_source_hashes(
    *,
    datasets: dict[
        tuple[
            str,
            str,
        ],
        Any,
    ],
    splits: list[str],
    reference_resolution: str,
    expected_unique_images: int,
    progress_every: int,
) -> None:
    """
    Hash each unique development image exactly once.
    """

    seen_paths: set[
        str
    ] = set()

    seen_hashes: set[
        str
    ] = set()

    total = 0

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "SOURCE IMAGE SHA-256 AUDIT"
    )

    LOGGER.info(
        "=" * 72
    )

    for split_name in splits:

        dataset = datasets[
            (
                split_name,
                reference_resolution,
            )
        ]

        for row in dataset.rows:

            relative_path = str(
                row[
                    "image_path_relative"
                ]
            )

            expected_sha = str(
                row[
                    "image_sha256"
                ]
            )

            absolute_path = Path(
                row[
                    "image_path"
                ]
            )

            if relative_path in seen_paths:

                raise RuntimeError(
                    "Development image path appears more than once "
                    "across project_train/dev_val:\n"
                    f"  {relative_path}"
                )

            if expected_sha in seen_hashes:

                raise RuntimeError(
                    "Development image SHA appears more than once "
                    "across project_train/dev_val:\n"
                    f"  {expected_sha}"
                )

            actual_sha = sha256_file(
                absolute_path
            )

            if actual_sha != expected_sha:

                raise RuntimeError(
                    "Source image SHA-256 mismatch:\n"
                    f"  split={split_name}\n"
                    f"  path={relative_path}\n"
                    f"  expected={expected_sha}\n"
                    f"  actual={actual_sha}"
                )

            seen_paths.add(
                relative_path
            )

            seen_hashes.add(
                expected_sha
            )

            total += 1

            if (
                progress_every > 0
                and
                total % progress_every == 0
            ):

                LOGGER.info(
                    "Source hashes verified: %d / %d",
                    total,
                    expected_unique_images,
                )

    if total != expected_unique_images:

        raise RuntimeError(
            "Unique source-image count mismatch:\n"
            f"  expected={expected_unique_images}\n"
            f"  actual={total}"
        )

    if len(
        seen_paths
    ) != expected_unique_images:

        raise RuntimeError(
            "Unique source-path reconciliation failed."
        )

    if len(
        seen_hashes
    ) != expected_unique_images:

        raise RuntimeError(
            "Unique source-hash reconciliation failed."
        )

    LOGGER.info(
        "[PASS] All %d unique development source images "
        "match their frozen SHA-256 values.",
        expected_unique_images,
    )


# ======================================================================
# Exhaustive preprocessing audit
# ======================================================================

def audit_preprocessing(
    *,
    datasets: dict[
        tuple[
            str,
            str,
        ],
        Any,
    ],
    splits: list[str],
    resolutions: list[str],
    expected_rows: int,
    aspect_error_limit: float,
    padding_zero_atol: float,
    include_tensor_sha256: bool,
    progress_every: int,
    partial_output_path: Path,
) -> dict[
    tuple[
        str,
        str,
    ],
    dict[str, Any],
]:
    """
    Run Dataset.__getitem__ exhaustively for every split/resolution.
    """

    total_processed = 0

    reference_identity: dict[
        tuple[
            str,
            int,
        ],
        tuple[
            str,
            int,
            int,
            int,
        ],
    ] = {}

    summaries: dict[
        tuple[
            str,
            str,
        ],
        dict[str, Any],
    ] = {}

    with partial_output_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        with torch.inference_mode():

            for split_name in splits:

                for resolution_index, resolution_name in enumerate(
                    resolutions
                ):

                    dataset = datasets[
                        (
                            split_name,
                            resolution_name,
                        )
                    ]

                    canvas = (
                        dataset
                        .preprocessor
                        .canvas
                    )

                    label_counts: Counter[int] = Counter()

                    resized_widths: list[int] = []
                    pad_left_values: list[int] = []
                    pad_right_values: list[int] = []

                    max_aspect_error = 0.0
                    max_padding_abs = 0.0

                    LOGGER.info(
                        "=" * 72
                    )

                    LOGGER.info(
                        "PREPROCESSING AUDIT | split=%s | resolution=%s",
                        split_name,
                        resolution_name,
                    )

                    LOGGER.info(
                        "=" * 72
                    )

                    for index in range(
                        len(
                            dataset
                        )
                    ):

                        row = dataset.rows[
                            index
                        ]

                        sample = dataset[
                            index
                        ]

                        tensor = sample[
                            "image"
                        ]

                        geometry = sample[
                            "geometry"
                        ]

                        # --------------------------------------------------
                        # Metadata must survive Dataset access unchanged.
                        # --------------------------------------------------

                        metadata_pairs = {
                            "image_path":
                                "image_path_relative",

                            "image_sha256":
                                "image_sha256",

                            "file_stem":
                                "file_stem",

                            "traffic_type":
                                "traffic_type",

                            "variant":
                                "variant",

                            "hardware_source":
                                "hardware_source",

                            "face_db":
                                "face_db",

                            "face_id":
                                "face_id",

                            "gender":
                                "gender",

                            "project_role":
                                "project_role",
                        }

                        for (
                            sample_key,
                            row_key,
                        ) in metadata_pairs.items():

                            if (
                                sample[
                                    sample_key
                                ]
                                !=
                                row[
                                    row_key
                                ]
                            ):

                                raise RuntimeError(
                                    "Dataset metadata changed during access:\n"
                                    f"  split={split_name}\n"
                                    f"  resolution={resolution_name}\n"
                                    f"  index={index}\n"
                                    f"  field={sample_key}\n"
                                    f"  expected={row[row_key]!r}\n"
                                    f"  actual={sample[sample_key]!r}"
                                )

                        expected_label = int(
                            row[
                                "label"
                            ]
                        )

                        actual_label = int(
                            sample[
                                "label"
                            ]
                        )

                        if actual_label != expected_label:

                            raise RuntimeError(
                                "Dataset label mismatch:\n"
                                f"  split={split_name}\n"
                                f"  index={index}\n"
                                f"  expected={expected_label}\n"
                                f"  actual={actual_label}"
                            )

                        label_counts[
                            actual_label
                        ] += 1

                        # --------------------------------------------------
                        # Tensor contract.
                        # --------------------------------------------------

                        expected_shape = (
                            3,
                            int(
                                canvas.canvas_height
                            ),
                            int(
                                canvas.canvas_width
                            ),
                        )

                        if tuple(
                            tensor.shape
                        ) != expected_shape:

                            raise RuntimeError(
                                "Tensor shape mismatch:\n"
                                f"  split={split_name}\n"
                                f"  resolution={resolution_name}\n"
                                f"  index={index}\n"
                                f"  expected={expected_shape}\n"
                                f"  actual={tuple(tensor.shape)}"
                            )

                        if (
                            tensor.dtype
                            != torch.float32
                        ):

                            raise RuntimeError(
                                "Tensor dtype mismatch:\n"
                                f"  expected=torch.float32\n"
                                f"  actual={tensor.dtype}"
                            )

                        if (
                            tensor.device.type
                            != "cpu"
                        ):

                            raise RuntimeError(
                                "Dataset preprocessing unexpectedly "
                                "returned a non-CPU tensor."
                            )

                        if not bool(
                            torch.isfinite(
                                tensor
                            ).all()
                        ):

                            raise RuntimeError(
                                "Tensor contains NaN or Inf:\n"
                                f"  split={split_name}\n"
                                f"  resolution={resolution_name}\n"
                                f"  index={index}"
                            )

                        # --------------------------------------------------
                        # Geometry contract.
                        # --------------------------------------------------

                        original_width = int(
                            geometry[
                                "original_width"
                            ]
                        )

                        original_height = int(
                            geometry[
                                "original_height"
                            ]
                        )

                        resized_width = int(
                            geometry[
                                "resized_width"
                            ]
                        )

                        resized_height = int(
                            geometry[
                                "resized_height"
                            ]
                        )

                        canvas_width = int(
                            geometry[
                                "canvas_width"
                            ]
                        )

                        canvas_height = int(
                            geometry[
                                "canvas_height"
                            ]
                        )

                        pad_left = int(
                            geometry[
                                "pad_left"
                            ]
                        )

                        pad_right = int(
                            geometry[
                                "pad_right"
                            ]
                        )

                        pad_top = int(
                            geometry[
                                "pad_top"
                            ]
                        )

                        pad_bottom = int(
                            geometry[
                                "pad_bottom"
                            ]
                        )

                        if (
                            original_width
                            <= original_height
                        ):

                            raise RuntimeError(
                                "Non-landscape source reached frozen "
                                "preprocessing:\n"
                                f"  {original_width}x{original_height}"
                            )

                        if (
                            resized_height
                            != int(
                                canvas.content_height
                            )
                        ):

                            raise RuntimeError(
                                "Resized height does not equal "
                                "frozen content height."
                            )

                        if (
                            canvas_height
                            != int(
                                canvas.canvas_height
                            )
                            or
                            canvas_width
                            != int(
                                canvas.canvas_width
                            )
                        ):

                            raise RuntimeError(
                                "Geometry canvas does not match "
                                "frozen CanvasSpec."
                            )

                        if (
                            pad_top != 0
                            or
                            pad_bottom != 0
                        ):

                            raise RuntimeError(
                                "Frozen preprocessing permits "
                                "horizontal padding only."
                            )

                        if (
                            resized_width
                            > canvas_width
                        ):

                            raise RuntimeError(
                                "Resized content exceeds canvas width."
                            )

                        horizontal_padding = (
                            canvas_width
                            - resized_width
                        )

                        expected_pad_left = (
                            horizontal_padding
                            // 2
                        )

                        expected_pad_right = (
                            horizontal_padding
                            - expected_pad_left
                        )

                        if (
                            pad_left
                            != expected_pad_left
                            or
                            pad_right
                            != expected_pad_right
                        ):

                            raise RuntimeError(
                                "Horizontal centering mismatch:\n"
                                f"  total_padding={horizontal_padding}\n"
                                f"  expected_left={expected_pad_left}\n"
                                f"  actual_left={pad_left}\n"
                                f"  expected_right={expected_pad_right}\n"
                                f"  actual_right={pad_right}"
                            )

                        if (
                            pad_left
                            + resized_width
                            + pad_right
                            != canvas_width
                        ):

                            raise RuntimeError(
                                "Horizontal geometry does not reconcile "
                                "to canvas width."
                            )

                        # --------------------------------------------------
                        # Aspect-preservation check.
                        #
                        # Integer output width may differ from the
                        # mathematically ideal width by < 1 pixel.
                        # --------------------------------------------------

                        ideal_width = (
                            original_width
                            * resized_height
                            / original_height
                        )

                        aspect_error = abs(
                            resized_width
                            - ideal_width
                        )

                        if not (
                            aspect_error
                            < aspect_error_limit
                        ):

                            raise RuntimeError(
                                "Aspect-preserving resize exceeds "
                                "integer-rounding tolerance:\n"
                                f"  split={split_name}\n"
                                f"  resolution={resolution_name}\n"
                                f"  path={sample['image_path']}\n"
                                f"  ideal_width={ideal_width:.12f}\n"
                                f"  resized_width={resized_width}\n"
                                f"  error={aspect_error:.12f}\n"
                                f"  limit={aspect_error_limit}"
                            )

                        # --------------------------------------------------
                        # Normalized padding must be zero.
                        # --------------------------------------------------

                        padding_max_abs = (
                            max_abs_padding_value(
                                tensor=tensor,
                                geometry=geometry,
                            )
                        )

                        if (
                            padding_max_abs
                            > padding_zero_atol
                        ):

                            raise RuntimeError(
                                "Normalized padding is not zero:\n"
                                f"  split={split_name}\n"
                                f"  resolution={resolution_name}\n"
                                f"  path={sample['image_path']}\n"
                                f"  max_abs={padding_max_abs:.12g}\n"
                                f"  atol={padding_zero_atol:.12g}"
                            )

                        # --------------------------------------------------
                        # Cross-resolution source identity.
                        # --------------------------------------------------

                        identity_key = (
                            split_name,
                            index,
                        )

                        identity_value = (
                            str(
                                sample[
                                    "image_path"
                                ]
                            ),
                            actual_label,
                            original_width,
                            original_height,
                        )

                        if resolution_index == 0:

                            reference_identity[
                                identity_key
                            ] = identity_value

                        else:

                            expected_identity = (
                                reference_identity[
                                    identity_key
                                ]
                            )

                            if (
                                identity_value
                                != expected_identity
                            ):

                                raise RuntimeError(
                                    "Source identity/geometry changed "
                                    "between resolutions:\n"
                                    f"  split={split_name}\n"
                                    f"  index={index}\n"
                                    f"  expected={expected_identity}\n"
                                    f"  actual={identity_value}"
                                )

                        tensor_hash = (
                            sha256_tensor(
                                tensor
                            )
                            if include_tensor_sha256
                            else ""
                        )

                        writer.writerow(
                            {
                                "split":
                                    split_name,

                                "resolution":
                                    resolution_name,

                                "manifest_index":
                                    index,

                                "image_path":
                                    sample[
                                        "image_path"
                                    ],

                                "image_sha256":
                                    sample[
                                        "image_sha256"
                                    ],

                                "file_stem":
                                    sample[
                                        "file_stem"
                                    ],

                                "traffic_type":
                                    sample[
                                        "traffic_type"
                                    ],

                                "label":
                                    actual_label,

                                "variant":
                                    sample[
                                        "variant"
                                    ],

                                "hardware_source":
                                    sample[
                                        "hardware_source"
                                    ],

                                "original_width":
                                    original_width,

                                "original_height":
                                    original_height,

                                "resized_width":
                                    resized_width,

                                "resized_height":
                                    resized_height,

                                "canvas_width":
                                    canvas_width,

                                "canvas_height":
                                    canvas_height,

                                "pad_left":
                                    pad_left,

                                "pad_right":
                                    pad_right,

                                "pad_top":
                                    pad_top,

                                "pad_bottom":
                                    pad_bottom,

                                "horizontal_padding":
                                    horizontal_padding,

                                "aspect_width_error_pixels":
                                    (
                                        f"{aspect_error:.12f}"
                                    ),

                                "padding_max_abs_normalized":
                                    (
                                        f"{padding_max_abs:.12g}"
                                    ),

                                "tensor_sha256":
                                    tensor_hash,
                            }
                        )

                        resized_widths.append(
                            resized_width
                        )

                        pad_left_values.append(
                            pad_left
                        )

                        pad_right_values.append(
                            pad_right
                        )

                        max_aspect_error = max(
                            max_aspect_error,
                            aspect_error,
                        )

                        max_padding_abs = max(
                            max_padding_abs,
                            padding_max_abs,
                        )

                        total_processed += 1

                        if (
                            progress_every > 0
                            and
                            total_processed % progress_every == 0
                        ):

                            LOGGER.info(
                                "Preprocessed tensors audited: %d / %d",
                                total_processed,
                                expected_rows,
                            )

                    expected_label_counts = {
                        0:
                            (
                                480
                                if split_name == "project_train"
                                else 153
                            ),

                        1:
                            (
                                960
                                if split_name == "project_train"
                                else 306
                            ),
                    }

                    if dict(
                        sorted(
                            label_counts.items()
                        )
                    ) != expected_label_counts:

                        raise RuntimeError(
                            "Observed labels do not match frozen "
                            "class counts:\n"
                            f"  split={split_name}\n"
                            f"  resolution={resolution_name}\n"
                            f"  expected={expected_label_counts}\n"
                            f"  actual={dict(label_counts)}"
                        )

                    summaries[
                        (
                            split_name,
                            resolution_name,
                        )
                    ] = {
                        "rows":
                            len(
                                dataset
                            ),

                        "resized_width_min":
                            min(
                                resized_widths
                            ),

                        "resized_width_max":
                            max(
                                resized_widths
                            ),

                        "pad_left_min":
                            min(
                                pad_left_values
                            ),

                        "pad_left_max":
                            max(
                                pad_left_values
                            ),

                        "pad_right_min":
                            min(
                                pad_right_values
                            ),

                        "pad_right_max":
                            max(
                                pad_right_values
                            ),

                        "max_aspect_error":
                            max_aspect_error,

                        "max_padding_abs":
                            max_padding_abs,

                        "label_counts":
                            dict(
                                sorted(
                                    label_counts.items()
                                )
                            ),
                    }

    if total_processed != expected_rows:

        raise RuntimeError(
            "Preprocessed-output row count mismatch:\n"
            f"  expected={expected_rows}\n"
            f"  actual={total_processed}"
        )

    return summaries


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively audit frozen ResNet-18 "
            "development preprocessing."
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
            "audit_resnet18_preprocessing_config.yaml"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configuration before creating evidence.
    # ------------------------------------------------------------------

    audit_config_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        audit_config_path
    )

    if (
        tool_cfg.get(
            "schema_version"
        )
        != 1
    ):

        raise ValueError(
            "Preprocessing audit config schema_version "
            "must currently equal 1."
        )

    # ------------------------------------------------------------------
    # Scientific audit requires a clean committed implementation.
    # ------------------------------------------------------------------

    commit_sha = require_clean_git()

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

    # ------------------------------------------------------------------
    # Validate exact frozen protocol BEFORE creating this audit log.
    # ------------------------------------------------------------------

    (
        validation_log_path,
        validation_log_sha,
    ) = run_validator(
        experiment_path=experiment_path,
        machine_path=machine_path,
    )

    machine = require_mapping(
        require_key(
            machine_cfg,
            "machine",
            "machine_config",
        ),
        "machine_config.machine",
    )

    machine_id = str(
        require_key(
            machine,
            "id",
            "machine_config.machine",
        )
    )

    (
        logger,
        log_path,
        output_path,
        partial_output_path,
    ) = configure_outputs(
        tool_cfg=tool_cfg,
        machine_id=machine_id,
    )

    try:

        audit_cfg = require_mapping(
            require_key(
                tool_cfg,
                "audit",
                "audit_config",
            ),
            "audit_config.audit",
        )

        splits = [
            str(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "splits",
                "audit_config.audit",
            )
        ]

        resolutions = [
            str(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "resolutions",
                "audit_config.audit",
            )
        ]

        if splits != [
            "project_train",
            "dev_val",
        ]:

            raise ValueError(
                "Audit split order must be exactly "
                "['project_train', 'dev_val']."
            )

        if resolutions != [
            "r256",
            "r512",
        ]:

            raise ValueError(
                "Audit resolution order must be exactly "
                "['r256', 'r512']."
            )

        expected_unique_images = int(
            require_key(
                audit_cfg,
                "expected_unique_source_images",
                "audit_config.audit",
            )
        )

        expected_preprocessed_rows = int(
            require_key(
                audit_cfg,
                "expected_preprocessed_rows",
                "audit_config.audit",
            )
        )

        if expected_unique_images != 1899:

            raise ValueError(
                "Expected unique development image count "
                "must equal the frozen value 1899."
            )

        if expected_preprocessed_rows != 3798:

            raise ValueError(
                "Expected preprocessing row count "
                "must equal 1899 x 2 = 3798."
            )

        if (
            require_key(
                audit_cfg,
                "source_hashing",
                "audit_config.audit",
            )
            != "exhaustive_once_per_image"
        ):

            raise ValueError(
                "Source hashing must be exhaustive_once_per_image."
            )

        padding_zero_atol = float(
            require_key(
                audit_cfg,
                "padding_zero_atol",
                "audit_config.audit",
            )
        )

        aspect_error_limit = float(
            require_key(
                audit_cfg,
                "aspect_ratio_width_error_lt_pixels",
                "audit_config.audit",
            )
        )

        include_tensor_sha256 = bool(
            require_key(
                audit_cfg,
                "tensor_sha256",
                "audit_config.audit",
            )
        )

        progress_every = int(
            require_key(
                audit_cfg,
                "progress_every",
                "audit_config.audit",
            )
        )

        if padding_zero_atol < 0.0:

            raise ValueError(
                "padding_zero_atol cannot be negative."
            )

        if (
            not math.isclose(
                aspect_error_limit,
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):

            raise ValueError(
                "Aspect-width error limit must be exactly 1.0 pixel."
            )

        if not include_tensor_sha256:

            raise ValueError(
                "This audit requires tensor_sha256=true."
            )

        # ------------------------------------------------------------------
        # Frozen status is now mandatory.
        # ------------------------------------------------------------------

        experiment = require_mapping(
            require_key(
                experiment_cfg,
                "experiment",
                "experiment_config",
            ),
            "experiment",
        )

        protocol_status = require_key(
            experiment,
            "protocol_status",
            "experiment",
        )

        if protocol_status != "frozen":

            raise RuntimeError(
                "Preprocessing audit requires a frozen protocol, got "
                f"{protocol_status!r}."
            )

        # ------------------------------------------------------------------
        # Audit provenance.
        # ------------------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 DEVELOPMENT PREPROCESSING AUDIT"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Git commit: %s",
            commit_sha,
        )

        logger.info(
            "Git working tree clean before evidence generation: True"
        )

        logger.info(
            "Audit script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        logger.info(
            "src/data.py SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "data.py"
            ),
        )

        logger.info(
            "Audit config SHA-256: %s",
            sha256_file(
                audit_config_path
            ),
        )

        logger.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        logger.info(
            "Experiment protocol status: %s",
            protocol_status,
        )

        logger.info(
            "Machine ID: %s",
            machine_id,
        )

        logger.info(
            "Canonical validator log: %s",
            validation_log_path,
        )

        logger.info(
            "Canonical validator log SHA-256: %s",
            validation_log_sha,
        )

        logger.info(
            "project_train + dev_val only"
        )

        logger.info(
            "held-out test: NOT ACCESSED"
        )

        logger.info(
            "Resolutions: %s",
            resolutions,
        )

        logger.info(
            "Expected unique source images: %d",
            expected_unique_images,
        )

        logger.info(
            "Expected preprocessing outputs: %d",
            expected_preprocessed_rows,
        )

        logger.info(
            "Tensor SHA-256 diagnostics: ENABLED"
        )

        # ------------------------------------------------------------------
        # Structural test-protection check.
        # ------------------------------------------------------------------

        validate_test_is_unavailable(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
        )

        logger.info(
            "[PASS] src.data structurally rejects held-out test access"
        )

        # ------------------------------------------------------------------
        # Construct all four development Datasets.
        # ------------------------------------------------------------------

        datasets: dict[
            tuple[
                str,
                str,
            ],
            Any,
        ] = {}

        for split_name in splits:

            for resolution_name in resolutions:

                dataset = build_fantasyid_dataset(
                    experiment_cfg=experiment_cfg,
                    machine_cfg=machine_cfg,
                    repo_root=REPO_ROOT,
                    split_name=split_name,
                    resolution_name=resolution_name,
                )

                datasets[
                    (
                        split_name,
                        resolution_name,
                    )
                ] = dataset

                logger.info(
                    "[PASS] Dataset construction | "
                    "split=%s | resolution=%s | rows=%d | "
                    "manifest_sha256=%s",
                    split_name,
                    resolution_name,
                    len(
                        dataset
                    ),
                    dataset.manifest_sha256,
                )

        # ------------------------------------------------------------------
        # Exhaustive source integrity.
        # ------------------------------------------------------------------

        audit_source_hashes(
            datasets=datasets,
            splits=splits,
            reference_resolution=resolutions[
                0
            ],
            expected_unique_images=expected_unique_images,
            progress_every=progress_every,
        )

        # ------------------------------------------------------------------
        # Exhaustive preprocessing.
        # ------------------------------------------------------------------

        summaries = audit_preprocessing(
            datasets=datasets,
            splits=splits,
            resolutions=resolutions,
            expected_rows=expected_preprocessed_rows,
            aspect_error_limit=aspect_error_limit,
            padding_zero_atol=padding_zero_atol,
            include_tensor_sha256=include_tensor_sha256,
            progress_every=progress_every,
            partial_output_path=partial_output_path,
        )

        # Only successful complete evidence receives the canonical name.
        partial_output_path.replace(
            output_path
        )

        output_sha = sha256_file(
            output_path
        )

        # ------------------------------------------------------------------
        # Summary.
        # ------------------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "PREPROCESSING SUMMARY"
        )

        logger.info(
            "=" * 72
        )

        for (
            split_name,
            resolution_name,
        ) in [
            (
                split_name,
                resolution_name,
            )
            for split_name
            in splits
            for resolution_name
            in resolutions
        ]:

            summary = summaries[
                (
                    split_name,
                    resolution_name,
                )
            ]

            logger.info(
                "%s / %s | rows=%d | "
                "resized_width=%d..%d | "
                "pad_left=%d..%d | "
                "pad_right=%d..%d | "
                "max_aspect_error=%.12f px | "
                "max_padding_abs=%.12g | "
                "labels=%s",
                split_name,
                resolution_name,
                summary[
                    "rows"
                ],
                summary[
                    "resized_width_min"
                ],
                summary[
                    "resized_width_max"
                ],
                summary[
                    "pad_left_min"
                ],
                summary[
                    "pad_left_max"
                ],
                summary[
                    "pad_right_min"
                ],
                summary[
                    "pad_right_max"
                ],
                summary[
                    "max_aspect_error"
                ],
                summary[
                    "max_padding_abs"
                ],
                summary[
                    "label_counts"
                ],
            )

        logger.info(
            "-" * 72
        )

        logger.info(
            "Audit CSV: %s",
            output_path,
        )

        logger.info(
            "Audit CSV SHA-256: %s",
            output_sha,
        )

        logger.info(
            "[PASS] Unique development source images audited: %d",
            expected_unique_images,
        )

        logger.info(
            "[PASS] Development preprocessing tensors audited: %d",
            expected_preprocessed_rows,
        )

        logger.info(
            "[PASS] All tensor shapes/dtypes/finite-value checks"
        )

        logger.info(
            "[PASS] All source-to-canvas geometry checks"
        )

        logger.info(
            "[PASS] All aspect-preservation checks"
        )

        logger.info(
            "[PASS] All normalized padding checks"
        )

        logger.info(
            "[PASS] Cross-resolution source identity checks"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 DEVELOPMENT PREPROCESSING AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 DEVELOPMENT PREPROCESSING AUDIT: FAIL"
        )

        if partial_output_path.exists():

            partial_output_path.unlink()

        return 1

    finally:

        for handler in list(
            logger.handlers
        ):

            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )