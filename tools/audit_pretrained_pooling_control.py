#!/usr/bin/env python3
"""
Pooling-control audit for ImageNet-pretrained ResNet-18 on FantasyID.

Purpose
-------
The preceding project_train cohort established that:

    layer4.1 attack/bona-fide representations
        become consistently more similar
    after
        global average pooling.

This follow-up asks whether that phenomenon is specific to the
attack-vs-bona-fide class boundary or is a more general consequence
of spatial averaging.

Two controls are used.

1. Empirical same-class control
-------------------------------

For each card + hardware capture:

    digital_1 vs bona fide
    digital_2 vs bona fide
    digital_1 vs digital_2

The third comparison is attack vs attack.

If layer4 -> avgpool reconvergence also occurs there, the phenomenon
cannot be described as specific to bona-fide/manipulated separation.

2. Mechanistic spatial control
------------------------------

For each of the 1,440 project_train images:

    layer4.1 tensor
        vs
    the same tensor circularly rolled by one spatial cell.

Rolling changes WHERE activations occur but does not alter their
values.

Global average pooling is permutation-invariant over H x W, so:

    GAP(A) == GAP(roll(A))

within floating-point tolerance.

This directly demonstrates that global average pooling removes spatial
arrangement information.

Important
---------
This experiment does NOT establish whether the information removed by
pooling is forensic information.

No FantasyID fine-tuning occurs.

Only frozen project_train is accessed.
dev_val is NOT used.
held-out test is NOT accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


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


PAIR_TYPES = (
    "digital_1_vs_bonafide",
    "digital_2_vs_bonafide",
    "digital_1_vs_digital_2",
)

IMAGE_ROLES = (
    "bonafide",
    "digital_1",
    "digital_2",
)

PAIR_METRICS = (
    "layer4_1_cosine",
    "avgpool_cosine",
    "cosine_reconvergence",
    "layer4_1_relative_l2",
    "avgpool_relative_l2",
    "relative_l2_reconvergence",
)


# ======================================================================
# Basic utilities
# ======================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

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
            yaml.safe_load(f)
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
    Run before creating the audit log so the tool does not dirty its
    own working tree before checking provenance.
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
            "Commit or stash changes before running this audit.\n\n"
            f"{status}"
        )

    return commit


def write_dataframe_exclusive(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as f:

        dataframe.to_csv(
            f,
            index=False,
        )


# ======================================================================
# Logging / output
# ======================================================================

def configure_outputs(
    tool_cfg: dict[str, Any],
) -> tuple[
    logging.Logger,
    Path,
    dict[str, Path],
    str,
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

    output_paths: dict[
        str,
        Path,
    ] = {}

    for (
        key,
        filename,
    ) in output_cfg.items():

        if key == "directory":
            continue

        output_paths[
            key
        ] = (
            output_dir
            / str(
                filename
            ).format(
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
        "audit_pretrained_pooling_control"
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
        output_paths,
        timestamp,
    )


# ======================================================================
# Frozen experiment provenance
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
# Frozen project_train
# ======================================================================

def load_project_train_manifest(
    experiment_cfg: dict[str, Any],
) -> tuple[
    list[dict[str, str]],
    Path,
]:

    split_cfg = (
        experiment_cfg[
            "data"
        ][
            "frozen_split"
        ]
    )

    manifest_path = resolve_repo_path(
        split_cfg[
            "project_train"
        ][
            "path"
        ]
    )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(
                f
            )
        )

    expected = int(
        split_cfg[
            "project_train"
        ][
            "images"
        ]
    )

    if len(
        rows
    ) != expected:

        raise ValueError(
            "project_train row count mismatch: "
            f"expected={expected}, "
            f"actual={len(rows)}"
        )

    return (
        rows,
        manifest_path,
    )


def build_capture_groups(
    *,
    manifest_rows: list[
        dict[str, str]
    ],
    expected_variants: set[str],
    expected_hardware: set[str],
) -> dict[
    tuple[str, str],
    dict[
        str,
        dict[str, str],
    ],
]:
    """
    Return:

        (file_stem, hardware)
            -> {
                bonafide: row,
                digital_1: row,
                digital_2: row,
            }
    """

    groups: dict[
        tuple[str, str],
        dict[
            str,
            dict[str, str],
        ],
    ] = defaultdict(
        dict
    )

    for row in (
        manifest_rows
    ):

        hardware = row[
            "hardware_source"
        ]

        if hardware not in expected_hardware:

            raise ValueError(
                "Unexpected hardware source: "
                f"{hardware}"
            )

        key = (
            row[
                "file_stem"
            ],
            hardware,
        )

        traffic = row[
            "traffic_type"
        ]

        if traffic == "bonafide":

            role = "bonafide"

        elif traffic == "attack":

            role = row[
                "variant"
            ]

            if role not in expected_variants:

                raise ValueError(
                    "Unexpected attack variant: "
                    f"{role}"
                )

        else:

            raise ValueError(
                "Unexpected traffic type: "
                f"{traffic}"
            )

        if role in groups[
            key
        ]:

            raise ValueError(
                "Duplicate capture role:\n"
                f"  key={key}\n"
                f"  role={role}"
            )

        groups[
            key
        ][
            role
        ] = row

    expected_roles = (
        {"bonafide"}
        | expected_variants
    )

    for (
        key,
        role_rows,
    ) in groups.items():

        actual_roles = set(
            role_rows
        )

        if actual_roles != expected_roles:

            raise ValueError(
                "Capture group composition mismatch:\n"
                f"  key={key}\n"
                f"  expected={sorted(expected_roles)}\n"
                f"  actual={sorted(actual_roles)}"
            )

    return dict(
        groups
    )


def verify_image(
    *,
    row: dict[str, str],
    dataset_root: Path,
) -> Path:

    path = (
        dataset_root
        / row[
            "image_path"
        ]
    ).resolve()

    try:

        path.relative_to(
            dataset_root
        )

    except ValueError as exc:

        raise ValueError(
            "Manifest image escapes dataset root."
        ) from exc

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    actual_sha = sha256_file(
        path
    )

    expected_sha = row[
        "image_sha256"
    ]

    if actual_sha != expected_sha:

        raise ValueError(
            "Image SHA-256 mismatch:\n"
            f"  path={row['image_path']}\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return path


# ======================================================================
# Image preprocessing
# ======================================================================

def prepare_image(
    *,
    path: Path,
    short_side: int,
    mean: list[float],
    std: list[float],
) -> tuple[
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
]:

    with Image.open(
        path
    ) as opened:

        image = opened.convert(
            "RGB"
        )

    original_size = (
        image.size
    )

    resized = TF.resize(
        image,
        size=short_side,
        interpolation=(
            InterpolationMode
            .BILINEAR
        ),
        antialias=True,
    )

    input_size = (
        resized.size
    )

    tensor = (
        TF.pil_to_tensor(
            resized
        )
        .float()
        .div(
            255.0
        )
    )

    tensor = TF.normalize(
        tensor,
        mean=mean,
        std=std,
    )

    return (
        tensor.unsqueeze(
            0
        ),
        original_size,
        input_size,
    )


# ======================================================================
# Capture layer4.1 and avgpool only
# ======================================================================

class RepresentationCapture:

    def __init__(
        self,
        model: nn.Module,
    ) -> None:

        self.layer4_1: (
            torch.Tensor
            | None
        ) = None

        self.avgpool: (
            torch.Tensor
            | None
        ) = None

        self.handles = [
            model.layer4[
                1
            ].register_forward_hook(
                self._layer4_hook
            ),

            model.avgpool.register_forward_hook(
                self._avgpool_hook
            ),
        ]

    def _layer4_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:

        self.layer4_1 = (
            output.detach()
        )

    def _avgpool_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:

        self.avgpool = (
            output.detach()
        )

    def run(
        self,
        *,
        model: nn.Module,
        tensor: torch.Tensor,
        device: torch.device,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        self.layer4_1 = None
        self.avgpool = None

        with torch.inference_mode():

            model(
                tensor.to(
                    device,
                    non_blocking=True,
                )
            )

        if self.layer4_1 is None:

            raise RuntimeError(
                "layer4.1 activation was not captured."
            )

        if self.avgpool is None:

            raise RuntimeError(
                "avgpool activation was not captured."
            )

        return {
            "layer4.1":
                self.layer4_1,

            "avgpool":
                self.avgpool,
        }

    def close(
        self,
    ) -> None:

        for handle in (
            self.handles
        ):

            handle.remove()

        self.handles.clear()


# ======================================================================
# Representation metrics
# ======================================================================

def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    a_flat = (
        a.reshape(-1)
        .float()
    )

    b_flat = (
        b.reshape(-1)
        .float()
    )

    return float(
        F.cosine_similarity(
            a_flat.unsqueeze(0),
            b_flat.unsqueeze(0),
        )[0].item()
    )


def relative_l2_distance(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    a_flat = (
        a.reshape(-1)
        .float()
    )

    b_flat = (
        b.reshape(-1)
        .float()
    )

    difference_norm = (
        torch.linalg.vector_norm(
            a_flat
            - b_flat
        )
    )

    mean_norm = (
        torch.linalg.vector_norm(
            a_flat
        )
        +
        torch.linalg.vector_norm(
            b_flat
        )
    ) / 2.0

    if (
        mean_norm.item()
        == 0.0
    ):

        return 0.0

    return float(
        (
            difference_norm
            / mean_norm
        ).item()
    )


def pair_metrics(
    a: dict[
        str,
        torch.Tensor,
    ],
    b: dict[
        str,
        torch.Tensor,
    ],
) -> dict[str, float]:

    layer4_cosine = cosine_similarity(
        a[
            "layer4.1"
        ],
        b[
            "layer4.1"
        ],
    )

    avgpool_cosine = cosine_similarity(
        a[
            "avgpool"
        ],
        b[
            "avgpool"
        ],
    )

    layer4_l2 = relative_l2_distance(
        a[
            "layer4.1"
        ],
        b[
            "layer4.1"
        ],
    )

    avgpool_l2 = relative_l2_distance(
        a[
            "avgpool"
        ],
        b[
            "avgpool"
        ],
    )

    return {
        "layer4_1_cosine":
            layer4_cosine,

        "avgpool_cosine":
            avgpool_cosine,

        "cosine_reconvergence":
            (
                avgpool_cosine
                -
                layer4_cosine
            ),

        "layer4_1_relative_l2":
            layer4_l2,

        "avgpool_relative_l2":
            avgpool_l2,

        "relative_l2_reconvergence":
            (
                layer4_l2
                -
                avgpool_l2
            ),
    }


# ======================================================================
# Mechanistic spatial-roll control
# ======================================================================

def spatial_roll_metrics(
    *,
    representation: dict[
        str,
        torch.Tensor,
    ],
    shift_y: int,
    shift_x: int,
    pooled_atol: float,
) -> dict[str, float]:

    layer4 = representation[
        "layer4.1"
    ]

    model_avgpool = representation[
        "avgpool"
    ]

    if layer4.ndim != 4:

        raise ValueError(
            "layer4.1 must be N x C x H x W."
        )

    rolled = torch.roll(
        layer4,
        shifts=(
            shift_y,
            shift_x,
        ),
        dims=(
            -2,
            -1,
        ),
    )

    original_pooled = (
        F.adaptive_avg_pool2d(
            layer4,
            output_size=(
                1,
                1,
            ),
        )
    )

    rolled_pooled = (
        F.adaptive_avg_pool2d(
            rolled,
            output_size=(
                1,
                1,
            ),
        )
    )

    # Confirm our independently computed pooling agrees with the
    # model's actual avgpool output.
    model_pool_error = float(
        (
            original_pooled
            - model_avgpool
        )
        .abs()
        .max()
        .item()
    )

    pooled_roll_error = float(
        (
            original_pooled
            - rolled_pooled
        )
        .abs()
        .max()
        .item()
    )

    if model_pool_error > pooled_atol:

        raise RuntimeError(
            "Independent adaptive pooling disagrees with "
            "model.avgpool:\n"
            f"  max_abs_error={model_pool_error}\n"
            f"  tolerance={pooled_atol}"
        )

    if pooled_roll_error > pooled_atol:

        raise RuntimeError(
            "Global average pooling was not invariant to the "
            "configured spatial roll:\n"
            f"  max_abs_error={pooled_roll_error}\n"
            f"  tolerance={pooled_atol}"
        )

    return {
        "layer4_original_vs_roll_cosine":
            cosine_similarity(
                layer4,
                rolled,
            ),

        "layer4_original_vs_roll_relative_l2":
            relative_l2_distance(
                layer4,
                rolled,
            ),

        "avgpool_original_vs_roll_cosine":
            cosine_similarity(
                original_pooled,
                rolled_pooled,
            ),

        "avgpool_original_vs_roll_relative_l2":
            relative_l2_distance(
                original_pooled,
                rolled_pooled,
            ),

        "model_avgpool_max_abs_error":
            model_pool_error,

        "roll_avgpool_max_abs_error":
            pooled_roll_error,
    }


# ======================================================================
# Model-weight provenance
# ======================================================================

def checkpoint_provenance(
    weights: ResNet18_Weights,
) -> dict[
    str,
    str | None,
]:

    filename = Path(
        urlparse(
            weights.url
        ).path
    ).name

    checkpoint_path = (
        Path(
            torch.hub.get_dir()
        )
        / "checkpoints"
        / filename
    )

    if checkpoint_path.is_file():

        return {
            "url":
                weights.url,

            "cached_path":
                str(
                    checkpoint_path
                ),

            "sha256":
                sha256_file(
                    checkpoint_path
                ),
        }

    return {
        "url":
            weights.url,

        "cached_path":
            None,

        "sha256":
            None,
    }


# ======================================================================
# Aggregation
# ======================================================================

def numeric_summary(
    values: pd.Series,
) -> dict[
    str,
    float | int,
]:

    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    n = len(
        clean
    )

    if n == 0:

        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }

    return {
        "n":
            n,

        "mean":
            float(
                clean.mean()
            ),

        "std":
            (
                float(
                    clean.std(
                        ddof=1
                    )
                )
                if n > 1
                else 0.0
            ),

        "min":
            float(
                clean.min()
            ),

        "q25":
            float(
                clean.quantile(
                    0.25
                )
            ),

        "median":
            float(
                clean.median()
            ),

        "q75":
            float(
                clean.quantile(
                    0.75
                )
            ),

        "max":
            float(
                clean.max()
            ),
    }


def build_card_summary(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average the three hardware captures within each card and pair type.

    160 cards × 3 comparison types = 480 rows.
    """

    aggregations: dict[
        str,
        tuple[str, str],
    ] = {
        "capture_count":
            (
                "hardware_source",
                "size",
            ),

        "hardware_count":
            (
                "hardware_source",
                "nunique",
            ),
    }

    for metric in (
        PAIR_METRICS
    ):

        aggregations[
            metric
        ] = (
            metric,
            "mean",
        )

    result = (
        pair_df
        .groupby(
            [
                "file_stem",
                "pair_type",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            **aggregations
        )
    )

    bad = result[
        (
            result[
                "capture_count"
            ]
            != 3
        )
        |
        (
            result[
                "hardware_count"
            ]
            != 3
        )
    ]

    if not bad.empty:

        raise ValueError(
            "Card-level pooling-control aggregation does not "
            "contain exactly three hardware captures.\n\n"
            f"{bad.head(10).to_string(index=False)}"
        )

    return result


def append_group_summary(
    output_rows: list[
        dict[str, Any]
    ],
    *,
    dataframe: pd.DataFrame,
    analysis_unit: str,
    group_type: str,
    group_column: str | None,
    metrics: tuple[str, ...],
) -> None:

    if group_column is None:

        groups = [
            (
                "ALL",
                dataframe,
            )
        ]

    else:

        groups = list(
            dataframe.groupby(
                group_column,
                sort=True,
                dropna=False,
            )
        )

    for (
        group_value,
        group_df,
    ) in groups:

        for metric in (
            metrics
        ):

            summary = numeric_summary(
                group_df[
                    metric
                ]
            )

            output_rows.append(
                {
                    "analysis_unit":
                        analysis_unit,

                    "group_type":
                        group_type,

                    "group_value":
                        str(
                            group_value
                        ),

                    "metric":
                        metric,

                    **summary,
                }
            )


def log_pair_type_summary(
    *,
    logger: logging.Logger,
    dataframe: pd.DataFrame,
    label: str,
) -> None:

    n = len(
        dataframe
    )

    positive_cosine = int(
        (
            dataframe[
                "cosine_reconvergence"
            ]
            > 0
        ).sum()
    )

    positive_l2 = int(
        (
            dataframe[
                "relative_l2_reconvergence"
            ]
            > 0
        ).sum()
    )

    logger.info(
        "%s | n=%d",
        label,
        n,
    )

    logger.info(
        "  median layer4.1 cosine: %.6f",
        dataframe[
            "layer4_1_cosine"
        ].median(),
    )

    logger.info(
        "  median avgpool cosine: %.6f",
        dataframe[
            "avgpool_cosine"
        ].median(),
    )

    logger.info(
        "  median cosine reconvergence: %.6f",
        dataframe[
            "cosine_reconvergence"
        ].median(),
    )

    logger.info(
        "  positive cosine reconvergence: "
        "%d / %d (%.2f%%)",
        positive_cosine,
        n,
        (
            100.0
            * positive_cosine
            / n
        ),
    )

    logger.info(
        "  median layer4.1 relative-L2: %.6f",
        dataframe[
            "layer4_1_relative_l2"
        ].median(),
    )

    logger.info(
        "  median avgpool relative-L2: %.6f",
        dataframe[
            "avgpool_relative_l2"
        ].median(),
    )

    logger.info(
        "  median relative-L2 reduction: %.6f",
        dataframe[
            "relative_l2_reconvergence"
        ].median(),
    )

    logger.info(
        "  positive relative-L2 reduction: "
        "%d / %d (%.2f%%)",
        positive_l2,
        n,
        (
            100.0
            * positive_l2
            / n
        ),
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Test whether pretrained ResNet layer4->avgpool "
            "reconvergence is specific to attack-vs-bona-fide "
            "comparison or is a more general pooling effect."
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
        default=(
            "configs/local.yaml"
        ),
    )

    parser.add_argument(
        "--control-config",
        default=(
            "tools/"
            "audit_pretrained_pooling_control_config.yaml"
        ),
    )

    args = parser.parse_args()

    tool_cfg_path = resolve_repo_path(
        args.control_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    # Must happen before the audit creates its own output files.
    commit_sha = require_clean_git()

    (
        logger,
        log_path,
        output_paths,
        _timestamp,
    ) = configure_outputs(
        tool_cfg
    )

    capture: (
        RepresentationCapture
        | None
    ) = None

    try:

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

        dataset_root = Path(
            machine_cfg[
                "paths"
            ][
                "dataset_root"
            ]
        ).expanduser().resolve()

        cfg = tool_cfg[
            "control"
        ]

        roll_cfg = cfg[
            "spatial_roll"
        ]

        shift_y = int(
            roll_cfg[
                "shift_y"
            ]
        )

        shift_x = int(
            roll_cfg[
                "shift_x"
            ]
        )

        pooled_atol = float(
            roll_cfg[
                "pooled_invariance_atol"
            ]
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "PRETRAINED RESNET-18 POOLING CONTROL AUDIT"
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
            "Control config SHA-256: %s",
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
            "Machine ID: %s",
            machine_cfg[
                "machine"
            ][
                "id"
            ],
        )

        logger.info(
            "Dataset root: %s",
            dataset_root,
        )

        logger.info(
            "Spatial roll: dy=%d dx=%d",
            shift_y,
            shift_x,
        )

        logger.info(
            "Pooled invariance tolerance: %.3e",
            pooled_atol,
        )

        logger.info(
            "dev_val: NOT USED"
        )

        logger.info(
            "held-out test: NOT ACCESSED"
        )

        # ----------------------------------------------------------
        # Frozen upstream gate.
        # ----------------------------------------------------------

        run_frozen_provenance_gate(
            experiment_path=experiment_path,
            machine_path=machine_path,
            logger=logger,
        )

        logger.info(
            "[PASS] Frozen Tech-1 provenance gate"
        )

        (
            manifest_rows,
            manifest_path,
        ) = load_project_train_manifest(
            experiment_cfg
        )

        expected_variants = set(
            cfg[
                "expected_attack_variants"
            ]
        )

        expected_hardware = set(
            cfg[
                "expected_hardware_sources"
            ]
        )

        capture_groups = build_capture_groups(
            manifest_rows=manifest_rows,
            expected_variants=expected_variants,
            expected_hardware=expected_hardware,
        )

        expected_capture_groups = int(
            cfg[
                "expected_capture_groups"
            ]
        )

        if len(
            capture_groups
        ) != expected_capture_groups:

            raise ValueError(
                "Unexpected capture-group count: "
                f"expected={expected_capture_groups}, "
                f"actual={len(capture_groups)}"
            )

        expected_cards = int(
            cfg[
                "expected_cards"
            ]
        )

        cards = {
            key[
                0
            ]
            for key
            in capture_groups
        }

        if len(
            cards
        ) != expected_cards:

            raise ValueError(
                "Unexpected card count: "
                f"expected={expected_cards}, "
                f"actual={len(cards)}"
            )

        logger.info(
            "project_train manifest: %s",
            manifest_path,
        )

        logger.info(
            "project_train manifest SHA-256: %s",
            sha256_file(
                manifest_path
            ),
        )

        logger.info(
            "Cards: %d",
            len(
                cards
            ),
        )

        logger.info(
            "Capture groups: %d",
            len(
                capture_groups
            ),
        )

        # ----------------------------------------------------------
        # Model.
        # ----------------------------------------------------------

        weights_name = cfg[
            "weights"
        ]

        if weights_name != "IMAGENET1K_V1":

            raise ValueError(
                "Pooling control is locked to IMAGENET1K_V1."
            )

        weights = (
            ResNet18_Weights
            .IMAGENET1K_V1
        )

        model = resnet18(
            weights=weights
        )

        model.eval()

        device = torch.device(
            machine_cfg[
                "runtime"
            ][
                "device"
            ]
        )

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        model.to(
            device
        )

        checkpoint = checkpoint_provenance(
            weights
        )

        logger.info(
            "Weights: ResNet18_Weights.%s",
            weights_name,
        )

        logger.info(
            "Weights URL: %s",
            checkpoint[
                "url"
            ],
        )

        logger.info(
            "Checkpoint path: %s",
            checkpoint[
                "cached_path"
            ],
        )

        logger.info(
            "Checkpoint SHA-256: %s",
            checkpoint[
                "sha256"
            ],
        )

        preset = weights.transforms()

        mean = list(
            preset.mean
        )

        std = list(
            preset.std
        )

        short_side = int(
            cfg[
                "resize_short_side"
            ]
        )

        if cfg[
            "crop"
        ] != "none":

            raise ValueError(
                "Pooling-control baseline must use crop=none."
            )

        require_same_original = bool(
            cfg[
                "require_same_original_size"
            ]
        )

        require_same_input = bool(
            cfg[
                "require_same_model_input_size"
            ]
        )

        progress_every = int(
            tool_cfg[
                "logging"
            ][
                "progress_every_captures"
            ]
        )

        logger.info(
            "Preprocessing: RGB -> resize short side %d "
            "with aspect ratio preserved -> ImageNet normalization",
            short_side,
        )

        logger.info(
            "Crop: NONE"
        )

        # ----------------------------------------------------------
        # Run.
        # ----------------------------------------------------------

        capture = RepresentationCapture(
            model
        )

        if device.type == "cuda":

            torch.cuda.synchronize(
                device
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )

        start_time = (
            time.perf_counter()
        )

        pair_rows: list[
            dict[str, Any]
        ] = []

        roll_rows: list[
            dict[str, Any]
        ] = []

        capture_count = 0
        forward_count = 0

        for (
            (
                file_stem,
                hardware,
            ),
            role_rows,
        ) in sorted(
            capture_groups.items()
        ):

            representations: dict[
                str,
                dict[
                    str,
                    torch.Tensor,
                ],
            ] = {}

            original_sizes: dict[
                str,
                tuple[int, int],
            ] = {}

            input_sizes: dict[
                str,
                tuple[int, int],
            ] = {}

            for role in IMAGE_ROLES:

                row = role_rows[
                    role
                ]

                image_path = verify_image(
                    row=row,
                    dataset_root=dataset_root,
                )

                (
                    tensor,
                    original_size,
                    input_size,
                ) = prepare_image(
                    path=image_path,
                    short_side=short_side,
                    mean=mean,
                    std=std,
                )

                representation = capture.run(
                    model=model,
                    tensor=tensor,
                    device=device,
                )

                forward_count += 1

                representations[
                    role
                ] = representation

                original_sizes[
                    role
                ] = original_size

                input_sizes[
                    role
                ] = input_size

                # ----------------------------------------------
                # Mechanistic spatial-roll control.
                # ----------------------------------------------

                roll_metrics = spatial_roll_metrics(
                    representation=representation,
                    shift_y=shift_y,
                    shift_x=shift_x,
                    pooled_atol=pooled_atol,
                )

                roll_rows.append(
                    {
                        "file_stem":
                            file_stem,

                        "hardware_source":
                            hardware,

                        "image_role":
                            role,

                        "image_path":
                            row[
                                "image_path"
                            ],

                        "image_sha256":
                            row[
                                "image_sha256"
                            ],

                        "layer4_shape":
                            str(
                                tuple(
                                    representation[
                                        "layer4.1"
                                    ].shape
                                )
                            ),

                        "shift_y":
                            shift_y,

                        "shift_x":
                            shift_x,

                        **roll_metrics,
                    }
                )

                del tensor

            # --------------------------------------------------
            # Strict alignment inside one capture group.
            # --------------------------------------------------

            if require_same_original:

                if len(
                    set(
                        original_sizes.values()
                    )
                ) != 1:

                    raise ValueError(
                        "Original dimensions differ within capture:\n"
                        f"  file_stem={file_stem}\n"
                        f"  hardware={hardware}\n"
                        f"  sizes={original_sizes}"
                    )

            if require_same_input:

                if len(
                    set(
                        input_sizes.values()
                    )
                ) != 1:

                    raise ValueError(
                        "Model-input dimensions differ within capture:\n"
                        f"  file_stem={file_stem}\n"
                        f"  hardware={hardware}\n"
                        f"  sizes={input_sizes}"
                    )

            comparison_specs = (
                (
                    "digital_1_vs_bonafide",
                    "digital_1",
                    "bonafide",
                ),
                (
                    "digital_2_vs_bonafide",
                    "digital_2",
                    "bonafide",
                ),
                (
                    "digital_1_vs_digital_2",
                    "digital_1",
                    "digital_2",
                ),
            )

            for (
                pair_type,
                role_a,
                role_b,
            ) in comparison_specs:

                metrics = pair_metrics(
                    representations[
                        role_a
                    ],
                    representations[
                        role_b
                    ],
                )

                row_a = role_rows[
                    role_a
                ]

                row_b = role_rows[
                    role_b
                ]

                pair_rows.append(
                    {
                        "file_stem":
                            file_stem,

                        "hardware_source":
                            hardware,

                        "pair_type":
                            pair_type,

                        "role_a":
                            role_a,

                        "role_b":
                            role_b,

                        "image_a_path":
                            row_a[
                                "image_path"
                            ],

                        "image_a_sha256":
                            row_a[
                                "image_sha256"
                            ],

                        "image_b_path":
                            row_b[
                                "image_path"
                            ],

                        "image_b_sha256":
                            row_b[
                                "image_sha256"
                            ],

                        "original_size":
                            str(
                                original_sizes[
                                    role_a
                                ]
                            ),

                        "model_input_size":
                            str(
                                input_sizes[
                                    role_a
                                ]
                            ),

                        **metrics,
                    }
                )

            capture_count += 1

            if (
                capture_count
                % progress_every
                == 0
                or capture_count
                == expected_capture_groups
            ):

                logger.info(
                    "Progress: %d / %d capture groups",
                    capture_count,
                    expected_capture_groups,
                )

            del representations

        if device.type == "cuda":

            torch.cuda.synchronize(
                device
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------------
        # Reconciliation.
        # ----------------------------------------------------------

        pair_df = pd.DataFrame(
            pair_rows
        )

        roll_df = pd.DataFrame(
            roll_rows
        )

        expected_pair_rows = (
            expected_capture_groups
            * len(
                PAIR_TYPES
            )
        )

        expected_roll_rows = (
            expected_capture_groups
            * len(
                IMAGE_ROLES
            )
        )

        if len(
            pair_df
        ) != expected_pair_rows:

            raise RuntimeError(
                "Pair row-count mismatch: "
                f"expected={expected_pair_rows}, "
                f"actual={len(pair_df)}"
            )

        if len(
            roll_df
        ) != expected_roll_rows:

            raise RuntimeError(
                "Spatial-roll row-count mismatch: "
                f"expected={expected_roll_rows}, "
                f"actual={len(roll_df)}"
            )

        pair_key_duplicates = int(
            pair_df.duplicated(
                subset=[
                    "file_stem",
                    "hardware_source",
                    "pair_type",
                ]
            ).sum()
        )

        if pair_key_duplicates != 0:

            raise RuntimeError(
                "Duplicate pooling-control pair keys: "
                f"{pair_key_duplicates}"
            )

        roll_key_duplicates = int(
            roll_df.duplicated(
                subset=[
                    "file_stem",
                    "hardware_source",
                    "image_role",
                ]
            ).sum()
        )

        if roll_key_duplicates != 0:

            raise RuntimeError(
                "Duplicate spatial-roll keys: "
                f"{roll_key_duplicates}"
            )

        card_df = build_card_summary(
            pair_df
        )

        expected_card_rows = (
            expected_cards
            * len(
                PAIR_TYPES
            )
        )

        if len(
            card_df
        ) != expected_card_rows:

            raise RuntimeError(
                "Card summary row-count mismatch: "
                f"expected={expected_card_rows}, "
                f"actual={len(card_df)}"
            )

        # ----------------------------------------------------------
        # Group summaries.
        # ----------------------------------------------------------

        group_rows: list[
            dict[str, Any]
        ] = []

        append_group_summary(
            group_rows,
            dataframe=pair_df,
            analysis_unit="capture_pair",
            group_type="pair_type",
            group_column="pair_type",
            metrics=PAIR_METRICS,
        )

        for pair_type in PAIR_TYPES:

            subset = pair_df[
                pair_df[
                    "pair_type"
                ]
                == pair_type
            ]

            append_group_summary(
                group_rows,
                dataframe=subset,
                analysis_unit="capture_pair",
                group_type="hardware_source",
                group_column="hardware_source",
                metrics=PAIR_METRICS,
            )

        append_group_summary(
            group_rows,
            dataframe=card_df,
            analysis_unit="card",
            group_type="pair_type",
            group_column="pair_type",
            metrics=PAIR_METRICS,
        )

        roll_metrics = (
            "layer4_original_vs_roll_cosine",
            "layer4_original_vs_roll_relative_l2",
            "avgpool_original_vs_roll_cosine",
            "avgpool_original_vs_roll_relative_l2",
            "model_avgpool_max_abs_error",
            "roll_avgpool_max_abs_error",
        )

        append_group_summary(
            group_rows,
            dataframe=roll_df,
            analysis_unit="image",
            group_type="image_role",
            group_column="image_role",
            metrics=roll_metrics,
        )

        group_df = pd.DataFrame(
            group_rows
        )

        # ----------------------------------------------------------
        # Save.
        # ----------------------------------------------------------

        write_dataframe_exclusive(
            pair_df,
            output_paths[
                "capture_pairs"
            ],
        )

        write_dataframe_exclusive(
            card_df,
            output_paths[
                "card_summary"
            ],
        )

        write_dataframe_exclusive(
            roll_df,
            output_paths[
                "spatial_roll"
            ],
        )

        write_dataframe_exclusive(
            group_df,
            output_paths[
                "group_summary"
            ],
        )

        # ----------------------------------------------------------
        # Key findings.
        # ----------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "EMPIRICAL PAIRING CONTROL"
        )

        logger.info(
            "=" * 72
        )

        for pair_type in (
            PAIR_TYPES
        ):

            subset = pair_df[
                pair_df[
                    "pair_type"
                ]
                == pair_type
            ]

            log_pair_type_summary(
                logger=logger,
                dataframe=subset,
                label=pair_type,
            )

        logger.info(
            "-" * 72
        )

        logger.info(
            "CARD-LEVEL CONTROL "
            "(three hardware captures averaged)"
        )

        for pair_type in (
            PAIR_TYPES
        ):

            subset = card_df[
                card_df[
                    "pair_type"
                ]
                == pair_type
            ]

            log_pair_type_summary(
                logger=logger,
                dataframe=subset,
                label=pair_type,
            )

        # ----------------------------------------------------------
        # Mechanistic GAP control.
        # ----------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "MECHANISTIC SPATIAL-ROLL CONTROL"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "Images tested: %d",
            len(
                roll_df
            ),
        )

        logger.info(
            "Median layer4 original-vs-roll cosine: %.6f",
            roll_df[
                "layer4_original_vs_roll_cosine"
            ].median(),
        )

        logger.info(
            "Median layer4 original-vs-roll relative-L2: %.6f",
            roll_df[
                "layer4_original_vs_roll_relative_l2"
            ].median(),
        )

        logger.info(
            "Median avgpool original-vs-roll cosine: %.9f",
            roll_df[
                "avgpool_original_vs_roll_cosine"
            ].median(),
        )

        logger.info(
            "Maximum avgpool original-vs-roll relative-L2: %.9e",
            roll_df[
                "avgpool_original_vs_roll_relative_l2"
            ].max(),
        )

        logger.info(
            "Maximum roll pooling absolute error: %.9e",
            roll_df[
                "roll_avgpool_max_abs_error"
            ].max(),
        )

        logger.info(
            "Maximum model-vs-independent avgpool error: %.9e",
            roll_df[
                "model_avgpool_max_abs_error"
            ].max(),
        )

        # ----------------------------------------------------------
        # Runtime/provenance.
        # ----------------------------------------------------------

        logger.info(
            "-" * 72
        )

        logger.info(
            "Capture groups: %d",
            capture_count,
        )

        logger.info(
            "Model forwards: %d",
            forward_count,
        )

        logger.info(
            "Empirical comparison rows: %d",
            len(
                pair_df
            ),
        )

        logger.info(
            "Card summary rows: %d",
            len(
                card_df
            ),
        )

        logger.info(
            "Spatial-roll rows: %d",
            len(
                roll_df
            ),
        )

        logger.info(
            "Elapsed seconds: %.3f",
            elapsed,
        )

        logger.info(
            "Forward throughput: %.3f images/s",
            (
                forward_count
                / elapsed
            ),
        )

        if device.type == "cuda":

            logger.info(
                "Peak CUDA allocated: %.3f GiB",
                (
                    torch.cuda
                    .max_memory_allocated(
                        device
                    )
                    / 1024**3
                ),
            )

            logger.info(
                "Peak CUDA reserved: %.3f GiB",
                (
                    torch.cuda
                    .max_memory_reserved(
                        device
                    )
                    / 1024**3
                ),
            )

        logger.info(
            "-" * 72
        )

        logger.info(
            "OUTPUT ARTIFACTS"
        )

        for (
            name,
            path,
        ) in output_paths.items():

            logger.info(
                "%s: %s",
                name,
                path,
            )

            logger.info(
                "%s SHA-256: %s",
                name,
                sha256_file(
                    path
                ),
            )

        logger.info(
            "Audit log: %s",
            log_path,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "PRETRAINED RESNET-18 POOLING CONTROL AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "PRETRAINED RESNET-18 "
            "POOLING CONTROL AUDIT: FAIL"
        )

        return 1

    finally:

        if capture is not None:

            capture.close()

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