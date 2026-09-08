#!/usr/bin/env python3
"""
Cohort-wide paired ImageNet-pretrained ResNet-18 representation audit.

Purpose
-------
Extend the successful single-pair exploratory probe to every aligned
attack / bona-fide pair in frozen FantasyID project_train.

For every attack image:

    attack
      ↕ same file_stem + same hardware_source
    bona fide

The model is NOT fine-tuned on FantasyID.

The audit measures representation difference at:

    stem_relu
    maxpool
    layer1.0
    layer1.1
    layer2.0
    layer2.1
    layer3.0
    layer3.1
    layer4.0
    layer4.1
    avgpool

It additionally measures whether each stage's attack/bona-fide
difference is:

    - concentrated in a few channels, or
    - broadly distributed across many channels.

Analysis levels
---------------
1. Capture-pair level

       960 attack ↔ bona-fide pairs.

2. Card-variant level

       Measurements are averaged over Huawei / iPhone / scan
       for the same:

           file_stem + attack variant

       giving:

           160 cards × 2 variants = 320 card-variant units.

This prevents three hardware captures of one manipulation from being
mistaken for three independent manipulation examples.

Important interpretation
------------------------
The channel-difference measurements are GLOBAL over each feature map.

They do NOT establish that a recurrent channel represents:
    - face identity,
    - text manipulation,
    - forgery,
    - or a classifier shortcut.

Every project_train attack has both face and text alteration, so
face-specific attribution requires a later ROI-aware analysis.

No spatial diagnostic PNGs are generated here. Relative, median, and
channel-normalised spatial maps are intentionally deferred until the
cohort identifies representative examples.

Data policy
-----------
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
from collections import Counter, defaultdict
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
from torchvision.models.resnet import BasicBlock
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


# ======================================================================
# Frozen annotation semantics
# ======================================================================

ALTERED_PROVENANCE = "altered"
FACE_FIELD = "face"


# ======================================================================
# ResNet stages
# ======================================================================

STAGE_NAMES = (
    "stem_relu",
    "maxpool",
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
    "avgpool",
)


BASE_STAGE_METRICS = (
    "cosine_similarity",
    "pearson_correlation",
    "euclidean_distance",
    "relative_l2_distance",
    "mean_absolute_difference",
    "max_absolute_difference",
)


PAIR_DERIVED_METRICS = (
    "layer4_1_cosine",
    "avgpool_cosine",
    "cosine_reconvergence",
    "layer4_1_relative_l2",
    "avgpool_relative_l2",
    "relative_l2_reconvergence",
    "top_k_feature_overlap",
    "top1_same_numeric",
)


# ======================================================================
# Basic utilities
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


def normalize_text(
    value: Any,
) -> str:
    """
    Normalize workbook values.

    Excel blanks / pandas NaN / explicit null-like markers become "".
    """

    if value is None:
        return ""

    try:

        if pd.isna(
            value
        ):
            return ""

    except TypeError:
        pass

    text = str(
        value
    ).strip()

    if text.casefold() in {
        "none",
        "<none>",
        "nan",
        "<nan>",
        "null",
        "<null>",
    }:

        return ""

    return text


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
    Require committed code before any output file is created.

    This check intentionally happens BEFORE creation of the audit log,
    otherwise an untracked audit log could make the run mark its own
    Git tree as dirty.
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

    status = result.stdout.strip()

    if status:

        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit or stash outstanding changes before running "
            "the cohort diagnostic.\n\n"
            f"{status}"
        )

    return commit


def write_dataframe_exclusive(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """
    Never silently overwrite an existing scientific result.
    """

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
# Logging and output paths
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
        "cohort_pretrained_resnet18"
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
# Frozen upstream provenance gate
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
# Frozen project_train manifest
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
            "project_train manifest row-count mismatch: "
            f"expected={expected}, "
            f"actual={len(rows)}"
        )

    return (
        rows,
        manifest_path,
    )


# ======================================================================
# Frozen altered-field annotations
# ======================================================================

def load_altered_field_sets(
    *,
    experiment_cfg: dict[str, Any],
    attack_paths: set[str],
) -> tuple[
    dict[str, str],
    Path,
]:

    discovery_cfg = (
        experiment_cfg[
            "data"
        ][
            "source_discovery"
        ]
    )

    workbook_path = resolve_repo_path(
        discovery_cfg[
            "workbook"
        ]
    )

    expected_sha = discovery_cfg[
        "sha256"
    ]

    actual_sha = sha256_file(
        workbook_path
    )

    if actual_sha != expected_sha:

        raise ValueError(
            "Frozen discovery workbook SHA-256 mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    regions = pd.read_excel(
        workbook_path,
        sheet_name="Regions",
        engine="openpyxl",
        usecols=[
            "image_path",
            "field_name",
            "region_provenance_raw",
        ],
    )

    image_paths = (
        regions[
            "image_path"
        ]
        .map(
            normalize_text
        )
    )

    provenance = (
        regions[
            "region_provenance_raw"
        ]
        .map(
            normalize_text
        )
        .str.casefold()
    )

    fields = (
        regions[
            "field_name"
        ]
        .map(
            normalize_text
        )
        .str.casefold()
    )

    altered_mask = (
        image_paths.isin(
            attack_paths
        )
        &
        provenance.eq(
            ALTERED_PROVENANCE
        )
    )

    altered = pd.DataFrame(
        {
            "image_path":
                image_paths[
                    altered_mask
                ],

            "field_name":
                fields[
                    altered_mask
                ],
        }
    )

    result: dict[
        str,
        str,
    ] = {}

    for (
        image_path,
        group,
    ) in altered.groupby(
        "image_path",
        sort=False,
    ):

        field_values = sorted(
            {
                field
                for field
                in group[
                    "field_name"
                ].tolist()
                if field
            }
        )

        has_face = (
            FACE_FIELD
            in field_values
        )

        has_text = any(
            field
            != FACE_FIELD
            for field
            in field_values
        )

        if not (
            has_face
            and has_text
        ):

            raise ValueError(
                "Frozen training attack does not contain "
                "confirmed face + text alterations:\n"
                f"  image={image_path}\n"
                f"  fields={field_values}"
            )

        result[
            image_path
        ] = "|".join(
            field_values
        )

    missing = (
        attack_paths
        - set(
            result
        )
    )

    if missing:

        raise ValueError(
            "Attack images missing altered-region annotations: "
            f"{sorted(missing)[:10]}"
        )

    return (
        result,
        workbook_path,
    )


# ======================================================================
# Capture pairing
# ======================================================================

def build_capture_groups(
    *,
    manifest_rows: list[
        dict[str, str]
    ],
    expected_variants: set[str],
    expected_hardware: set[str],
) -> tuple[
    dict[
        tuple[str, str],
        dict[str, str],
    ],
    dict[
        tuple[str, str],
        list[
            dict[str, str]
        ],
    ],
]:

    bonafide: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    attacks: dict[
        tuple[str, str],
        list[
            dict[str, str]
        ],
    ] = defaultdict(
        list
    )

    for row in manifest_rows:

        hardware = row[
            "hardware_source"
        ]

        if hardware not in expected_hardware:

            raise ValueError(
                "Unexpected project_train hardware source: "
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

            if key in bonafide:

                raise ValueError(
                    "Duplicate bona-fide capture key: "
                    f"{key}"
                )

            bonafide[
                key
            ] = row

        elif traffic == "attack":

            attacks[
                key
            ].append(
                row
            )

        else:

            raise ValueError(
                "Unexpected project_train traffic type: "
                f"{traffic}"
            )

    if set(
        bonafide
    ) != set(
        attacks
    ):

        missing_attack = (
            set(
                bonafide
            )
            - set(
                attacks
            )
        )

        missing_bonafide = (
            set(
                attacks
            )
            - set(
                bonafide
            )
        )

        raise ValueError(
            "Attack/bona-fide capture-key mismatch:\n"
            f"  no attack={sorted(missing_attack)[:10]}\n"
            f"  no bona-fide={sorted(missing_bonafide)[:10]}"
        )

    for (
        key,
        rows,
    ) in attacks.items():

        variants = {
            row[
                "variant"
            ]
            for row
            in rows
        }

        if variants != expected_variants:

            raise ValueError(
                "Unexpected attack variant composition:\n"
                f"  capture={key}\n"
                f"  expected={sorted(expected_variants)}\n"
                f"  actual={sorted(variants)}"
            )

        if len(
            rows
        ) != len(
            expected_variants
        ):

            raise ValueError(
                "Duplicate or missing attack variants for "
                f"capture {key}"
            )

    return (
        bonafide,
        attacks,
    )


# ======================================================================
# Source image integrity
# ======================================================================

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
            "Manifest image path escapes dataset root."
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
# Input preparation
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

    resized_size = (
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
        resized_size,
    )


# ======================================================================
# Persistent ResNet activation hooks
# ======================================================================

class StageCapture:
    """
    Capture only stage boundaries needed for the cohort metrics.

    Persistent hooks avoid registering and removing hooks for each
    of the 1,440 model forwards.
    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:

        self.activations: dict[
            str,
            torch.Tensor,
        ] = {}

        self.handles = []

        def make_hook(
            name: str,
        ):

            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: torch.Tensor,
            ) -> None:

                self.activations[
                    name
                ] = (
                    output.detach()
                )

            return hook

        # Stem output.
        self.handles.append(
            model.relu.register_forward_hook(
                make_hook(
                    "stem_relu"
                )
            )
        )

        # Max-pool output.
        self.handles.append(
            model.maxpool.register_forward_hook(
                make_hook(
                    "maxpool"
                )
            )
        )

        # Residual block outputs.
        for (
            name,
            module,
        ) in model.named_modules():

            if isinstance(
                module,
                BasicBlock,
            ):

                self.handles.append(
                    module.register_forward_hook(
                        make_hook(
                            name
                        )
                    )
                )

        # Final global average pooling.
        self.handles.append(
            model.avgpool.register_forward_hook(
                make_hook(
                    "avgpool"
                )
            )
        )

    def run(
        self,
        *,
        model: nn.Module,
        tensor: torch.Tensor,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        dict[
            str,
            torch.Tensor,
        ],
    ]:

        self.activations = {}

        with torch.inference_mode():

            logits = model(
                tensor.to(
                    device,
                    non_blocking=True,
                )
            )

        missing = (
            set(
                STAGE_NAMES
            )
            - set(
                self.activations
            )
        )

        if missing:

            raise RuntimeError(
                "Required activation stages were not captured: "
                f"{sorted(missing)}"
            )

        return (
            logits.detach(),
            dict(
                self.activations
            ),
        )

    def close(
        self,
    ) -> None:

        for handle in (
            self.handles
        ):

            handle.remove()

        self.handles.clear()


# ======================================================================
# Representation similarity
# ======================================================================

def representation_metrics(
    attack: torch.Tensor,
    bonafide: torch.Tensor,
) -> dict[str, float]:

    if attack.shape != bonafide.shape:

        raise ValueError(
            "Activation shapes differ:\n"
            f"  attack={tuple(attack.shape)}\n"
            f"  bonafide={tuple(bonafide.shape)}"
        )

    a = (
        attack
        .reshape(-1)
        .float()
    )

    b = (
        bonafide
        .reshape(-1)
        .float()
    )

    difference = (
        a - b
    )

    absolute_difference = (
        difference.abs()
    )

    cosine = (
        F.cosine_similarity(
            a.unsqueeze(
                0
            ),
            b.unsqueeze(
                0
            ),
        )[0]
    )

    a_centered = (
        a - a.mean()
    )

    b_centered = (
        b - b.mean()
    )

    pearson_denominator = (
        torch.linalg.vector_norm(
            a_centered
        )
        *
        torch.linalg.vector_norm(
            b_centered
        )
    )

    if (
        pearson_denominator
        .item()
        == 0.0
    ):

        pearson = float(
            "nan"
        )

    else:

        pearson = float(
            (
                torch.dot(
                    a_centered,
                    b_centered,
                )
                /
                pearson_denominator
            ).item()
        )

    euclidean = (
        torch.linalg.vector_norm(
            difference
        )
    )

    mean_norm = (
        (
            torch.linalg
            .vector_norm(
                a
            )
        )
        +
        (
            torch.linalg
            .vector_norm(
                b
            )
        )
    ) / 2.0

    if (
        mean_norm.item()
        == 0.0
    ):

        relative_l2 = 0.0

    else:

        relative_l2 = float(
            (
                euclidean
                /
                mean_norm
            ).item()
        )

    return {
        "cosine_similarity":
            float(
                cosine.item()
            ),

        "pearson_correlation":
            pearson,

        "euclidean_distance":
            float(
                euclidean.item()
            ),

        "relative_l2_distance":
            relative_l2,

        "mean_absolute_difference":
            float(
                absolute_difference
                .mean()
                .item()
            ),

        "max_absolute_difference":
            float(
                absolute_difference
                .max()
                .item()
            ),
    }


# ======================================================================
# Channel concentration
# ======================================================================

def concentration_metric_names(
    top_ks: list[int],
) -> tuple[str, ...]:

    names = [
        f"channel_diff_top{k}_share"
        for k in top_ks
    ]

    names.extend(
        [
            "channel_diff_effective_channels",
            "channel_diff_effective_fraction",
        ]
    )

    return tuple(
        names
    )


def channel_difference_profile(
    attack: torch.Tensor,
    bonafide: torch.Tensor,
    *,
    top_ks: list[int],
) -> tuple[
    dict[str, float],
    torch.Tensor,
]:
    """
    Measure whether representation difference is concentrated in a
    few channels or spread across many.

    For channel c:

        d_c = mean_H,W |A_attack[c] - A_bonafide[c]|

    Top-k share:

        sum(k largest d_c) / sum(all d_c)

    Effective number of changed channels:

        p_c = d_c / sum(d)

        N_eff = 1 / sum(p_c^2)

    Interpretation:

        N_eff ~= 1
            almost all difference is carried by one channel.

        N_eff ~= C
            difference is broadly distributed over channels.

    effective_fraction = N_eff / C

    This is a GLOBAL stage-level quantity. It is not face-ROI-specific
    or text-ROI-specific.
    """

    if attack.shape != bonafide.shape:

        raise ValueError(
            "Activation shapes differ:\n"
            f"  attack={tuple(attack.shape)}\n"
            f"  bonafide={tuple(bonafide.shape)}"
        )

    if attack.ndim != 4:

        raise ValueError(
            "Expected N x C x H x W activation, got "
            f"{tuple(attack.shape)}"
        )

    if attack.shape[0] != 1:

        raise ValueError(
            "Channel-concentration audit expects batch size 1."
        )

    difference = (
        attack
        - bonafide
    ).abs()[0].float()

    channels = int(
        difference.shape[0]
    )

    # C × (H*W)
    flattened = (
        difference
        .reshape(
            channels,
            -1,
        )
    )

    # One difference magnitude for each channel.
    channel_difference = (
        flattened
        .mean(
            dim=1
        )
    )

    for k in top_ks:

        if k <= 0:

            raise ValueError(
                "Channel concentration top-k must be positive: "
                f"{k}"
            )

        if k > channels:

            raise ValueError(
                "Channel concentration top-k exceeds stage "
                "channel count:\n"
                f"  k={k}\n"
                f"  channels={channels}"
            )

    total = (
        channel_difference
        .sum()
    )

    metrics: dict[
        str,
        float,
    ] = {}

    if total.item() == 0.0:

        for k in top_ks:

            metrics[
                f"channel_diff_top{k}_share"
            ] = 0.0

        metrics[
            "channel_diff_effective_channels"
        ] = 0.0

        metrics[
            "channel_diff_effective_fraction"
        ] = 0.0

        return (
            metrics,
            channel_difference.detach(),
        )

    for k in top_ks:

        strongest = (
            torch.topk(
                channel_difference,
                k=k,
            )
            .values
        )

        metrics[
            f"channel_diff_top{k}_share"
        ] = float(
            (
                strongest.sum()
                /
                total
            ).item()
        )

    probabilities = (
        channel_difference
        /
        total
    )

    inverse_concentration = (
        probabilities
        .square()
        .sum()
    )

    effective_channels = float(
        (
            1.0
            /
            inverse_concentration
        ).item()
    )

    metrics[
        "channel_diff_effective_channels"
    ] = effective_channels

    metrics[
        "channel_diff_effective_fraction"
    ] = (
        effective_channels
        /
        channels
    )

    return (
        metrics,
        channel_difference.detach(),
    )


# ======================================================================
# Avgpool top-k overlap
# ======================================================================

def top_k_overlap(
    attack_pool: torch.Tensor,
    bonafide_pool: torch.Tensor,
    k: int,
) -> float:

    attack_vector = (
        attack_pool
        .reshape(-1)
    )

    bonafide_vector = (
        bonafide_pool
        .reshape(-1)
    )

    if k > attack_vector.numel():

        raise ValueError(
            "top_k_features exceeds pooled feature count."
        )

    attack_top = set(
        torch.topk(
            attack_vector.abs(),
            k=k,
        ).indices.tolist()
    )

    bonafide_top = set(
        torch.topk(
            bonafide_vector.abs(),
            k=k,
        ).indices.tolist()
    )

    return (
        len(
            attack_top
            &
            bonafide_top
        )
        /
        k
    )


# ======================================================================
# Pretrained weight provenance
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
# Card-aware stage aggregation
# ======================================================================

def build_card_variant_stage_metrics(
    stage_df: pd.DataFrame,
    *,
    metric_names: tuple[str, ...],
) -> pd.DataFrame:

    aggregations: dict[
        str,
        tuple[str, str],
    ] = {
        "capture_pair_count":
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
        metric_names
    ):

        aggregations[
            metric
        ] = (
            metric,
            "mean",
        )

    result = (
        stage_df
        .groupby(
            [
                "file_stem",
                "variant",
                "altered_fields_unique",
                "stage",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            **aggregations
        )
    )

    bad = result[
        (
            result[
                "capture_pair_count"
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
            "Card-variant stage aggregation does not contain "
            "exactly three hardware captures per stage.\n\n"
            f"{bad.head(10).to_string(index=False)}"
        )

    return result


def build_card_variant_summary(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        pair_df
        .groupby(
            [
                "file_stem",
                "variant",
                "altered_fields_unique",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            capture_pair_count=(
                "hardware_source",
                "size",
            ),

            hardware_count=(
                "hardware_source",
                "nunique",
            ),

            layer4_1_cosine=(
                "layer4_1_cosine",
                "mean",
            ),

            avgpool_cosine=(
                "avgpool_cosine",
                "mean",
            ),

            cosine_reconvergence=(
                "cosine_reconvergence",
                "mean",
            ),

            layer4_1_relative_l2=(
                "layer4_1_relative_l2",
                "mean",
            ),

            avgpool_relative_l2=(
                "avgpool_relative_l2",
                "mean",
            ),

            relative_l2_reconvergence=(
                "relative_l2_reconvergence",
                "mean",
            ),

            top_k_feature_overlap=(
                "top_k_feature_overlap",
                "mean",
            ),

            top1_same_numeric=(
                "top1_same_numeric",
                "mean",
            ),
        )
    )

    bad = result[
        (
            result[
                "capture_pair_count"
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
            "Card-variant pair aggregation does not contain "
            "exactly three hardware captures per unit."
        )

    return result


# ======================================================================
# Generic descriptive summaries
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


def append_stage_group_summary(
    output_rows: list[
        dict[str, Any]
    ],
    *,
    dataframe: pd.DataFrame,
    analysis_unit: str,
    group_type: str,
    group_column: str | None,
    metric_names: tuple[str, ...],
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

        for (
            stage,
            stage_df,
        ) in group_df.groupby(
            "stage",
            sort=False,
        ):

            for metric in (
                metric_names
            ):

                summary = numeric_summary(
                    stage_df[
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

                        "stage":
                            stage,

                        "metric":
                            metric,

                        **summary,
                    }
                )


def append_pair_group_summary(
    output_rows: list[
        dict[str, Any]
    ],
    *,
    dataframe: pd.DataFrame,
    analysis_unit: str,
    group_type: str,
    group_column: str | None,
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
            PAIR_DERIVED_METRICS
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

                    "stage":
                        "PAIR_DERIVED",

                    "metric":
                        metric,

                    **summary,
                }
            )


# ======================================================================
# Layer4 channel recurrence
# ======================================================================

def build_card_variant_channel_records(
    capture_records: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Average channel-difference vectors over the three hardware captures
    belonging to the same card + attack variant.
    """

    grouped: dict[
        tuple[
            str,
            str,
            str,
        ],
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for record in (
        capture_records
    ):

        key = (
            record[
                "file_stem"
            ],
            record[
                "variant"
            ],
            record[
                "altered_fields_unique"
            ],
        )

        grouped[
            key
        ].append(
            record
        )

    result: list[
        dict[str, Any]
    ] = []

    for (
        (
            file_stem,
            variant,
            altered_fields,
        ),
        records,
    ) in sorted(
        grouped.items()
    ):

        hardware = {
            record[
                "hardware_source"
            ]
            for record
            in records
        }

        if len(
            records
        ) != 3:

            raise ValueError(
                "Expected three capture records for "
                "card-variant channel aggregation:\n"
                f"  file_stem={file_stem}\n"
                f"  variant={variant}\n"
                f"  records={len(records)}"
            )

        if len(
            hardware
        ) != 3:

            raise ValueError(
                "Expected three distinct hardware sources for "
                "card-variant channel aggregation:\n"
                f"  file_stem={file_stem}\n"
                f"  variant={variant}\n"
                f"  hardware={sorted(hardware)}"
            )

        vectors = torch.stack(
            [
                record[
                    "channel_difference"
                ]
                for record
                in records
            ],
            dim=0,
        )

        result.append(
            {
                "file_stem":
                    file_stem,

                "variant":
                    variant,

                "altered_fields_unique":
                    altered_fields,

                "channel_difference":
                    vectors.mean(
                        dim=0
                    ),
            }
        )

    return result


def append_channel_frequency_rows(
    output_rows: list[
        dict[str, Any]
    ],
    *,
    records: list[
        dict[str, Any]
    ],
    analysis_unit: str,
    group_type: str,
    group_value: str,
    stage: str,
    top_k: int,
) -> None:
    """
    Report how often every channel appears among the top-k difference
    channels across the supplied analysis units.
    """

    if not records:
        return

    vectors = torch.stack(
        [
            record[
                "channel_difference"
            ]
            for record
            in records
        ],
        dim=0,
    ).float()

    unit_count = int(
        vectors.shape[0]
    )

    channel_count = int(
        vectors.shape[1]
    )

    if top_k > channel_count:

        raise ValueError(
            "recurrence_top_k exceeds channel count:\n"
            f"  top_k={top_k}\n"
            f"  channels={channel_count}"
        )

    counts = Counter()

    for vector in vectors:

        indices = (
            torch.topk(
                vector,
                k=top_k,
            )
            .indices
            .tolist()
        )

        counts.update(
            indices
        )

    expected_total = (
        unit_count
        *
        top_k
    )

    actual_total = sum(
        counts.values()
    )

    if actual_total != expected_total:

        raise RuntimeError(
            "Channel top-k frequency reconciliation failed:\n"
            f"  expected={expected_total}\n"
            f"  actual={actual_total}"
        )

    mean_difference = (
        vectors.mean(
            dim=0
        )
    )

    median_difference = (
        vectors.median(
            dim=0
        ).values
    )

    ordered_channels = sorted(
        range(
            channel_count
        ),
        key=lambda channel: (
            -counts[
                channel
            ],
            -float(
                mean_difference[
                    channel
                ]
            ),
            channel,
        ),
    )

    frequency_rank = {
        channel: rank
        for (
            rank,
            channel,
        ) in enumerate(
            ordered_channels,
            start=1,
        )
    }

    for channel in range(
        channel_count
    ):

        appearance_count = int(
            counts[
                channel
            ]
        )

        output_rows.append(
            {
                "analysis_unit":
                    analysis_unit,

                "group_type":
                    group_type,

                "group_value":
                    group_value,

                "stage":
                    stage,

                "top_k":
                    top_k,

                "unit_count":
                    unit_count,

                "channel_index":
                    channel,

                "appearance_count":
                    appearance_count,

                "appearance_rate":
                    (
                        appearance_count
                        /
                        unit_count
                    ),

                "mean_channel_difference":
                    float(
                        mean_difference[
                            channel
                        ]
                    ),

                "median_channel_difference":
                    float(
                        median_difference[
                            channel
                        ]
                    ),

                "frequency_rank":
                    frequency_rank[
                        channel
                    ],
            }
        )


def append_grouped_channel_frequency(
    output_rows: list[
        dict[str, Any]
    ],
    *,
    records: list[
        dict[str, Any]
    ],
    analysis_unit: str,
    group_type: str,
    group_field: str | None,
    stage: str,
    top_k: int,
) -> None:

    if group_field is None:

        append_channel_frequency_rows(
            output_rows,
            records=records,
            analysis_unit=analysis_unit,
            group_type=group_type,
            group_value="ALL",
            stage=stage,
            top_k=top_k,
        )

        return

    values = sorted(
        {
            str(
                record[
                    group_field
                ]
            )
            for record
            in records
        }
    )

    for value in values:

        subset = [
            record
            for record
            in records
            if str(
                record[
                    group_field
                ]
            )
            == value
        ]

        append_channel_frequency_rows(
            output_rows,
            records=subset,
            analysis_unit=analysis_unit,
            group_type=group_type,
            group_value=value,
            stage=stage,
            top_k=top_k,
        )


def build_channel_frequency_table(
    *,
    capture_records: list[
        dict[str, Any]
    ],
    card_variant_records: list[
        dict[str, Any]
    ],
    stage: str,
    top_k: int,
) -> pd.DataFrame:

    rows: list[
        dict[str, Any]
    ] = []

    # Capture-pair descriptive analysis.
    append_grouped_channel_frequency(
        rows,
        records=capture_records,
        analysis_unit="capture_pair",
        group_type="overall",
        group_field=None,
        stage=stage,
        top_k=top_k,
    )

    append_grouped_channel_frequency(
        rows,
        records=capture_records,
        analysis_unit="capture_pair",
        group_type="variant",
        group_field="variant",
        stage=stage,
        top_k=top_k,
    )

    append_grouped_channel_frequency(
        rows,
        records=capture_records,
        analysis_unit="capture_pair",
        group_type="hardware_source",
        group_field="hardware_source",
        stage=stage,
        top_k=top_k,
    )

    append_grouped_channel_frequency(
        rows,
        records=capture_records,
        analysis_unit="capture_pair",
        group_type="altered_fields_unique",
        group_field="altered_fields_unique",
        stage=stage,
        top_k=top_k,
    )

    # Card-variant level is the more appropriate unit for recurrence
    # because the three hardware repetitions have already been averaged.
    append_grouped_channel_frequency(
        rows,
        records=card_variant_records,
        analysis_unit="card_variant",
        group_type="overall",
        group_field=None,
        stage=stage,
        top_k=top_k,
    )

    append_grouped_channel_frequency(
        rows,
        records=card_variant_records,
        analysis_unit="card_variant",
        group_type="variant",
        group_field="variant",
        stage=stage,
        top_k=top_k,
    )

    append_grouped_channel_frequency(
        rows,
        records=card_variant_records,
        analysis_unit="card_variant",
        group_type="altered_fields_unique",
        group_field="altered_fields_unique",
        stage=stage,
        top_k=top_k,
    )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Key-result logging
# ======================================================================

def log_pair_summary(
    *,
    logger: logging.Logger,
    pair_df: pd.DataFrame,
    label: str,
) -> None:

    n = len(
        pair_df
    )

    reconverged = int(
        (
            pair_df[
                "cosine_reconvergence"
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
        pair_df[
            "layer4_1_cosine"
        ].median(),
    )

    logger.info(
        "  median avgpool cosine: %.6f",
        pair_df[
            "avgpool_cosine"
        ].median(),
    )

    logger.info(
        "  median cosine reconvergence "
        "(avgpool - layer4.1): %.6f",
        pair_df[
            "cosine_reconvergence"
        ].median(),
    )

    logger.info(
        "  avgpool cosine > layer4.1 cosine: "
        "%d / %d (%.2f%%)",
        reconverged,
        n,
        (
            100.0
            * reconverged
            / n
        ),
    )

    logger.info(
        "  median relative-L2 reduction "
        "(layer4.1 - avgpool): %.6f",
        pair_df[
            "relative_l2_reconvergence"
        ].median(),
    )

    logger.info(
        "  ImageNet top-1 agreement: %.2f%%",
        (
            100.0
            * pair_df[
                "top1_same_numeric"
            ].mean()
        ),
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run paired pretrained ResNet-18 representation "
            "analysis across every frozen project_train attack."
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
        "--cohort-config",
        default=(
            "tools/"
            "cohort_pretrained_resnet18_config.yaml"
        ),
    )

    args = parser.parse_args()

    tool_cfg_path = resolve_repo_path(
        args.cohort_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    # --------------------------------------------------------------
    # Check cleanliness BEFORE the tool creates its own log.
    # --------------------------------------------------------------

    commit_sha = require_clean_git()

    (
        logger,
        log_path,
        output_paths,
        _timestamp,
    ) = configure_outputs(
        tool_cfg
    )

    capture: StageCapture | None = None

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
            "cohort"
        ]

        concentration_cfg = tool_cfg[
            "channel_concentration"
        ]

        concentration_top_ks = sorted(
            [
                int(k)
                for k
                in concentration_cfg[
                    "top_k_shares"
                ]
            ]
        )

        if len(
            concentration_top_ks
        ) != len(
            set(
                concentration_top_ks
            )
        ):

            raise ValueError(
                "Duplicate channel-concentration top-k values."
            )

        recurrence_stage = str(
            concentration_cfg[
                "recurrence_stage"
            ]
        )

        recurrence_top_k = int(
            concentration_cfg[
                "recurrence_top_k"
            ]
        )

        if recurrence_stage not in STAGE_NAMES:

            raise ValueError(
                "Invalid recurrence_stage: "
                f"{recurrence_stage}"
            )

        if recurrence_top_k not in concentration_top_ks:

            raise ValueError(
                "recurrence_top_k must also be present in "
                "channel_concentration.top_k_shares."
            )

        concentration_metrics = (
            concentration_metric_names(
                concentration_top_ks
            )
        )

        all_stage_metrics = (
            BASE_STAGE_METRICS
            +
            concentration_metrics
        )

        # Cross-layer summary deliberately emphasizes scale-free metrics.
        # Absolute difference values remain available in the raw CSV.
        grouped_stage_metrics = (
            "cosine_similarity",
            "pearson_correlation",
            "relative_l2_distance",
            "mean_absolute_difference",
        ) + concentration_metrics

        logger.info(
            "=" * 72
        )

        logger.info(
            "PRETRAINED RESNET-18 FULL PROJECT_TRAIN PAIRED COHORT"
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
            "Cohort config SHA-256: %s",
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
            "Channel concentration top-k shares: %s",
            concentration_top_ks,
        )

        logger.info(
            "Channel recurrence stage: %s",
            recurrence_stage,
        )

        logger.info(
            "Channel recurrence top-k: %d",
            recurrence_top_k,
        )

        logger.info(
            "Channel concentration is GLOBAL over each feature map; "
            "it is not face-ROI-specific."
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

        attack_rows = [
            row
            for row
            in manifest_rows
            if row[
                "traffic_type"
            ]
            == "attack"
        ]

        bonafide_rows = [
            row
            for row
            in manifest_rows
            if row[
                "traffic_type"
            ]
            == "bonafide"
        ]

        expected_attack_pairs = int(
            cfg[
                "expected_attack_pairs"
            ]
        )

        expected_bonafide = int(
            cfg[
                "expected_bonafide_captures"
            ]
        )

        if len(
            attack_rows
        ) != expected_attack_pairs:

            raise ValueError(
                "Unexpected project_train attack count: "
                f"expected={expected_attack_pairs}, "
                f"actual={len(attack_rows)}"
            )

        if len(
            bonafide_rows
        ) != expected_bonafide:

            raise ValueError(
                "Unexpected project_train bona-fide count: "
                f"expected={expected_bonafide}, "
                f"actual={len(bonafide_rows)}"
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

        (
            bonafide_by_capture,
            attacks_by_capture,
        ) = build_capture_groups(
            manifest_rows=manifest_rows,
            expected_variants=expected_variants,
            expected_hardware=expected_hardware,
        )

        if len(
            bonafide_by_capture
        ) != expected_bonafide:

            raise ValueError(
                "Unexpected number of aligned bona-fide "
                "capture groups."
            )

        attack_paths = {
            row[
                "image_path"
            ]
            for row
            in attack_rows
        }

        (
            altered_fields,
            workbook_path,
        ) = load_altered_field_sets(
            experiment_cfg=experiment_cfg,
            attack_paths=attack_paths,
        )

        expected_cards = int(
            experiment_cfg[
                "data"
            ][
                "frozen_split"
            ][
                "project_train"
            ][
                "cards"
            ]
        )

        unique_cards = {
            row[
                "file_stem"
            ]
            for row
            in attack_rows
        }

        if len(
            unique_cards
        ) != expected_cards:

            raise ValueError(
                "Unexpected project_train card count: "
                f"expected={expected_cards}, "
                f"actual={len(unique_cards)}"
            )

        expected_card_variant_units = (
            expected_cards
            *
            len(
                expected_variants
            )
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
            "Frozen discovery workbook: %s",
            workbook_path,
        )

        logger.info(
            "Frozen discovery workbook SHA-256: %s",
            sha256_file(
                workbook_path
            ),
        )

        logger.info(
            "Cards: %d",
            expected_cards,
        )

        logger.info(
            "Aligned bona-fide captures: %d",
            len(
                bonafide_by_capture
            ),
        )

        logger.info(
            "Attack/bona-fide pairs: %d",
            len(
                attack_rows
            ),
        )

        logger.info(
            "Card-variant analysis units: %d",
            expected_card_variant_units,
        )

        # ----------------------------------------------------------
        # Model.
        # ----------------------------------------------------------

        weights_name = cfg[
            "weights"
        ]

        if weights_name != "IMAGENET1K_V1":

            raise ValueError(
                "Current pretrained cohort baseline is locked "
                "to IMAGENET1K_V1."
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
            device.type
            == "cuda"
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
                "This exploratory cohort baseline must use crop=none."
            )

        logger.info(
            "Preprocessing: RGB -> resize short side %d "
            "with aspect ratio preserved -> ImageNet normalization",
            short_side,
        )

        logger.info(
            "Crop: NONE"
        )

        logger.warning(
            "ImageNet predictions remain diagnostic only because "
            "the normal ImageNet centre crop is intentionally omitted."
        )

        categories = weights.meta[
            "categories"
        ]

        top_k_features = int(
            cfg[
                "top_k_features"
            ]
        )

        progress_every = int(
            tool_cfg[
                "logging"
            ][
                "progress_every_pairs"
            ]
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

        # ----------------------------------------------------------
        # Cohort forward pass.
        # ----------------------------------------------------------

        capture = StageCapture(
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

        stage_rows: list[
            dict[str, Any]
        ] = []

        pair_rows: list[
            dict[str, Any]
        ] = []

        recurrence_capture_records: list[
            dict[str, Any]
        ] = []

        pair_count = 0
        forward_image_count = 0

        # ----------------------------------------------------------
        # A bona-fide representation is calculated once per physical
        # capture and reused for digital_1 and digital_2.
        # ----------------------------------------------------------

        for capture_key in sorted(
            bonafide_by_capture
        ):

            bonafide_row = (
                bonafide_by_capture[
                    capture_key
                ]
            )

            bonafide_path = verify_image(
                row=bonafide_row,
                dataset_root=dataset_root,
            )

            (
                bonafide_tensor,
                bonafide_original_size,
                bonafide_input_size,
            ) = prepare_image(
                path=bonafide_path,
                short_side=short_side,
                mean=mean,
                std=std,
            )

            (
                bonafide_logits,
                bonafide_activations,
            ) = capture.run(
                model=model,
                tensor=bonafide_tensor,
                device=device,
            )

            forward_image_count += 1

            bonafide_probabilities = (
                bonafide_logits[
                    0
                ]
                .softmax(
                    dim=0
                )
            )

            bonafide_top1 = int(
                bonafide_probabilities
                .argmax()
                .item()
            )

            for attack_row in sorted(
                attacks_by_capture[
                    capture_key
                ],
                key=lambda row:
                    row[
                        "variant"
                    ],
            ):

                attack_path = verify_image(
                    row=attack_row,
                    dataset_root=dataset_root,
                )

                (
                    attack_tensor,
                    attack_original_size,
                    attack_input_size,
                ) = prepare_image(
                    path=attack_path,
                    short_side=short_side,
                    mean=mean,
                    std=std,
                )

                if (
                    require_same_original
                    and
                    attack_original_size
                    != bonafide_original_size
                ):

                    raise ValueError(
                        "Attack/bona-fide original dimensions differ:\n"
                        f"  attack={attack_row['image_path']} "
                        f"{attack_original_size}\n"
                        f"  bonafide={bonafide_row['image_path']} "
                        f"{bonafide_original_size}"
                    )

                if (
                    require_same_input
                    and
                    attack_input_size
                    != bonafide_input_size
                ):

                    raise ValueError(
                        "Attack/bona-fide model-input dimensions differ:\n"
                        f"  attack={attack_input_size}\n"
                        f"  bonafide={bonafide_input_size}"
                    )

                (
                    attack_logits,
                    attack_activations,
                ) = capture.run(
                    model=model,
                    tensor=attack_tensor,
                    device=device,
                )

                forward_image_count += 1

                altered_field_set = (
                    altered_fields[
                        attack_row[
                            "image_path"
                        ]
                    ]
                )

                pair_id = (
                    f"{attack_row['file_stem']}::"
                    f"{attack_row['variant']}::"
                    f"{attack_row['hardware_source']}"
                )

                pair_stage_metrics: dict[
                    str,
                    dict[str, float],
                ] = {}

                # --------------------------------------------------
                # Stage-level similarity + channel concentration.
                # --------------------------------------------------

                for stage in STAGE_NAMES:

                    base_metrics = (
                        representation_metrics(
                            attack_activations[
                                stage
                            ],
                            bonafide_activations[
                                stage
                            ],
                        )
                    )

                    (
                        channel_metrics,
                        channel_difference,
                    ) = channel_difference_profile(
                        attack_activations[
                            stage
                        ],
                        bonafide_activations[
                            stage
                        ],
                        top_ks=concentration_top_ks,
                    )

                    combined_metrics = {
                        **base_metrics,
                        **channel_metrics,
                    }

                    pair_stage_metrics[
                        stage
                    ] = combined_metrics

                    stage_rows.append(
                        {
                            "pair_id":
                                pair_id,

                            "file_stem":
                                attack_row[
                                    "file_stem"
                                ],

                            "variant":
                                attack_row[
                                    "variant"
                                ],

                            "hardware_source":
                                attack_row[
                                    "hardware_source"
                                ],

                            "altered_fields_unique":
                                altered_field_set,

                            "attack_image_path":
                                attack_row[
                                    "image_path"
                                ],

                            "bonafide_image_path":
                                bonafide_row[
                                    "image_path"
                                ],

                            "stage":
                                stage,

                            "shape":
                                str(
                                    tuple(
                                        attack_activations[
                                            stage
                                        ].shape
                                    )
                                ),

                            **combined_metrics,
                        }
                    )

                    if stage == recurrence_stage:

                        recurrence_capture_records.append(
                            {
                                "file_stem":
                                    attack_row[
                                        "file_stem"
                                    ],

                                "variant":
                                    attack_row[
                                        "variant"
                                    ],

                                "hardware_source":
                                    attack_row[
                                        "hardware_source"
                                    ],

                                "altered_fields_unique":
                                    altered_field_set,

                                "channel_difference":
                                    (
                                        channel_difference
                                        .detach()
                                        .float()
                                        .cpu()
                                        .clone()
                                    ),
                            }
                        )

                # --------------------------------------------------
                # Derived layer4 -> avgpool behaviour.
                # --------------------------------------------------

                layer4_metrics = (
                    pair_stage_metrics[
                        "layer4.1"
                    ]
                )

                avgpool_metrics = (
                    pair_stage_metrics[
                        "avgpool"
                    ]
                )

                cosine_reconvergence = (
                    avgpool_metrics[
                        "cosine_similarity"
                    ]
                    -
                    layer4_metrics[
                        "cosine_similarity"
                    ]
                )

                relative_l2_reconvergence = (
                    layer4_metrics[
                        "relative_l2_distance"
                    ]
                    -
                    avgpool_metrics[
                        "relative_l2_distance"
                    ]
                )

                attack_pool = (
                    attack_activations[
                        "avgpool"
                    ]
                )

                bonafide_pool = (
                    bonafide_activations[
                        "avgpool"
                    ]
                )

                attack_probabilities = (
                    attack_logits[
                        0
                    ]
                    .softmax(
                        dim=0
                    )
                )

                attack_top1 = int(
                    attack_probabilities
                    .argmax()
                    .item()
                )

                top1_same = (
                    attack_top1
                    ==
                    bonafide_top1
                )

                pair_rows.append(
                    {
                        "pair_id":
                            pair_id,

                        "file_stem":
                            attack_row[
                                "file_stem"
                            ],

                        "variant":
                            attack_row[
                                "variant"
                            ],

                        "hardware_source":
                            attack_row[
                                "hardware_source"
                            ],

                        "altered_fields_unique":
                            altered_field_set,

                        "attack_image_path":
                            attack_row[
                                "image_path"
                            ],

                        "attack_image_sha256":
                            attack_row[
                                "image_sha256"
                            ],

                        "bonafide_image_path":
                            bonafide_row[
                                "image_path"
                            ],

                        "bonafide_image_sha256":
                            bonafide_row[
                                "image_sha256"
                            ],

                        "original_size":
                            str(
                                attack_original_size
                            ),

                        "model_input_size":
                            str(
                                attack_input_size
                            ),

                        "layer4_1_cosine":
                            layer4_metrics[
                                "cosine_similarity"
                            ],

                        "avgpool_cosine":
                            avgpool_metrics[
                                "cosine_similarity"
                            ],

                        "cosine_reconvergence":
                            cosine_reconvergence,

                        "layer4_1_relative_l2":
                            layer4_metrics[
                                "relative_l2_distance"
                            ],

                        "avgpool_relative_l2":
                            avgpool_metrics[
                                "relative_l2_distance"
                            ],

                        "relative_l2_reconvergence":
                            relative_l2_reconvergence,

                        "top_k_feature_overlap":
                            top_k_overlap(
                                attack_pool,
                                bonafide_pool,
                                top_k_features,
                            ),

                        "attack_top1_index":
                            attack_top1,

                        "attack_top1_name":
                            categories[
                                attack_top1
                            ],

                        "attack_top1_probability":
                            float(
                                attack_probabilities[
                                    attack_top1
                                ].item()
                            ),

                        "bonafide_top1_index":
                            bonafide_top1,

                        "bonafide_top1_name":
                            categories[
                                bonafide_top1
                            ],

                        "bonafide_top1_probability":
                            float(
                                bonafide_probabilities[
                                    bonafide_top1
                                ].item()
                            ),

                        "top1_same":
                            top1_same,

                        "top1_same_numeric":
                            float(
                                top1_same
                            ),
                    }
                )

                pair_count += 1

                if (
                    pair_count
                    % progress_every
                    == 0
                    or
                    pair_count
                    == expected_attack_pairs
                ):

                    logger.info(
                        "Progress: %d / %d attack pairs",
                        pair_count,
                        expected_attack_pairs,
                    )

                # Attack tensors are no longer needed.
                del attack_activations
                del attack_logits
                del attack_tensor

            # Bona-fide representation was reused for both attack
            # variants and can now be released.
            del bonafide_activations
            del bonafide_logits
            del bonafide_tensor

        if pair_count != expected_attack_pairs:

            raise RuntimeError(
                "Final attack-pair count mismatch: "
                f"expected={expected_attack_pairs}, "
                f"actual={pair_count}"
            )

        if device.type == "cuda":

            torch.cuda.synchronize(
                device
            )

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        # ----------------------------------------------------------
        # Build primary result tables.
        # ----------------------------------------------------------

        stage_df = pd.DataFrame(
            stage_rows
        )

        pair_df = pd.DataFrame(
            pair_rows
        )

        expected_stage_rows = (
            expected_attack_pairs
            *
            len(
                STAGE_NAMES
            )
        )

        if len(
            stage_df
        ) != expected_stage_rows:

            raise RuntimeError(
                "Stage metric row-count mismatch: "
                f"expected={expected_stage_rows}, "
                f"actual={len(stage_df)}"
            )

        card_stage_df = (
            build_card_variant_stage_metrics(
                stage_df,
                metric_names=all_stage_metrics,
            )
        )

        card_pair_df = (
            build_card_variant_summary(
                pair_df
            )
        )

        if len(
            card_pair_df
        ) != expected_card_variant_units:

            raise RuntimeError(
                "Card-variant unit count mismatch: "
                f"expected={expected_card_variant_units}, "
                f"actual={len(card_pair_df)}"
            )

        expected_card_stage_rows = (
            expected_card_variant_units
            *
            len(
                STAGE_NAMES
            )
        )

        if len(
            card_stage_df
        ) != expected_card_stage_rows:

            raise RuntimeError(
                "Card-variant stage row-count mismatch: "
                f"expected={expected_card_stage_rows}, "
                f"actual={len(card_stage_df)}"
            )

        # ----------------------------------------------------------
        # Channel recurrence.
        # ----------------------------------------------------------

        if len(
            recurrence_capture_records
        ) != expected_attack_pairs:

            raise RuntimeError(
                "Recurrence capture-record count mismatch: "
                f"expected={expected_attack_pairs}, "
                f"actual={len(recurrence_capture_records)}"
            )

        recurrence_card_records = (
            build_card_variant_channel_records(
                recurrence_capture_records
            )
        )

        if len(
            recurrence_card_records
        ) != expected_card_variant_units:

            raise RuntimeError(
                "Recurrence card-variant record count mismatch: "
                f"expected={expected_card_variant_units}, "
                f"actual={len(recurrence_card_records)}"
            )

        channel_frequency_df = (
            build_channel_frequency_table(
                capture_records=recurrence_capture_records,
                card_variant_records=recurrence_card_records,
                stage=recurrence_stage,
                top_k=recurrence_top_k,
            )
        )

        # ----------------------------------------------------------
        # Descriptive grouped summaries.
        # ----------------------------------------------------------

        group_rows: list[
            dict[str, Any]
        ] = []

        # Capture-pair stage results.
        append_stage_group_summary(
            group_rows,
            dataframe=stage_df,
            analysis_unit="capture_pair",
            group_type="overall",
            group_column=None,
            metric_names=grouped_stage_metrics,
        )

        for group_column in (
            "variant",
            "hardware_source",
            "altered_fields_unique",
        ):

            append_stage_group_summary(
                group_rows,
                dataframe=stage_df,
                analysis_unit="capture_pair",
                group_type=group_column,
                group_column=group_column,
                metric_names=grouped_stage_metrics,
            )

        # Card-variant stage results.
        append_stage_group_summary(
            group_rows,
            dataframe=card_stage_df,
            analysis_unit="card_variant",
            group_type="overall",
            group_column=None,
            metric_names=grouped_stage_metrics,
        )

        for group_column in (
            "variant",
            "altered_fields_unique",
        ):

            append_stage_group_summary(
                group_rows,
                dataframe=card_stage_df,
                analysis_unit="card_variant",
                group_type=group_column,
                group_column=group_column,
                metric_names=grouped_stage_metrics,
            )

        # Capture-pair derived layer4 -> avgpool behaviour.
        append_pair_group_summary(
            group_rows,
            dataframe=pair_df,
            analysis_unit="capture_pair",
            group_type="overall",
            group_column=None,
        )

        for group_column in (
            "variant",
            "hardware_source",
            "altered_fields_unique",
        ):

            append_pair_group_summary(
                group_rows,
                dataframe=pair_df,
                analysis_unit="capture_pair",
                group_type=group_column,
                group_column=group_column,
            )

        # Card-variant derived behaviour.
        append_pair_group_summary(
            group_rows,
            dataframe=card_pair_df,
            analysis_unit="card_variant",
            group_type="overall",
            group_column=None,
        )

        for group_column in (
            "variant",
            "altered_fields_unique",
        ):

            append_pair_group_summary(
                group_rows,
                dataframe=card_pair_df,
                analysis_unit="card_variant",
                group_type=group_column,
                group_column=group_column,
            )

        group_df = pd.DataFrame(
            group_rows
        )

        # ----------------------------------------------------------
        # Persist outputs.
        # ----------------------------------------------------------

        write_dataframe_exclusive(
            pair_df,
            output_paths[
                "pair_summary"
            ],
        )

        write_dataframe_exclusive(
            stage_df,
            output_paths[
                "pair_stage_metrics"
            ],
        )

        write_dataframe_exclusive(
            card_pair_df,
            output_paths[
                "card_variant_summary"
            ],
        )

        write_dataframe_exclusive(
            card_stage_df,
            output_paths[
                "card_variant_stage_metrics"
            ],
        )

        write_dataframe_exclusive(
            group_df,
            output_paths[
                "group_summary"
            ],
        )

        write_dataframe_exclusive(
            channel_frequency_df,
            output_paths[
                "layer4_channel_frequency"
            ],
        )

        # ----------------------------------------------------------
        # Main cohort findings.
        # ----------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "COHORT KEY FINDINGS"
        )

        logger.info(
            "=" * 72
        )

        log_pair_summary(
            logger=logger,
            pair_df=pair_df,
            label="ALL CAPTURE PAIRS",
        )

        for (
            variant,
            subset,
        ) in pair_df.groupby(
            "variant",
            sort=True,
        ):

            log_pair_summary(
                logger=logger,
                pair_df=subset,
                label=(
                    f"VARIANT {variant}"
                ),
            )

        for (
            hardware,
            subset,
        ) in pair_df.groupby(
            "hardware_source",
            sort=True,
        ):

            log_pair_summary(
                logger=logger,
                pair_df=subset,
                label=(
                    f"HARDWARE {hardware}"
                ),
            )

        # ----------------------------------------------------------
        # Channel concentration findings.
        # ----------------------------------------------------------

        logger.info(
            "-" * 72
        )

        logger.info(
            "CHANNEL-DIFFERENCE CONCENTRATION"
        )

        for (
            analysis_label,
            dataframe,
        ) in (
            (
                "CAPTURE PAIRS",
                stage_df,
            ),
            (
                "CARD-VARIANT UNITS",
                card_stage_df,
            ),
        ):

            stage_subset = dataframe[
                dataframe[
                    "stage"
                ]
                == recurrence_stage
            ]

            logger.info(
                "%s | %s",
                analysis_label,
                recurrence_stage,
            )

            for k in (
                concentration_top_ks
            ):

                column = (
                    f"channel_diff_top"
                    f"{k}_share"
                )

                logger.info(
                    "  median top-%d difference share: %.6f",
                    k,
                    stage_subset[
                        column
                    ].median(),
                )

            logger.info(
                "  median effective changed channels: %.3f",
                stage_subset[
                    "channel_diff_effective_channels"
                ].median(),
            )

            logger.info(
                "  median effective channel fraction: %.6f",
                stage_subset[
                    "channel_diff_effective_fraction"
                ].median(),
            )

        # ----------------------------------------------------------
        # Most recurrent layer4 channels, using card-variant units.
        # ----------------------------------------------------------

        primary_frequency = (
            channel_frequency_df[
                (
                    channel_frequency_df[
                        "analysis_unit"
                    ]
                    == "card_variant"
                )
                &
                (
                    channel_frequency_df[
                        "group_type"
                    ]
                    == "overall"
                )
            ]
            .sort_values(
                [
                    "frequency_rank",
                    "channel_index",
                ]
            )
            .head(
                15
            )
        )

        logger.info(
            "-" * 72
        )

        logger.info(
            "Most recurrent %s top-%d difference channels "
            "across card-variant units:",
            recurrence_stage,
            recurrence_top_k,
        )

        for (
            _,
            row,
        ) in primary_frequency.iterrows():

            logger.info(
                "  channel=%3d "
                "appearance=%3d/%3d "
                "(%.2f%%) "
                "mean_difference=%.6f",
                int(
                    row[
                        "channel_index"
                    ]
                ),
                int(
                    row[
                        "appearance_count"
                    ]
                ),
                int(
                    row[
                        "unit_count"
                    ]
                ),
                (
                    100.0
                    *
                    float(
                        row[
                            "appearance_rate"
                        ]
                    )
                ),
                float(
                    row[
                        "mean_channel_difference"
                    ]
                ),
            )

        # ----------------------------------------------------------
        # Runtime and output provenance.
        # ----------------------------------------------------------

        logger.info(
            "-" * 72
        )

        logger.info(
            "Capture-pair rows: %d",
            len(
                pair_df
            ),
        )

        logger.info(
            "Card-variant units: %d",
            len(
                card_pair_df
            ),
        )

        logger.info(
            "Capture-pair stage rows: %d",
            len(
                stage_df
            ),
        )

        logger.info(
            "Card-variant stage rows: %d",
            len(
                card_stage_df
            ),
        )

        logger.info(
            "Forward images: %d",
            forward_image_count,
        )

        logger.info(
            "Elapsed seconds: %.3f",
            elapsed,
        )

        logger.info(
            "End-to-end forward-image throughput: %.3f images/s",
            (
                forward_image_count
                /
                elapsed
            ),
        )

        logger.info(
            "End-to-end attack-pair throughput: %.3f pairs/s",
            (
                pair_count
                /
                elapsed
            ),
        )

        if device.type == "cuda":

            peak_allocated = (
                torch.cuda
                .max_memory_allocated(
                    device
                )
            )

            peak_reserved = (
                torch.cuda
                .max_memory_reserved(
                    device
                )
            )

            logger.info(
                "Peak CUDA allocated: %.3f GiB",
                (
                    peak_allocated
                    /
                    1024**3
                ),
            )

            logger.info(
                "Peak CUDA reserved: %.3f GiB",
                (
                    peak_reserved
                    /
                    1024**3
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
            "PRETRAINED RESNET-18 FULL PROJECT_TRAIN "
            "PAIRED COHORT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "PRETRAINED RESNET-18 FULL PROJECT_TRAIN "
            "PAIRED COHORT: FAIL"
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