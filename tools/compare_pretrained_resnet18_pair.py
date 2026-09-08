#!/usr/bin/env python3
"""
Quantitative paired comparison of ImageNet-pretrained ResNet-18
representations for one FantasyID attack / bona-fide pair.

Purpose:
- quantify whether the pretrained representation actually changes;
- compare representations at every residual stage;
- visualize spatial difference maps while alignment still exists;
- compare the complete 512-dimensional avgpool vectors;
- distinguish strong activation from actual FC-class contribution.

Important:
- ImageNet-pretrained ResNet-18 only;
- NO FantasyID fine-tuning;
- project_train images only;
- full document retained;
- short side resized to 256;
- NO center crop;
- ImageNet normalization;
- held-out test not accessed.

This is an exploratory representation diagnostic, not a forgery detector.
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

import matplotlib.pyplot as plt
import numpy as np
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
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config, load_machine_config


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}

    if not isinstance(value, dict):
        raise TypeError(
            f"Top level of YAML must be a mapping: {path}"
        )

    return value


def git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def configure_logger(
    config: dict[str, Any],
) -> tuple[logging.Logger, Path, str]:

    cfg = config["logging"]

    log_dir = resolve_repo_path(cfg["directory"])
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d_%H%M%S_%fZ"
    )

    filename = cfg["filename"].format(
        timestamp=timestamp
    )

    log_path = log_dir / filename

    level = getattr(
        logging,
        str(cfg["level"]).upper(),
    )

    logger = logging.getLogger(
        "pretrained_resnet18_pair"
    )

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)

    handler = logging.FileHandler(
        log_path,
        mode="x",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    formatter.converter = time.gmtime

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger, log_path, timestamp


# ----------------------------------------------------------------------
# Frozen manifest handling
# ----------------------------------------------------------------------

def load_project_train_manifest(
    experiment_cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], Path]:

    manifest_path = resolve_repo_path(
        experiment_cfg[
            "data"
        ][
            "frozen_split"
        ][
            "project_train"
        ][
            "path"
        ]
    )

    rows: dict[str, dict[str, str]] = {}

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            path = row["image_path"]

            if path in rows:
                raise ValueError(
                    f"Duplicate image_path: {path}"
                )

            rows[path] = row

    return rows, manifest_path


def get_manifest_row(
    requested_path: str,
    rows: dict[str, dict[str, str]],
) -> dict[str, str]:

    path = requested_path.replace("\\", "/")

    while path.startswith("./"):
        path = path[2:]

    if path not in rows:
        raise ValueError(
            "Image is not present in the frozen "
            "project_train manifest:\n"
            f"  {path}"
        )

    return rows[path]


def verify_image(
    *,
    row: dict[str, str],
    dataset_root: Path,
) -> Path:

    image_path = (
        dataset_root / row["image_path"]
    ).resolve()

    try:
        image_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            "Image path escapes dataset root."
        ) from exc

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    actual = sha256_file(image_path)
    expected = row["image_sha256"]

    if actual != expected:
        raise ValueError(
            "Source SHA-256 mismatch:\n"
            f"  image:    {row['image_path']}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    return image_path


# ----------------------------------------------------------------------
# Input preparation
# ----------------------------------------------------------------------

def prepare_image(
    *,
    path: Path,
    short_side: int,
    mean: list[float],
    std: list[float],
) -> tuple[
    Image.Image,
    Image.Image,
    torch.Tensor,
]:

    with Image.open(path) as opened:
        original = opened.convert("RGB")

    resized = TF.resize(
        original,
        size=short_side,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    tensor = (
        TF.pil_to_tensor(resized)
        .float()
        / 255.0
    )

    tensor = TF.normalize(
        tensor,
        mean=mean,
        std=std,
    )

    return (
        original,
        resized,
        tensor.unsqueeze(0),
    )


# ----------------------------------------------------------------------
# Activation capture
# ----------------------------------------------------------------------

def register_stage_hooks(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
) -> list[Any]:
    """
    Capture semantically useful stage boundaries:

    stem_relu
    maxpool
    layer1.0
    layer1.1
    layer2.0
    ...
    layer4.1
    avgpool
    """

    handles = []

    def hook_for(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:

            activations[name] = (
                output
                .detach()
                .float()
                .cpu()
                .clone()
            )

        return hook

    handles.append(
        model.relu.register_forward_hook(
            hook_for("stem_relu")
        )
    )

    handles.append(
        model.maxpool.register_forward_hook(
            hook_for("maxpool")
        )
    )

    for name, module in model.named_modules():

        if isinstance(module, BasicBlock):

            handles.append(
                module.register_forward_hook(
                    hook_for(name)
                )
            )

    handles.append(
        model.avgpool.register_forward_hook(
            hook_for("avgpool")
        )
    )

    return handles


def forward_capture(
    *,
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:

    activations: dict[
        str,
        torch.Tensor,
    ] = {}

    handles = register_stage_hooks(
        model,
        activations,
    )

    try:

        with torch.inference_mode():

            logits = model(
                tensor.to(device)
            ).cpu()

    finally:

        for handle in handles:
            handle.remove()

    return logits, activations


# ----------------------------------------------------------------------
# Similarity metrics
# ----------------------------------------------------------------------

def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    a = a.reshape(-1)
    b = b.reshape(-1)

    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
    )


def euclidean_distance(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    return float(
        torch.linalg.vector_norm(
            a.reshape(-1)
            - b.reshape(-1)
        ).item()
    )


def relative_l2_distance(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """
    Normalize L2 difference by the average representation norm.
    """

    a = a.reshape(-1)
    b = b.reshape(-1)

    difference = torch.linalg.vector_norm(
        a - b
    )

    scale = (
        torch.linalg.vector_norm(a)
        + torch.linalg.vector_norm(b)
    ) / 2.0

    if scale.item() == 0:
        return 0.0

    return float(
        (difference / scale).item()
    )


def pearson_correlation(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    x = a.reshape(-1).numpy()
    y = b.reshape(-1).numpy()

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def representation_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float]:

    if a.shape != b.shape:
        raise ValueError(
            "Representation shapes differ:\n"
            f"  attack:   {tuple(a.shape)}\n"
            f"  bonafide: {tuple(b.shape)}"
        )

    difference = (
        a - b
    ).abs()

    return {
        "cosine_similarity":
            cosine_similarity(a, b),

        "pearson_correlation":
            pearson_correlation(a, b),

        "euclidean_distance":
            euclidean_distance(a, b),

        "relative_l2_distance":
            relative_l2_distance(a, b),

        "mean_absolute_difference":
            float(
                difference.mean().item()
            ),

        "max_absolute_difference":
            float(
                difference.max().item()
            ),
    }


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------

def robust_normalize(
    array: np.ndarray,
) -> np.ndarray:

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    low = float(
        np.percentile(array, 1)
    )

    high = float(
        np.percentile(array, 99)
    )

    if high <= low:

        low = float(array.min())
        high = float(array.max())

    if high <= low:

        return np.zeros_like(
            array
        )

    result = (
        array - low
    ) / (
        high - low
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def spatial_difference_map(
    a: torch.Tensor,
    b: torch.Tensor,
) -> np.ndarray:
    """
    Mean absolute attack-vs-bona-fide difference across channels.

    This says WHERE the representations differ most.

    It does NOT say whether that difference affects a prediction.
    """

    difference = (
        a[0] - b[0]
    ).abs().mean(
        dim=0
    )

    return difference.numpy()


def save_stage_difference_overview(
    *,
    attack_activations: dict[str, torch.Tensor],
    bonafide_activations: dict[str, torch.Tensor],
    output_path: Path,
    dpi: int,
) -> None:

    stage_names = [
        name
        for name in attack_activations
        if name != "avgpool"
    ]

    ncols = 3
    nrows = math.ceil(
        len(stage_names)
        / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            5 * ncols,
            3.6 * nrows,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, name in zip(
        axes,
        stage_names,
    ):

        attack = attack_activations[
            name
        ]

        bonafide = bonafide_activations[
            name
        ]

        difference = (
            spatial_difference_map(
                attack,
                bonafide,
            )
        )

        raw_mean = float(
            difference.mean()
        )

        axis.imshow(
            robust_normalize(
                difference
            )
        )

        axis.set_title(
            f"{name}\n"
            f"{tuple(attack.shape)}\n"
            f"mean |Δ|={raw_mean:.4f}"
        )

        axis.axis("off")

    fig.suptitle(
        "Attack vs bona-fide ResNet-18 representation difference\n"
        "mean absolute channel difference; "
        "visual normalization is per-panel",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_avgpool_scatter(
    *,
    attack_vector: np.ndarray,
    bonafide_vector: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:

    fig, axis = plt.subplots(
        figsize=(8, 8)
    )

    axis.scatter(
        attack_vector,
        bonafide_vector,
        s=18,
    )

    low = min(
        attack_vector.min(),
        bonafide_vector.min(),
    )

    high = max(
        attack_vector.max(),
        bonafide_vector.max(),
    )

    axis.plot(
        [low, high],
        [low, high],
        linestyle="--",
    )

    axis.set_xlabel(
        "Attack pooled activation"
    )

    axis.set_ylabel(
        "Bona-fide pooled activation"
    )

    axis.set_title(
        "512-dimensional pretrained ResNet-18 representation\n"
        "each point = one pooled feature"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_top_avgpool_changes(
    *,
    attack_vector: np.ndarray,
    bonafide_vector: np.ndarray,
    top_k: int,
    output_path: Path,
    dpi: int,
) -> None:

    delta = (
        attack_vector
        - bonafide_vector
    )

    indices = np.argsort(
        np.abs(delta)
    )[-top_k:][::-1]

    positions = np.arange(
        len(indices)
    )

    width = 0.38

    fig, axis = plt.subplots(
        figsize=(14, 6)
    )

    axis.bar(
        positions - width / 2,
        attack_vector[indices],
        width=width,
        label="attack",
    )

    axis.bar(
        positions + width / 2,
        bonafide_vector[indices],
        width=width,
        label="bona fide",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        indices,
        rotation=90,
    )

    axis.set_xlabel(
        "Feature index"
    )

    axis.set_ylabel(
        "Pooled activation"
    )

    axis.set_title(
        "Pooled features with largest attack-vs-bona-fide change"
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


# ----------------------------------------------------------------------
# FC contribution analysis
# ----------------------------------------------------------------------

def class_feature_contributions(
    *,
    vector: torch.Tensor,
    model: nn.Module,
    class_index: int,
) -> torch.Tensor:
    """
    Contribution from pooled feature c to class k:

        contribution[c] = W[k,c] * z[c]

    Bias is separate and is not attributed to an input feature.
    """

    z = vector.reshape(-1)

    weights = (
        model.fc.weight[
            class_index
        ]
        .detach()
        .cpu()
    )

    return weights * z


def save_fc_contribution_difference(
    *,
    attack_contributions: torch.Tensor,
    bonafide_contributions: torch.Tensor,
    class_name: str,
    top_k: int,
    output_path: Path,
    dpi: int,
) -> None:

    attack = attack_contributions.numpy()
    bonafide = bonafide_contributions.numpy()

    delta = attack - bonafide

    indices = np.argsort(
        np.abs(delta)
    )[-top_k:][::-1]

    positions = np.arange(
        len(indices)
    )

    width = 0.38

    fig, axis = plt.subplots(
        figsize=(14, 6)
    )

    axis.bar(
        positions - width / 2,
        attack[indices],
        width=width,
        label="attack",
    )

    axis.bar(
        positions + width / 2,
        bonafide[indices],
        width=width,
        label="bona fide",
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        indices,
        rotation=90,
    )

    axis.set_xlabel(
        "Pooled feature index"
    )

    axis.set_ylabel(
        "FC contribution = activation × class weight"
    )

    axis.set_title(
        f"Features with largest contribution change\n"
        f"ImageNet class: {class_name}"
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


# ----------------------------------------------------------------------
# CSV outputs
# ----------------------------------------------------------------------

def save_stage_metrics(
    *,
    path: Path,
    attack_activations: dict[str, torch.Tensor],
    bonafide_activations: dict[str, torch.Tensor],
) -> None:

    fieldnames = [
        "stage",
        "shape",
        "cosine_similarity",
        "pearson_correlation",
        "euclidean_distance",
        "relative_l2_distance",
        "mean_absolute_difference",
        "max_absolute_difference",
    ]

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for stage in attack_activations:

            metrics = representation_metrics(
                attack_activations[stage],
                bonafide_activations[stage],
            )

            writer.writerow(
                {
                    "stage": stage,
                    "shape": str(
                        tuple(
                            attack_activations[
                                stage
                            ].shape
                        )
                    ),
                    **metrics,
                }
            )


def save_avgpool_csv(
    *,
    path: Path,
    attack: torch.Tensor,
    bonafide: torch.Tensor,
    attack_contrib: torch.Tensor | None,
    bonafide_contrib: torch.Tensor | None,
) -> None:

    attack = attack.reshape(-1)
    bonafide = bonafide.reshape(-1)

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "feature_index",
                "attack_activation",
                "bonafide_activation",
                "activation_delta",
                "attack_fc_contribution",
                "bonafide_fc_contribution",
                "fc_contribution_delta",
            ]
        )

        for index in range(
            attack.numel()
        ):

            if attack_contrib is None:

                a_contrib = ""
                b_contrib = ""
                contrib_delta = ""

            else:

                a_contrib = float(
                    attack_contrib[index]
                )

                b_contrib = float(
                    bonafide_contrib[index]
                )

                contrib_delta = (
                    a_contrib
                    - b_contrib
                )

            writer.writerow(
                [
                    index,
                    float(attack[index]),
                    float(bonafide[index]),
                    float(
                        attack[index]
                        - bonafide[index]
                    ),
                    a_contrib,
                    b_contrib,
                    contrib_delta,
                ]
            )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Compare an aligned FantasyID attack / bona-fide pair "
            "through ImageNet-pretrained ResNet-18."
        )
    )

    parser.add_argument(
        "--attack",
        required=True,
        help=(
            "Exact image_path from frozen project_train manifest."
        ),
    )

    parser.add_argument(
        "--bonafide",
        required=True,
        help=(
            "Exact image_path from frozen project_train manifest."
        ),
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
        "--comparison-config",
        default=(
            "tools/"
            "compare_pretrained_resnet18_pair_config.yaml"
        ),
    )

    args = parser.parse_args()

    comparison_config_path = (
        resolve_repo_path(
            args.comparison_config
        )
    )

    comparison_cfg = load_yaml(
        comparison_config_path
    )

    logger, log_path, timestamp = (
        configure_logger(
            comparison_cfg
        )
    )

    try:

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

        dataset_root = Path(
            machine_cfg[
                "paths"
            ][
                "dataset_root"
            ]
        ).expanduser().resolve()

        manifest_rows, manifest_path = (
            load_project_train_manifest(
                experiment_cfg
            )
        )

        attack_row = get_manifest_row(
            args.attack,
            manifest_rows,
        )

        bonafide_row = get_manifest_row(
            args.bonafide,
            manifest_rows,
        )

        if attack_row["traffic_type"] != "attack":

            raise ValueError(
                "--attack image is not labelled attack."
            )

        if bonafide_row["traffic_type"] != "bonafide":

            raise ValueError(
                "--bonafide image is not labelled bonafide."
            )

        cfg = comparison_cfg[
            "comparison"
        ]

        if cfg.get(
            "require_same_file_stem",
            True,
        ):

            if (
                attack_row["file_stem"]
                != bonafide_row["file_stem"]
            ):

                raise ValueError(
                    "Pair does not share the same file_stem."
                )

        if cfg.get(
            "require_same_hardware_source",
            True,
        ):

            if (
                attack_row["hardware_source"]
                != bonafide_row["hardware_source"]
            ):

                raise ValueError(
                    "Pair does not share the same hardware source."
                )

        attack_path = verify_image(
            row=attack_row,
            dataset_root=dataset_root,
        )

        bonafide_path = verify_image(
            row=bonafide_row,
            dataset_root=dataset_root,
        )

        weights_name = cfg[
            "weights"
        ]

        if weights_name != "IMAGENET1K_V1":

            raise ValueError(
                "Current comparison is locked to "
                "IMAGENET1K_V1."
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

        model.to(device)

        preset = weights.transforms()

        mean = list(
            preset.mean
        )

        std = list(
            preset.std
        )

        short_side = int(
            cfg["resize_short_side"]
        )

        (
            attack_original,
            attack_resized,
            attack_tensor,
        ) = prepare_image(
            path=attack_path,
            short_side=short_side,
            mean=mean,
            std=std,
        )

        (
            bonafide_original,
            bonafide_resized,
            bonafide_tensor,
        ) = prepare_image(
            path=bonafide_path,
            short_side=short_side,
            mean=mean,
            std=std,
        )

        if cfg.get(
            "require_same_original_size",
            True,
        ):

            if (
                attack_original.size
                != bonafide_original.size
            ):

                raise ValueError(
                    "Original image dimensions differ; "
                    "strict spatial difference maps would "
                    "not be directly aligned."
                )

        if (
            attack_resized.size
            != bonafide_resized.size
        ):

            raise ValueError(
                "Model-input dimensions differ."
            )

        (
            attack_logits,
            attack_activations,
        ) = forward_capture(
            model=model,
            tensor=attack_tensor,
            device=device,
        )

        (
            bonafide_logits,
            bonafide_activations,
        ) = forward_capture(
            model=model,
            tensor=bonafide_tensor,
            device=device,
        )

        if (
            attack_activations.keys()
            != bonafide_activations.keys()
        ):

            raise RuntimeError(
                "Captured stage sets differ."
            )

        output_dir = (
            REPO_ROOT
            / "runs"
            / (
                "exploratory_pretrained_resnet18_pair_"
                f"{timestamp}"
            )
            / "diagnostics"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        logger.info("=" * 72)
        logger.info(
            "PRETRAINED RESNET-18 PAIRED REPRESENTATION COMPARISON"
        )
        logger.info("=" * 72)

        logger.info(
            "Git commit: %s",
            git_commit_sha(),
        )

        logger.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        logger.info(
            "Comparison config SHA-256: %s",
            sha256_file(
                comparison_config_path
            ),
        )

        logger.info(
            "project_train manifest SHA-256: %s",
            sha256_file(
                manifest_path
            ),
        )

        logger.info(
            "Attack: %s",
            attack_row["image_path"],
        )

        logger.info(
            "Attack SHA-256: %s",
            attack_row["image_sha256"],
        )

        logger.info(
            "Bona fide: %s",
            bonafide_row["image_path"],
        )

        logger.info(
            "Bona-fide SHA-256: %s",
            bonafide_row["image_sha256"],
        )

        logger.info(
            "Original dimensions: %s",
            attack_original.size,
        )

        logger.info(
            "Model-input dimensions: %s",
            attack_resized.size,
        )

        logger.info(
            "Resize short side=%d, aspect ratio preserved",
            short_side,
        )

        logger.info(
            "Crop=NONE"
        )

        logger.info(
            "Held-out test=NOT ACCESSED"
        )

        # --------------------------------------------------------------
        # Stage-level numerical comparison
        # --------------------------------------------------------------

        logger.info("-" * 72)
        logger.info(
            "STAGE REPRESENTATION SIMILARITY"
        )

        for stage in attack_activations:

            metrics = representation_metrics(
                attack_activations[stage],
                bonafide_activations[stage],
            )

            logger.info(
                "%-10s shape=%-18s "
                "cos=%.6f corr=%.6f relL2=%.6f "
                "mean|delta|=%.6f",
                stage,
                str(
                    tuple(
                        attack_activations[
                            stage
                        ].shape
                    )
                ),
                metrics[
                    "cosine_similarity"
                ],
                metrics[
                    "pearson_correlation"
                ],
                metrics[
                    "relative_l2_distance"
                ],
                metrics[
                    "mean_absolute_difference"
                ],
            )

        save_stage_metrics(
            path=(
                output_dir
                / "stage_similarity.csv"
            ),
            attack_activations=attack_activations,
            bonafide_activations=bonafide_activations,
        )

        save_stage_difference_overview(
            attack_activations=attack_activations,
            bonafide_activations=bonafide_activations,
            output_path=(
                output_dir
                / "stage_difference_overview.png"
            ),
            dpi=int(
                cfg["dpi"]
            ),
        )

        # --------------------------------------------------------------
        # Final pooled representation
        # --------------------------------------------------------------

        attack_pool = (
            attack_activations[
                "avgpool"
            ].reshape(-1)
        )

        bonafide_pool = (
            bonafide_activations[
                "avgpool"
            ].reshape(-1)
        )

        pool_metrics = representation_metrics(
            attack_pool,
            bonafide_pool,
        )

        logger.info("-" * 72)
        logger.info(
            "FINAL 512-DIMENSIONAL REPRESENTATION"
        )

        for name, value in pool_metrics.items():

            logger.info(
                "%-28s %.8f",
                name,
                value,
            )

        top_k = int(
            cfg["top_k_features"]
        )

        attack_top = set(
            torch.topk(
                attack_pool.abs(),
                k=top_k,
            ).indices.tolist()
        )

        bonafide_top = set(
            torch.topk(
                bonafide_pool.abs(),
                k=top_k,
            ).indices.tolist()
        )

        overlap = (
            len(
                attack_top
                & bonafide_top
            )
            / top_k
        )

        logger.info(
            "Top-%d pooled-feature overlap: %.4f (%d/%d)",
            top_k,
            overlap,
            len(
                attack_top
                & bonafide_top
            ),
            top_k,
        )

        save_avgpool_scatter(
            attack_vector=(
                attack_pool.numpy()
            ),
            bonafide_vector=(
                bonafide_pool.numpy()
            ),
            output_path=(
                output_dir
                / "avgpool_attack_vs_bonafide_scatter.png"
            ),
            dpi=int(
                cfg["dpi"]
            ),
        )

        save_top_avgpool_changes(
            attack_vector=(
                attack_pool.numpy()
            ),
            bonafide_vector=(
                bonafide_pool.numpy()
            ),
            top_k=top_k,
            output_path=(
                output_dir
                / "avgpool_largest_changes.png"
            ),
            dpi=int(
                cfg["dpi"]
            ),
        )

        # --------------------------------------------------------------
        # ImageNet predictions
        # --------------------------------------------------------------

        categories = weights.meta[
            "categories"
        ]

        attack_probabilities = (
            attack_logits[0]
            .softmax(dim=0)
        )

        bonafide_probabilities = (
            bonafide_logits[0]
            .softmax(dim=0)
        )

        attack_top_class = int(
            attack_probabilities.argmax()
        )

        bonafide_top_class = int(
            bonafide_probabilities.argmax()
        )

        logger.info("-" * 72)
        logger.info(
            "IMAGENET TOP-1 DIAGNOSTIC"
        )

        logger.info(
            "Attack: class=%d name=%s probability=%.6f",
            attack_top_class,
            categories[
                attack_top_class
            ],
            float(
                attack_probabilities[
                    attack_top_class
                ]
            ),
        )

        logger.info(
            "Bona fide: class=%d name=%s probability=%.6f",
            bonafide_top_class,
            categories[
                bonafide_top_class
            ],
            float(
                bonafide_probabilities[
                    bonafide_top_class
                ]
            ),
        )

        # --------------------------------------------------------------
        # FC contribution analysis
        # --------------------------------------------------------------

        attack_contrib = None
        bonafide_contrib = None

        if (
            attack_top_class
            == bonafide_top_class
        ):

            common_class = (
                attack_top_class
            )

            class_name = categories[
                common_class
            ]

            logger.info(
                "Both inputs share top-1 ImageNet class: %s",
                class_name,
            )

            attack_contrib = (
                class_feature_contributions(
                    vector=attack_pool,
                    model=model,
                    class_index=common_class,
                )
            )

            bonafide_contrib = (
                class_feature_contributions(
                    vector=bonafide_pool,
                    model=model,
                    class_index=common_class,
                )
            )

            contribution_delta = (
                attack_contrib
                - bonafide_contrib
            )

            important = torch.topk(
                contribution_delta.abs(),
                k=top_k,
            ).indices.tolist()

            logger.info(
                "Largest changes in actual FC contribution "
                "(activation x class weight):"
            )

            for index in important[:10]:

                logger.info(
                    "  feature=%3d "
                    "attack=% .6f "
                    "bonafide=% .6f "
                    "delta=% .6f",
                    index,
                    float(
                        attack_contrib[index]
                    ),
                    float(
                        bonafide_contrib[index]
                    ),
                    float(
                        contribution_delta[index]
                    ),
                )

            save_fc_contribution_difference(
                attack_contributions=attack_contrib,
                bonafide_contributions=bonafide_contrib,
                class_name=class_name,
                top_k=top_k,
                output_path=(
                    output_dir
                    / "fc_contribution_largest_changes.png"
                ),
                dpi=int(
                    cfg["dpi"]
                ),
            )

        else:

            logger.warning(
                "Top-1 ImageNet classes differ. "
                "A common-class FC contribution comparison "
                "was not generated."
            )

        # --------------------------------------------------------------
        # Complete 512-feature CSV
        # --------------------------------------------------------------

        save_avgpool_csv(
            path=(
                output_dir
                / "avgpool_pair.csv"
            ),
            attack=attack_pool,
            bonafide=bonafide_pool,
            attack_contrib=attack_contrib,
            bonafide_contrib=bonafide_contrib,
        )

        # --------------------------------------------------------------
        # Pair metadata
        # --------------------------------------------------------------

        metadata = {
            "schema_version": 1,

            "model": {
                "architecture": "resnet18",
                "weights": "IMAGENET1K_V1",
                "fine_tuned_on_fantasyid": False,
            },

            "preprocessing": {
                "resize_short_side": short_side,
                "preserve_aspect_ratio": True,
                "crop": None,
                "normalization": "ImageNet",
            },

            "attack": {
                "image_path":
                    attack_row["image_path"],
                "image_sha256":
                    attack_row["image_sha256"],
                "original_size_wh":
                    list(
                        attack_original.size
                    ),
            },

            "bonafide": {
                "image_path":
                    bonafide_row["image_path"],
                "image_sha256":
                    bonafide_row["image_sha256"],
                "original_size_wh":
                    list(
                        bonafide_original.size
                    ),
            },

            "avgpool_metrics":
                pool_metrics,

            "top_k_feature_overlap":
                overlap,

            "interpretation": {
                "activation_difference_is_not_prediction_importance":
                    True,

                "fc_contribution_definition":
                    "pooled_activation * ImageNet_fc_weight",
            },
        }

        with (
            output_dir
            / "pair_metadata.yaml"
        ).open(
            "x",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                metadata,
                f,
                sort_keys=False,
            )

        logger.info("=" * 72)
        logger.info(
            "PRETRAINED RESNET-18 PAIRED "
            "REPRESENTATION COMPARISON: PASS"
        )
        logger.info("=" * 72)

        logger.info(
            "Output directory: %s",
            output_dir,
        )

        return 0

    except Exception:

        logger.exception(
            "PRETRAINED RESNET-18 PAIRED "
            "REPRESENTATION COMPARISON: FAIL"
        )

        return 1

    finally:

        for handler in logger.handlers:
            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())