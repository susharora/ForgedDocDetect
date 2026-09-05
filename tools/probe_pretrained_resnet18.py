#!/usr/bin/env python3
"""
Explore ImageNet-pretrained ResNet-18 activations on frozen FantasyID
project_train images before any FantasyID fine-tuning.

Important:
- uses ResNet18_Weights.IMAGENET1K_V1;
- model remains completely unchanged;
- only frozen project_train images are permitted;
- source image SHA-256 is verified against the frozen manifest;
- full document is retained;
- short side is resized to 256 while preserving aspect ratio;
- NO center crop is applied;
- ImageNet normalization is applied;
- spatial activations are visualised layer-by-layer;
- final 512-dimensional avgpool representation is visualised separately.

These visualisations show activation, NOT causal feature importance.
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
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

import torchvision
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
        "pretrained_resnet18_probe"
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
            image_path = row["image_path"]

            if image_path in rows:
                raise ValueError(
                    f"Duplicate image_path in manifest: "
                    f"{image_path}"
                )

            rows[image_path] = row

    return rows, manifest_path


def normalize_requested_path(value: str) -> str:
    value = value.replace("\\", "/")

    while value.startswith("./"):
        value = value[2:]

    return value


def resolve_requested_images(
    *,
    cli_images: list[str],
    probe_config: dict[str, Any],
    manifest_rows: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    configured = probe_config[
        "probe"
    ].get("images", [])

    requested = [
        normalize_requested_path(x)
        for x in configured + cli_images
    ]

    if not requested:
        raise ValueError(
            "No images requested. Supply one or more "
            "--image arguments or configure probe.images."
        )

    # Preserve order while removing duplicates.
    requested = list(dict.fromkeys(requested))

    selected: list[dict[str, str]] = []

    for image_path in requested:
        if image_path not in manifest_rows:
            raise ValueError(
                "Requested image is not present in the "
                "frozen project_train manifest:\n"
                f"  {image_path}\n\n"
                "This probe deliberately refuses dev_val "
                "and held-out test images."
            )

        selected.append(
            manifest_rows[image_path]
        )

    return selected


def checkpoint_provenance(
    weights: ResNet18_Weights,
) -> dict[str, str | None]:
    filename = Path(
        urlparse(weights.url).path
    ).name

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / filename
    )

    if checkpoint.is_file():
        return {
            "url": weights.url,
            "cached_path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }

    return {
        "url": weights.url,
        "cached_path": None,
        "sha256": None,
    }


def resize_full_document(
    image: Image.Image,
    short_side: int,
) -> Image.Image:
    """
    Preserve the complete document and aspect ratio.

    Passing an integer to torchvision resize makes the shorter
    side equal to that value while preserving aspect ratio.
    """
    return TF.resize(
        image,
        size=short_side,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def prepare_tensor(
    image: Image.Image,
    *,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    tensor = TF.pil_to_tensor(image).float() / 255.0

    tensor = TF.normalize(
        tensor,
        mean=mean,
        std=std,
    )

    return tensor.unsqueeze(0)


def register_activation_hooks(
    model: nn.Module,
    capture_cfg: dict[str, Any],
    activations: dict[str, torch.Tensor],
) -> list[Any]:
    """
    Capture:
      - stem output;
      - every Conv2d output if requested;
      - every residual BasicBlock output;
      - maxpool;
      - avgpool.

    Hook tensors are immediately cloned to CPU so the diagnostic
    does not retain the GPU computation graph.
    """

    handles = []

    def make_hook(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            activations[name] = (
                output.detach()
                .float()
                .cpu()
                .clone()
            )

        return hook

    if capture_cfg.get("stem", True):
        handles.append(
            model.relu.register_forward_hook(
                make_hook("stem_relu")
            )
        )

    if capture_cfg.get(
        "internal_convolutions",
        True,
    ):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(
                    module.register_forward_hook(
                        make_hook(
                            f"conv::{name}"
                        )
                    )
                )

    if capture_cfg.get(
        "residual_block_outputs",
        True,
    ):
        for name, module in model.named_modules():
            if isinstance(module, BasicBlock):
                handles.append(
                    module.register_forward_hook(
                        make_hook(
                            f"block::{name}"
                        )
                    )
                )

    if capture_cfg.get("maxpool", True):
        handles.append(
            model.maxpool.register_forward_hook(
                make_hook("maxpool")
            )
        )

    handles.append(
        model.avgpool.register_forward_hook(
            make_hook("avgpool")
        )
    )

    return handles


def safe_name(name: str) -> str:
    return (
        name.replace("::", "__")
        .replace(".", "_")
        .replace("/", "_")
    )


def robust_normalize(
    array: np.ndarray,
) -> np.ndarray:
    """
    Visualization normalization only.

    Uses robust percentiles so a few extreme pixels do not flatten
    the visible structure.
    """
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
            array,
            dtype=np.float32,
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


def save_top_channels(
    *,
    name: str,
    activation: torch.Tensor,
    output_dir: Path,
    top_k: int,
    dpi: int,
) -> None:
    """
    Save the most strongly responding channels for one spatial
    activation tensor.

    Ranking is by mean absolute activation over H x W.

    This means "strong response", NOT "importance to prediction".
    """
    if activation.ndim != 4:
        return

    tensor = activation[0]

    if tensor.shape[-2:] == (1, 1):
        return

    scores = tensor.abs().mean(
        dim=(1, 2)
    )

    k = min(
        top_k,
        tensor.shape[0],
    )

    indices = torch.topk(
        scores,
        k=k,
    ).indices.tolist()

    ncols = 4
    nrows = math.ceil(
        k / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            3.2 * ncols,
            3.0 * nrows,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, channel_index in zip(
        axes,
        indices,
    ):
        channel = (
            tensor[channel_index]
            .abs()
            .numpy()
        )

        axis.imshow(
            robust_normalize(
                channel
            )
        )

        axis.set_title(
            f"ch {channel_index}\n"
            f"mean|a|="
            f"{scores[channel_index]:.4f}"
        )
        axis.axis("off")

    fig.suptitle(
        f"{name}\n"
        f"shape={tuple(activation.shape)} "
        "| strongest channels by mean |activation|",
        fontsize=11,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{safe_name(name)}__top_channels.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def spatial_magnitude_map(
    activation: torch.Tensor,
) -> np.ndarray:
    tensor = activation[0]

    magnitude = tensor.abs().mean(
        dim=0
    )

    return robust_normalize(
        magnitude.numpy()
    )


def save_network_overview(
    *,
    activations: dict[str, torch.Tensor],
    resized_image: Image.Image,
    output_path: Path,
    dpi: int,
) -> None:
    """
    One-page progression through the network.

    To keep this readable, the overview uses:
      stem,
      maxpool,
      residual block outputs.

    Detailed Conv2d channel grids are saved separately.
    """
    overview_names = [
        name
        for name in activations
        if (
            name == "stem_relu"
            or name == "maxpool"
            or name.startswith("block::")
        )
    ]

    n_items = 1 + len(
        overview_names
    )

    ncols = 3
    nrows = math.ceil(
        n_items / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            5.0 * ncols,
            3.7 * nrows,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis in axes:
        axis.axis("off")

    axes[0].imshow(
        resized_image
    )
    axes[0].set_title(
        "Full document model input\nNO CROP"
    )
    axes[0].axis("off")

    for axis, name in zip(
        axes[1:],
        overview_names,
    ):
        activation = activations[
            name
        ]

        axis.imshow(
            spatial_magnitude_map(
                activation
            )
        )

        axis.set_title(
            f"{name}\n"
            f"{tuple(activation.shape)}"
        )
        axis.axis("off")

    fig.suptitle(
        "ResNet-18 feature progression\n"
        "mean absolute activation across channels "
        "(activation magnitude, NOT importance)",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_avgpool_plots(
    *,
    avgpool: torch.Tensor,
    output_dir: Path,
    dpi: int,
) -> None:
    vector = (
        avgpool[0]
        .reshape(-1)
        .numpy()
    )

    # Full 512-dimensional representation.
    fig, axis = plt.subplots(
        figsize=(14, 5)
    )

    axis.plot(
        np.arange(
            len(vector)
        ),
        vector,
    )

    axis.set_xlabel(
        "Pooled feature index"
    )
    axis.set_ylabel(
        "Activation"
    )
    axis.set_title(
        "ResNet-18 final average-pooled representation\n"
        f"{len(vector)} features; spatial dimensions have been removed"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "avgpool_512_feature_vector.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Strongest pooled features.
    top_k = min(
        32,
        len(vector),
    )

    indices = np.argsort(
        np.abs(vector)
    )[-top_k:][::-1]

    fig, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.bar(
        np.arange(top_k),
        vector[indices],
    )

    axis.set_xticks(
        np.arange(top_k)
    )
    axis.set_xticklabels(
        indices,
        rotation=90,
    )

    axis.set_xlabel(
        "Feature index"
    )
    axis.set_ylabel(
        "Activation"
    )

    axis.set_title(
        "Strongest final pooled ResNet-18 features"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "avgpool_top32_features.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_imagenet_predictions(
    *,
    logits: torch.Tensor,
    categories: list[str],
    output_dir: Path,
    top_k: int,
    dpi: int,
    logger: logging.Logger,
) -> None:
    probabilities = logits[
        0
    ].softmax(
        dim=0
    ).cpu()

    values, indices = torch.topk(
        probabilities,
        k=top_k,
    )

    labels = [
        categories[i]
        for i in indices.tolist()
    ]

    probs = values.numpy()

    logger.info(
        "Top ImageNet predictions "
        "(exploratory because center crop was deliberately omitted):"
    )

    for rank, (
        label,
        probability,
    ) in enumerate(
        zip(
            labels,
            probs,
        ),
        start=1,
    ):
        logger.info(
            "  %2d | %-30s %.6f",
            rank,
            label,
            probability,
        )

    fig, axis = plt.subplots(
        figsize=(10, 6)
    )

    order = np.arange(
        len(labels)
    )

    axis.barh(
        order,
        probs,
    )

    axis.set_yticks(
        order
    )
    axis.set_yticklabels(
        labels
    )
    axis.invert_yaxis()

    axis.set_xlabel(
        "Softmax probability"
    )
    axis.set_title(
        "ImageNet-pretrained ResNet-18 predictions\n"
        "diagnostic only: full document retained, no center crop"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "imagenet_top_predictions.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_metadata(
    *,
    path: Path,
    row: dict[str, str],
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
    model_checkpoint: dict[str, str | None],
    weights_name: str,
) -> None:
    metadata = {
        "image": {
            "path": row["image_path"],
            "sha256": row["image_sha256"],
            "traffic_type": row["traffic_type"],
            "variant": row["variant"],
            "hardware_source": row["hardware_source"],
            "original_size_wh": list(
                original_size
            ),
            "model_input_size_wh": list(
                resized_size
            ),
        },
        "model": {
            "architecture": "resnet18",
            "weights": weights_name,
            "fine_tuned_on_fantasyid": False,
            "checkpoint": model_checkpoint,
        },
        "preprocessing": {
            "rgb": True,
            "resize_short_side": 256,
            "preserve_aspect_ratio": True,
            "crop": None,
            "normalization": "ImageNet ResNet18 weight normalization",
        },
        "interpretation_warning": (
            "Feature activation magnitude is not causal "
            "prediction importance."
        ),
    }

    with path.open(
        "x",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            metadata,
            f,
            sort_keys=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise ImageNet-pretrained ResNet-18 "
            "activations on frozen project_train images."
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
        "--probe-config",
        default=(
            "tools/"
            "probe_pretrained_resnet18_config.yaml"
        ),
    )

    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help=(
            "Exact image_path from the frozen project_train "
            "manifest. May be supplied multiple times."
        ),
    )

    args = parser.parse_args()

    probe_config_path = resolve_repo_path(
        args.probe_config
    )

    probe_config = load_yaml(
        probe_config_path
    )

    logger, log_path, timestamp = (
        configure_logger(
            probe_config
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

        manifest_rows, manifest_path = (
            load_project_train_manifest(
                experiment_cfg
            )
        )

        selected_rows = (
            resolve_requested_images(
                cli_images=args.image,
                probe_config=probe_config,
                manifest_rows=manifest_rows,
            )
        )

        dataset_root = Path(
            machine_cfg[
                "paths"
            ][
                "dataset_root"
            ]
        ).expanduser().resolve()

        device_name = machine_cfg[
            "runtime"
        ][
            "device"
        ]

        device = torch.device(
            device_name
        )

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        weights_name = probe_config[
            "probe"
        ][
            "weights"
        ]

        if weights_name != "IMAGENET1K_V1":
            raise ValueError(
                "This exploratory baseline is currently "
                "locked to IMAGENET1K_V1."
            )

        weights = (
            ResNet18_Weights
            .IMAGENET1K_V1
        )

        logger.info("=" * 72)
        logger.info(
            "PRETRAINED RESNET-18 FEATURE PROBE"
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
            "Probe config SHA-256: %s",
            sha256_file(
                probe_config_path
            ),
        )
        logger.info(
            "Frozen project_train manifest: %s",
            manifest_path,
        )
        logger.info(
            "Frozen project_train manifest SHA-256: %s",
            sha256_file(
                manifest_path
            ),
        )
        logger.info(
            "torch=%s torchvision=%s",
            torch.__version__,
            torchvision.__version__,
        )
        logger.info(
            "device=%s",
            device,
        )

        # This may download the official weight file the first time.
        model = resnet18(
            weights=weights
        )

        model.eval()
        model.to(device)

        checkpoint_info = (
            checkpoint_provenance(
                weights
            )
        )

        logger.info(
            "Weights enum: ResNet18_Weights.%s",
            weights_name,
        )
        logger.info(
            "Weights URL: %s",
            checkpoint_info["url"],
        )
        logger.info(
            "Cached checkpoint: %s",
            checkpoint_info[
                "cached_path"
            ],
        )
        logger.info(
            "Checkpoint SHA-256: %s",
            checkpoint_info[
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

        logger.info(
            "ImageNet mean=%s",
            mean,
        )
        logger.info(
            "ImageNet std=%s",
            std,
        )

        resize_short_side = int(
            probe_config[
                "probe"
            ][
                "resize_short_side"
            ]
        )

        if probe_config[
            "probe"
        ][
            "crop"
        ] != "none":
            raise ValueError(
                "This probe is explicitly defined "
                "as a no-crop diagnostic."
            )

        logger.info(
            "Resize short side=%d, aspect ratio preserved",
            resize_short_side,
        )
        logger.info(
            "Crop=NONE"
        )
        logger.warning(
            "ImageNet predictions are exploratory because "
            "the official 224x224 center crop is deliberately omitted."
        )

        capture_cfg = probe_config[
            "probe"
        ][
            "capture"
        ]

        vis_cfg = probe_config[
            "probe"
        ][
            "visualisation"
        ]

        output_root = (
            REPO_ROOT
            / "runs"
            / (
                "exploratory_pretrained_resnet18_"
                f"{timestamp}"
            )
            / "diagnostics"
        )

        output_root.mkdir(
            parents=True,
            exist_ok=False,
        )

        logger.info(
            "Diagnostic output root: %s",
            output_root,
        )

        for row in selected_rows:
            relative_path = row[
                "image_path"
            ]

            image_path = (
                dataset_root
                / relative_path
            ).resolve()

            if not image_path.is_file():
                raise FileNotFoundError(
                    image_path
                )

            actual_sha = sha256_file(
                image_path
            )

            expected_sha = row[
                "image_sha256"
            ]

            if actual_sha != expected_sha:
                raise ValueError(
                    "Source image SHA-256 mismatch:\n"
                    f"  {relative_path}\n"
                    f"  expected={expected_sha}\n"
                    f"  actual={actual_sha}"
                )

            image_id = (
                f"{image_path.stem}_"
                f"{expected_sha[:12]}"
            )

            image_output_dir = (
                output_root
                / image_id
            )

            image_output_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            logger.info("-" * 72)
            logger.info(
                "Image: %s",
                relative_path,
            )
            logger.info(
                "SHA-256: %s",
                expected_sha,
            )
            logger.info(
                "traffic_type=%s variant=%s "
                "hardware_source=%s",
                row["traffic_type"],
                row["variant"],
                row["hardware_source"],
            )

            with Image.open(
                image_path
            ) as opened:
                image = opened.convert(
                    "RGB"
                )

            original_size = image.size

            resized = resize_full_document(
                image,
                resize_short_side,
            )

            logger.info(
                "Original size W x H: %s",
                original_size,
            )
            logger.info(
                "Model input size W x H: %s",
                resized.size,
            )

            resized.save(
                image_output_dir
                / "full_document_model_input.png"
            )

            input_tensor = prepare_tensor(
                resized,
                mean=mean,
                std=std,
            ).to(
                device
            )

            activations: dict[
                str,
                torch.Tensor,
            ] = {}

            handles = (
                register_activation_hooks(
                    model,
                    capture_cfg,
                    activations,
                )
            )

            try:
                with torch.inference_mode():
                    logits = model(
                        input_tensor
                    ).cpu()
            finally:
                for handle in handles:
                    handle.remove()

            logger.info(
                "Captured %d activation tensors.",
                len(activations),
            )

            for (
                activation_name,
                activation,
            ) in activations.items():
                logger.info(
                    "Activation %-35s shape=%s",
                    activation_name,
                    tuple(
                        activation.shape
                    ),
                )

            # One-page progression through the residual network.
            save_network_overview(
                activations=activations,
                resized_image=resized,
                output_path=(
                    image_output_dir
                    / "network_overview.png"
                ),
                dpi=int(
                    vis_cfg["dpi"]
                ),
            )

            layer_dir = (
                image_output_dir
                / "layer_feature_maps"
            )
            layer_dir.mkdir()

            # Detailed channels for every captured spatial tensor.
            for (
                activation_name,
                activation,
            ) in activations.items():
                save_top_channels(
                    name=activation_name,
                    activation=activation,
                    output_dir=layer_dir,
                    top_k=int(
                        vis_cfg[
                            "top_channels_per_layer"
                        ]
                    ),
                    dpi=int(
                        vis_cfg["dpi"]
                    ),
                )

            avgpool = activations.get(
                "avgpool"
            )

            if avgpool is None:
                raise RuntimeError(
                    "avgpool activation was not captured."
                )

            logger.info(
                "avgpool output shape=%s",
                tuple(
                    avgpool.shape
                ),
            )

            logger.info(
                "avgpool flattened dimensionality=%d",
                avgpool.numel(),
            )

            save_avgpool_plots(
                avgpool=avgpool,
                output_dir=image_output_dir,
                dpi=int(
                    vis_cfg["dpi"]
                ),
            )

            save_imagenet_predictions(
                logits=logits,
                categories=weights.meta[
                    "categories"
                ],
                output_dir=image_output_dir,
                top_k=int(
                    vis_cfg[
                        "top_imagenet_predictions"
                    ]
                ),
                dpi=int(
                    vis_cfg["dpi"]
                ),
                logger=logger,
            )

            save_metadata(
                path=(
                    image_output_dir
                    / "probe_metadata.yaml"
                ),
                row=row,
                original_size=original_size,
                resized_size=resized.size,
                model_checkpoint=(
                    checkpoint_info
                ),
                weights_name=(
                    weights_name
                ),
            )

            logger.info(
                "[PASS] Completed image probe: %s",
                relative_path,
            )

        logger.info("=" * 72)
        logger.info(
            "PRETRAINED RESNET-18 FEATURE PROBE: PASS"
        )
        logger.info("=" * 72)

        return 0

    except Exception:
        logger.exception(
            "PRETRAINED RESNET-18 FEATURE PROBE: FAIL"
        )
        return 1

    finally:
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())