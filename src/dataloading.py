"""
Development DataLoader construction for the frozen Tech-2 protocol.

This module connects:
    frozen manifests
        ->
    validated deterministic Dataset/preprocessing
        ->
    reproducibly seeded DataLoaders

Supported partitions:
    project_train
    dev_val

Held-out test access is deliberately unsupported.

No model or training logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from src.data import (
    FantasyIDManifestDataset,
    build_fantasyid_dataset,
)

from src.reproducibility import (
    make_dataloader_generator,
    seed_dataloader_worker,
    validate_run_seed,
)


# ======================================================================
# Small config helpers
# ======================================================================

def require_mapping(
    value: Any,
    label: str,
) -> Mapping[str, Any]:

    if not isinstance(
        value,
        Mapping,
    ):

        raise TypeError(
            f"{label} must be a mapping, "
            f"got {type(value).__name__}: {value!r}"
        )

    return value


def require_key(
    mapping: Mapping[str, Any],
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
# Loader bundle
# ======================================================================

@dataclass
class DevelopmentDataLoaders:
    """
    One resolution + one run seed.

    The generator objects are deliberately exposed so the later
    checkpoint implementation can capture/restore their exact state.
    """

    resolution_name: str
    run_seed: int

    project_train_dataset: FantasyIDManifestDataset
    dev_val_dataset: FantasyIDManifestDataset

    project_train: DataLoader
    dev_val: DataLoader

    project_train_generator: torch.Generator
    dev_val_generator: torch.Generator


# ======================================================================
# Builder
# ======================================================================

def build_development_dataloaders(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    repo_root: Path,
    resolution_name: str,
    run_seed: int,
) -> DevelopmentDataLoaders:
    """
    Construct reproducible project_train and dev_val DataLoaders.

    This function intentionally does NOT call
    configure_run_reproducibility().

    The future run/training entry point must call:

        configure_run_reproducibility(...)

    before model construction.

    Keeping that explicit prevents the DataLoader builder from silently
    reseeding global RNG state in the middle of a run.
    """

    validate_run_seed(
        experiment_cfg=experiment_cfg,
        run_seed=run_seed,
    )

    # ------------------------------------------------------------------
    # Frozen scientific DataLoader settings.
    # ------------------------------------------------------------------

    dataloader_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "dataloader",
            "experiment_config",
        ),
        "dataloader",
    )

    batch_size = int(
        require_key(
            dataloader_cfg,
            "batch_size",
            "dataloader",
        )
    )

    if batch_size != 32:

        raise ValueError(
            "Frozen Tech-2 protocol requires "
            f"batch_size=32, got {batch_size}."
        )

    num_workers_source = require_key(
        dataloader_cfg,
        "num_workers_source",
        "dataloader",
    )

    if (
        num_workers_source
        != "machine_config.runtime.num_workers"
    ):

        raise ValueError(
            "Frozen protocol requires DataLoader worker count "
            "to come from machine_config.runtime.num_workers."
        )

    train_cfg = require_mapping(
        require_key(
            dataloader_cfg,
            "project_train",
            "dataloader",
        ),
        "dataloader.project_train",
    )

    dev_cfg = require_mapping(
        require_key(
            dataloader_cfg,
            "dev_val",
            "dataloader",
        ),
        "dataloader.dev_val",
    )

    train_shuffle = require_key(
        train_cfg,
        "shuffle",
        "dataloader.project_train",
    )

    train_drop_last = require_key(
        train_cfg,
        "drop_last",
        "dataloader.project_train",
    )

    dev_shuffle = require_key(
        dev_cfg,
        "shuffle",
        "dataloader.dev_val",
    )

    dev_drop_last = require_key(
        dev_cfg,
        "drop_last",
        "dataloader.dev_val",
    )

    if train_shuffle is not True:

        raise ValueError(
            "Frozen project_train DataLoader requires shuffle=true."
        )

    if train_drop_last is not False:

        raise ValueError(
            "Frozen project_train DataLoader requires drop_last=false."
        )

    if dev_shuffle is not False:

        raise ValueError(
            "Frozen dev_val DataLoader requires shuffle=false."
        )

    if dev_drop_last is not False:

        raise ValueError(
            "Frozen dev_val DataLoader requires drop_last=false."
        )

    # ------------------------------------------------------------------
    # Machine-only worker count.
    # ------------------------------------------------------------------

    runtime_cfg = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
    )

    num_workers = require_key(
        runtime_cfg,
        "num_workers",
        "machine_config.runtime",
    )

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

        raise TypeError(
            "machine_config.runtime.num_workers must be "
            f"a non-negative integer, got {num_workers!r}"
        )

    # ------------------------------------------------------------------
    # Construct the already-validated datasets.
    # ------------------------------------------------------------------

    project_train_dataset = (
        build_fantasyid_dataset(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=repo_root,
            split_name="project_train",
            resolution_name=resolution_name,
        )
    )

    dev_val_dataset = (
        build_fantasyid_dataset(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            repo_root=repo_root,
            split_name="dev_val",
            resolution_name=resolution_name,
        )
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Train and dev each receive an independent generator object.
    #
    # Both are seeded from the same frozen run_seed, but iteration of
    # dev cannot advance or otherwise perturb the mutable generator state
    # used for train shuffling.
    #
    # This also gives later checkpoint code explicit generator states to
    # capture and restore.
    # ------------------------------------------------------------------

    project_train_generator = (
        make_dataloader_generator(
            run_seed
        )
    )

    dev_val_generator = (
        make_dataloader_generator(
            run_seed
        )
    )

    # ------------------------------------------------------------------
    # DataLoaders.
    #
    # We deliberately leave purely runtime/performance PyTorch options
    # such as pin_memory/prefetch_factor at their framework defaults for
    # now. They are not scientific selection variables and have not been
    # added to the frozen protocol.
    # ------------------------------------------------------------------

    project_train_loader = DataLoader(
        dataset=project_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=project_train_generator,
    )

    dev_val_loader = DataLoader(
        dataset=dev_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=dev_val_generator,
    )

    return DevelopmentDataLoaders(
        resolution_name=resolution_name,
        run_seed=run_seed,
        project_train_dataset=project_train_dataset,
        dev_val_dataset=dev_val_dataset,
        project_train=project_train_loader,
        dev_val=dev_val_loader,
        project_train_generator=project_train_generator,
        dev_val_generator=dev_val_generator,
    )