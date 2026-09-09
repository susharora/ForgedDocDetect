"""
Frozen run-level reproducibility utilities for Tech-2.

This module owns:
- Python RNG seeding;
- NumPy RNG seeding;
- torch CPU RNG seeding;
- torch CUDA RNG seeding;
- deterministic PyTorch settings;
- cuDNN deterministic settings;
- TF32 disabling;
- CUBLAS_WORKSPACE_CONFIG;
- DataLoader generator creation;
- DataLoader worker seeding.

It does NOT:
- construct a model;
- construct a Dataset;
- construct a DataLoader;
- train anything.

Important
---------
configure_run_reproducibility() must be called before model
initialisation and before the first CUDA computation of a run.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


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
# Reproducibility state
# ======================================================================

@dataclass(
    frozen=True
)
class ReproducibilityState:
    """
    Concrete run-level deterministic settings that were applied.
    """

    run_seed: int

    cudnn_benchmark: bool
    cudnn_deterministic: bool

    deterministic_algorithms: bool

    cublas_workspace_config: str

    allow_tf32_matmul: bool
    allow_tf32_cudnn: bool


# ======================================================================
# Seed-plan validation
# ======================================================================

def validate_run_seed(
    *,
    experiment_cfg: Mapping[str, Any],
    run_seed: int,
) -> None:
    """
    Restrict primary development runs to the frozen seed plan.
    """

    if (
        not isinstance(
            run_seed,
            int,
        )
        or isinstance(
            run_seed,
            bool,
        )
    ):

        raise TypeError(
            f"run_seed must be an integer, got {run_seed!r}"
        )

    reproducibility = require_mapping(
        require_key(
            experiment_cfg,
            "reproducibility",
            "experiment_config",
        ),
        "reproducibility",
    )

    seed_plan = require_mapping(
        require_key(
            reproducibility,
            "seed_plan",
            "reproducibility",
        ),
        "reproducibility.seed_plan",
    )

    screening_seed = int(
        require_key(
            seed_plan,
            "screening_seed",
            "reproducibility.seed_plan",
        )
    )

    confirmation_seeds = [
        int(
            value
        )
        for value
        in require_key(
            seed_plan,
            "confirmation_seeds",
            "reproducibility.seed_plan",
        )
    ]

    allowed_seeds = set(
        confirmation_seeds
    )

    allowed_seeds.add(
        screening_seed
    )

    if run_seed not in allowed_seeds:

        raise ValueError(
            "run_seed is outside the frozen development seed plan:\n"
            f"  run_seed={run_seed}\n"
            f"  allowed={sorted(allowed_seeds)}"
        )


# ======================================================================
# Global deterministic configuration
# ======================================================================

def configure_run_reproducibility(
    *,
    experiment_cfg: Mapping[str, Any],
    run_seed: int,
) -> ReproducibilityState:
    """
    Apply the frozen deterministic run contract.

    Call this once near the start of every scientific run, before:
    - model construction;
    - classifier initialization;
    - DataLoader iteration;
    - CUDA computation.

    Merely importing torch does not normally initialize CUDA, but if
    CUDA has already been initialized with an incompatible
    CUBLAS_WORKSPACE_CONFIG this function fails rather than silently
    pretending deterministic cuBLAS was configured.
    """

    validate_run_seed(
        experiment_cfg=experiment_cfg,
        run_seed=run_seed,
    )

    reproducibility = require_mapping(
        require_key(
            experiment_cfg,
            "reproducibility",
            "experiment_config",
        ),
        "reproducibility",
    )

    determinism = require_mapping(
        require_key(
            reproducibility,
            "determinism",
            "reproducibility",
        ),
        "reproducibility.determinism",
    )

    cudnn_benchmark = require_key(
        determinism,
        "cudnn_benchmark",
        "reproducibility.determinism",
    )

    cudnn_deterministic = require_key(
        determinism,
        "cudnn_deterministic",
        "reproducibility.determinism",
    )

    deterministic_algorithms = require_key(
        determinism,
        "deterministic_algorithms",
        "reproducibility.determinism",
    )

    cublas_workspace_config = str(
        require_key(
            determinism,
            "cublas_workspace_config",
            "reproducibility.determinism",
        )
    )

    allow_tf32_matmul = require_key(
        determinism,
        "allow_tf32_matmul",
        "reproducibility.determinism",
    )

    allow_tf32_cudnn = require_key(
        determinism,
        "allow_tf32_cudnn",
        "reproducibility.determinism",
    )

    failure_policy = require_mapping(
        require_key(
            determinism,
            "failure_policy",
            "reproducibility.determinism",
        ),
        "reproducibility.determinism.failure_policy",
    )

    nondeterministic_policy = require_key(
        failure_policy,
        "nondeterministic_required_operation",
        (
            "reproducibility."
            "determinism."
            "failure_policy"
        ),
    )

    # ------------------------------------------------------------------
    # Reassert the exact frozen scientific values.
    # ------------------------------------------------------------------

    if cudnn_benchmark is not False:

        raise ValueError(
            "Frozen protocol requires cudnn_benchmark=false."
        )

    if cudnn_deterministic is not True:

        raise ValueError(
            "Frozen protocol requires cudnn_deterministic=true."
        )

    if deterministic_algorithms is not True:

        raise ValueError(
            "Frozen protocol requires "
            "deterministic_algorithms=true."
        )

    if cublas_workspace_config != ":4096:8":

        raise ValueError(
            "Frozen protocol requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8."
        )

    if allow_tf32_matmul is not False:

        raise ValueError(
            "Frozen protocol requires TF32 matmul disabled."
        )

    if allow_tf32_cudnn is not False:

        raise ValueError(
            "Frozen protocol requires TF32 cuDNN disabled."
        )

    if nondeterministic_policy != "fail":

        raise ValueError(
            "Frozen protocol requires nondeterministic "
            "required operations to fail."
        )

    # ------------------------------------------------------------------
    # CUBLAS_WORKSPACE_CONFIG must be established before CUDA work.
    # ------------------------------------------------------------------

    existing_workspace_config = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    )

    if (
        torch.cuda.is_initialized()
        and existing_workspace_config
        != cublas_workspace_config
    ):

        raise RuntimeError(
            "CUDA was already initialized before the frozen "
            "CUBLAS_WORKSPACE_CONFIG was established:\n"
            f"  expected={cublas_workspace_config!r}\n"
            f"  existing={existing_workspace_config!r}"
        )

    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = cublas_workspace_config

    # ------------------------------------------------------------------
    # Global RNGs.
    # ------------------------------------------------------------------

    random.seed(
        run_seed
    )

    np.random.seed(
        run_seed
    )

    torch.manual_seed(
        run_seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            run_seed
        )

    # ------------------------------------------------------------------
    # Deterministic backend contract.
    # ------------------------------------------------------------------

    torch.backends.cudnn.benchmark = (
        False
    )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.use_deterministic_algorithms(
        True,
        warn_only=False,
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        False
    )

    torch.backends.cudnn.allow_tf32 = (
        False
    )

    return ReproducibilityState(
        run_seed=run_seed,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        deterministic_algorithms=True,
        cublas_workspace_config=(
            cublas_workspace_config
        ),
        allow_tf32_matmul=False,
        allow_tf32_cudnn=False,
    )


# ======================================================================
# DataLoader-local generator
# ======================================================================

def make_dataloader_generator(
    run_seed: int,
) -> torch.Generator:
    """
    Return a fresh CPU generator seeded exactly with run_seed.

    A fresh generator is created for each split/branch rather than
    sharing one mutable generator between train and dev.
    """

    if (
        not isinstance(
            run_seed,
            int,
        )
        or isinstance(
            run_seed,
            bool,
        )
    ):

        raise TypeError(
            f"run_seed must be an integer, got {run_seed!r}"
        )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        run_seed
    )

    return generator


# ======================================================================
# DataLoader worker RNG
# ======================================================================

def seed_dataloader_worker(
    worker_id: int,
) -> None:
    """
    Seed Python and NumPy inside one PyTorch DataLoader worker.

    PyTorch itself assigns each worker a deterministic torch seed from:

        DataLoader generator
            ->
        base_seed
            ->
        base_seed + worker_id

    torch.initial_seed() therefore already represents the worker's
    PyTorch seed.

    NumPy's legacy global RNG requires a 32-bit seed, so the worker seed
    is reduced modulo 2**32 for Python/NumPy.

    This function is top-level so it remains picklable under worker
    multiprocessing start methods.
    """

    if worker_id < 0:

        raise ValueError(
            f"worker_id cannot be negative: {worker_id}"
        )

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(
        worker_seed
    )

    np.random.seed(
        worker_seed
    )