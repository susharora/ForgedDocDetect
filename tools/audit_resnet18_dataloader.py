#!/usr/bin/env python3
"""
Audit the frozen Tech-2 DataLoader and reproducibility contract.

This audit validates:

1. the frozen experiment validator still passes;
2. execution starts from a clean Git state;
3. run seeds 8, 9 and 10 are accepted;
4. Python RNG is reproducible per run seed;
5. NumPy RNG is reproducible per run seed;
6. torch CPU RNG is reproducible per run seed;
7. torch CUDA RNG is reproducible per run seed;
8. deterministic backend settings are actually active;
9. DataLoader worker Python/NumPy/torch RNG state is deterministic;
10. worker seeds follow PyTorch base_seed + worker_id semantics;
11. repeated fresh loaders reproduce the same shuffled train order;
12. different run seeds produce different train orders;
13. dev order remains frozen manifest order;
14. dev iteration does not alter the independent train generator;
15. r256 and r512 use the same train ordering for a given seed;
16. actual multi-worker FantasyID loaders follow the reference order;
17. actual batch count, batch size, tensor shape and metadata are correct;
18. project_train and dev_val each cover every sample exactly once.

No model is constructed.
No loss is computed.
No optimizer is constructed.
No training occurs.
No held-out test data are accessed.

Detailed evidence is written to ./logs/.
No print() is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, get_worker_info


# ======================================================================
# Repository imports
# ======================================================================

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

from src.dataloading import (
    build_development_dataloaders,
)

from src.reproducibility import (
    configure_run_reproducibility,
    make_dataloader_generator,
    seed_dataloader_worker,
)


LOGGER = logging.getLogger(
    "audit_resnet18_dataloader"
)


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


# ======================================================================
# Generic utilities
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


def sha256_bytes(
    value: bytes,
) -> str:

    return hashlib.sha256(
        value
    ).hexdigest()


def sha256_indices(
    values: list[int],
) -> str:

    encoded = ",".join(
        str(
            value
        )
        for value
        in values
    ).encode(
        "utf-8"
    )

    return sha256_bytes(
        encoded
    )


def sha256_generator_state(
    generator: torch.Generator,
) -> str:

    state = (
        generator
        .get_state()
        .contiguous()
        .numpy()
        .tobytes()
    )

    return sha256_bytes(
        state
    )


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
# Clean Git gate
# ======================================================================

def require_clean_git() -> str:
    """
    Capture Git state before validator/audit artifacts are created.
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
            "Commit/remove outstanding files before "
            "running the DataLoader audit.\n\n"
            f"{status}"
        )

    return (
        commit_result
        .stdout
        .strip()
    )


# ======================================================================
# Frozen validator
# ======================================================================

def run_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> tuple[
    Path,
    str,
]:

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

    if (
        match.group(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Validator did not return PASS."
        )

    log_path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = match.group(
        "sha256"
    )

    if not log_path.is_file():
        raise FileNotFoundError(
            log_path
        )

    actual_sha = sha256_file(
        log_path
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Validator artifact SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    return (
        log_path,
        expected_sha,
    )


# ======================================================================
# Audit logging/output
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

    values = {
        "machine_id":
            machine_id,

        "timestamp":
            timestamp,
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
            **values
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
            **values
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
# Global RNG audit
# ======================================================================

def capture_global_rng_fingerprint(
    *,
    device: str,
    require_cuda: bool,
) -> dict[str, str]:

    python_values = [
        random.random()
        for _
        in range(
            8
        )
    ]

    numpy_values = np.random.random(
        8
    ).astype(
        np.float64,
        copy=False,
    )

    torch_cpu_values = torch.rand(
        8,
        dtype=torch.float32,
        device="cpu",
    )

    python_bytes = json.dumps(
        python_values,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    result = {
        "python":
            sha256_bytes(
                python_bytes
            ),

        "numpy":
            sha256_bytes(
                numpy_values
                .tobytes()
            ),

        "torch_cpu":
            sha256_bytes(
                torch_cpu_values
                .contiguous()
                .numpy()
                .tobytes()
            ),
    }

    if require_cuda:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA RNG probe required but CUDA is unavailable."
            )

        torch_cuda_values = torch.rand(
            8,
            dtype=torch.float32,
            device=device,
        ).cpu()

        result[
            "torch_cuda"
        ] = sha256_bytes(
            torch_cuda_values
            .contiguous()
            .numpy()
            .tobytes()
        )

    return result


def validate_backend_state() -> None:

    if torch.backends.cudnn.benchmark is not False:
        raise RuntimeError(
            "cudnn.benchmark is not False."
        )

    if torch.backends.cudnn.deterministic is not True:
        raise RuntimeError(
            "cudnn.deterministic is not True."
        )

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "torch deterministic algorithms are not enabled."
        )

    if torch.backends.cuda.matmul.allow_tf32 is not False:
        raise RuntimeError(
            "TF32 matmul is not disabled."
        )

    if torch.backends.cudnn.allow_tf32 is not False:
        raise RuntimeError(
            "TF32 cuDNN is not disabled."
        )

    if (
        os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        )
        != ":4096:8"
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG is not :4096:8."
        )


def audit_global_rngs(
    *,
    experiment_cfg: dict[str, Any],
    seeds: list[int],
    device: str,
    require_cuda: bool,
) -> dict[str, Any]:

    results: dict[str, Any] = {}

    full_fingerprints: set[
        str
    ] = set()

    for seed in seeds:

        state_a = configure_run_reproducibility(
            experiment_cfg=experiment_cfg,
            run_seed=seed,
        )

        validate_backend_state()

        fingerprint_a = capture_global_rng_fingerprint(
            device=device,
            require_cuda=require_cuda,
        )

        state_b = configure_run_reproducibility(
            experiment_cfg=experiment_cfg,
            run_seed=seed,
        )

        validate_backend_state()

        fingerprint_b = capture_global_rng_fingerprint(
            device=device,
            require_cuda=require_cuda,
        )

        if state_a != state_b:
            raise RuntimeError(
                f"ReproducibilityState mismatch for seed {seed}."
            )

        if fingerprint_a != fingerprint_b:
            raise RuntimeError(
                "Global RNG streams were not reproducible:\n"
                f"  seed={seed}\n"
                f"  first={fingerprint_a}\n"
                f"  second={fingerprint_b}"
            )

        combined = sha256_bytes(
            json.dumps(
                fingerprint_a,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )

        if combined in full_fingerprints:
            raise RuntimeError(
                "Different frozen run seeds produced identical "
                "global RNG fingerprints."
            )

        full_fingerprints.add(
            combined
        )

        results[
            str(
                seed
            )
        ] = {
            "fingerprints":
                fingerprint_a,

            "combined_sha256":
                combined,
        }

        LOGGER.info(
            "[PASS] Global RNG reproducibility | seed=%d | sha256=%s",
            seed,
            combined,
        )

    LOGGER.info(
        "[PASS] Seeds 8/9/10 produce distinct global RNG streams"
    )

    return results


# ======================================================================
# Worker RNG probe
# ======================================================================

class WorkerRNGProbeDataset(
    Dataset[
        dict[str, Any]
    ]
):

    def __init__(
        self,
        length: int,
    ) -> None:

        self.length = int(
            length
        )

    def __len__(
        self,
    ) -> int:

        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        worker = get_worker_info()

        if worker is None:
            raise RuntimeError(
                "Worker RNG probe unexpectedly ran in the main process."
            )

        return {
            "index":
                int(
                    index
                ),

            "worker_id":
                int(
                    worker.id
                ),

            "torch_initial_seed":
                int(
                    torch.initial_seed()
                ),

            "python_random":
                float(
                    random.random()
                ),

            "numpy_random":
                float(
                    np.random.random()
                ),

            "torch_random":
                float(
                    torch.rand(
                        (),
                        dtype=torch.float64,
                    ).item()
                ),
        }


def collect_worker_probe(
    *,
    run_seed: int,
    num_workers: int,
    items_per_worker: int,
) -> list[
    dict[str, Any]
]:

    dataset = WorkerRNGProbeDataset(
        num_workers
        * items_per_worker
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(
            run_seed
        ),
    )

    records: list[
        dict[str, Any]
    ] = []

    for batch in loader:

        records.append(
            {
                "index":
                    int(
                        batch[
                            "index"
                        ].item()
                    ),

                "worker_id":
                    int(
                        batch[
                            "worker_id"
                        ].item()
                    ),

                "torch_initial_seed":
                    int(
                        batch[
                            "torch_initial_seed"
                        ].item()
                    ),

                "python_random":
                    float(
                        batch[
                            "python_random"
                        ].item()
                    ),

                "numpy_random":
                    float(
                        batch[
                            "numpy_random"
                        ].item()
                    ),

                "torch_random":
                    float(
                        batch[
                            "torch_random"
                        ].item()
                    ),
            }
        )

    return records


def audit_worker_rngs(
    *,
    seeds: list[int],
    num_workers: int,
    items_per_worker: int,
) -> dict[str, Any]:

    results: dict[str, Any] = {}

    probe_hashes: set[
        str
    ] = set()

    for seed in seeds:

        first = collect_worker_probe(
            run_seed=seed,
            num_workers=num_workers,
            items_per_worker=items_per_worker,
        )

        second = collect_worker_probe(
            run_seed=seed,
            num_workers=num_workers,
            items_per_worker=items_per_worker,
        )

        if first != second:
            raise RuntimeError(
                "Worker RNG probe was not reproducible:\n"
                f"  seed={seed}"
            )

        observed_worker_ids = sorted(
            {
                int(
                    record[
                        "worker_id"
                    ]
                )
                for record
                in first
            }
        )

        expected_worker_ids = list(
            range(
                num_workers
            )
        )

        if observed_worker_ids != expected_worker_ids:
            raise RuntimeError(
                "Worker probe did not observe every configured worker:\n"
                f"  expected={expected_worker_ids}\n"
                f"  actual={observed_worker_ids}"
            )

        first_record_by_worker: dict[
            int,
            dict[str, Any],
        ] = {}

        for record in first:

            worker_id = int(
                record[
                    "worker_id"
                ]
            )

            if worker_id not in first_record_by_worker:
                first_record_by_worker[
                    worker_id
                ] = record

        worker_zero_seed = int(
            first_record_by_worker[
                0
            ][
                "torch_initial_seed"
            ]
        )

        worker_summaries: dict[
            str,
            Any,
        ] = {}

        for worker_id in expected_worker_ids:

            record = first_record_by_worker[
                worker_id
            ]

            full_seed = int(
                record[
                    "torch_initial_seed"
                ]
            )

            expected_full_seed = (
                worker_zero_seed
                + worker_id
            )

            if full_seed != expected_full_seed:
                raise RuntimeError(
                    "Worker torch seed does not follow "
                    "base_seed + worker_id:\n"
                    f"  worker={worker_id}\n"
                    f"  expected={expected_full_seed}\n"
                    f"  actual={full_seed}"
                )

            worker_seed_32 = (
                full_seed
                % (2**32)
            )

            expected_python = (
                random.Random(
                    worker_seed_32
                )
                .random()
            )

            expected_numpy = float(
                np.random.RandomState(
                    worker_seed_32
                )
                .random_sample()
            )

            torch_generator = torch.Generator(
                device="cpu"
            )

            torch_generator.manual_seed(
                full_seed
            )

            expected_torch = float(
                torch.rand(
                    (),
                    dtype=torch.float64,
                    generator=torch_generator,
                ).item()
            )

            if (
                record[
                    "python_random"
                ]
                != expected_python
            ):
                raise RuntimeError(
                    "Worker Python RNG does not match "
                    "seed_dataloader_worker contract:\n"
                    f"  worker={worker_id}"
                )

            if (
                record[
                    "numpy_random"
                ]
                != expected_numpy
            ):
                raise RuntimeError(
                    "Worker NumPy RNG does not match "
                    "seed_dataloader_worker contract:\n"
                    f"  worker={worker_id}"
                )

            if (
                record[
                    "torch_random"
                ]
                != expected_torch
            ):
                raise RuntimeError(
                    "Worker torch RNG does not match "
                    "PyTorch worker seed:\n"
                    f"  worker={worker_id}"
                )

            worker_summaries[
                str(
                    worker_id
                )
            ] = {
                "torch_initial_seed":
                    full_seed,

                "seed_mod_2_32":
                    worker_seed_32,
            }

        probe_hash = sha256_bytes(
            json.dumps(
                first,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )

        if probe_hash in probe_hashes:
            raise RuntimeError(
                "Different run seeds produced identical worker probes."
            )

        probe_hashes.add(
            probe_hash
        )

        results[
            str(
                seed
            )
        ] = {
            "sha256":
                probe_hash,

            "workers":
                worker_summaries,
        }

        LOGGER.info(
            "[PASS] Worker RNG contract | seed=%d | workers=%d | sha256=%s",
            seed,
            num_workers,
            probe_hash,
        )

    return results


# ======================================================================
# Lightweight reference order
# ======================================================================

class IndexDataset(
    Dataset[int]
):

    def __init__(
        self,
        length: int,
    ) -> None:

        self.length = int(
            length
        )

    def __len__(
        self,
    ) -> int:

        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> int:

        return int(
            index
        )


def reference_loader_order(
    *,
    length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    run_seed: int,
) -> list[int]:

    loader = DataLoader(
        dataset=IndexDataset(
            length
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(
            run_seed
        ),
    )

    order: list[int] = []

    for batch in loader:

        order.extend(
            int(
                value
            )
            for value
            in batch.tolist()
        )

    return order


def build_reference_orders(
    *,
    seeds: list[int],
    train_rows: int,
    dev_rows: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:

    results: dict[str, Any] = {}

    train_hashes: set[
        str
    ] = set()

    expected_dev = list(
        range(
            dev_rows
        )
    )

    expected_dev_hash = sha256_indices(
        expected_dev
    )

    for seed in seeds:

        train_a = reference_loader_order(
            length=train_rows,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            run_seed=seed,
        )

        train_b = reference_loader_order(
            length=train_rows,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            run_seed=seed,
        )

        if train_a != train_b:
            raise RuntimeError(
                "Repeated fresh train DataLoaders did not reproduce "
                f"the same order for seed {seed}."
            )

        if sorted(
            train_a
        ) != list(
            range(
                train_rows
            )
        ):
            raise RuntimeError(
                "Reference train loader does not contain each "
                "training index exactly once."
            )

        train_hash = sha256_indices(
            train_a
        )

        if train_hash in train_hashes:
            raise RuntimeError(
                "Different frozen run seeds produced identical "
                "train shuffle order."
            )

        train_hashes.add(
            train_hash
        )

        dev_order = reference_loader_order(
            length=dev_rows,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            run_seed=seed,
        )

        if dev_order != expected_dev:
            raise RuntimeError(
                "Reference dev loader changed manifest order."
            )

        results[
            str(
                seed
            )
        ] = {
            "train_order_sha256":
                train_hash,

            "train_first_16_indices":
                train_a[
                    :16
                ],

            "train_last_16_indices":
                train_a[
                    -16:
                ],

            "dev_order_sha256":
                expected_dev_hash,
        }

        LOGGER.info(
            "[PASS] Reference DataLoader order | seed=%d | train=%s | dev=%s",
            seed,
            train_hash,
            expected_dev_hash,
        )

    LOGGER.info(
        "[PASS] Seeds 8/9/10 produce distinct train permutations"
    )

    return results


# ======================================================================
# Actual FantasyID loader audit
# ======================================================================

def validate_batch(
    *,
    batch: dict[str, Any],
    dataset: Any,
    split_name: str,
    expected_batch_size: int,
    expected_height: int,
    expected_width: int,
    path_to_index: dict[str, int],
) -> list[int]:

    images = batch[
        "image"
    ]

    labels = batch[
        "label"
    ]

    if tuple(
        images.shape
    ) != (
        expected_batch_size,
        3,
        expected_height,
        expected_width,
    ):
        raise RuntimeError(
            "Batch image shape mismatch:\n"
            f"  split={split_name}\n"
            f"  expected="
            f"{(expected_batch_size, 3, expected_height, expected_width)}\n"
            f"  actual={tuple(images.shape)}"
        )

    if images.dtype != torch.float32:
        raise RuntimeError(
            f"{split_name} image dtype is not float32."
        )

    if images.device.type != "cpu":
        raise RuntimeError(
            f"{split_name} DataLoader unexpectedly returned GPU tensors."
        )

    if tuple(
        labels.shape
    ) != (
        expected_batch_size,
    ):
        raise RuntimeError(
            f"{split_name} label shape mismatch."
        )

    if labels.dtype != torch.int64:
        raise RuntimeError(
            f"{split_name} labels are not int64."
        )

    paths = list(
        batch[
            "image_path"
        ]
    )

    traffic_types = list(
        batch[
            "traffic_type"
        ]
    )

    image_hashes = list(
        batch[
            "image_sha256"
        ]
    )

    roles = list(
        batch[
            "project_role"
        ]
    )

    if not (
        len(
            paths
        )
        == len(
            traffic_types
        )
        == len(
            image_hashes
        )
        == len(
            roles
        )
        == expected_batch_size
    ):
        raise RuntimeError(
            f"{split_name} metadata batch length mismatch."
        )

    geometry = batch[
        "geometry"
    ]

    required_geometry = {
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
    }

    if set(
        geometry
    ) != required_geometry:
        raise RuntimeError(
            "Collated geometry fields changed unexpectedly."
        )

    for (
        name,
        value,
    ) in geometry.items():

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise RuntimeError(
                f"Geometry field {name} was not collated to a tensor."
            )

        if tuple(
            value.shape
        ) != (
            expected_batch_size,
        ):
            raise RuntimeError(
                f"Geometry field {name} has incorrect batch shape."
            )

        if value.dtype != torch.int64:
            raise RuntimeError(
                f"Geometry field {name} is not int64."
            )

    indices: list[int] = []

    for offset, path in enumerate(
        paths
    ):

        index = path_to_index[
            str(
                path
            )
        ]

        row = dataset.rows[
            index
        ]

        actual_label = int(
            labels[
                offset
            ].item()
        )

        if actual_label != int(
            row[
                "label"
            ]
        ):
            raise RuntimeError(
                "Label/manifest mismatch during collated loading."
            )

        if (
            traffic_types[
                offset
            ]
            != row[
                "traffic_type"
            ]
        ):
            raise RuntimeError(
                "traffic_type/manifest mismatch during loading."
            )

        if (
            image_hashes[
                offset
            ]
            != row[
                "image_sha256"
            ]
        ):
            raise RuntimeError(
                "image_sha256/manifest mismatch during loading."
            )

        if (
            roles[
                offset
            ]
            != split_name
        ):
            raise RuntimeError(
                "project_role changed during loading."
            )

        indices.append(
            index
        )

    return indices


def collect_actual_loader_order(
    *,
    loader: DataLoader,
    dataset: Any,
    split_name: str,
    expected_rows: int,
    expected_batches: int,
    expected_last_batch_size: int,
) -> tuple[
    list[int],
    dict[int, int],
]:

    canvas = (
        dataset
        .preprocessor
        .canvas
    )

    path_to_index = {
        str(
            row[
                "image_path_relative"
            ]
        ):
            index

        for index, row
        in enumerate(
            dataset.rows
        )
    }

    if len(
        path_to_index
    ) != expected_rows:
        raise RuntimeError(
            f"{split_name} path-to-index mapping is not unique."
        )

    if len(
        loader
    ) != expected_batches:
        raise RuntimeError(
            f"{split_name} DataLoader batch-count mismatch:\n"
            f"  expected={expected_batches}\n"
            f"  actual={len(loader)}"
        )

    observed_order: list[int] = []

    label_counts: Counter[int] = Counter()

    for batch_index, batch in enumerate(
        loader
    ):

        if batch_index == (
            expected_batches
            - 1
        ):
            expected_batch_size = (
                expected_last_batch_size
            )
        else:
            expected_batch_size = int(
                loader.batch_size
            )

        indices = validate_batch(
            batch=batch,
            dataset=dataset,
            split_name=split_name,
            expected_batch_size=expected_batch_size,
            expected_height=int(
                canvas.canvas_height
            ),
            expected_width=int(
                canvas.canvas_width
            ),
            path_to_index=path_to_index,
        )

        observed_order.extend(
            indices
        )

        label_counts.update(
            int(
                value
            )
            for value
            in batch[
                "label"
            ].tolist()
        )

    if len(
        observed_order
    ) != expected_rows:
        raise RuntimeError(
            f"{split_name} row-count mismatch after iteration."
        )

    if sorted(
        observed_order
    ) != list(
        range(
            expected_rows
        )
    ):
        raise RuntimeError(
            f"{split_name} did not yield every manifest row exactly once."
        )

    return (
        observed_order,
        dict(
            sorted(
                label_counts.items()
            )
        ),
    )


def audit_actual_loaders(
    *,
    experiment_cfg: dict[str, Any],
    machine_cfg: dict[str, Any],
    seeds: list[int],
    resolutions: list[str],
    reference_orders: dict[str, Any],
    expected: dict[str, int],
    num_workers: int,
) -> dict[str, Any]:

    results: dict[str, Any] = {}

    train_hash_by_seed_and_resolution: dict[
        tuple[int, str],
        str,
    ] = {}

    dev_hashes: set[
        str
    ] = set()

    for resolution_name in resolutions:

        resolution_results: dict[
            str,
            Any,
        ] = {}

        for seed in seeds:

            configure_run_reproducibility(
                experiment_cfg=experiment_cfg,
                run_seed=seed,
            )

            bundle = build_development_dataloaders(
                experiment_cfg=experiment_cfg,
                machine_cfg=machine_cfg,
                repo_root=REPO_ROOT,
                resolution_name=resolution_name,
                run_seed=seed,
            )

            if (
                bundle.project_train.num_workers
                != num_workers
            ):
                raise RuntimeError(
                    "Train loader worker count does not match machine config."
                )

            if (
                bundle.dev_val.num_workers
                != num_workers
            ):
                raise RuntimeError(
                    "Dev loader worker count does not match machine config."
                )

            if (
                bundle.project_train.batch_size
                != expected[
                    "batch_size"
                ]
            ):
                raise RuntimeError(
                    "Train DataLoader batch size changed."
                )

            if (
                bundle.dev_val.batch_size
                != expected[
                    "batch_size"
                ]
            ):
                raise RuntimeError(
                    "Dev DataLoader batch size changed."
                )

            # ----------------------------------------------------------
            # Prove dev uses a completely independent generator.
            # ----------------------------------------------------------

            train_generator_before_dev = (
                sha256_generator_state(
                    bundle.project_train_generator
                )
            )

            dev_generator_before = (
                sha256_generator_state(
                    bundle.dev_val_generator
                )
            )

            dev_order, dev_labels = (
                collect_actual_loader_order(
                    loader=bundle.dev_val,
                    dataset=bundle.dev_val_dataset,
                    split_name="dev_val",
                    expected_rows=expected[
                        "dev_val_rows"
                    ],
                    expected_batches=expected[
                        "dev_val_batches"
                    ],
                    expected_last_batch_size=expected[
                        "dev_val_last_batch_size"
                    ],
                )
            )

            train_generator_after_dev = (
                sha256_generator_state(
                    bundle.project_train_generator
                )
            )

            dev_generator_after = (
                sha256_generator_state(
                    bundle.dev_val_generator
                )
            )

            if (
                train_generator_before_dev
                != train_generator_after_dev
            ):
                raise RuntimeError(
                    "Iterating dev_val altered the independent "
                    "project_train generator."
                )

            if (
                dev_generator_before
                == dev_generator_after
            ):
                raise RuntimeError(
                    "dev_val generator state did not advance "
                    "during DataLoader iteration."
                )

            expected_dev_order = list(
                range(
                    expected[
                        "dev_val_rows"
                    ]
                )
            )

            if dev_order != expected_dev_order:
                raise RuntimeError(
                    "Actual dev_val DataLoader does not preserve "
                    "frozen manifest order."
                )

            dev_hash = sha256_indices(
                dev_order
            )

            dev_hashes.add(
                dev_hash
            )

            # ----------------------------------------------------------
            # Now execute the actual shuffled project_train loader.
            # ----------------------------------------------------------

            train_generator_before_train = (
                sha256_generator_state(
                    bundle.project_train_generator
                )
            )

            train_order, train_labels = (
                collect_actual_loader_order(
                    loader=bundle.project_train,
                    dataset=bundle.project_train_dataset,
                    split_name="project_train",
                    expected_rows=expected[
                        "project_train_rows"
                    ],
                    expected_batches=expected[
                        "project_train_batches"
                    ],
                    expected_last_batch_size=expected[
                        "project_train_last_batch_size"
                    ],
                )
            )

            train_generator_after_train = (
                sha256_generator_state(
                    bundle.project_train_generator
                )
            )

            if (
                train_generator_before_train
                == train_generator_after_train
            ):
                raise RuntimeError(
                    "project_train generator state did not advance."
                )

            expected_train_hash = (
                reference_orders[
                    str(
                        seed
                    )
                ][
                    "train_order_sha256"
                ]
            )

            actual_train_hash = sha256_indices(
                train_order
            )

            if actual_train_hash != expected_train_hash:
                raise RuntimeError(
                    "Actual FantasyID train order does not match "
                    "the independently reconstructed multi-worker "
                    "DataLoader order:\n"
                    f"  seed={seed}\n"
                    f"  resolution={resolution_name}\n"
                    f"  expected={expected_train_hash}\n"
                    f"  actual={actual_train_hash}"
                )

            if train_labels != {
                0:
                    480,

                1:
                    960,
            }:
                raise RuntimeError(
                    "Actual project_train class counts changed."
                )

            if dev_labels != {
                0:
                    153,

                1:
                    306,
            }:
                raise RuntimeError(
                    "Actual dev_val class counts changed."
                )

            train_hash_by_seed_and_resolution[
                (
                    seed,
                    resolution_name,
                )
            ] = actual_train_hash

            resolution_results[
                str(
                    seed
                )
            ] = {
                "project_train_order_sha256":
                    actual_train_hash,

                "dev_val_order_sha256":
                    dev_hash,

                "project_train_first_16_indices":
                    train_order[
                        :16
                    ],

                "project_train_last_16_indices":
                    train_order[
                        -16:
                    ],

                "project_train_label_counts":
                    train_labels,

                "dev_val_label_counts":
                    dev_labels,

                "train_generator_initial_sha256":
                    train_generator_before_train,

                "train_generator_after_epoch_sha256":
                    train_generator_after_train,

                "dev_generator_initial_sha256":
                    dev_generator_before,

                "dev_generator_after_epoch_sha256":
                    dev_generator_after,
            }

            LOGGER.info(
                "[PASS] Actual DataLoaders | resolution=%s | "
                "seed=%d | train_order=%s | dev_order=%s",
                resolution_name,
                seed,
                actual_train_hash,
                dev_hash,
            )

        results[
            resolution_name
        ] = resolution_results

    # ------------------------------------------------------------------
    # Same seed must imply same train order at both resolutions.
    # ------------------------------------------------------------------

    for seed in seeds:

        r256_hash = train_hash_by_seed_and_resolution[
            (
                seed,
                "r256",
            )
        ]

        r512_hash = train_hash_by_seed_and_resolution[
            (
                seed,
                "r512",
            )
        ]

        if r256_hash != r512_hash:
            raise RuntimeError(
                "Same run seed produced different training order "
                "between r256 and r512:\n"
                f"  seed={seed}\n"
                f"  r256={r256_hash}\n"
                f"  r512={r512_hash}"
            )

    if len(
        dev_hashes
    ) != 1:
        raise RuntimeError(
            "dev_val order changed across seed or resolution."
        )

    LOGGER.info(
        "[PASS] Same run seed gives identical train order "
        "for r256 and r512"
    )

    LOGGER.info(
        "[PASS] dev_val manifest order is identical across "
        "all seeds and resolutions"
    )

    return results


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit Tech-2 DataLoader and RNG reproducibility."
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
            "audit_resnet18_dataloader_config.yaml"
        ),
    )

    args = parser.parse_args()

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
            "DataLoader audit schema_version must equal 1."
        )

    # ------------------------------------------------------------------
    # Must be clean before validator/audit artifacts are generated.
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

    runtime = require_mapping(
        require_key(
            machine_cfg,
            "runtime",
            "machine_config",
        ),
        "machine_config.runtime",
    )

    device = str(
        require_key(
            runtime,
            "device",
            "machine_config.runtime",
        )
    )

    num_workers = int(
        require_key(
            runtime,
            "num_workers",
            "machine_config.runtime",
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

        seeds = [
            int(
                value
            )
            for value
            in require_key(
                audit_cfg,
                "seeds",
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

        expected = {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in require_mapping(
                require_key(
                    audit_cfg,
                    "expected",
                    "audit_config.audit",
                ),
                "audit_config.audit.expected",
            ).items()
        }

        worker_probe_cfg = require_mapping(
            require_key(
                audit_cfg,
                "worker_probe",
                "audit_config.audit",
            ),
            "audit_config.audit.worker_probe",
        )

        items_per_worker = int(
            require_key(
                worker_probe_cfg,
                "items_per_worker",
                "audit_config.audit.worker_probe",
            )
        )

        require_positive_workers = bool(
            require_key(
                worker_probe_cfg,
                "require_positive_num_workers",
                "audit_config.audit.worker_probe",
            )
        )

        require_cuda_rng_probe = bool(
            require_key(
                audit_cfg,
                "require_cuda_rng_probe",
                "audit_config.audit",
            )
        )

        if seeds != [
            8,
            9,
            10,
        ]:
            raise ValueError(
                "Audit seeds must be exactly [8, 9, 10]."
            )

        if resolutions != [
            "r256",
            "r512",
        ]:
            raise ValueError(
                "Audit resolutions must be exactly "
                "['r256', 'r512']."
            )

        frozen_expected = {
            "project_train_rows":
                1440,

            "dev_val_rows":
                459,

            "batch_size":
                32,

            "project_train_batches":
                45,

            "project_train_last_batch_size":
                32,

            "dev_val_batches":
                15,

            "dev_val_last_batch_size":
                11,
        }

        if expected != frozen_expected:
            raise ValueError(
                "Audit expected values do not match the frozen "
                "Tech-2 DataLoader contract:\n"
                f"  expected={frozen_expected}\n"
                f"  actual={expected}"
            )

        if items_per_worker < 2:
            raise ValueError(
                "worker_probe.items_per_worker must be >= 2."
            )

        if (
            require_positive_workers
            and num_workers <= 0
        ):
            raise RuntimeError(
                "This audit is intended to validate the configured "
                "multi-worker DataLoader, but num_workers <= 0."
            )

        if (
            require_cuda_rng_probe
            and not device.startswith(
                "cuda"
            )
        ):
            raise RuntimeError(
                "CUDA RNG probe is required but machine device "
                f"is {device!r}."
            )

        experiment = require_mapping(
            require_key(
                experiment_cfg,
                "experiment",
                "experiment_config",
            ),
            "experiment",
        )

        if (
            require_key(
                experiment,
                "protocol_status",
                "experiment",
            )
            != "frozen"
        ):
            raise RuntimeError(
                "DataLoader audit requires protocol_status=frozen."
            )

        # ------------------------------------------------------------------
        # Provenance
        # ------------------------------------------------------------------

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 DATALOADER + RNG REPRODUCIBILITY AUDIT"
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

        for source_path in (
            "src/data.py",
            "src/reproducibility.py",
            "src/dataloading.py",
        ):

            logger.info(
                "%s SHA-256: %s",
                source_path,
                sha256_file(
                    REPO_ROOT
                    / source_path
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
            "Machine ID: %s",
            machine_id,
        )

        logger.info(
            "Device: %s",
            device,
        )

        logger.info(
            "Configured num_workers: %d",
            num_workers,
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
            "Seeds: %s",
            seeds,
        )

        logger.info(
            "Resolutions: %s",
            resolutions,
        )

        logger.info(
            "held-out test: NOT ACCESSED"
        )

        # ------------------------------------------------------------------
        # Global RNGs / backend state.
        # ------------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Global reproducibility contract ---"
        )

        global_rng_results = audit_global_rngs(
            experiment_cfg=experiment_cfg,
            seeds=seeds,
            device=device,
            require_cuda=require_cuda_rng_probe,
        )

        # ------------------------------------------------------------------
        # Worker RNG contract.
        # ------------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- DataLoader worker RNG contract ---"
        )

        worker_results = audit_worker_rngs(
            seeds=seeds,
            num_workers=num_workers,
            items_per_worker=items_per_worker,
        )

        # ------------------------------------------------------------------
        # Cheap but complete order reconstruction.
        # ------------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Multi-worker reference ordering ---"
        )

        reference_orders = build_reference_orders(
            seeds=seeds,
            train_rows=expected[
                "project_train_rows"
            ],
            dev_rows=expected[
                "dev_val_rows"
            ],
            batch_size=expected[
                "batch_size"
            ],
            num_workers=num_workers,
        )

        # ------------------------------------------------------------------
        # Actual FantasyID DataLoaders.
        #
        # This is the expensive part: every actual development loader is
        # fully iterated at both resolutions and all three run seeds.
        # ------------------------------------------------------------------

        logger.info(
            ""
        )

        logger.info(
            "--- Actual FantasyID multi-worker loaders ---"
        )

        actual_loader_results = audit_actual_loaders(
            experiment_cfg=experiment_cfg,
            machine_cfg=machine_cfg,
            seeds=seeds,
            resolutions=resolutions,
            reference_orders=reference_orders,
            expected=expected,
            num_workers=num_workers,
        )

        # ------------------------------------------------------------------
        # Final evidence.
        # ------------------------------------------------------------------

        result = {
            "schema_version":
                1,

            "status":
                "PASS",

            "machine":
                {
                    "id":
                        machine_id,

                    "device":
                        device,

                    "num_workers":
                        num_workers,
                },

            "provenance":
                {
                    "git_commit":
                        commit_sha,

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

                    "experiment_config_sha256":
                        sha256_file(
                            experiment_path
                        ),

                    "data_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "data.py"
                        ),

                    "reproducibility_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "reproducibility.py"
                        ),

                    "dataloading_module_sha256":
                        sha256_file(
                            REPO_ROOT
                            / "src"
                            / "dataloading.py"
                        ),

                    "validator_log":
                        str(
                            validation_log_path
                        ),

                    "validator_log_sha256":
                        validation_log_sha,
                },

            "global_rng":
                global_rng_results,

            "worker_rng":
                worker_results,

            "reference_orders":
                reference_orders,

            "actual_loaders":
                actual_loader_results,

            "interpretation":
                (
                    "PASS establishes that the frozen run seeds, "
                    "global RNGs, worker RNGs, multi-worker shuffle "
                    "ordering, dev ordering, generator isolation, "
                    "batch geometry and actual FantasyID DataLoaders "
                    "behave according to the frozen Tech-2 contract. "
                    "It does not validate model construction or training."
                ),
        }

        with partial_output_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                result,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        partial_output_path.replace(
            output_path
        )

        output_sha = sha256_file(
            output_path
        )

        logger.info(
            ""
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "AUDIT SUMMARY"
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "[PASS] Python RNG reproducibility for seeds 8/9/10"
        )

        logger.info(
            "[PASS] NumPy RNG reproducibility for seeds 8/9/10"
        )

        logger.info(
            "[PASS] torch CPU RNG reproducibility for seeds 8/9/10"
        )

        if require_cuda_rng_probe:
            logger.info(
                "[PASS] torch CUDA RNG reproducibility for seeds 8/9/10"
            )

        logger.info(
            "[PASS] deterministic backend flags active"
        )

        logger.info(
            "[PASS] configured worker seeding deterministic"
        )

        logger.info(
            "[PASS] repeated fresh train DataLoaders reproduce order"
        )

        logger.info(
            "[PASS] different seeds produce different train orders"
        )

        logger.info(
            "[PASS] dev_val order remains manifest order"
        )

        logger.info(
            "[PASS] dev iteration does not perturb train generator"
        )

        logger.info(
            "[PASS] same seed gives same order at r256 and r512"
        )

        logger.info(
            "[PASS] actual FantasyID loaders match reference ordering"
        )

        logger.info(
            "[PASS] batch counts, sizes, dtypes and geometry"
        )

        logger.info(
            "Audit result: %s",
            output_path,
        )

        logger.info(
            "Audit result SHA-256: %s",
            output_sha,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "RESNET-18 DATALOADER + RNG REPRODUCIBILITY AUDIT: PASS"
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception:

        logger.exception(
            "RESNET-18 DATALOADER + RNG REPRODUCIBILITY AUDIT: FAIL"
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