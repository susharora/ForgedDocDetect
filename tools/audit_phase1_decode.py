#!/usr/bin/env python3
"""
Phase-1 raw image decoding and geometry audit.

The audit consumes only the frozen Tech-1 project_train and dev_val
manifests.

It verifies:
- frozen experiment provenance before starting;
- every manifest image exists;
- every source image SHA-256 matches the frozen manifest;
- every image decodes successfully with Pillow;
- RGB conversion succeeds without changing dimensions;
- file stems and project roles agree with the frozen manifests;
- image format, colour mode, dimensions, aspect ratios and EXIF
  orientation are characterised.

It deliberately performs:
- no resize;
- no crop;
- no augmentation;
- no held-out test access.

All audit output is written to a timestamped file under ./logs/.
There are no print() calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config, load_machine_config


EXIF_ORIENTATION_TAG = 274

REQUIRED_COLUMNS = {
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "project_role",
}


def sha256_file(path: Path) -> str:
    """Return SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve repo-relative paths while preserving absolute paths."""
    path = Path(path_value).expanduser()

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load YAML and require a mapping at the top level."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(
            f"Top level of YAML must be a mapping: {path}"
        )

    return data


def require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:
    """Require a value to be a mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping.")

    return value


def require_key(
    mapping: dict[str, Any],
    key: str,
    label: str,
) -> Any:
    """Return one required mapping value."""
    if key not in mapping:
        raise KeyError(f"Missing required key: {label}.{key}")

    return mapping[key]


def load_audit_config(path: Path) -> dict[str, Any]:
    """Load and validate operational settings for this audit tool."""
    config = load_yaml_mapping(path)

    if config.get("schema_version") != 1:
        raise ValueError(
            "audit config schema_version must currently equal 1"
        )

    logging_cfg = require_mapping(
        require_key(config, "logging", "audit_config"),
        "audit_config.logging",
    )

    required = {
        "directory",
        "filename",
        "level",
        "progress_every",
    }

    missing = required - set(logging_cfg)

    if missing:
        raise KeyError(
            "Audit logging configuration is missing: "
            f"{sorted(missing)}"
        )

    filename = logging_cfg["filename"]

    if not isinstance(filename, str):
        raise TypeError("logging.filename must be a string")

    if Path(filename).name != filename:
        raise ValueError(
            "logging.filename must contain a filename only, "
            "not a directory"
        )

    if "{timestamp}" not in filename:
        raise ValueError(
            "logging.filename must contain {timestamp} so audit logs "
            "cannot silently overwrite one another"
        )

    progress_every = logging_cfg["progress_every"]

    if (
        not isinstance(progress_every, int)
        or isinstance(progress_every, bool)
        or progress_every <= 0
    ):
        raise ValueError(
            "logging.progress_every must be a positive integer"
        )

    return config


def configure_logger(
    audit_config: dict[str, Any],
) -> tuple[logging.Logger, Path]:
    """Create one unique UTC-timestamped audit log."""
    logging_cfg = audit_config["logging"]

    log_dir = resolve_repo_path(logging_cfg["directory"])
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d_%H%M%S_%fZ"
    )

    filename = logging_cfg["filename"].format(
        timestamp=timestamp
    )

    log_path = log_dir / filename

    level_name = str(logging_cfg["level"]).upper()
    level = getattr(logging, level_name, None)

    if not isinstance(level, int):
        raise ValueError(
            f"Unsupported logging level: {level_name}"
        )

    logger = logging.getLogger("audit_phase1_decode")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    # mode="x" guarantees that an existing audit log can never
    # be silently overwritten.
    handler = logging.FileHandler(
        log_path,
        mode="x",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Use UTC in the audit log rather than machine-local time.
    formatter.converter = time.gmtime

    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)

    return logger, log_path


def git_commit_sha() -> str:
    """Return the exact Git commit currently checked out."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def run_frozen_provenance_gate(
    *,
    experiment_path: Path,
    machine_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Execute the existing frozen Tech-1 validator and capture all of its
    output inside this audit log instead of emitting it to the terminal.
    """
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "tools"
            / "validate_experiment_config.py"
        ),
        "--config",
        str(experiment_path),
        "--machine-config",
        str(machine_path),
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    logger.info(
        "Frozen provenance validator return code: %d",
        result.returncode,
    )

    for line in result.stdout.splitlines():
        logger.info("PROVENANCE_GATE | %s", line)

    for line in result.stderr.splitlines():
        logger.error("PROVENANCE_GATE_STDERR | %s", line)

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen Tech-1 provenance validation failed. "
            "See audit log for details."
        )


def resolve_dataset_image(
    dataset_root: Path,
    relative_path: str,
) -> Path:
    """Resolve one frozen manifest path and reject path traversal."""
    image_path = (
        dataset_root / relative_path
    ).resolve()

    try:
        image_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            "Manifest image path escapes dataset root: "
            f"{relative_path}"
        ) from exc

    return image_path


def load_manifest(
    path: Path,
) -> list[dict[str, str]]:
    """Load one frozen image-level manifest."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"Manifest has no header: {path}"
            )

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Manifest missing required columns "
                f"{sorted(missing)}: {path}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"Manifest contains no rows: {path}"
        )

    return rows


def audit_manifest(
    *,
    split_name: str,
    manifest_path: Path,
    expected_rows: int,
    dataset_root: Path,
    progress_every: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Decode and audit every image in one frozen manifest."""
    rows = load_manifest(manifest_path)

    if len(rows) != expected_rows:
        raise ValueError(
            f"{split_name} manifest row count mismatch: "
            f"expected={expected_rows}, actual={len(rows)}"
        )

    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    exif_orientations: Counter[str] = Counter()

    traffic_types: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    hardware_sources: Counter[str] = Counter()

    sizes_by_traffic: dict[
        str,
        Counter[tuple[int, int]],
    ] = defaultdict(Counter)

    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []

    missing_files: list[str] = []
    hash_mismatches: list[
        tuple[str, str, str]
    ] = []
    decode_failures: list[
        tuple[str, str]
    ] = []
    rgb_failures: list[
        tuple[str, str]
    ] = []
    stem_mismatches: list[
        tuple[str, str, str]
    ] = []
    role_mismatches: list[
        tuple[str, str]
    ] = []

    orientation_nontrivial = 0

    logger.info(
        "Starting split audit: split=%s manifest=%s rows=%d",
        split_name,
        manifest_path,
        len(rows),
    )

    for index, row in enumerate(rows, start=1):
        relative_path = row["image_path"]

        image_path = resolve_dataset_image(
            dataset_root,
            relative_path,
        )

        if not image_path.is_file():
            missing_files.append(relative_path)
            continue

        expected_hash = row["image_sha256"].lower()
        actual_hash = sha256_file(image_path).lower()

        if actual_hash != expected_hash:
            hash_mismatches.append(
                (
                    relative_path,
                    expected_hash,
                    actual_hash,
                )
            )

        expected_stem = row["file_stem"]
        actual_stem = image_path.stem

        if actual_stem != expected_stem:
            stem_mismatches.append(
                (
                    relative_path,
                    expected_stem,
                    actual_stem,
                )
            )

        if row["project_role"] != split_name:
            role_mismatches.append(
                (
                    relative_path,
                    row["project_role"],
                )
            )

        try:
            with Image.open(image_path) as image:
                # Force the pixel payload to decode.
                image.load()

                image_format = image.format or "UNKNOWN"
                mode = image.mode
                width, height = image.size

                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Invalid decoded dimensions "
                        f"{width}x{height}"
                    )

                exif = image.getexif()
                orientation = exif.get(
                    EXIF_ORIENTATION_TAG
                )

                formats[image_format] += 1
                modes[mode] += 1
                sizes[(width, height)] += 1

                orientation_key = (
                    "none"
                    if orientation is None
                    else str(orientation)
                )

                exif_orientations[
                    orientation_key
                ] += 1

                if orientation not in (None, 1):
                    orientation_nontrivial += 1

                traffic_type = row["traffic_type"]

                traffic_types[traffic_type] += 1
                variants[row["variant"]] += 1
                hardware_sources[
                    row["hardware_source"]
                ] += 1

                sizes_by_traffic[
                    traffic_type
                ][(width, height)] += 1

                widths.append(width)
                heights.append(height)
                aspect_ratios.append(
                    width / height
                )

                try:
                    rgb = image.convert("RGB")
                    rgb.load()

                    if rgb.mode != "RGB":
                        raise ValueError(
                            "RGB conversion returned "
                            f"mode={rgb.mode}"
                        )

                    if rgb.size != image.size:
                        raise ValueError(
                            "RGB conversion changed "
                            "image dimensions"
                        )

                except Exception as exc:
                    rgb_failures.append(
                        (
                            relative_path,
                            repr(exc),
                        )
                    )

        except Exception as exc:
            decode_failures.append(
                (
                    relative_path,
                    repr(exc),
                )
            )

        if (
            index % progress_every == 0
            or index == len(rows)
        ):
            logger.info(
                "Progress | split=%s decoded=%d/%d",
                split_name,
                index,
                len(rows),
            )

    return {
        "split_name": split_name,
        "rows": len(rows),
        "formats": formats,
        "modes": modes,
        "sizes": sizes,
        "sizes_by_traffic": sizes_by_traffic,
        "exif_orientations": exif_orientations,
        "orientation_nontrivial": orientation_nontrivial,
        "traffic_types": traffic_types,
        "variants": variants,
        "hardware_sources": hardware_sources,
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        "missing_files": missing_files,
        "hash_mismatches": hash_mismatches,
        "decode_failures": decode_failures,
        "rgb_failures": rgb_failures,
        "stem_mismatches": stem_mismatches,
        "role_mismatches": role_mismatches,
    }


def log_counter(
    logger: logging.Logger,
    label: str,
    counter: Counter[Any],
    *,
    limit: int | None = None,
) -> None:
    """Write one Counter into the audit log."""
    logger.info("%s:", label)

    for value, count in counter.most_common(limit):
        logger.info(
            "  %-30s %6d",
            str(value),
            count,
        )

    if (
        limit is not None
        and len(counter) > limit
    ):
        logger.info(
            "  ... %d additional values",
            len(counter) - limit,
        )


def log_failures(
    *,
    logger: logging.Logger,
    label: str,
    values: list[Any],
) -> None:
    """Record one integrity check and representative failures."""
    if not values:
        logger.info(
            "[PASS] %s: 0",
            label,
        )
        return

    logger.error(
        "[FAIL] %s: %d",
        label,
        len(values),
    )

    for example in values[:10]:
        logger.error(
            "       %r",
            example,
        )


def log_split_report(
    result: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Write one split's full audit summary."""
    logger.info("=" * 72)
    logger.info(
        "SPLIT REPORT: %s",
        result["split_name"],
    )
    logger.info("=" * 72)

    logger.info(
        "Manifest rows: %d",
        result["rows"],
    )

    log_counter(
        logger,
        "Traffic type",
        result["traffic_types"],
    )

    log_counter(
        logger,
        "Image format",
        result["formats"],
    )

    log_counter(
        logger,
        "Decoded colour mode",
        result["modes"],
    )

    log_counter(
        logger,
        "EXIF orientation",
        result["exif_orientations"],
    )

    log_counter(
        logger,
        "Most common decoded dimensions (W x H)",
        result["sizes"],
        limit=20,
    )

    widths = result["widths"]
    heights = result["heights"]
    ratios = result["aspect_ratios"]

    if widths:
        logger.info("Geometry range:")
        logger.info(
            "  width: %d .. %d",
            min(widths),
            max(widths),
        )
        logger.info(
            "  height: %d .. %d",
            min(heights),
            max(heights),
        )
        logger.info(
            "  aspect ratio: %.6f .. %.6f",
            min(ratios),
            max(ratios),
        )
        logger.info(
            "  unique sizes: %d",
            len(result["sizes"]),
        )

    logger.info(
        "Images with non-trivial EXIF orientation: %d",
        result["orientation_nontrivial"],
    )

    log_counter(
        logger,
        "Variant",
        result["variants"],
    )

    log_counter(
        logger,
        "Hardware source",
        result["hardware_sources"],
    )

    logger.info(
        "Decoded dimensions by traffic type:"
    )

    for traffic_type in sorted(
        result["sizes_by_traffic"]
    ):
        counter = result[
            "sizes_by_traffic"
        ][traffic_type]

        logger.info(
            "  %s: %d unique size(s)",
            traffic_type,
            len(counter),
        )

        for size, count in counter.most_common(10):
            logger.info(
                "    %-24s %6d",
                str(size),
                count,
            )

        if len(counter) > 10:
            logger.info(
                "    ... %d additional sizes",
                len(counter) - 10,
            )

    logger.info("Integrity checks:")

    failure_groups = {
        "missing files":
            result["missing_files"],
        "source SHA-256 mismatches":
            result["hash_mismatches"],
        "decode failures":
            result["decode_failures"],
        "RGB conversion failures":
            result["rgb_failures"],
        "file-stem mismatches":
            result["stem_mismatches"],
        "project-role mismatches":
            result["role_mismatches"],
    }

    for label, values in failure_groups.items():
        log_failures(
            logger=logger,
            label=label,
            values=values,
        )

    if any(failure_groups.values()):
        raise RuntimeError(
            f"{result['split_name']} audit failed "
            "one or more integrity checks."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw FantasyID decoding and geometry "
            "from frozen Tech-1 manifests."
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
            "audit_phase1_decode_config.yaml"
        ),
    )

    args = parser.parse_args()

    audit_config_path = resolve_repo_path(
        args.audit_config
    )

    audit_config = load_audit_config(
        audit_config_path
    )

    logger, log_path = configure_logger(
        audit_config
    )

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
            machine_cfg["paths"]["dataset_root"]
        ).expanduser().resolve()

        progress_every = audit_config[
            "logging"
        ]["progress_every"]

        script_path = Path(__file__).resolve()

        logger.info("=" * 72)
        logger.info(
            "PHASE-1 RAW DECODE / GEOMETRY AUDIT"
        )
        logger.info("=" * 72)

        logger.info(
            "Audit log: %s",
            log_path,
        )

        logger.info(
            "Timestamp UTC: %s",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )

        logger.info(
            "Git commit SHA: %s",
            git_commit_sha(),
        )

        logger.info(
            "Audit script: %s",
            script_path.relative_to(
                REPO_ROOT
            ),
        )
        logger.info(
            "Audit script SHA-256: %s",
            sha256_file(script_path),
        )

        logger.info(
            "Audit config: %s",
            audit_config_path.relative_to(
                REPO_ROOT
            ),
        )
        logger.info(
            "Audit config SHA-256: %s",
            sha256_file(
                audit_config_path
            ),
        )

        logger.info(
            "Experiment config: %s",
            experiment_path,
        )
        logger.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        logger.info(
            "Machine ID: %s",
            machine_cfg["machine"]["id"],
        )

        logger.info(
            "Dataset root: %s",
            dataset_root,
        )

        logger.info(
            "Decoder: Pillow %s",
            Image.__version__,
        )

        logger.info(
            "Colour conversion test: RGB"
        )
        logger.info("Resize: NONE")
        logger.info("Crop: NONE")
        logger.info("Augmentation: NONE")
        logger.info(
            "Held-out test: NOT ACCESSED"
        )

        logger.info(
            "Running frozen Tech-1 "
            "configuration/provenance gate."
        )

        run_frozen_provenance_gate(
            experiment_path=experiment_path,
            machine_path=machine_path,
            logger=logger,
        )

        logger.info(
            "[PASS] Frozen Tech-1 "
            "configuration/provenance gate"
        )

        split_cfg = experiment_cfg[
            "data"
        ]["frozen_split"]

        specifications = (
            (
                "project_train",
                resolve_repo_path(
                    split_cfg[
                        "project_train"
                    ]["path"]
                ),
                split_cfg[
                    "project_train"
                ]["images"],
            ),
            (
                "dev_val",
                resolve_repo_path(
                    split_cfg[
                        "dev_val"
                    ]["path"]
                ),
                split_cfg[
                    "dev_val"
                ]["images"],
            ),
        )

        results = []

        for (
            split_name,
            manifest_path,
            expected_rows,
        ) in specifications:
            logger.info(
                "Manifest | split=%s "
                "path=%s sha256=%s",
                split_name,
                manifest_path,
                sha256_file(
                    manifest_path
                ),
            )

            result = audit_manifest(
                split_name=split_name,
                manifest_path=manifest_path,
                expected_rows=expected_rows,
                dataset_root=dataset_root,
                progress_every=progress_every,
                logger=logger,
            )

            results.append(result)

        for result in results:
            log_split_report(
                result,
                logger,
            )

        total_rows = sum(
            result["rows"]
            for result in results
        )

        total_nontrivial_orientation = sum(
            result[
                "orientation_nontrivial"
            ]
            for result in results
        )

        logger.info("=" * 72)
        logger.info(
            "PHASE-1 RAW DECODE / "
            "GEOMETRY AUDIT: PASS"
        )
        logger.info("=" * 72)

        logger.info(
            "Total frozen images audited: %d",
            total_rows,
        )

        logger.info(
            "Non-trivial EXIF orientations: %d",
            total_nontrivial_orientation,
        )

        logger.info(
            "No resize/crop/augmentation "
            "decision has been applied."
        )

        return 0

    except Exception:
        logger.exception(
            "PHASE-1 RAW DECODE / "
            "GEOMETRY AUDIT: FAIL"
        )
        return 1

    finally:
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())