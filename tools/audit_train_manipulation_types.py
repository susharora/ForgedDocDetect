#!/usr/bin/env python3
"""
Audit manipulation composition in the frozen FantasyID project_train split.

Questions answered:

1. Does every frozen training attack image contain at least one
   altered face region?
2. Does every frozen training attack image contain at least one
   altered non-face/text region?
3. Are there any face-only, text-only, unknown, or annotation-empty
   attack images?
4. Which fields are altered, and how frequently?
5. Does every attack capture have the aligned bona-fide image with the
   same file_stem + hardware_source?

Definitions are taken from the frozen Tech-1 annotation semantics:

    altered region:
        region_provenance_raw == "altered"

    face region:
        field_name == "face"

    text region:
        any altered region whose non-empty field_name != "face"

Only project_train is inspected.
dev_val and held-out test are not used.
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

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config, load_machine_config


FACE_FIELD = "face"
ALTERED_PROVENANCE = "altered"

REQUIRED_REGION_COLUMNS = {
    "split",
    "traffic_type",
    "variant",
    "hardware_source",
    "image_path",
    "file_stem",
    "region_index",
    "field_name",
    "region_provenance_raw",
}

OUTPUT_COLUMNS = [
    "image_path",
    "image_sha256",
    "file_stem",
    "variant",
    "hardware_source",
    "altered_region_count",
    "altered_fields_unique",
    "has_altered_face",
    "has_altered_text",
    "has_unknown_altered_field",
    "manip_type",
    "paired_bonafide_available",
    "paired_bonafide_path",
]


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
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}

    if not isinstance(value, dict):
        raise TypeError(
            f"Top level of YAML must be a mapping: {path}"
        )

    return value


def normalize_text(value: Any) -> str:
    """
    Normalize Excel/pandas values for annotation comparison.

    Empty Excel cells and pandas NaN become "".
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    text = str(value).strip()

    # Defensive handling for explicit null-like text that may have
    # survived spreadsheet serialization.
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
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def configure_logger(
    tool_cfg: dict[str, Any],
) -> tuple[logging.Logger, Path, Path, str]:

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d_%H%M%S_%fZ"
    )

    logging_cfg = tool_cfg["logging"]
    output_cfg = tool_cfg["output"]

    log_dir = resolve_repo_path(
        logging_cfg["directory"]
    )
    output_dir = resolve_repo_path(
        output_cfg["directory"]
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
        / logging_cfg["filename"].format(
            timestamp=timestamp
        )
    )

    csv_path = (
        output_dir
        / output_cfg["filename"].format(
            timestamp=timestamp
        )
    )

    level = getattr(
        logging,
        str(logging_cfg["level"]).upper(),
    )

    logger = logging.getLogger(
        "audit_train_manipulation_types"
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

    return (
        logger,
        log_path,
        csv_path,
        timestamp,
    )


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

    for line in result.stdout.splitlines():
        logger.info(
            "PROVENANCE_GATE | %s",
            line,
        )

    for line in result.stderr.splitlines():
        logger.error(
            "PROVENANCE_GATE_STDERR | %s",
            line,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen Tech-1 provenance gate failed."
        )


def load_project_train_manifest(
    experiment_cfg: dict[str, Any],
) -> tuple[
    list[dict[str, str]],
    Path,
]:

    split_cfg = experiment_cfg[
        "data"
    ][
        "frozen_split"
    ]

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
            csv.DictReader(f)
        )

    expected_rows = split_cfg[
        "project_train"
    ][
        "images"
    ]

    if len(rows) != expected_rows:
        raise ValueError(
            "project_train manifest row count mismatch: "
            f"expected={expected_rows}, actual={len(rows)}"
        )

    return rows, manifest_path


def load_frozen_regions(
    experiment_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Path]:

    discovery_cfg = experiment_cfg[
        "data"
    ][
        "source_discovery"
    ]

    workbook_path = resolve_repo_path(
        discovery_cfg["workbook"]
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

    regions_df = pd.read_excel(
        workbook_path,
        sheet_name="Regions",
        engine="openpyxl",
    )

    missing = (
        REQUIRED_REGION_COLUMNS
        - set(regions_df.columns)
    )

    if missing:
        raise ValueError(
            "Frozen Regions sheet is missing required columns: "
            f"{sorted(missing)}"
        )

    return regions_df, workbook_path


def classify_attack(
    altered_regions: pd.DataFrame,
) -> dict[str, Any]:

    fields = [
        normalize_text(value).casefold()
        for value
        in altered_regions[
            "field_name"
        ].tolist()
    ]

    known_fields = [
        field
        for field in fields
        if field
    ]

    has_face = (
        FACE_FIELD
        in known_fields
    )

    has_text = any(
        field != FACE_FIELD
        for field in known_fields
    )

    has_unknown = (
        len(known_fields)
        != len(fields)
    )

    if has_face and has_text:
        manip_type = "face_and_text"

    elif has_face:
        manip_type = "face_only"

    elif has_text:
        manip_type = "text_only"

    elif len(fields) == 0:
        manip_type = "no_altered_regions"

    else:
        manip_type = "unknown"

    return {
        "altered_region_count":
            len(altered_regions),

        "altered_fields_unique":
            "|".join(
                sorted(
                    set(known_fields)
                )
            ),

        "has_altered_face":
            has_face,

        "has_altered_text":
            has_text,

        "has_unknown_altered_field":
            has_unknown,

        "manip_type":
            manip_type,
    }


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Audit face/text manipulation composition "
            "in frozen FantasyID project_train."
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
            "audit_train_manipulation_types_config.yaml"
        ),
    )

    args = parser.parse_args()

    tool_cfg_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    (
        logger,
        log_path,
        csv_path,
        _timestamp,
    ) = configure_logger(
        tool_cfg
    )

    try:

        experiment_cfg, experiment_path = (
            load_experiment_config(
                args.config
            )
        )

        _, machine_path = (
            load_machine_config(
                args.machine_config,
                required=True,
            )
        )

        logger.info("=" * 72)
        logger.info(
            "FROZEN PROJECT_TRAIN MANIPULATION-TYPE AUDIT"
        )
        logger.info("=" * 72)

        logger.info(
            "Git commit: %s",
            git_commit_sha(),
        )

        logger.info(
            "Audit script SHA-256: %s",
            sha256_file(
                Path(__file__).resolve()
            ),
        )

        logger.info(
            "Audit config SHA-256: %s",
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
            "Definition | altered region: "
            "region_provenance_raw == %r",
            ALTERED_PROVENANCE,
        )

        logger.info(
            "Definition | face field: "
            "field_name == %r",
            FACE_FIELD,
        )

        logger.info(
            "Definition | text field: "
            "non-empty altered field_name != %r",
            FACE_FIELD,
        )

        logger.info(
            "dev_val: NOT USED"
        )

        logger.info(
            "held-out test: NOT ACCESSED"
        )

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

        (
            regions_df,
            workbook_path,
        ) = load_frozen_regions(
            experiment_cfg
        )

        logger.info(
            "Frozen workbook: %s",
            workbook_path,
        )

        logger.info(
            "Frozen workbook SHA-256: %s",
            sha256_file(
                workbook_path
            ),
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

        # ----------------------------------------------------------
        # Restrict strictly to frozen project_train image paths.
        # ----------------------------------------------------------

        manifest_by_path = {
            row["image_path"]: row
            for row in manifest_rows
        }

        attack_rows = [
            row
            for row in manifest_rows
            if row["traffic_type"] == "attack"
        ]

        bonafide_rows = [
            row
            for row in manifest_rows
            if row["traffic_type"] == "bonafide"
        ]

        attack_paths = {
            row["image_path"]
            for row in attack_rows
        }

        project_regions = (
            regions_df[
                regions_df[
                    "image_path"
                ].isin(
                    set(
                        manifest_by_path
                    )
                )
            ]
            .copy()
        )

        logger.info(
            "project_train manifest images: %d",
            len(manifest_rows),
        )

        logger.info(
            "project_train attack images: %d",
            len(attack_rows),
        )

        logger.info(
            "project_train bona-fide images: %d",
            len(bonafide_rows),
        )

        logger.info(
            "Regions belonging to project_train images: %d",
            len(project_regions),
        )

        # ----------------------------------------------------------
        # Build bona-fide capture lookup.
        #
        # Same card + same hardware = aligned physical capture.
        # ----------------------------------------------------------

        bonafide_lookup: dict[
            tuple[str, str],
            str,
        ] = {}

        duplicate_bonafide_keys = []

        for row in bonafide_rows:

            key = (
                row["file_stem"],
                row["hardware_source"],
            )

            if key in bonafide_lookup:

                duplicate_bonafide_keys.append(
                    key
                )

            else:

                bonafide_lookup[
                    key
                ] = row[
                    "image_path"
                ]

        if duplicate_bonafide_keys:

            raise ValueError(
                "Duplicate bona-fide capture keys found: "
                f"{duplicate_bonafide_keys[:10]}"
            )

        # ----------------------------------------------------------
        # Altered-region lookup by attack image.
        # ----------------------------------------------------------

        provenance_series = (
            project_regions[
                "region_provenance_raw"
            ]
            .map(normalize_text)
            .str.casefold()
        )

        altered_regions = (
            project_regions[
                provenance_series
                == ALTERED_PROVENANCE
            ]
            .copy()
        )

        altered_by_image = {
            image_path: group
            for image_path, group
            in altered_regions.groupby(
                "image_path",
                sort=False,
            )
        }

        audit_rows = []

        manipulation_counts = Counter()
        variant_counts = defaultdict(Counter)
        field_counts = Counter()
        field_set_counts = Counter()

        missing_pair_images = []

        for row in attack_rows:

            image_path = row[
                "image_path"
            ]

            image_altered = (
                altered_by_image.get(
                    image_path
                )
            )

            if image_altered is None:

                image_altered = (
                    altered_regions.iloc[
                        0:0
                    ]
                )

            classification = (
                classify_attack(
                    image_altered
                )
            )

            pair_key = (
                row["file_stem"],
                row["hardware_source"],
            )

            paired_path = (
                bonafide_lookup.get(
                    pair_key
                )
            )

            paired_available = (
                paired_path is not None
            )

            if not paired_available:

                missing_pair_images.append(
                    image_path
                )

            result = {
                "image_path":
                    image_path,

                "image_sha256":
                    row["image_sha256"],

                "file_stem":
                    row["file_stem"],

                "variant":
                    row["variant"],

                "hardware_source":
                    row["hardware_source"],

                **classification,

                "paired_bonafide_available":
                    paired_available,

                "paired_bonafide_path":
                    paired_path or "",
            }

            audit_rows.append(
                result
            )

            manipulation_counts[
                classification[
                    "manip_type"
                ]
            ] += 1

            variant_counts[
                row["variant"]
            ][
                classification[
                    "manip_type"
                ]
            ] += 1

            field_set_counts[
                classification[
                    "altered_fields_unique"
                ]
            ] += 1

            for field in (
                classification[
                    "altered_fields_unique"
                ]
                .split("|")
            ):

                if field:
                    field_counts[
                        field
                    ] += 1

        # ----------------------------------------------------------
        # Persist complete image-level audit.
        # ----------------------------------------------------------

        with csv_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=OUTPUT_COLUMNS,
            )

            writer.writeheader()

            writer.writerows(
                audit_rows
            )

        # ----------------------------------------------------------
        # Log summary.
        # ----------------------------------------------------------

        logger.info("-" * 72)
        logger.info(
            "MANIPULATION COMPOSITION"
        )

        for (
            manip_type,
            count,
        ) in sorted(
            manipulation_counts.items()
        ):

            logger.info(
                "  %-22s %d",
                manip_type,
                count,
            )

        logger.info(
            "Composition by attack variant:"
        )

        for variant in sorted(
            variant_counts
        ):

            logger.info(
                "  %s: %s",
                variant,
                dict(
                    variant_counts[
                        variant
                    ]
                ),
            )

        logger.info("-" * 72)
        logger.info(
            "ALTERED FIELD PRESENCE"
        )

        for field, count in (
            field_counts.most_common()
        ):

            logger.info(
                "  %-30s %d images",
                field,
                count,
            )

        logger.info("-" * 72)
        logger.info(
            "DISTINCT ALTERED FIELD SETS"
        )

        for (
            field_set,
            count,
        ) in (
            field_set_counts
            .most_common()
        ):

            logger.info(
                "  %-60s %d",
                field_set or "<EMPTY>",
                count,
            )

        n_attack = len(
            attack_rows
        )

        n_face = sum(
            row[
                "has_altered_face"
            ]
            for row in audit_rows
        )

        n_text = sum(
            row[
                "has_altered_text"
            ]
            for row in audit_rows
        )

        n_both = sum(
            row[
                "manip_type"
            ]
            == "face_and_text"
            for row in audit_rows
        )

        n_unknown = sum(
            row[
                "has_unknown_altered_field"
            ]
            for row in audit_rows
        )

        n_pairs = sum(
            row[
                "paired_bonafide_available"
            ]
            for row in audit_rows
        )

        logger.info("-" * 72)
        logger.info(
            "FINAL CHECKS"
        )

        logger.info(
            "Attack images with altered face: %d / %d",
            n_face,
            n_attack,
        )

        logger.info(
            "Attack images with altered text: %d / %d",
            n_text,
            n_attack,
        )

        logger.info(
            "Attack images classified face_and_text: %d / %d",
            n_both,
            n_attack,
        )

        logger.info(
            "Attack images with unknown/blank altered field: %d",
            n_unknown,
        )

        logger.info(
            "Attack images with aligned bona-fide capture: %d / %d",
            n_pairs,
            n_attack,
        )

        if missing_pair_images:

            logger.error(
                "Missing aligned bona-fide pairs: %d",
                len(
                    missing_pair_images
                ),
            )

            for path in (
                missing_pair_images[:10]
            ):

                logger.error(
                    "  %s",
                    path,
                )

        # ----------------------------------------------------------
        # Scientific gate.
        # ----------------------------------------------------------

        failures = []

        if n_face != n_attack:

            failures.append(
                "not every attack image has an altered face region"
            )

        if n_text != n_attack:

            failures.append(
                "not every attack image has an altered text region"
            )

        if n_both != n_attack:

            failures.append(
                "not every attack image classifies as face_and_text"
            )

        if n_unknown != 0:

            failures.append(
                "one or more altered regions have blank/unknown field_name"
            )

        if n_pairs != n_attack:

            failures.append(
                "not every attack image has an aligned bona-fide capture"
            )

        if failures:

            for failure in failures:

                logger.error(
                    "[FAIL] %s",
                    failure,
                )

            raise RuntimeError(
                "Training manipulation-type audit failed."
            )

        logger.info(
            "[PASS] Every frozen project_train attack "
            "contains altered face + altered text."
        )

        logger.info(
            "[PASS] Every frozen project_train attack "
            "has an aligned bona-fide capture."
        )

        logger.info(
            "Audit CSV: %s",
            csv_path,
        )

        logger.info("=" * 72)
        logger.info(
            "FROZEN PROJECT_TRAIN MANIPULATION-TYPE AUDIT: PASS"
        )
        logger.info("=" * 72)

        return 0

    except Exception:

        logger.exception(
            "FROZEN PROJECT_TRAIN "
            "MANIPULATION-TYPE AUDIT: FAIL"
        )

        return 1

    finally:

        for handler in logger.handlers:

            handler.flush()
            handler.close()

        logger.handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())