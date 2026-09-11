#!/usr/bin/env python3
"""
Exhaustively validate development-only counterfactual restoration.

For every frozen dev_val attack:

    original attack
        +
    aligned dev_val bona-fide capture

construct:

    face restored
        -> manipulated text remains

    text restored
        -> manipulated face remains

    both restored
        -> all annotated manipulated regions restored

Validation performed
--------------------
1. every dev attack has an aligned bona-fide capture;
2. only frozen dev_val paths are used;
3. attack output geometry never changes;
4. face mode restores only face fields;
5. text mode restores only non-face fields;
6. both mode is exactly the union of face + text semantic matches;
7. no output pixel changes outside destination restoration boxes;
8. every requested restoration changes at least one pixel;
9. semantic/spatial matching quality is measured exhaustively;
10. weak geometric matches are surfaced rather than silently ignored;
11. a small deterministic visual cohort is rendered for inspection.

Important
---------
This audit does NOT:
- load a model;
- load a checkpoint;
- perform inference;
- train;
- select a resolution;
- derive a threshold;
- access held-out test images;
- perform Grad-CAM.

The counterfactual images are diagnostic composites and must not be
described as genuine naturally generated face-only/text-only attacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from PIL import Image


# ======================================================================
# Repository
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

from src.counterfactual_restoration import (
    SEMANTIC_SOURCE_FIELD_ALIASES,
    RestorationResult,
    restore_counterfactual,
)


LOGGER = logging.getLogger(
    "audit_counterfactual_restoration"
)


# ======================================================================
# Constants
# ======================================================================

REQUIRED_REGION_COLUMNS = {
    "image_path",
    "region_index",
    "field_name",
    "region_provenance_raw",
    "x",
    "y",
    "width",
    "height",
}


CSV_COLUMNS = [
    "image_path",
    "bonafide_path",
    "file_stem",
    "variant",
    "hardware_source",

    "attack_width",
    "attack_height",
    "bonafide_width",
    "bonafide_height",
    "pair_dimensions_equal",

    "face_region_count",
    "text_region_count",
    "both_region_count",

    "face_min_normalized_iou",
    "text_min_normalized_iou",
    "both_min_normalized_iou",

    "face_min_common_visible_fraction",
    "text_min_common_visible_fraction",
    "both_min_common_visible_fraction",

    "face_resize_count",
    "text_resize_count",
    "both_resize_count",

    "face_mask_pixels",
    "face_changed_pixels",
    "face_changed_fraction_of_mask",

    "text_mask_pixels",
    "text_changed_pixels",
    "text_changed_fraction_of_mask",

    "both_mask_pixels",
    "both_changed_pixels",
    "both_changed_fraction_of_mask",

    "overall_min_normalized_iou",
    "overall_min_common_visible_fraction",

    "visual_sample",
]


VALIDATION_HANDOFF_PATTERN = re.compile(
    r"^VALIDATION_ARTIFACT"
    r" \| status=(?P<status>[A-Z]+)"
    r" \| path=(?P<path>.+)"
    r" \| sha256=(?P<sha256>[0-9a-f]{64})$"
)


# ======================================================================
# Generic helpers
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
            "Tool YAML top level must be a mapping."
        )

    return value


def load_semantic_alias_validation(
    *,
    tool_cfg: Mapping[str, Any],
) -> dict[str, Any]:

    cfg = (
        tool_cfg[
            "semantic_alias_validation"
        ]
    )

    evidence_cfg = (
        cfg[
            "evidence"
        ]
    )

    evidence_path = resolve_repo_path(
        evidence_cfg[
            "path"
        ]
    )

    expected_sha = str(
        evidence_cfg[
            "sha256"
        ]
    )

    actual_sha = sha256_file(
        evidence_path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Semantic-parent probe SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    evidence = load_yaml(
        evidence_path
    )

    if (
        evidence.get(
            "status"
        )
        != "PROBE_ONLY_NOT_FROZEN"
    ):

        raise RuntimeError(
            "Semantic-parent probe has unexpected status."
        )

    expected_map: dict[
        str,
        str,
    ] = {}

    expected_counts: dict[
        tuple[str, str],
        int,
    ] = {}

    for (
        attack_field,
        specification,
    ) in cfg[
        "expected"
    ].items():

        source_field = str(
            specification[
                "source_field"
            ]
        )

        occurrences = int(
            specification[
                "occurrences"
            ]
        )

        expected_map[
            attack_field
        ] = source_field

        expected_counts[
            (
                attack_field,
                source_field,
            )
        ] = occurrences

    if (
        dict(
            SEMANTIC_SOURCE_FIELD_ALIASES
        )
        != expected_map
    ):

        raise RuntimeError(
            "Primitive semantic alias map differs from "
            "the audit's evidence-bound contract."
        )

    probe_summary = (
        evidence[
            "candidate_summary_by_attack_field"
        ]
    )

    if set(
        probe_summary
    ) != set(
        expected_map
    ):

        raise RuntimeError(
            "Probe semantic fields differ from expected aliases."
        )

    for (
        attack_field,
        source_field,
    ) in expected_map.items():

        record = (
            probe_summary[
                attack_field
            ]
        )

        if (
            record[
                "dominant_candidate_parent"
            ]
            != source_field
            or not record[
                "candidate_is_unanimous"
            ]
            or not record[
                "candidate_meets_probe_coverage_floor"
            ]
        ):

            raise RuntimeError(
                "Probe does not support configured semantic alias:\n"
                f"  {attack_field!r} -> {source_field!r}"
            )

    if int(
        evidence[
            "unmatched_semantic_occurrences"
        ]
    ) != sum(
        expected_counts.values()
    ):

        raise RuntimeError(
            "Probe unmatched occurrence count differs "
            "from configured alias count."
        )

    return {
        "evidence_path":
            evidence_path,

        "evidence_sha256":
            actual_sha,

        "expected_counts":
            expected_counts,
    }

def cell_text(
    value: Any,
) -> str:

    if value is None:

        return ""

    try:

        if pd.isna(
            value
        ):

            return ""

    except TypeError:

        pass

    return str(
        value
    ).strip()


# ======================================================================
# Git / canonical validator
# ======================================================================

def require_clean_git() -> str:

    commit = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status = subprocess.run(
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
    ).stdout.strip()

    if status:

        raise RuntimeError(
            "Git working tree must be clean before the audit:\n"
            f"{status}"
        )

    if len(
        commit
    ) != 40:

        raise RuntimeError(
            "Git HEAD is not a full SHA."
        )

    return commit


def run_canonical_validator(
    *,
    experiment_path: Path,
    machine_path: Path,
) -> dict[str, str]:

    result = subprocess.run(
        [
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
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Canonical experiment validator failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    handoffs = []

    for line in (
        result.stdout
        .splitlines()
    ):

        match = (
            VALIDATION_HANDOFF_PATTERN
            .fullmatch(
                line.strip()
            )
        )

        if match is not None:

            handoffs.append(
                match
            )

    if len(
        handoffs
    ) != 1:

        raise RuntimeError(
            "Canonical validator did not produce exactly one "
            "VALIDATION_ARTIFACT handoff."
        )

    match = (
        handoffs[
            0
        ]
    )

    if (
        match.group(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Canonical validator did not report PASS."
        )

    path = Path(
        match.group(
            "path"
        )
    ).expanduser().resolve()

    expected_sha = (
        match.group(
            "sha256"
        )
    )

    if not path.is_file():

        raise FileNotFoundError(
            "Canonical validation artifact is missing:\n"
            f"  {path}"
        )

    actual_sha = sha256_file(
        path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Canonical validation artifact SHA mismatch."
        )

    return {
        "path":
            str(
                path
            ),

        "sha256":
            actual_sha,
    }


# ======================================================================
# Audit outputs / logging
# ======================================================================

def configure_outputs(
    *,
    tool_cfg: Mapping[str, Any],
) -> tuple[
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

    logging_cfg = (
        tool_cfg[
            "logging"
        ]
    )

    output_cfg = (
        tool_cfg[
            "output"
        ]
    )

    audit_cfg = (
        tool_cfg[
            "audit"
        ]
    )

    log_dir = resolve_repo_path(
        logging_cfg[
            "directory"
        ]
    )

    csv_dir = resolve_repo_path(
        output_cfg[
            "directory"
        ]
    )

    visual_root = resolve_repo_path(
        audit_cfg[
            "visual_root"
        ]
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_dir = (
        visual_root
        / timestamp
    )

    visual_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path = (
        log_dir
        / str(
            logging_cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    csv_path = (
        csv_dir
        / str(
            output_cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    LOGGER.handlers.clear()
    LOGGER.propagate = False

    LOGGER.setLevel(
        getattr(
            logging,
            str(
                logging_cfg[
                    "level"
                ]
            ).upper(),
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
        log_path,
        csv_path,
        visual_dir,
    )


# ======================================================================
# Frozen dev manifest
# ======================================================================

def load_dev_manifest(
    *,
    experiment_cfg: Mapping[str, Any],
) -> tuple[
    list[
        dict[
            str,
            str,
        ]
    ],
    Path,
]:

    dev_cfg = (
        experiment_cfg[
            "data"
        ][
            "frozen_split"
        ][
            "dev_val"
        ]
    )

    path = resolve_repo_path(
        dev_cfg[
            "path"
        ]
    )

    expected_sha = str(
        dev_cfg[
            "sha256"
        ]
    )

    actual_sha = sha256_file(
        path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Frozen dev_val manifest SHA mismatch:\n"
            f"  expected={expected_sha}\n"
            f"  actual={actual_sha}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    expected_rows = int(
        dev_cfg[
            "images"
        ]
    )

    if len(
        rows
    ) != expected_rows:

        raise RuntimeError(
            "Frozen dev_val row count mismatch."
        )

    seen_paths: set[
        str
    ] = set()

    for (
        index,
        row,
    ) in enumerate(
        rows
    ):

        if (
            row[
                "project_role"
            ]
            != "dev_val"
        ):

            raise RuntimeError(
                "Unexpected project_role:\n"
                f"  row={index}\n"
                f"  value={row['project_role']!r}"
            )

        image_path = (
            row[
                "image_path"
            ]
            .replace(
                "\\",
                "/",
            )
        )

        # Our internal dev_val is carved only from FantasyID train.
        # Any non-train path is a hard stop.
        if not image_path.startswith(
            "train/"
        ):

            raise RuntimeError(
                "Development audit encountered non-train image path:\n"
                f"  {image_path}"
            )

        if image_path in seen_paths:

            raise RuntimeError(
                "Duplicate image path in dev manifest:\n"
                f"  {image_path}"
            )

        seen_paths.add(
            image_path
        )

    return (
        rows,
        path,
    )


# ======================================================================
# Frozen region annotations
# ======================================================================

def load_required_regions(
    *,
    experiment_cfg: Mapping[str, Any],
    required_paths: set[str],
) -> tuple[
    dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ],
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

    expected_sha = str(
        discovery_cfg[
            "sha256"
        ]
    )

    actual_sha = sha256_file(
        workbook_path
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            "Frozen discovery workbook SHA mismatch."
        )

    regions = pd.read_excel(
        workbook_path,
        sheet_name="Regions",
        engine="openpyxl",
    )

    missing_columns = (
        REQUIRED_REGION_COLUMNS
        - set(
            regions.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            "Frozen Regions sheet missing columns:\n"
            f"  {sorted(missing_columns)}"
        )

    path_keys = (
        regions[
            "image_path"
        ]
        .map(
            cell_text
        )
    )

    selected = (
        regions.loc[
            path_keys.isin(
                required_paths
            )
        ]
        .copy()
    )

    selected[
        "_audit_image_path"
    ] = (
        selected[
            "image_path"
        ]
        .map(
            cell_text
        )
    )

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for (
        image_path,
        frame,
    ) in selected.groupby(
        "_audit_image_path",
        sort=False,
    ):

        grouped[
            str(
                image_path
            )
        ] = (
            frame
            .drop(
                columns=[
                    "_audit_image_path"
                ]
            )
            .to_dict(
                orient="records"
            )
        )

    missing_paths = (
        required_paths
        - set(
            grouped
        )
    )

    if missing_paths:

        raise RuntimeError(
            "Region annotations missing for development paths:\n"
            f"  first_missing="
            f"{sorted(missing_paths)[:10]}"
        )

    # Strong guard: every row that will actually be used belongs to
    # one of the explicitly requested dev_val paths.
    unexpected_used_paths = (
        set(
            grouped
        )
        - required_paths
    )

    if unexpected_used_paths:

        raise RuntimeError(
            "Unexpected annotation paths entered development audit."
        )

    return (
        grouped,
        workbook_path,
    )


# ======================================================================
# Dataset path guard
# ======================================================================

def resolve_dataset_image(
    *,
    dataset_root: Path,
    relative_path: str,
) -> Path:

    if not (
        relative_path
        .replace(
            "\\",
            "/",
        )
        .startswith(
            "train/"
        )
    ):

        raise RuntimeError(
            "Counterfactual audit refuses non-train image path:\n"
            f"  {relative_path}"
        )

    path = (
        dataset_root
        / relative_path
    ).resolve()

    try:

        path.relative_to(
            dataset_root
        )

    except ValueError as exc:

        raise RuntimeError(
            "Image path escapes dataset root:\n"
            f"  {path}"
        ) from exc

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    return path


# ======================================================================
# Restoration contract checks
# ======================================================================

def match_map(
    result: RestorationResult,
) -> dict[
    tuple[
        str,
        int,
    ],
    tuple[
        str,
        int,
    ],
]:

    mapping: dict[
        tuple[
            str,
            int,
        ],
        tuple[
            str,
            int,
        ],
    ] = {}

    for match in (
        result.matches
    ):

        key = (
            match.field_name,
            match.attack_region_index,
        )

        if key in mapping:

            raise RuntimeError(
                "Duplicate attack-region match key:\n"
                f"  {key}"
            )

        mapping[
            key
        ] = (
            match.source_field_name,
            match.bonafide_region_index,
        )

    return mapping

def validate_semantic_source_contract(
    result: RestorationResult,
) -> None:

    for match in (
        result.matches
    ):

        if (
            match.source_field_name
            == match.field_name
        ):
            continue

        permitted_source = (
            SEMANTIC_SOURCE_FIELD_ALIASES.get(
                match.field_name
            )
        )

        if (
            permitted_source
            != match.source_field_name
        ):

            raise RuntimeError(
                "Restoration used an unapproved semantic alias:\n"
                f"  attack_field={match.field_name!r}\n"
                f"  source_field={match.source_field_name!r}"
            )

def validate_cross_mode_consistency(
    *,
    face: RestorationResult,
    text: RestorationResult,
    both: RestorationResult,
) -> None:

    for result in (
        face,
        text,
        both,
    ):

        validate_semantic_source_contract(
            result
        )

    if face.mode != "face":

        raise RuntimeError(
            "Face restoration returned wrong mode."
        )

    if text.mode != "text":

        raise RuntimeError(
            "Text restoration returned wrong mode."
        )

    if both.mode != "both":

        raise RuntimeError(
            "Both restoration returned wrong mode."
        )

    if any(
        match.field_name
        != "face"

        for match
        in face.matches
    ):

        raise RuntimeError(
            "Face restoration contains non-face field."
        )

    if any(
        match.field_name
        == "face"

        for match
        in text.matches
    ):

        raise RuntimeError(
            "Text restoration contains a face field."
        )

    face_map = match_map(
        face
    )

    text_map = match_map(
        text
    )

    both_map = match_map(
        both
    )

    if (
        set(
            face_map
        )
        & set(
            text_map
        )
    ):

        raise RuntimeError(
            "Face/text destination-region sets overlap."
        )

    expected_both = {
        **face_map,
        **text_map,
    }

    if both_map != expected_both:

        raise RuntimeError(
            "Both restoration is not exactly the union "
            "of face + text semantic matches."
        )

    if (
        len(
            both.applied
        )
        != (
            len(
                face.applied
            )
            + len(
                text.applied
            )
        )
    ):

        raise RuntimeError(
            "Both-restoration applied-region count "
            "does not equal face + text."
        )


# ======================================================================
# Pixel locality
# ======================================================================

def restoration_mask(
    *,
    image_size: tuple[
        int,
        int,
    ],
    result: RestorationResult,
) -> np.ndarray:

    width, height = (
        image_size
    )

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.bool_,
    )

    for applied in (
        result.applied
    ):

        (
            left,
            top,
            right,
            bottom,
        ) = (
            applied
            .destination_box
        )

        if not (
            0 <= left < right <= width
            and
            0 <= top < bottom <= height
        ):

            raise RuntimeError(
                "Restoration destination box is outside "
                "the actual attack image:\n"
                f"  box={applied.destination_box}\n"
                f"  image_size={image_size}"
            )

        mask[
            top:bottom,
            left:right,
        ] = True

    return mask


def validate_pixel_locality(
    *,
    original: Image.Image,
    result: RestorationResult,
    label: str,
) -> dict[str, Any]:

    if result.image.mode != "RGB":

        raise RuntimeError(
            f"{label}: restored image is not RGB."
        )

    if (
        result.image.size
        != original.size
    ):

        raise RuntimeError(
            f"{label}: restoration changed full-image geometry."
        )

    original_array = np.asarray(
        original,
        dtype=np.uint8,
    )

    restored_array = np.asarray(
        result.image,
        dtype=np.uint8,
    )

    if (
        original_array.shape
        != restored_array.shape
    ):

        raise RuntimeError(
            f"{label}: array shape mismatch."
        )

    changed = np.any(
        original_array
        != restored_array,
        axis=2,
    )

    mask = restoration_mask(
        image_size=(
            original.size
        ),
        result=(
            result
        ),
    )

    outside = (
        changed
        & ~mask
    )

    outside_count = int(
        outside.sum()
    )

    if outside_count != 0:

        raise RuntimeError(
            f"{label}: pixels changed outside restoration "
            f"destination boxes: {outside_count}"
        )

    mask_pixels = int(
        mask.sum()
    )

    changed_pixels = int(
        changed.sum()
    )

    if mask_pixels <= 0:

        raise RuntimeError(
            f"{label}: restoration mask contains zero pixels."
        )

    if changed_pixels <= 0:

        raise RuntimeError(
            f"{label}: restoration changed zero pixels."
        )

    if changed_pixels > mask_pixels:

        raise RuntimeError(
            f"{label}: changed pixel count exceeds mask."
        )

    return {
        "mask_pixels":
            mask_pixels,

        "changed_pixels":
            changed_pixels,

        "changed_fraction":
            (
                changed_pixels
                / mask_pixels
            ),
    }


# ======================================================================
# Restoration statistics
# ======================================================================

def restoration_statistics(
    result: RestorationResult,
) -> dict[str, Any]:

    if not result.applied:

        raise RuntimeError(
            "Restoration contains zero applied regions."
        )

    minimum_iou = min(
        item.normalized_iou

        for item
        in result.applied
    )

    minimum_visible = min(
        item.common_visible_fraction

        for item
        in result.applied
    )

    if not (
        0.0
        <= minimum_iou
        <= 1.0
    ):

        raise RuntimeError(
            "Invalid normalized IoU."
        )

    if not (
        0.0
        < minimum_visible
        <= 1.0
    ):

        raise RuntimeError(
            "Invalid common visible fraction."
        )

    return {
        "region_count":
            len(
                result.applied
            ),

        "minimum_iou":
            minimum_iou,

        "minimum_visible":
            minimum_visible,

        "resize_count":
            sum(
                int(
                    item.resize_required
                )

                for item
                in result.applied
            ),
    }


# ======================================================================
# Visual evidence
# ======================================================================

def safe_component(
    value: str,
) -> str:

    cleaned = "".join(
        character

        if (
            character.isalnum()
            or character in "-_."
        )

        else "-"

        for character
        in value
    )

    return (
        cleaned
        or "empty"
    )


def restoration_metadata(
    result: RestorationResult,
) -> dict[str, Any]:

    return {
        "mode":
            result.mode,

        "matches":
            [
                {
                    "field_name":
                        match.field_name,

                    "source_field_name":
                        match.source_field_name,

                    "attack_region_index":
                        match.attack_region_index,

                    "bonafide_region_index":
                        match.bonafide_region_index,

                    "attack_box":
                        {
                            "x":
                                match.attack_box.x,

                            "y":
                                match.attack_box.y,

                            "width":
                                match.attack_box.width,

                            "height":
                                match.attack_box.height,
                        },

                    "bonafide_box":
                        {
                            "x":
                                match.bonafide_box.x,

                            "y":
                                match.bonafide_box.y,

                            "width":
                                match.bonafide_box.width,

                            "height":
                                match.bonafide_box.height,
                        },

                    "normalized_iou":
                        match.normalized_iou,

                    "normalized_center_distance":
                        match.normalized_center_distance,
                }

                for match
                in result.matches
            ],

        "applied":
            [
                {
                    "field_name":
                        item.field_name,

                    "attack_region_index":
                        item.attack_region_index,

                    "bonafide_region_index":
                        item.bonafide_region_index,

                    "normalized_iou":
                        item.normalized_iou,

                    "source_crop_box":
                        list(
                            item.source_crop_box
                        ),

                    "destination_box":
                        list(
                            item.destination_box
                        ),

                    "common_visible_fraction":
                        item.common_visible_fraction,

                    "source_patch_size":
                        list(
                            item.source_patch_size
                        ),

                    "destination_patch_size":
                        list(
                            item.destination_patch_size
                        ),

                    "resize_required":
                        item.resize_required,
                }

                for item
                in result.applied
            ],
    }


def render_visual_case(
    *,
    visual_dir: Path,
    attack_row: Mapping[str, str],
    bonafide_path_relative: str,
    attack: Image.Image,
    bonafide: Image.Image,
    face: RestorationResult,
    text: RestorationResult,
    both: RestorationResult,
) -> None:

    case_name = (
        f"{safe_component(attack_row['variant'])}"
        "__"
        f"{safe_component(attack_row['hardware_source'])}"
        "__"
        f"{safe_component(attack_row['file_stem'])}"
    )

    case_dir = (
        visual_dir
        / case_name
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    images = (
        (
            "01_original_attack.png",
            attack,
        ),
        (
            "02_face_restored__text_remains.png",
            face.image,
        ),
        (
            "03_text_restored__face_remains.png",
            text.image,
        ),
        (
            "04_both_restored.png",
            both.image,
        ),
        (
            "05_bonafide_reference.png",
            bonafide,
        ),
    )

    for (
        filename,
        image,
    ) in images:

        image.save(
            case_dir
            / filename,
            format="PNG",
            optimize=False,
        )

    metadata = {
        "attack_image_path":
            attack_row[
                "image_path"
            ],

        "bonafide_image_path":
            bonafide_path_relative,

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

        "interpretation":
            {
                "face_restored":
                    (
                        "Altered face region(s) restored; "
                        "manipulated text remains."
                    ),

                "text_restored":
                    (
                        "Altered non-face region(s) restored; "
                        "manipulated face remains."
                    ),

                "both_restored":
                    (
                        "All known altered regions restored. "
                        "This is not claimed to be identical "
                        "to the natural bona-fide capture."
                    ),
            },

        "face":
            restoration_metadata(
                face
            ),

        "text":
            restoration_metadata(
                text
            ),

        "both":
            restoration_metadata(
                both
            ),
    }

    with (
        case_dir
        / "metadata.yaml"
    ).open(
        "x",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            metadata,
            file,
            sort_keys=False,
        )


# ======================================================================
# Main
# ======================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively audit development-only "
            "counterfactual restoration."
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
            "audit_counterfactual_restoration_config.yaml"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Must happen before either validator or audit log creation.
    # ------------------------------------------------------------------

    git_commit = (
        require_clean_git()
    )

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

    validator_artifact = (
        run_canonical_validator(
            experiment_path=(
                experiment_path
            ),
            machine_path=(
                machine_path
            ),
        )
    )

    tool_cfg_path = resolve_repo_path(
        args.audit_config
    )

    tool_cfg = load_yaml(
        tool_cfg_path
    )

    (
        log_path,
        csv_path,
        visual_dir,
    ) = configure_outputs(
        tool_cfg=(
            tool_cfg
        )
    )

    try:

        audit_cfg = (
            tool_cfg[
                "audit"
            ]
        )

        probe_minimum_iou = float(
            audit_cfg[
                "probe_minimum_normalized_iou"
            ]
        )

        acceptance_minimum_iou = float(
            audit_cfg[
                "acceptance_minimum_normalized_iou"
            ]
        )

        samples_per_group = int(
            audit_cfg[
                "samples_per_variant_hardware"
            ]
        )

        weakest_examples = int(
            audit_cfg[
                "weakest_examples"
            ]
        )

        weakest_matches_to_log = int(
            audit_cfg[
                "weakest_matches_to_log"
            ]
        )

        if probe_minimum_iou != 0.0:

            raise RuntimeError(
                "Exhaustive audit requires "
                "probe_minimum_normalized_iou=0.0."
            )

        if not (
            0.0
            < acceptance_minimum_iou
            <= 1.0
        ):

            raise RuntimeError(
                "Invalid acceptance IoU."
            )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "DEV COUNTERFACTUAL RESTORATION AUDIT"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            git_commit,
        )

        LOGGER.info(
            "Audit script SHA-256: %s",
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
        )

        LOGGER.info(
            "Restoration primitive SHA-256: %s",
            sha256_file(
                REPO_ROOT
                / "src"
                / "counterfactual_restoration.py"
            ),
        )

        LOGGER.info(
            "Audit config SHA-256: %s",
            sha256_file(
                tool_cfg_path
            ),
        )

        LOGGER.info(
            "Experiment config SHA-256: %s",
            sha256_file(
                experiment_path
            ),
        )

        LOGGER.info(
            "Canonical validator: %s",
            validator_artifact[
                "path"
            ],
        )

        LOGGER.info(
            "Canonical validator SHA-256: %s",
            validator_artifact[
                "sha256"
            ],
        )

        LOGGER.info(
            "Machine: %s",
            machine_cfg[
                "machine"
            ][
                "id"
            ],
        )

        LOGGER.info(
            "Probe match floor: %.6f",
            probe_minimum_iou,
        )

        LOGGER.info(
            "Acceptance minimum normalized IoU: %.6f",
            acceptance_minimum_iou,
        )

        LOGGER.info(
            "Held-out test images: NOT ACCESSED"
        )

        LOGGER.info(
            "Frozen workbook rows used: required dev_val paths only"
        )

        # ==============================================================
        # Development cohort / pairing
        # ==============================================================

        (
            manifest_rows,
            manifest_path,
        ) = load_dev_manifest(
            experiment_cfg=(
                experiment_cfg
            )
        )

        LOGGER.info(
            "Frozen dev manifest: %s",
            manifest_path,
        )

        LOGGER.info(
            "Frozen dev manifest SHA-256: %s",
            sha256_file(
                manifest_path
            ),
        )

        attack_rows: list[
            dict[
                str,
                str,
            ]
        ] = []

        bonafide_by_key: dict[
            tuple[
                str,
                str,
            ],
            dict[
                str,
                str,
            ],
        ] = {}

        for row in (
            manifest_rows
        ):

            traffic_type = (
                row[
                    "traffic_type"
                ]
            )

            if traffic_type == "attack":

                attack_rows.append(
                    row
                )

            elif traffic_type == "bonafide":

                key = (
                    row[
                        "file_stem"
                    ],
                    row[
                        "hardware_source"
                    ],
                )

                if key in bonafide_by_key:

                    raise RuntimeError(
                        "Duplicate dev bona-fide pairing key:\n"
                        f"  {key}"
                    )

                bonafide_by_key[
                    key
                ] = row

            else:

                raise RuntimeError(
                    "Unexpected dev traffic_type:\n"
                    f"  {traffic_type!r}"
                )

        expected_attack_count = int(
            experiment_cfg[
                "class_contract"
            ][
                "dev_val_counts"
            ][
                "attack"
            ]
        )

        if len(
            attack_rows
        ) != expected_attack_count:

            raise RuntimeError(
                "Unexpected number of dev attacks:\n"
                f"  expected={expected_attack_count}\n"
                f"  actual={len(attack_rows)}"
            )

        pair_by_attack_path: dict[
            str,
            dict[
                str,
                str,
            ],
        ] = {}

        required_paths: set[
            str
        ] = set()

        for attack_row in (
            attack_rows
        ):

            key = (
                attack_row[
                    "file_stem"
                ],
                attack_row[
                    "hardware_source"
                ],
            )

            if key not in bonafide_by_key:

                raise RuntimeError(
                    "Dev attack lacks aligned bona-fide capture:\n"
                    f"  {attack_row['image_path']}"
                )

            bonafide_row = (
                bonafide_by_key[
                    key
                ]
            )

            attack_path = (
                attack_row[
                    "image_path"
                ]
            )

            pair_by_attack_path[
                attack_path
            ] = (
                bonafide_row
            )

            required_paths.add(
                attack_path
            )

            required_paths.add(
                bonafide_row[
                    "image_path"
                ]
            )

        LOGGER.info(
            "Development attacks: %d",
            len(
                attack_rows
            ),
        )

        LOGGER.info(
            "Aligned bona-fide captures: %d",
            len(
                bonafide_by_key
            ),
        )

        LOGGER.info(
            "Required dev image paths for restoration audit: %d",
            len(
                required_paths
            ),
        )

        # ==============================================================
        # Frozen annotations
        # ==============================================================

        (
            regions_by_path,
            workbook_path,
        ) = load_required_regions(
            experiment_cfg=(
                experiment_cfg
            ),
            required_paths=(
                required_paths
            ),
        )

        LOGGER.info(
            "Frozen discovery workbook: %s",
            workbook_path,
        )

        LOGGER.info(
            "Frozen discovery workbook SHA-256: %s",
            sha256_file(
                workbook_path
            ),
        )

        # ==============================================================
        # Deterministic representative sample candidates
        # ==============================================================

        grouped_attacks: dict[
            tuple[
                str,
                str,
            ],
            list[
                dict[
                    str,
                    str,
                ]
            ],
        ] = defaultdict(
            list
        )

        for row in (
            attack_rows
        ):

            grouped_attacks[
                (
                    row[
                        "variant"
                    ],
                    row[
                        "hardware_source"
                    ],
                )
            ].append(
                row
            )

        normal_visual_paths: set[
            str
        ] = set()

        for group in (
            grouped_attacks.values()
        ):

            group.sort(
                key=lambda row:
                    row[
                        "image_path"
                    ]
            )

            for row in group[
                :samples_per_group
            ]:

                normal_visual_paths.add(
                    row[
                        "image_path"
                    ]
                )

        # ==============================================================
        # Exhaustive restoration pass
        # ==============================================================

        dataset_root = resolve_repo_path(
            machine_cfg[
                "paths"
            ][
                "dataset_root"
            ]
        )

        csv_rows: list[
            dict[
                str,
                Any,
            ]
        ] = []

        all_match_records: list[
            dict[
                str,
                Any,
            ]
        ] = []

        dimension_mismatch_pairs = 0

        total_resize_count = 0

        all_iou_values: list[
            float
        ] = []

        all_visible_values: list[
            float
        ] = []

        changed_fractions: list[
            float
        ] = []

        ordered_attacks = sorted(
            attack_rows,
            key=lambda row:
                row[
                    "image_path"
                ],
        )

        for (
            index,
            attack_row,
        ) in enumerate(
            ordered_attacks,
            start=1,
        ):

            attack_relative = (
                attack_row[
                    "image_path"
                ]
            )

            bonafide_row = (
                pair_by_attack_path[
                    attack_relative
                ]
            )

            bonafide_relative = (
                bonafide_row[
                    "image_path"
                ]
            )

            attack_path = (
                resolve_dataset_image(
                    dataset_root=(
                        dataset_root
                    ),
                    relative_path=(
                        attack_relative
                    ),
                )
            )

            bonafide_path = (
                resolve_dataset_image(
                    dataset_root=(
                        dataset_root
                    ),
                    relative_path=(
                        bonafide_relative
                    ),
                )
            )

            with Image.open(
                attack_path
            ) as image:

                attack = (
                    image.convert(
                        "RGB"
                    )
                )

            with Image.open(
                bonafide_path
            ) as image:

                bonafide = (
                    image.convert(
                        "RGB"
                    )
                )

            pair_dimensions_equal = (
                attack.size
                == bonafide.size
            )

            if not pair_dimensions_equal:

                dimension_mismatch_pairs += 1

            attack_region_rows = (
                regions_by_path[
                    attack_relative
                ]
            )

            bonafide_region_rows = (
                regions_by_path[
                    bonafide_relative
                ]
            )

            face = restore_counterfactual(
                attack_image=(
                    attack
                ),
                bonafide_image=(
                    bonafide
                ),
                attack_rows=(
                    attack_region_rows
                ),
                bonafide_rows=(
                    bonafide_region_rows
                ),
                mode="face",
                minimum_normalized_iou=(
                    probe_minimum_iou
                ),
            )

            text = restore_counterfactual(
                attack_image=(
                    attack
                ),
                bonafide_image=(
                    bonafide
                ),
                attack_rows=(
                    attack_region_rows
                ),
                bonafide_rows=(
                    bonafide_region_rows
                ),
                mode="text",
                minimum_normalized_iou=(
                    probe_minimum_iou
                ),
            )

            both = restore_counterfactual(
                attack_image=(
                    attack
                ),
                bonafide_image=(
                    bonafide
                ),
                attack_rows=(
                    attack_region_rows
                ),
                bonafide_rows=(
                    bonafide_region_rows
                ),
                mode="both",
                minimum_normalized_iou=(
                    probe_minimum_iou
                ),
            )

            validate_cross_mode_consistency(
                face=(
                    face
                ),
                text=(
                    text
                ),
                both=(
                    both
                ),
            )

            face_pixels = (
                validate_pixel_locality(
                    original=(
                        attack
                    ),
                    result=(
                        face
                    ),
                    label="face",
                )
            )

            text_pixels = (
                validate_pixel_locality(
                    original=(
                        attack
                    ),
                    result=(
                        text
                    ),
                    label="text",
                )
            )

            both_pixels = (
                validate_pixel_locality(
                    original=(
                        attack
                    ),
                    result=(
                        both
                    ),
                    label="both",
                )
            )

            face_stats = (
                restoration_statistics(
                    face
                )
            )

            text_stats = (
                restoration_statistics(
                    text
                )
            )

            both_stats = (
                restoration_statistics(
                    both
                )
            )

            overall_min_iou = min(
                face_stats[
                    "minimum_iou"
                ],
                text_stats[
                    "minimum_iou"
                ],
                both_stats[
                    "minimum_iou"
                ],
            )

            overall_min_visible = min(
                face_stats[
                    "minimum_visible"
                ],
                text_stats[
                    "minimum_visible"
                ],
                both_stats[
                    "minimum_visible"
                ],
            )

            total_resize_count += (
                face_stats[
                    "resize_count"
                ]
                +
                text_stats[
                    "resize_count"
                ]
                +
                both_stats[
                    "resize_count"
                ]
            )

            changed_fractions.extend(
                [
                    face_pixels[
                        "changed_fraction"
                    ],
                    text_pixels[
                        "changed_fraction"
                    ],
                    both_pixels[
                        "changed_fraction"
                    ],
                ]
            )

            # Use "both" once when collecting match statistics because
            # face/text modes are exact subsets of it.
            applied_by_key = {
                (
                    item.field_name,
                    item.source_field_name,
                    item.attack_region_index,
                    item.bonafide_region_index,
                ):                 
                    item

                for item
                in both.applied
            }

            for match in (
                both.matches
            ):

                key = (
                    match.field_name,
                    match.source_field_name,
                    match.attack_region_index,
                    match.bonafide_region_index,
                )

                applied = (
                    applied_by_key[
                        key
                    ]
                )

                all_iou_values.append(
                    match.normalized_iou
                )

                all_visible_values.append(
                    applied.common_visible_fraction
                )

                all_match_records.append(
                    {
                        "image_path":
                            attack_relative,

                        "source_field_name":
                            match.source_field_name,

                        "bonafide_path":
                            bonafide_relative,

                        "variant":
                            attack_row[
                                "variant"
                            ],

                        "hardware_source":
                            attack_row[
                                "hardware_source"
                            ],

                        "file_stem":
                            attack_row[
                                "file_stem"
                            ],

                        "field_name":
                            match.field_name,

                        "attack_region_index":
                            match.attack_region_index,

                        "bonafide_region_index":
                            match.bonafide_region_index,

                        "normalized_iou":
                            match.normalized_iou,

                        "normalized_center_distance":
                            (
                                match
                                .normalized_center_distance
                            ),

                        "common_visible_fraction":
                            (
                                applied
                                .common_visible_fraction
                            ),

                        "resize_required":
                            applied.resize_required,
                    }
                )

            csv_rows.append(
                {
                    "image_path":
                        attack_relative,

                    "bonafide_path":
                        bonafide_relative,

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

                    "attack_width":
                        attack.width,

                    "attack_height":
                        attack.height,

                    "bonafide_width":
                        bonafide.width,

                    "bonafide_height":
                        bonafide.height,

                    "pair_dimensions_equal":
                        pair_dimensions_equal,

                    "face_region_count":
                        face_stats[
                            "region_count"
                        ],

                    "text_region_count":
                        text_stats[
                            "region_count"
                        ],

                    "both_region_count":
                        both_stats[
                            "region_count"
                        ],

                    "face_min_normalized_iou":
                        face_stats[
                            "minimum_iou"
                        ],

                    "text_min_normalized_iou":
                        text_stats[
                            "minimum_iou"
                        ],

                    "both_min_normalized_iou":
                        both_stats[
                            "minimum_iou"
                        ],

                    "face_min_common_visible_fraction":
                        face_stats[
                            "minimum_visible"
                        ],

                    "text_min_common_visible_fraction":
                        text_stats[
                            "minimum_visible"
                        ],

                    "both_min_common_visible_fraction":
                        both_stats[
                            "minimum_visible"
                        ],

                    "face_resize_count":
                        face_stats[
                            "resize_count"
                        ],

                    "text_resize_count":
                        text_stats[
                            "resize_count"
                        ],

                    "both_resize_count":
                        both_stats[
                            "resize_count"
                        ],

                    "face_mask_pixels":
                        face_pixels[
                            "mask_pixels"
                        ],

                    "face_changed_pixels":
                        face_pixels[
                            "changed_pixels"
                        ],

                    "face_changed_fraction_of_mask":
                        face_pixels[
                            "changed_fraction"
                        ],

                    "text_mask_pixels":
                        text_pixels[
                            "mask_pixels"
                        ],

                    "text_changed_pixels":
                        text_pixels[
                            "changed_pixels"
                        ],

                    "text_changed_fraction_of_mask":
                        text_pixels[
                            "changed_fraction"
                        ],

                    "both_mask_pixels":
                        both_pixels[
                            "mask_pixels"
                        ],

                    "both_changed_pixels":
                        both_pixels[
                            "changed_pixels"
                        ],

                    "both_changed_fraction_of_mask":
                        both_pixels[
                            "changed_fraction"
                        ],

                    "overall_min_normalized_iou":
                        overall_min_iou,

                    "overall_min_common_visible_fraction":
                        overall_min_visible,

                    "visual_sample":
                        False,
                }
            )

            if (
                index % 25
                == 0
                or index
                == len(
                    ordered_attacks
                )
            ):

                LOGGER.info(
                    "Processed %d / %d development attacks",
                    index,
                    len(
                        ordered_attacks
                    ),
                )

        if len(
            csv_rows
        ) != expected_attack_count:

            raise RuntimeError(
                "Not every dev attack completed restoration."
            )

        # ==============================================================
        # Select weak cases for visual inspection
        # ==============================================================

        weakest_image_rows = sorted(
            csv_rows,
            key=lambda row: (
                row[
                    "overall_min_normalized_iou"
                ],
                row[
                    "image_path"
                ],
            ),
        )

        weak_visual_paths = {
            row[
                "image_path"
            ]

            for row
            in weakest_image_rows[
                :weakest_examples
            ]
        }

        visual_paths = (
            normal_visual_paths
            | weak_visual_paths
        )

        csv_by_path = {
            row[
                "image_path"
            ]:
                row

            for row
            in csv_rows
        }

        for path in (
            visual_paths
        ):

            csv_by_path[
                path
            ][
                "visual_sample"
            ] = True

        # ==============================================================
        # Render only selected visual cases
        # ==============================================================

        attack_by_path = {
            row[
                "image_path"
            ]:
                row

            for row
            in attack_rows
        }

        for attack_relative in sorted(
            visual_paths
        ):

            attack_row = (
                attack_by_path[
                    attack_relative
                ]
            )

            bonafide_row = (
                pair_by_attack_path[
                    attack_relative
                ]
            )

            bonafide_relative = (
                bonafide_row[
                    "image_path"
                ]
            )

            attack_path = resolve_dataset_image(
                dataset_root=(
                    dataset_root
                ),
                relative_path=(
                    attack_relative
                ),
            )

            bonafide_path = resolve_dataset_image(
                dataset_root=(
                    dataset_root
                ),
                relative_path=(
                    bonafide_relative
                ),
            )

            with Image.open(
                attack_path
            ) as image:

                attack = (
                    image.convert(
                        "RGB"
                    )
                )

            with Image.open(
                bonafide_path
            ) as image:

                bonafide = (
                    image.convert(
                        "RGB"
                    )
                )

            attack_region_rows = (
                regions_by_path[
                    attack_relative
                ]
            )

            bonafide_region_rows = (
                regions_by_path[
                    bonafide_relative
                ]
            )

            face = restore_counterfactual(
                attack_image=attack,
                bonafide_image=bonafide,
                attack_rows=attack_region_rows,
                bonafide_rows=bonafide_region_rows,
                mode="face",
                minimum_normalized_iou=(
                    probe_minimum_iou
                ),
            )

            text = restore_counterfactual(
                attack_image=attack,
                bonafide_image=bonafide,
                attack_rows=attack_region_rows,
                bonafide_rows=bonafide_region_rows,
                mode="text",
                minimum_normalized_iou=(
                    probe_minimum_iou
                ),
            )

            both = restore_counterfactual(
                attack_image=attack,
                bonafide_image=bonafide,
                attack_rows=attack_region_rows,
                bonafide_rows=bonafide_region_rows,
                mode="both",
                minimum_normalized_iou=(
                    probe_minimum_iou
                ),
            )

            render_visual_case(
                visual_dir=(
                    visual_dir
                ),
                attack_row=(
                    attack_row
                ),
                bonafide_path_relative=(
                    bonafide_relative
                ),
                attack=(
                    attack
                ),
                bonafide=(
                    bonafide
                ),
                face=(
                    face
                ),
                text=(
                    text
                ),
                both=(
                    both
                ),
            )

        # ==============================================================
        # Canonical CSV
        # ==============================================================

        with csv_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    CSV_COLUMNS
                ),
                extrasaction="raise",
                lineterminator="\n",
            )

            writer.writeheader()

            writer.writerows(
                csv_rows
            )

        # ==============================================================
        # Aggregate evidence
        # ==============================================================

        if not all_iou_values:

            raise RuntimeError(
                "No restoration matches were recorded."
            )

        alias_validation = (
            load_semantic_alias_validation(
                tool_cfg=tool_cfg
            )
        )

        observed_alias_counts = Counter(
            (
                row[
                    "field_name"
                ],
                row[
                    "source_field_name"
                ],
            )

            for row
            in all_match_records

            if (
                row[
                    "field_name"
                ]
                != row[
                    "source_field_name"
                ]
            )
        )

        expected_alias_counts = (
            alias_validation[
                "expected_counts"
            ]
        )

        if (
            dict(
                observed_alias_counts
            )
            != expected_alias_counts
        ):

            raise RuntimeError(
                "Observed semantic-alias usage differs "
                "from probe-backed expectation:\n"
                f"  expected={expected_alias_counts}\n"
                f"  observed={dict(observed_alias_counts)}"
            )

        LOGGER.info(
            "[PASS] semantic-parent evidence SHA-256 = %s",
            alias_validation[
                "evidence_sha256"
            ],
        )

        for (
            attack_field,
            source_field,
        ), count in sorted(
            observed_alias_counts.items()
        ):

            LOGGER.info(
                "[PASS] semantic alias %s -> %s | occurrences=%d",
                attack_field,
                source_field,
                count,
            )


        weak_matches = [
            row

            for row
            in all_match_records

            if (
                row[
                    "normalized_iou"
                ]
                < acceptance_minimum_iou
            )
        ]

        weakest_matches = sorted(
            all_match_records,
            key=lambda row: (
                row[
                    "normalized_iou"
                ],
                row[
                    "image_path"
                ],
                row[
                    "attack_region_index"
                ],
            ),
        )[
            :weakest_matches_to_log
        ]

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "AUDIT SUMMARY"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "[PASS] dev attacks processed: %d / %d",
            len(
                csv_rows
            ),
            expected_attack_count,
        )

        LOGGER.info(
            "[PASS] every attack has aligned dev bona-fide capture"
        )

        LOGGER.info(
            "[PASS] face mode restored face fields only"
        )

        LOGGER.info(
            "[PASS] text mode restored non-face fields only"
        )

        LOGGER.info(
            "[PASS] both mode equals semantic union of face + text"
        )

        LOGGER.info(
            "[PASS] every counterfactual preserved attack geometry"
        )

        LOGGER.info(
            "[PASS] no changed pixel occurred outside "
            "restoration destination boxes"
        )

        LOGGER.info(
            "[PASS] every requested restoration changed pixels"
        )

        LOGGER.info(
            "Attack/bona-fide dimension-mismatch pairs: %d",
            dimension_mismatch_pairs,
        )

        LOGGER.info(
            "Unique altered-region matches audited: %d",
            len(
                all_match_records
            ),
        )

        LOGGER.info(
            "Normalized IoU | min=%.6f | median=%.6f",
            min(
                all_iou_values
            ),
            statistics.median(
                all_iou_values
            ),
        )

        LOGGER.info(
            "Common visible fraction | min=%.6f | median=%.6f",
            min(
                all_visible_values
            ),
            statistics.median(
                all_visible_values
            ),
        )

        LOGGER.info(
            "Changed fraction within restoration mask | "
            "min=%.6f | median=%.6f",
            min(
                changed_fractions
            ),
            statistics.median(
                changed_fractions
            ),
        )

        LOGGER.info(
            "Patch resize operations across face/text/both: %d",
            total_resize_count,
        )

        LOGGER.info(
            "Matches below acceptance IoU %.6f: %d",
            acceptance_minimum_iou,
            len(
                weak_matches
            ),
        )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "WEAKEST SEMANTIC/SPATIAL MATCHES"
        )

        for row in (
            weakest_matches
        ):

            LOGGER.info(
                "  IoU=%.6f | center_distance=%.6f | "
                "visible=%.6f | field=%s | "
                "attack_region=%d | bona_region=%d | "
                "%s",
                row[
                    "normalized_iou"
                ],
                row[
                    "normalized_center_distance"
                ],
                row[
                    "common_visible_fraction"
                ],
                row[
                    "field_name"
                ],
                row[
                    "attack_region_index"
                ],
                row[
                    "bonafide_region_index"
                ],
                row[
                    "image_path"
                ],
            )

        LOGGER.info(
            ""
        )

        LOGGER.info(
            "Visual cases rendered: %d",
            len(
                visual_paths
            ),
        )

        LOGGER.info(
            "Visual directory: %s",
            visual_dir,
        )

        LOGGER.info(
            "Audit CSV: %s",
            csv_path,
        )

        LOGGER.info(
            "Audit CSV SHA-256: %s",
            sha256_file(
                csv_path
            ),
        )

        LOGGER.info(
            "Held-out test images: NOT ACCESSED"
        )

        # --------------------------------------------------------------
        # The spatial threshold is evaluated only AFTER all 306 attacks
        # have been processed so weak cases remain visible in evidence.
        # --------------------------------------------------------------

        if weak_matches:

            LOGGER.error(
                "Counterfactual geometry acceptance FAILED: "
                "%d semantic matches have normalized IoU < %.6f.",
                len(
                    weak_matches
                ),
                acceptance_minimum_iou,
            )

            LOGGER.info(
                "=" * 72
            )

            LOGGER.info(
                "DEV COUNTERFACTUAL RESTORATION AUDIT: FAIL"
            )

            LOGGER.info(
                "=" * 72
            )

            return 1

        LOGGER.info(
            "[PASS] every semantic match satisfies "
            "minimum normalized IoU %.6f",
            acceptance_minimum_iou,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "DEV COUNTERFACTUAL RESTORATION AUDIT: PASS"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "DEV COUNTERFACTUAL RESTORATION AUDIT: FAIL"
        )

        return 1

    finally:

        for handler in list(
            LOGGER.handlers
        ):

            handler.flush()
            handler.close()

        LOGGER.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )