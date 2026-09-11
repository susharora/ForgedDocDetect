#!/usr/bin/env python3
"""Dev-only metadata probe for counterfactual semantic parent correspondences.

For altered dev attack regions where the exact field_name does not exist
in the aligned bona-fide annotation, rank possible bona-fide parent fields
using normalized geometry.

No pixels are copied. No images, models, checkpoints or test data are opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger(
    "probe_counterfactual_semantic_parent_map"
)


def resolve(
    value: str | Path,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        ROOT
        / path
    ).resolve()


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


def require_sha(
    path: Path,
    expected: str,
    label: str,
) -> str:

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch:\n"
            f"  expected={expected}\n"
            f"  actual={actual}"
        )

    return actual


def load_yaml(
    path: Path,
) -> dict[str, Any]:

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


def clean_git() -> tuple[
    str,
    str,
]:

    def git(
        *args: str,
    ) -> str:

        return subprocess.run(
            [
                "git",
                *args,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = git(
        "rev-parse",
        "HEAD",
    )

    branch = git(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    status = git(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )

    if status:
        raise RuntimeError(
            "Git tree must be clean before probe:\n"
            f"{status}"
        )

    return (
        commit,
        branch,
    )


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

    result = str(
        value
    ).strip()

    if result.casefold() in {
        "",
        "none",
        "nan",
        "null",
    }:
        return ""

    return result


def normalized_text(
    value: Any,
) -> str:

    return cell_text(
        value
    ).casefold()


def integer_like(
    value: Any,
    label: str,
) -> int:

    number = float(
        value
    )

    if not math.isfinite(
        number
    ):
        raise RuntimeError(
            f"{label} is non-finite."
        )

    rounded = round(
        number
    )

    if not math.isclose(
        number,
        rounded,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError(
            f"{label} is not integer-like: {number}"
        )

    return int(
        rounded
    )


def configure_logging(
    config: dict[str, Any],
    timestamp: str,
) -> Path:

    log_config = (
        config[
            "logging"
        ]
    )

    directory = resolve(
        log_config[
            "directory"
        ]
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        directory
        / log_config[
            "filename"
        ].format(
            timestamp=timestamp
        )
    )

    LOGGER.handlers.clear()
    LOGGER.propagate = False

    LOGGER.setLevel(
        getattr(
            logging,
            str(
                log_config[
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

    return log_path


def normalized_box(
    region: dict[str, Any],
    size: tuple[int, int],
) -> tuple[
    float,
    float,
    float,
    float,
]:

    image_width, image_height = (
        size
    )

    return (
        region[
            "x"
        ]
        / image_width,

        region[
            "y"
        ]
        / image_height,

        (
            region[
                "x"
            ]
            + region[
                "width"
            ]
        )
        / image_width,

        (
            region[
                "y"
            ]
            + region[
                "height"
            ]
        )
        / image_height,
    )


def geometry(
    attack: dict[str, Any],
    attack_size: tuple[int, int],
    source: dict[str, Any],
    source_size: tuple[int, int],
) -> tuple[
    float,
    float,
    float,
]:

    ax0, ay0, ax1, ay1 = (
        normalized_box(
            attack,
            attack_size,
        )
    )

    bx0, by0, bx1, by1 = (
        normalized_box(
            source,
            source_size,
        )
    )

    intersection = (
        max(
            0.0,
            min(
                ax1,
                bx1,
            )
            - max(
                ax0,
                bx0,
            ),
        )
        *
        max(
            0.0,
            min(
                ay1,
                by1,
            )
            - max(
                ay0,
                by0,
            ),
        )
    )

    attack_area = max(
        0.0,
        (
            ax1 - ax0
        )
        * (
            ay1 - ay0
        ),
    )

    source_area = max(
        0.0,
        (
            bx1 - bx0
        )
        * (
            by1 - by0
        ),
    )

    union = (
        attack_area
        + source_area
        - intersection
    )

    coverage = (
        0.0
        if attack_area == 0.0
        else intersection / attack_area
    )

    iou = (
        0.0
        if union == 0.0
        else intersection / union
    )

    center_distance = math.hypot(
        (
            ax0
            + ax1
            - bx0
            - bx1
        )
        / 2.0,
        (
            ay0
            + ay1
            - by0
            - by1
        )
        / 2.0,
    )

    return (
        coverage,
        iou,
        center_distance,
    )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "tools/"
            "probe_counterfactual_semantic_parent_map_config.yaml"
        ),
    )

    args = parser.parse_args()

    commit, branch = (
        clean_git()
    )

    config = load_yaml(
        resolve(
            args.config
        )
    )

    experiment_ref = (
        config[
            "experiment_config"
        ]
    )

    experiment_path = resolve(
        experiment_ref[
            "path"
        ]
    )

    experiment_sha = require_sha(
        experiment_path,
        experiment_ref[
            "sha256"
        ],
        "Experiment config",
    )

    experiment = load_yaml(
        experiment_path
    )

    # --------------------------------------------------------------
    # Frozen dev manifest.
    # --------------------------------------------------------------

    dev_config = (
        experiment[
            "data"
        ][
            "frozen_split"
        ][
            "dev_val"
        ]
    )

    dev_path = resolve(
        dev_config[
            "path"
        ]
    )

    dev_sha = require_sha(
        dev_path,
        dev_config[
            "sha256"
        ],
        "Frozen dev_val manifest",
    )

    with dev_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        dev_rows = list(
            csv.DictReader(
                file
            )
        )

    if len(
        dev_rows
    ) != 459:
        raise RuntimeError(
            "Expected exactly 459 dev rows."
        )

    if any(
        row[
            "project_role"
        ]
        != "dev_val"
        for row
        in dev_rows
    ):
        raise RuntimeError(
            "Unexpected project_role in dev manifest."
        )

    if any(
        not row[
            "image_path"
        ]
        .replace(
            "\\",
            "/",
        )
        .startswith(
            "train/"
        )
        for row
        in dev_rows
    ):
        raise RuntimeError(
            "Probe encountered a non-train path."
        )

    bonafide: dict[
        tuple[str, str],
        str,
    ] = {}

    attacks: list[
        dict[str, str]
    ] = []

    for row in dev_rows:

        key = (
            row[
                "file_stem"
            ],
            row[
                "hardware_source"
            ],
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

        if (
            row[
                "traffic_type"
            ]
            == "bonafide"
        ):

            if key in bonafide:
                raise RuntimeError(
                    f"Duplicate bona-fide key: {key}"
                )

            bonafide[
                key
            ] = image_path

        elif (
            row[
                "traffic_type"
            ]
            == "attack"
        ):
            attacks.append(
                row
            )

    if (
        len(
            bonafide
        )
        != 153
        or len(
            attacks
        )
        != 306
    ):
        raise RuntimeError(
            "Expected 153 aligned bona-fide "
            "and 306 attack rows."
        )

    # --------------------------------------------------------------
    # Frozen discovery metadata.
    # --------------------------------------------------------------

    source_config = (
        experiment[
            "data"
        ][
            "source_discovery"
        ]
    )

    workbook_path = resolve(
        source_config[
            "workbook"
        ]
    )

    workbook_sha = require_sha(
        workbook_path,
        source_config[
            "sha256"
        ],
        "Frozen discovery workbook",
    )

    required_paths = {
        row[
            "image_path"
        ]
        .replace(
            "\\",
            "/",
        )
        for row
        in dev_rows
    }

    images = pd.read_excel(
        workbook_path,
        sheet_name="Images",
        engine="openpyxl",
    )

    regions_frame = pd.read_excel(
        workbook_path,
        sheet_name="Regions",
        engine="openpyxl",
    )

    required_image_columns = {
        "image_path",
        "image_width",
        "image_height",
    }

    required_region_columns = {
        "image_path",
        "region_index",
        "field_name",
        "region_provenance_raw",
        "x",
        "y",
        "width",
        "height",
    }

    missing = (
        required_image_columns
        - set(
            images.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Images sheet missing columns: "
            f"{sorted(missing)}"
        )

    missing = (
        required_region_columns
        - set(
            regions_frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Regions sheet missing columns: "
            f"{sorted(missing)}"
        )

    sizes: dict[
        str,
        tuple[int, int],
    ] = {}

    for _, row in (
        images.iterrows()
    ):

        image_path = (
            cell_text(
                row[
                    "image_path"
                ]
            )
            .replace(
                "\\",
                "/",
            )
        )

        if image_path not in required_paths:
            continue

        sizes[
            image_path
        ] = (
            integer_like(
                row[
                    "image_width"
                ],
                "image_width",
            ),
            integer_like(
                row[
                    "image_height"
                ],
                "image_height",
            ),
        )

    regions: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for _, row in (
        regions_frame.iterrows()
    ):

        image_path = (
            cell_text(
                row[
                    "image_path"
                ]
            )
            .replace(
                "\\",
                "/",
            )
        )

        if image_path not in required_paths:
            continue

        regions[
            image_path
        ].append(
            {
                "region_index":
                    integer_like(
                        row[
                            "region_index"
                        ],
                        "region_index",
                    ),

                "field_name":
                    normalized_text(
                        row[
                            "field_name"
                        ]
                    ),

                "provenance":
                    normalized_text(
                        row[
                            "region_provenance_raw"
                        ]
                    ),

                "x":
                    integer_like(
                        row[
                            "x"
                        ],
                        "x",
                    ),

                "y":
                    integer_like(
                        row[
                            "y"
                        ],
                        "y",
                    ),

                "width":
                    integer_like(
                        row[
                            "width"
                        ],
                        "width",
                    ),

                "height":
                    integer_like(
                        row[
                            "height"
                        ],
                        "height",
                    ),
            }
        )

    if (
        set(
            sizes
        )
        != required_paths
    ):
        raise RuntimeError(
            "Images sheet does not cover all dev paths."
        )

    if (
        set(
            regions
        )
        != required_paths
    ):
        raise RuntimeError(
            "Regions sheet does not cover all dev paths."
        )

    coverage_floor = float(
        config[
            "probe"
        ][
            "minimum_attack_coverage_for_parent_candidate"
        ]
    )

    if not (
        0.0
        <= coverage_floor
        <= 1.0
    ):
        raise ValueError(
            "Coverage floor must be in [0,1]."
        )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
    )

    configure_logging(
        config,
        timestamp,
    )

    output_config = (
        config[
            "output"
        ]
    )

    output_directory = resolve(
        output_config[
            "directory"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / output_config[
            "csv_filename"
        ].format(
            timestamp=timestamp
        )
    )

    yaml_path = (
        output_directory
        / output_config[
            "yaml_filename"
        ].format(
            timestamp=timestamp
        )
    )

    try:

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "COUNTERFACTUAL SEMANTIC PARENT-MAP PROBE"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Git commit: %s",
            commit,
        )

        LOGGER.info(
            "[PASS] experiment config SHA-256 = %s",
            experiment_sha,
        )

        LOGGER.info(
            "[PASS] dev manifest SHA-256 = %s",
            dev_sha,
        )

        LOGGER.info(
            "[PASS] discovery workbook SHA-256 = %s",
            workbook_sha,
        )

        LOGGER.info(
            "[PASS] scope = 306 dev attacks + "
            "153 aligned bona-fide"
        )

        LOGGER.info(
            "[PASS] images opened: FALSE"
        )

        LOGGER.info(
            "[PASS] held-out test accessed: FALSE"
        )

        evidence: list[
            dict[str, Any]
        ] = []

        exact_match_count = 0

        for attack_row in attacks:

            attack_path = (
                attack_row[
                    "image_path"
                ]
                .replace(
                    "\\",
                    "/",
                )
            )

            key = (
                attack_row[
                    "file_stem"
                ],
                attack_row[
                    "hardware_source"
                ],
            )

            if key not in bonafide:
                raise RuntimeError(
                    "Missing aligned bona-fide "
                    f"for {key}"
                )

            bona_path = (
                bonafide[
                    key
                ]
            )

            altered = [
                region
                for region
                in regions[
                    attack_path
                ]
                if (
                    region[
                        "provenance"
                    ]
                    == "altered"
                )
            ]

            sources = [
                region
                for region
                in regions[
                    bona_path
                ]
                if (
                    region[
                        "field_name"
                    ]
                    and region[
                        "provenance"
                    ]
                    != "altered"
                )
            ]

            source_fields = {
                region[
                    "field_name"
                ]
                for region
                in sources
            }

            for destination in altered:

                attack_field = (
                    destination[
                        "field_name"
                    ]
                )

                if not attack_field:
                    raise RuntimeError(
                        "Blank altered field in "
                        f"{attack_path}"
                    )

                if attack_field in source_fields:

                    exact_match_count += 1
                    continue

                ranked = []

                for source in sources:

                    coverage, iou, distance = (
                        geometry(
                            destination,
                            sizes[
                                attack_path
                            ],
                            source,
                            sizes[
                                bona_path
                            ],
                        )
                    )

                    ranked.append(
                        (
                            -coverage,
                            -iou,
                            distance,
                            source[
                                "region_index"
                            ],
                            source,
                        )
                    )

                if not ranked:
                    raise RuntimeError(
                        "No bona-fide semantic "
                        f"regions for {bona_path}"
                    )

                ranked.sort(
                    key=lambda item:
                        item[
                            :4
                        ]
                )

                best = (
                    ranked[
                        0
                    ]
                )

                source = (
                    best[
                        4
                    ]
                )

                second_field = ""
                second_coverage = None

                for candidate in (
                    ranked[
                        1:
                    ]
                ):

                    if (
                        candidate[
                            4
                        ][
                            "field_name"
                        ]
                        != source[
                            "field_name"
                        ]
                    ):

                        second_field = (
                            candidate[
                                4
                            ][
                                "field_name"
                            ]
                        )

                        second_coverage = (
                            -candidate[
                                0
                            ]
                        )

                        break

                evidence.append(
                    {
                        "attack_image_path":
                            attack_path,

                        "bonafide_image_path":
                            bona_path,

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

                        "attack_region_index":
                            destination[
                                "region_index"
                            ],

                        "attack_field":
                            attack_field,

                        "candidate_parent_field":
                            source[
                                "field_name"
                            ],

                        "candidate_parent_region_index":
                            source[
                                "region_index"
                            ],

                        "attack_coverage_by_candidate":
                            -best[
                                0
                            ],

                        "normalized_iou":
                            -best[
                                1
                            ],

                        "normalized_center_distance":
                            best[
                                2
                            ],

                        "second_candidate_field":
                            second_field,

                        "second_candidate_attack_coverage":
                            second_coverage,
                    }
                )

        by_field: dict[
            str,
            list[
                dict[str, Any]
            ],
        ] = defaultdict(
            list
        )

        for row in evidence:

            by_field[
                row[
                    "attack_field"
                ]
            ].append(
                row
            )

        summary: dict[
            str,
            Any,
        ] = {}

        for (
            attack_field,
            field_rows,
        ) in sorted(
            by_field.items()
        ):

            counts = Counter(
                row[
                    "candidate_parent_field"
                ]
                for row
                in field_rows
            )

            parent, parent_count = (
                counts.most_common(
                    1
                )[
                    0
                ]
            )

            dominant_rows = [
                row
                for row
                in field_rows
                if (
                    row[
                        "candidate_parent_field"
                    ]
                    == parent
                )
            ]

            coverages = [
                float(
                    row[
                        "attack_coverage_by_candidate"
                    ]
                )
                for row
                in dominant_rows
            ]

            summary[
                attack_field
            ] = {
                "unmatched_occurrences":
                    len(
                        field_rows
                    ),

                "candidate_parent_counts":
                    dict(
                        sorted(
                            counts.items()
                        )
                    ),

                "dominant_candidate_parent":
                    parent,

                "dominance_fraction":
                    (
                        parent_count
                        / len(
                            field_rows
                        )
                    ),

                "dominant_candidate_min_attack_coverage":
                    min(
                        coverages
                    ),

                "dominant_candidate_median_attack_coverage":
                    statistics.median(
                        coverages
                    ),

                "dominant_candidate_max_attack_coverage":
                    max(
                        coverages
                    ),

                "candidate_is_unanimous":
                    (
                        parent_count
                        == len(
                            field_rows
                        )
                    ),

                "candidate_meets_probe_coverage_floor":
                    (
                        parent_count
                        == len(
                            field_rows
                        )
                        and min(
                            coverages
                        )
                        >= coverage_floor
                    ),
            }

            LOGGER.info(
                "field=%s | unmatched=%d | "
                "candidate=%s | dominance=%.6f | "
                "min_coverage=%.6f | "
                "median_coverage=%.6f | "
                "floor_pass=%s",
                attack_field,
                len(
                    field_rows
                ),
                parent,
                parent_count
                / len(
                    field_rows
                ),
                min(
                    coverages
                ),
                statistics.median(
                    coverages
                ),
                summary[
                    attack_field
                ][
                    "candidate_meets_probe_coverage_floor"
                ],
            )

        columns = [
            "attack_image_path",
            "bonafide_image_path",
            "file_stem",
            "variant",
            "hardware_source",
            "attack_region_index",
            "attack_field",
            "candidate_parent_field",
            "candidate_parent_region_index",
            "attack_coverage_by_candidate",
            "normalized_iou",
            "normalized_center_distance",
            "second_candidate_field",
            "second_candidate_attack_coverage",
        ]

        with csv_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(
                evidence
            )

        csv_sha = sha256_file(
            csv_path
        )

        artifact = {
            "schema_version":
                1,

            "artifact_type":
                "counterfactual_semantic_parent_map_probe",

            "status":
                "PROBE_ONLY_NOT_FROZEN",

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git":
                {
                    "commit_sha":
                        commit,

                    "branch":
                        branch,
                },

            "experiment_config":
                {
                    "path":
                        str(
                            experiment_path.relative_to(
                                ROOT
                            )
                        ),

                    "sha256":
                        experiment_sha,
                },

            "frozen_dev_manifest":
                {
                    "path":
                        str(
                            dev_path.relative_to(
                                ROOT
                            )
                        ),

                    "sha256":
                        dev_sha,

                    "rows":
                        459,
                },

            "frozen_discovery_workbook":
                {
                    "path":
                        str(
                            workbook_path.relative_to(
                                ROOT
                            )
                        ),

                    "sha256":
                        workbook_sha,
                },

            "probe_rule":
                {
                    "scope":
                        (
                            "altered fields lacking an exact "
                            "same-field bona-fide source"
                        ),

                    "ranking":
                        [
                            (
                                "highest normalized attack-box "
                                "coverage by candidate source"
                            ),
                            "highest normalized IoU",
                            (
                                "smallest normalized "
                                "center distance"
                            ),
                            (
                                "lowest bona-fide "
                                "region_index"
                            ),
                        ],

                    "minimum_attack_coverage_for_parent_candidate":
                        coverage_floor,

                    "automatic_mapping_applied":
                        False,
                },

            "exact_same_field_altered_occurrences":
                exact_match_count,

            "unmatched_semantic_occurrences":
                len(
                    evidence
                ),

            "candidate_summary_by_attack_field":
                summary,

            "evidence_csv":
                {
                    "path":
                        str(
                            csv_path.relative_to(
                                ROOT
                            )
                        ),

                    "sha256":
                        csv_sha,

                    "rows":
                        len(
                            evidence
                        ),
                },

            "scientific_boundaries":
                {
                    "probe_only":
                        True,

                    "restoration_performed":
                        False,

                    "pixels_copied":
                        False,

                    "images_opened":
                        False,

                    "model_loaded":
                        False,

                    "checkpoint_loaded":
                        False,

                    "inference_performed":
                        False,

                    "threshold_derived_or_modified":
                        False,

                    "held_out_test_accessed":
                        False,
                },
        }

        with yaml_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                artifact,
                file,
                sort_keys=False,
            )

        yaml_sha = sha256_file(
            yaml_path
        )

        LOGGER.info(
            "Exact-field altered occurrences: %d",
            exact_match_count,
        )

        LOGGER.info(
            "Unmatched semantic occurrences: %d",
            len(
                evidence
            ),
        )

        LOGGER.info(
            "Probe CSV: %s",
            csv_path,
        )

        LOGGER.info(
            "Probe CSV SHA-256: %s",
            csv_sha,
        )

        LOGGER.info(
            "Probe YAML: %s",
            yaml_path,
        )

        LOGGER.info(
            "Probe YAML SHA-256: %s",
            yaml_sha,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "COUNTERFACTUAL SEMANTIC PARENT-MAP PROBE: COMPLETE"
        )

        LOGGER.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOGGER.exception(
            "COUNTERFACTUAL SEMANTIC PARENT-MAP PROBE: FAIL"
        )

        for output in (
            csv_path,
            yaml_path,
        ):

            if output.exists():
                output.unlink()

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