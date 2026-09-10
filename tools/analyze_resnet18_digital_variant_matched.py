#!/usr/bin/env python3
"""Post-hoc matched digital-variant control for the frozen ResNet-18 baseline.

Each matched unit is one dev_val file_stem x hardware_source combination:
bonafide, digital_1 and digital_2 scores from dev_val are compared with the
corresponding digital_3 score from the official test set. There are 51 stems x
3 hardware sources = 153 units per seed.

This is diagnostic only. It reads persisted prediction CSVs and frozen threshold
artifacts; it opens no images, loads no model/checkpoint, performs no inference,
and derives or modifies no threshold.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]

LOG = logging.getLogger(
    "analyze_resnet18_digital_variant_matched"
)

SEEDS = (
    8,
    9,
    10,
)

HARDWARE = (
    "huawei",
    "iphone15pro",
    "scan",
)


DEV_COLUMNS = (
    "row_index",
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "target",
    "bonafide_logit_fp32",
    "attack_logit_fp32",
    "attack_margin_float64",
    "p_attack_float64",
    "prediction_fixed_0_5",
    "prediction_fpr10",
)


TEST_COLUMNS = (
    "row_index",
    "source_workbook_row",
    "image_path",
    "image_sha256",
    "verified_image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "face_db",
    "face_id",
    "gender",
    "target",
    "bonafide_logit_fp32",
    "attack_logit_fp32",
    "attack_margin_float64",
    "p_attack_float64",
    "prediction_fixed_0_5",
    "prediction_fpr10",
)


PAIR_COLUMNS = (
    "seed",
    "file_stem",
    "hardware_source",

    "dev_bonafide_path",
    "dev_digital1_path",
    "dev_digital2_path",
    "test_digital3_path",

    "p_bonafide",
    "p_digital1",
    "p_digital2",
    "p_digital3",

    "margin_bonafide",
    "margin_digital1",
    "margin_digital2",
    "margin_digital3",

    "fixed_bonafide_attack",
    "fixed_digital1_attack",
    "fixed_digital2_attack",
    "fixed_digital3_attack",

    "fpr10_bonafide_attack",
    "fpr10_digital1_attack",
    "fpr10_digital2_attack",
    "fpr10_digital3_attack",

    "delta_p_d3_minus_d1",
    "delta_p_d3_minus_d2",
    "delta_p_d3_minus_mean_d1_d2",

    "delta_margin_d3_minus_d1",
    "delta_margin_d3_minus_d2",
    "delta_margin_d3_minus_mean_d1_d2",

    "d3_below_both_dev_attacks",
    "d3_above_matched_bonafide",

    "dev_attacks_fixed_detected_then_d3_missed",
    "dev_attacks_fpr10_detected_then_d3_missed",
)


def path(
    value: str | Path,
) -> Path:

    p = Path(
        value
    ).expanduser()

    if p.is_absolute():
        return p.resolve()

    return (
        ROOT
        / p
    ).resolve()


def rel(
    p: Path,
) -> str:

    return (
        p.resolve()
        .relative_to(
            ROOT
        )
        .as_posix()
    )


def digest(
    p: Path,
) -> str:

    h = hashlib.sha256()

    with p.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def require_sha(
    p: Path,
    expected: str,
    label: str,
) -> str:

    if not p.is_file():
        raise FileNotFoundError(
            p
        )

    actual = digest(
        p
    )

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: "
            f"expected={expected}, "
            f"actual={actual}"
        )

    return actual


def yload(
    p: Path,
) -> dict[str, Any]:

    with p.open(
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
            f"{p} must contain a YAML mapping"
        )

    return value


def mp(
    value: Any,
    label: str,
) -> Mapping[Any, Any]:

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{label} must be a mapping"
        )

    return value


def seed_item(
    value: Mapping[Any, Any],
    seed: int,
) -> Any:

    if seed in value:
        return value[
            seed
        ]

    if str(seed) in value:
        return value[
            str(seed)
        ]

    raise KeyError(
        f"Missing seed {seed}"
    )


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
            "Git tree must be clean before analysis:\n"
            f"{status}"
        )

    return (
        commit,
        branch,
    )


def setup_log(
    config: Mapping[Any, Any],
    timestamp: str,
) -> Path:

    cfg = mp(
        config[
            "logging"
        ],
        "logging",
    )

    directory = path(
        str(
            cfg[
                "directory"
            ]
        )
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logfile = (
        directory
        / str(
            cfg[
                "filename"
            ]
        ).format(
            timestamp=timestamp
        )
    )

    LOG.handlers.clear()
    LOG.propagate = False

    LOG.setLevel(
        getattr(
            logging,
            str(
                cfg[
                    "level"
                ]
            ).upper(),
        )
    )

    handler = logging.FileHandler(
        logfile,
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

    LOG.addHandler(
        handler
    )

    return logfile


def rows(
    p: Path,
    columns: Sequence[str],
) -> list[dict[str, str]]:

    with p.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        if tuple(
            reader.fieldnames
            or ()
        ) != tuple(
            columns
        ):
            raise RuntimeError(
                f"Unexpected CSV columns in {p}"
            )

        return list(
            reader
        )


def sigmoid(
    x: float,
) -> float:

    if x >= 0:

        z = math.exp(
            -x
        )

        return (
            1.0
            / (
                1.0
                + z
            )
        )

    z = math.exp(
        x
    )

    return (
        z
        / (
            1.0
            + z
        )
    )


def checked_score(
    row: Mapping[str, str],
    threshold: float,
    label: str,
) -> dict[str, Any]:

    expected_target = (
        0
        if row[
            "traffic_type"
        ] == "bonafide"
        else 1
        if row[
            "traffic_type"
        ] == "attack"
        else -1
    )

    if int(
        row[
            "target"
        ]
    ) != expected_target:
        raise RuntimeError(
            f"{label}: class polarity mismatch"
        )

    z0 = float(
        row[
            "bonafide_logit_fp32"
        ]
    )

    z1 = float(
        row[
            "attack_logit_fp32"
        ]
    )

    margin = (
        z1
        - z0
    )

    p = sigmoid(
        margin
    )

    stored_margin = float(
        row[
            "attack_margin_float64"
        ]
    )

    stored_p = float(
        row[
            "p_attack_float64"
        ]
    )

    if not math.isclose(
        stored_margin,
        margin,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{label}: margin mismatch"
        )

    if not math.isclose(
        stored_p,
        p,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{label}: p_attack mismatch"
        )

    fixed = int(
        p
        >= 0.5
    )

    fpr10 = int(
        p
        >= threshold
    )

    if (
        fixed
        != int(
            row[
                "prediction_fixed_0_5"
            ]
        )
    ):
        raise RuntimeError(
            f"{label}: fixed prediction mismatch"
        )

    if (
        fpr10
        != int(
            row[
                "prediction_fpr10"
            ]
        )
    ):
        raise RuntimeError(
            f"{label}: FPR10 prediction mismatch"
        )

    return {
        "path":
            row[
                "image_path"
            ],

        "p":
            p,

        "margin":
            margin,

        "fixed":
            fixed,

        "fpr10":
            fpr10,
    }


def dev_map(
    dev_rows: Sequence[
        Mapping[str, str]
    ],
    threshold: float,
) -> tuple[
    dict[
        tuple[str, str],
        dict[
            str,
            dict[str, Any],
        ],
    ],
    set[str],
]:

    if len(
        dev_rows
    ) != 459:
        raise RuntimeError(
            "Dev predictions must contain 459 rows"
        )

    stems = {
        row[
            "file_stem"
        ]
        for row
        in dev_rows
    }

    if len(
        stems
    ) != 51:
        raise RuntimeError(
            "Expected 51 dev stems, "
            f"observed {len(stems)}"
        )

    result: dict[
        tuple[str, str],
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    for i, row in enumerate(
        dev_rows
    ):

        hardware = row[
            "hardware_source"
        ]

        if hardware not in HARDWARE:
            raise RuntimeError(
                f"Unexpected dev hardware {hardware!r}"
            )

        role = (
            "bonafide"
            if (
                row[
                    "traffic_type"
                ] == "bonafide"
                and row[
                    "variant"
                ] == ""
            )
            else row[
                "variant"
            ]
        )

        if role not in {
            "bonafide",
            "digital_1",
            "digital_2",
        }:
            raise RuntimeError(
                f"Unexpected dev role {role!r}"
            )

        key = (
            row[
                "file_stem"
            ],
            hardware,
        )

        bucket = result.setdefault(
            key,
            {},
        )

        if role in bucket:
            raise RuntimeError(
                f"Duplicate dev role {role} for {key}"
            )

        bucket[
            role
        ] = checked_score(
            row,
            threshold,
            f"dev row {i}",
        )

    expected = {
        (
            stem,
            hw,
        )
        for stem
        in stems
        for hw
        in HARDWARE
    }

    if set(
        result
    ) != expected:
        raise RuntimeError(
            "Dev prediction CSV does not form "
            "the expected stem x hardware keys"
        )

    for key, bucket in (
        result.items()
    ):

        if set(
            bucket
        ) != {
            "bonafide",
            "digital_1",
            "digital_2",
        }:
            raise RuntimeError(
                f"Incomplete dev matched unit {key}"
            )

    return (
        result,
        stems,
    )


def d3_map(
    test_rows: Sequence[
        Mapping[str, str]
    ],
    stems: set[str],
    threshold: float,
) -> dict[
    tuple[str, str],
    dict[str, Any],
]:

    if len(
        test_rows
    ) != 1385:
        raise RuntimeError(
            "Test predictions must contain 1385 rows"
        )

    result: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for i, row in enumerate(
        test_rows
    ):

        if not (
            row[
                "traffic_type"
            ] == "attack"
            and row[
                "variant"
            ] == "digital_3"
            and row[
                "file_stem"
            ] in stems
        ):
            continue

        if (
            row[
                "verified_image_sha256"
            ]
            != row[
                "image_sha256"
            ]
        ):
            raise RuntimeError(
                f"test row {i}: "
                "persisted SHA verification mismatch"
            )

        key = (
            row[
                "file_stem"
            ],
            row[
                "hardware_source"
            ],
        )

        if key in result:
            raise RuntimeError(
                f"Duplicate digital_3 matched key {key}"
            )

        result[
            key
        ] = checked_score(
            row,
            threshold,
            f"test row {i}",
        )

    expected = {
        (
            stem,
            hw,
        )
        for stem
        in stems
        for hw
        in HARDWARE
    }

    if set(
        result
    ) != expected:
        raise RuntimeError(
            "Test digital_3 does not provide "
            "the same 153 stem x hardware keys"
        )

    return result


def paired_rows(
    seed: int,
    dev: Mapping[
        tuple[str, str],
        Mapping[
            str,
            Mapping[str, Any],
        ],
    ],
    d3: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
) -> list[dict[str, Any]]:

    rank = {
        name:
            i
        for i, name
        in enumerate(
            HARDWARE
        )
    }

    output: list[
        dict[str, Any]
    ] = []

    for stem, hardware in sorted(
        dev,
        key=lambda k: (
            k[
                0
            ],
            rank[
                k[
                    1
                ]
            ],
        ),
    ):

        b = dev[
            (
                stem,
                hardware,
            )
        ][
            "bonafide"
        ]

        d1 = dev[
            (
                stem,
                hardware,
            )
        ][
            "digital_1"
        ]

        d2 = dev[
            (
                stem,
                hardware,
            )
        ][
            "digital_2"
        ]

        d3s = d3[
            (
                stem,
                hardware,
            )
        ]

        output.append(
            {
                "seed":
                    seed,

                "file_stem":
                    stem,

                "hardware_source":
                    hardware,

                "dev_bonafide_path":
                    b[
                        "path"
                    ],

                "dev_digital1_path":
                    d1[
                        "path"
                    ],

                "dev_digital2_path":
                    d2[
                        "path"
                    ],

                "test_digital3_path":
                    d3s[
                        "path"
                    ],

                "p_bonafide":
                    b[
                        "p"
                    ],

                "p_digital1":
                    d1[
                        "p"
                    ],

                "p_digital2":
                    d2[
                        "p"
                    ],

                "p_digital3":
                    d3s[
                        "p"
                    ],

                "margin_bonafide":
                    b[
                        "margin"
                    ],

                "margin_digital1":
                    d1[
                        "margin"
                    ],

                "margin_digital2":
                    d2[
                        "margin"
                    ],

                "margin_digital3":
                    d3s[
                        "margin"
                    ],

                "fixed_bonafide_attack":
                    b[
                        "fixed"
                    ],

                "fixed_digital1_attack":
                    d1[
                        "fixed"
                    ],

                "fixed_digital2_attack":
                    d2[
                        "fixed"
                    ],

                "fixed_digital3_attack":
                    d3s[
                        "fixed"
                    ],

                "fpr10_bonafide_attack":
                    b[
                        "fpr10"
                    ],

                "fpr10_digital1_attack":
                    d1[
                        "fpr10"
                    ],

                "fpr10_digital2_attack":
                    d2[
                        "fpr10"
                    ],

                "fpr10_digital3_attack":
                    d3s[
                        "fpr10"
                    ],

                "delta_p_d3_minus_d1":
                    (
                        d3s[
                            "p"
                        ]
                        - d1[
                            "p"
                        ]
                    ),

                "delta_p_d3_minus_d2":
                    (
                        d3s[
                            "p"
                        ]
                        - d2[
                            "p"
                        ]
                    ),

                "delta_p_d3_minus_mean_d1_d2":
                    (
                        d3s[
                            "p"
                        ]
                        - (
                            d1[
                                "p"
                            ]
                            + d2[
                                "p"
                            ]
                        )
                        / 2
                    ),

                "delta_margin_d3_minus_d1":
                    (
                        d3s[
                            "margin"
                        ]
                        - d1[
                            "margin"
                        ]
                    ),

                "delta_margin_d3_minus_d2":
                    (
                        d3s[
                            "margin"
                        ]
                        - d2[
                            "margin"
                        ]
                    ),

                "delta_margin_d3_minus_mean_d1_d2":
                    (
                        d3s[
                            "margin"
                        ]
                        - (
                            d1[
                                "margin"
                            ]
                            + d2[
                                "margin"
                            ]
                        )
                        / 2
                    ),

                "d3_below_both_dev_attacks":
                    int(
                        d3s[
                            "p"
                        ]
                        < min(
                            d1[
                                "p"
                            ],
                            d2[
                                "p"
                            ],
                        )
                    ),

                "d3_above_matched_bonafide":
                    int(
                        d3s[
                            "p"
                        ]
                        > b[
                            "p"
                        ]
                    ),

                "dev_attacks_fixed_detected_then_d3_missed":
                    int(
                        d1[
                            "fixed"
                        ] == 1
                        and d2[
                            "fixed"
                        ] == 1
                        and d3s[
                            "fixed"
                        ] == 0
                    ),

                "dev_attacks_fpr10_detected_then_d3_missed":
                    int(
                        d1[
                            "fpr10"
                        ] == 1
                        and d2[
                            "fpr10"
                        ] == 1
                        and d3s[
                            "fpr10"
                        ] == 0
                    ),
            }
        )

    return output


def median(
    rs: Sequence[
        Mapping[str, Any]
    ],
    field: str,
) -> float:

    return float(
        statistics.median(
            float(
                r[
                    field
                ]
            )
            for r
            in rs
        )
    )


def rate(
    rs: Sequence[
        Mapping[str, Any]
    ],
    field: str,
) -> float:

    return (
        sum(
            float(
                r[
                    field
                ]
            )
            for r
            in rs
        )
        / len(
            rs
        )
    )


def summarize(
    rs: Sequence[
        Mapping[str, Any]
    ],
) -> dict[
    str,
    float | int,
]:

    return {
        "n_units":
            len(
                rs
            ),

        **{
            f"median_p_{name}":
                median(
                    rs,
                    f"p_{name}",
                )
            for name
            in (
                "bonafide",
                "digital1",
                "digital2",
                "digital3",
            )
        },

        **{
            f"fixed_positive_rate_{name}":
                rate(
                    rs,
                    f"fixed_{name}_attack",
                )
            for name
            in (
                "bonafide",
                "digital1",
                "digital2",
                "digital3",
            )
        },

        **{
            f"fpr10_positive_rate_{name}":
                rate(
                    rs,
                    f"fpr10_{name}_attack",
                )
            for name
            in (
                "bonafide",
                "digital1",
                "digital2",
                "digital3",
            )
        },

        "rate_d3_below_both_dev_attacks":
            rate(
                rs,
                "d3_below_both_dev_attacks",
            ),

        "rate_d3_above_matched_bonafide":
            rate(
                rs,
                "d3_above_matched_bonafide",
            ),

        "rate_dev_attacks_fixed_detected_then_d3_missed":
            rate(
                rs,
                "dev_attacks_fixed_detected_then_d3_missed",
            ),

        "rate_dev_attacks_fpr10_detected_then_d3_missed":
            rate(
                rs,
                "dev_attacks_fpr10_detected_then_d3_missed",
            ),

        **{
            f"median_{field}":
                median(
                    rs,
                    field,
                )
            for field
            in (
                "delta_p_d3_minus_d1",
                "delta_p_d3_minus_d2",
                "delta_p_d3_minus_mean_d1_d2",
                "delta_margin_d3_minus_d1",
                "delta_margin_d3_minus_d2",
                "delta_margin_d3_minus_mean_d1_d2",
            )
        },
    }


def aggregate(
    values: Mapping[
        int,
        Mapping[
            str,
            float | int,
        ],
    ],
) -> dict[str, Any]:

    counts = {
        int(
            values[
                s
            ][
                "n_units"
            ]
        )
        for s
        in SEEDS
    }

    if len(
        counts
    ) != 1:
        raise RuntimeError(
            "Matched support differs across seeds"
        )

    result: dict[
        str,
        Any,
    ] = {
        "n_units":
            next(
                iter(
                    counts
                )
            )
    }

    for key in values[
        8
    ]:

        if key == "n_units":
            continue

        xs = [
            float(
                values[
                    s
                ][
                    key
                ]
            )
            for s
            in SEEDS
        ]

        result[
            key
        ] = {
            "mean":
                sum(
                    xs
                )
                / 3,

            "min":
                min(
                    xs
                ),

            "max":
                max(
                    xs
                ),
        }

    return result


def write_csv(
    p: Path,
    data: Sequence[
        Mapping[str, Any]
    ],
) -> str:

    with p.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=PAIR_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )

        writer.writeheader()

        for row in data:

            writer.writerow(
                {
                    key:
                        format(
                            row[
                                key
                            ],
                            ".17g",
                        )
                        if isinstance(
                            row[
                                key
                            ],
                            float,
                        )
                        else row[
                            key
                        ]
                    for key
                    in PAIR_COLUMNS
                }
            )

    return digest(
        p
    )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "tools/"
            "analyze_resnet18_digital_variant_matched_config.yaml"
        ),
    )

    args = parser.parse_args()

    commit, branch = clean_git()

    config = yload(
        path(
            args.config
        )
    )

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H%M%S_%fZ"
        )
    )

    final_cfg = mp(
        config[
            "frozen_final_summary"
        ],
        "frozen_final_summary",
    )

    final_path = path(
        str(
            final_cfg[
                "path"
            ]
        )
    )

    final_sha = require_sha(
        final_path,
        str(
            final_cfg[
                "sha256"
            ]
        ),
        "Frozen final summary",
    )

    final = yload(
        final_path
    )

    if (
        final.get(
            "status"
        )
        != "FROZEN_FINAL_BASELINE_TEST_SUMMARY"
    ):
        raise RuntimeError(
            "Frozen final summary status is invalid"
        )

    subgroup_cfg = mp(
        config[
            "completed_subgroup_diagnostic"
        ],
        "completed_subgroup_diagnostic",
    )

    subgroup_path = path(
        str(
            subgroup_cfg[
                "path"
            ]
        )
    )

    subgroup_sha = require_sha(
        subgroup_path,
        str(
            subgroup_cfg[
                "sha256"
            ]
        ),
        "Subgroup diagnostic",
    )

    subgroup = yload(
        subgroup_path
    )

    if (
        subgroup.get(
            "status"
        )
        != "POST_HOC_DIAGNOSTIC_NOT_MODEL_SELECTION"
        or subgroup[
            "frozen_baseline"
        ][
            "sha256"
        ]
        != final_sha
    ):
        raise RuntimeError(
            "Subgroup diagnostic prerequisite is invalid"
        )

    out_cfg = mp(
        config[
            "output"
        ],
        "output",
    )

    out_dir = path(
        str(
            out_cfg[
                "directory"
            ]
        )
    )

    if (
        out_dir.exists()
        and list(
            out_dir.glob(
                "resnet18_digital_variant_matched_control_*.yaml"
            )
        )
    ):
        raise RuntimeError(
            "Matched digital-variant control already exists"
        )

    setup_log(
        config,
        stamp,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / str(
            out_cfg[
                "csv_filename"
            ]
        ).format(
            timestamp=stamp
        )
    )

    yaml_path = (
        out_dir
        / str(
            out_cfg[
                "yaml_filename"
            ]
        ).format(
            timestamp=stamp
        )
    )

    try:

        LOG.info(
            "=" * 72
        )

        LOG.info(
            "RESNET-18 MATCHED DIGITAL-VARIANT POST-HOC CONTROL"
        )

        LOG.info(
            "=" * 72
        )

        LOG.info(
            "Git commit: %s",
            commit,
        )

        LOG.info(
            "[PASS] frozen final summary SHA-256 = %s",
            final_sha,
        )

        LOG.info(
            "[PASS] subgroup diagnostic SHA-256 = %s",
            subgroup_sha,
        )

        per_seed = mp(
            final[
                "per_seed"
            ],
            "final.per_seed",
        )

        all_pairs: list[
            dict[str, Any]
        ] = []

        summaries: dict[
            int,
            dict[
                str,
                dict[
                    str,
                    float | int,
                ],
            ],
        ] = {}

        inputs: dict[
            int,
            dict[str, Any],
        ] = {}

        for seed in SEEDS:

            info = mp(
                seed_item(
                    per_seed,
                    seed,
                ),
                f"per_seed.{seed}",
            )

            source = mp(
                info[
                    "source"
                ],
                "source",
            )

            checkpoint = mp(
                info[
                    "checkpoint"
                ],
                "checkpoint",
            )

            threshold_path = path(
                str(
                    source[
                        "threshold_artifact_path"
                    ]
                )
            )

            threshold_sha = require_sha(
                threshold_path,
                str(
                    source[
                        "threshold_artifact_sha256"
                    ]
                ),
                f"Seed {seed} threshold artifact",
            )

            threshold_artifact = yload(
                threshold_path
            )

            t_checkpoint = mp(
                threshold_artifact[
                    "checkpoint"
                ],
                "threshold.checkpoint",
            )

            for field in (
                "seed",
                "run_id",
                "file_sha256",
                "model_state_sha256",
            ):

                if (
                    t_checkpoint[
                        field
                    ]
                    != checkpoint[
                        field
                    ]
                ):
                    raise RuntimeError(
                        f"Seed {seed} checkpoint mismatch: {field}"
                    )

            threshold = float(
                info[
                    "thresholds"
                ][
                    "controlled_fpr10"
                ]
            )

            if (
                threshold
                != float(
                    threshold_artifact[
                        "controlled_fpr10"
                    ][
                        "threshold"
                    ]
                )
            ):
                raise RuntimeError(
                    f"Seed {seed} frozen threshold mismatch"
                )

            dev_info = mp(
                threshold_artifact[
                    "predictions"
                ],
                "threshold.predictions",
            )

            dev_path = path(
                str(
                    dev_info[
                        "path"
                    ]
                )
            )

            dev_sha = require_sha(
                dev_path,
                str(
                    dev_info[
                        "sha256"
                    ]
                ),
                f"Seed {seed} dev predictions",
            )

            test_path = path(
                str(
                    source[
                        "predictions_path"
                    ]
                )
            )

            test_sha = require_sha(
                test_path,
                str(
                    source[
                        "predictions_sha256"
                    ]
                ),
                f"Seed {seed} test predictions",
            )

            dev_scores, stems = dev_map(
                rows(
                    dev_path,
                    DEV_COLUMNS,
                ),
                threshold,
            )

            d3_scores = d3_map(
                rows(
                    test_path,
                    TEST_COLUMNS,
                ),
                stems,
                threshold,
            )

            pairs = paired_rows(
                seed,
                dev_scores,
                d3_scores,
            )

            if len(
                pairs
            ) != 153:
                raise RuntimeError(
                    f"Seed {seed}: "
                    "expected 153 pairs, "
                    f"observed {len(pairs)}"
                )

            all_pairs.extend(
                pairs
            )

            scope_summary = {
                "ALL":
                    summarize(
                        pairs
                    )
            }

            for hw in HARDWARE:

                subset = [
                    r
                    for r
                    in pairs
                    if r[
                        "hardware_source"
                    ] == hw
                ]

                if len(
                    subset
                ) != 51:
                    raise RuntimeError(
                        f"Seed {seed} {hw}: expected 51 pairs"
                    )

                scope_summary[
                    hw
                ] = summarize(
                    subset
                )

            summaries[
                seed
            ] = scope_summary

            s = scope_summary[
                "ALL"
            ]

            LOG.info(
                "Seed %d | "
                "median p bona/d1/d2/d3=%.6g/%.6g/%.6g/%.6g | "
                "fixed d1/d2/d3=%.4f/%.4f/%.4f | "
                "d3-below-both=%.4f | "
                "fixed-collapse=%.4f",
                seed,
                s[
                    "median_p_bonafide"
                ],
                s[
                    "median_p_digital1"
                ],
                s[
                    "median_p_digital2"
                ],
                s[
                    "median_p_digital3"
                ],
                s[
                    "fixed_positive_rate_digital1"
                ],
                s[
                    "fixed_positive_rate_digital2"
                ],
                s[
                    "fixed_positive_rate_digital3"
                ],
                s[
                    "rate_d3_below_both_dev_attacks"
                ],
                s[
                    "rate_dev_attacks_fixed_detected_then_d3_missed"
                ],
            )

            inputs[
                seed
            ] = {
                "threshold":
                    threshold,

                "checkpoint":
                    dict(
                        checkpoint
                    ),

                "threshold_artifact_path":
                    rel(
                        threshold_path
                    ),

                "threshold_artifact_sha256":
                    threshold_sha,

                "dev_predictions_path":
                    rel(
                        dev_path
                    ),

                "dev_predictions_sha256":
                    dev_sha,

                "test_predictions_path":
                    rel(
                        test_path
                    ),

                "test_predictions_sha256":
                    test_sha,
            }

        if len(
            all_pairs
        ) != 459:
            raise RuntimeError(
                "Expected 459 seed x matched-unit rows"
            )

        agg = {
            scope:
                aggregate(
                    {
                        seed:
                            summaries[
                                seed
                            ][
                                scope
                            ]
                        for seed
                        in SEEDS
                    }
                )
            for scope
            in (
                "ALL",
                *HARDWARE,
            )
        }

        pair_sha = write_csv(
            csv_path,
            all_pairs,
        )

        artifact = {
            "schema_version":
                1,

            "artifact_type":
                "resnet18_digital_variant_matched_posthoc_control",

            "status":
                "POST_HOC_MATCHED_CONTROL_COMPLETE_NOT_MODEL_SELECTION",

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

            "frozen_final_summary":
                {
                    "path":
                        rel(
                            final_path
                        ),

                    "sha256":
                        final_sha,
                },

            "prerequisite_subgroup_diagnostic":
                {
                    "path":
                        rel(
                            subgroup_path
                        ),

                    "sha256":
                        subgroup_sha,
                },

            "design":
                {
                    "matching_key":
                        [
                            "file_stem",
                            "hardware_source",
                        ],

                    "dev_stems":
                        51,

                    "hardware_sources":
                        list(
                            HARDWARE
                        ),

                    "matched_units_per_seed":
                        153,

                    "dev_members":
                        [
                            "bonafide",
                            "digital_1",
                            "digital_2",
                        ],

                    "test_member":
                        "digital_3",

                    "interpretation_limit":
                        (
                            "dev_val participated in model selection and "
                            "digital_3 is from the official test split; "
                            "matching controls file_stem and hardware but "
                            "not split or generation-pipeline differences"
                        ),
                },

            "inputs_by_seed":
                inputs,

            "summary_by_seed":
                summaries,

            "aggregate_mean_min_max_by_scope":
                agg,

            "matched_rows":
                {
                    "path":
                        rel(
                            csv_path
                        ),

                    "sha256":
                        pair_sha,

                    "rows":
                        459,
                },

            "scientific_boundaries":
                {
                    "post_hoc":
                        True,

                    "used_for_model_selection":
                        False,

                    "used_for_checkpoint_selection":
                        False,

                    "used_for_threshold_selection":
                        False,

                    "threshold_derived":
                        False,

                    "threshold_modified":
                        False,

                    "images_opened":
                        False,

                    "model_loaded":
                        False,

                    "checkpoint_loaded":
                        False,

                    "inference_performed":
                        False,

                    "persisted_predictions_only":
                        True,
                },
        }

        with yaml_path.open(
            "x",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                artifact,
                f,
                sort_keys=False,
            )

        yaml_sha = digest(
            yaml_path
        )

        overall = agg[
            "ALL"
        ]

        LOG.info(
            ""
        )

        LOG.info(
            "Aggregate fixed d1/d2/d3 positive-rate mean = "
            "%.4f / %.4f / %.4f",
            overall[
                "fixed_positive_rate_digital1"
            ][
                "mean"
            ],
            overall[
                "fixed_positive_rate_digital2"
            ][
                "mean"
            ],
            overall[
                "fixed_positive_rate_digital3"
            ][
                "mean"
            ],
        )

        LOG.info(
            "Aggregate d3-below-both mean/min/max = "
            "%.4f / %.4f / %.4f",
            overall[
                "rate_d3_below_both_dev_attacks"
            ][
                "mean"
            ],
            overall[
                "rate_d3_below_both_dev_attacks"
            ][
                "min"
            ],
            overall[
                "rate_d3_below_both_dev_attacks"
            ][
                "max"
            ],
        )

        LOG.info(
            "Aggregate fixed-collapse mean/min/max = "
            "%.4f / %.4f / %.4f",
            overall[
                "rate_dev_attacks_fixed_detected_then_d3_missed"
            ][
                "mean"
            ],
            overall[
                "rate_dev_attacks_fixed_detected_then_d3_missed"
            ][
                "min"
            ],
            overall[
                "rate_dev_attacks_fixed_detected_then_d3_missed"
            ][
                "max"
            ],
        )

        LOG.info(
            "[PASS] exactly 153 matched stem x hardware units per seed"
        )

        LOG.info(
            "[PASS] dev/test scores bound to the same frozen checkpoint per seed"
        )

        LOG.info(
            "[PASS] thresholds reused unchanged from frozen dev artifacts"
        )

        LOG.info(
            "[PASS] images/models/checkpoints/inference accessed: FALSE"
        )

        LOG.info(
            "[PASS] model/checkpoint/threshold selection: FALSE"
        )

        LOG.info(
            "Matched CSV: %s",
            csv_path,
        )

        LOG.info(
            "Matched CSV SHA-256: %s",
            pair_sha,
        )

        LOG.info(
            "Matched YAML: %s",
            yaml_path,
        )

        LOG.info(
            "Matched YAML SHA-256: %s",
            yaml_sha,
        )

        LOG.info(
            "=" * 72
        )

        LOG.info(
            "MATCHED DIGITAL-VARIANT POST-HOC CONTROL: COMPLETE"
        )

        LOG.info(
            "=" * 72
        )

        return 0

    except Exception:

        LOG.exception(
            "MATCHED DIGITAL-VARIANT POST-HOC CONTROL: FAIL"
        )

        for p in (
            csv_path,
            yaml_path,
        ):

            if p.exists():
                p.unlink()

        return 1

    finally:

        for handler in list(
            LOG.handlers
        ):

            handler.flush()
            handler.close()

        LOG.handlers.clear()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )