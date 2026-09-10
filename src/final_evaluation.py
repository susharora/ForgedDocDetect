"""
Frozen Tech-2 held-out-test access and final binary evaluation primitives.

This module provides two deliberately separate capabilities.

1. Held-out-test Dataset access
-------------------------------
The canonical test CSV is first validated as metadata only.

Actual image access occurs ONLY through HeldOutTestManifestDataset.__getitem__.

For every accessed test image:

    canonical relative path
        ->
    exact frozen file SHA-256 verification
        ->
    Pillow decode
        ->
    frozen ResNet-18 preprocessing

The Dataset requires an explicit access token so importing or validating
the module cannot accidentally open held-out pixels.

2. Final binary metrics
-----------------------
Native class convention:

    bonafide = 0
    attack   = 1

Threshold decisions use:

    p_attack = softmax(float64([z0,z1]))[1]

with:

    attack iff p_attack >= threshold

Two operating points are evaluated:

    fixed threshold = 0.5
    frozen seed-specific FPR10 threshold

AUROC uses:

    z1 - z0

which is exactly ranking-equivalent to two-class p_attack.

No threshold derivation exists in this module.
No training/model-selection logic exists in this module.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.data import (
    DeterministicDocumentPreprocessor,
    build_preprocessor_from_config,
)

from src.development_metrics import (
    attack_ranking_score_from_logits,
    binary_auroc_pairwise,
)

from src.fpr10 import (
    apply_attack_threshold,
    attack_probability_from_logits,
)


# ======================================================================
# Frozen constants
# ======================================================================

HELD_OUT_TEST_ACCESS_TOKEN = (
    "OPEN_FROZEN_HELD_OUT_TEST_FOR_FINAL_EVALUATION"
)


HELD_OUT_MANIFEST_COLUMNS = (
    "source_workbook_row",
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "label",
    "variant",
    "hardware_source",
    "face_db",
    "face_id",
    "gender",
    "project_role",
)


EXPECTED_TEST_ROWS = 1385

EXPECTED_TEST_CLASS_COUNTS = {
    "bonafide":
        300,

    "attack":
        1085,
}


EXPECTED_TEST_GROUP_COUNTS = {
    (
        "bonafide",
        "",
    ):
        300,

    (
        "attack",
        "digital_3",
    ):
        786,

    (
        "attack",
        "facedancer",
    ):
        150,

    (
        "attack",
        "textdiffuserft_bfei",
    ):
        149,
}


# ======================================================================
# Records
# ======================================================================

@dataclass(
    frozen=True
)
class FinalEvaluationContract:

    bonafide_index: int
    attack_index: int
    positive_class: str

    fixed_threshold: float

    threshold_score: str
    threshold_score_dtype: str

    auroc_score: str

    required_metrics: tuple[str, ...]


@dataclass(
    frozen=True
)
class HeldOutManifestContract:

    metadata_path: str
    metadata_sha256: str

    manifest_path: str
    manifest_sha256: str

    rows: int

    bonafide_count: int
    attack_count: int

    project_role: str


@dataclass(
    frozen=True
)
class PerClassMetrics:

    support: int

    precision: float
    recall: float
    f1: float


@dataclass(
    frozen=True
)
class OperatingPointMetrics:

    name: str
    threshold: float

    sample_count: int

    bonafide_support: int
    attack_support: int

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    accuracy: float

    support_weighted_f1: float

    bonafide: PerClassMetrics
    attack: PerClassMetrics

    true_positive_rate: float
    true_negative_rate: float

    false_positive_rate: float
    false_negative_rate: float

    half_total_error_rate: float
    balanced_accuracy: float

    apcer: float
    bpcer: float

    matthews_correlation_coefficient: float | None


@dataclass(
    frozen=True
)
class FinalBinaryEvaluation:

    sample_count: int

    bonafide_count: int
    attack_count: int

    auroc: float

    auroc_positive_negative_pair_count: int
    auroc_strict_attack_wins: int
    auroc_tied_pairs: int

    auroc_score_definition: str

    fixed_0_5: OperatingPointMetrics

    controlled_fpr10: OperatingPointMetrics


# ======================================================================
# Generic helpers
# ======================================================================

def _require_mapping(
    value: Any,
    label: str,
) -> Mapping[Any, Any]:

    if not isinstance(
        value,
        Mapping,
    ):

        raise TypeError(
            f"{label} must be a mapping."
        )

    return value


def _sha256_file(
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


def _load_yaml(
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


def _safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:

    denominator_value = float(
        denominator
    )

    if denominator_value <= 0.0:

        raise RuntimeError(
            "Required metric denominator is not positive."
        )

    return (
        float(
            numerator
        )
        / denominator_value
    )


def _precision_or_zero(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:

        return 0.0

    return (
        numerator
        / denominator
    )


def _f1_from_precision_recall(
    *,
    precision: float,
    recall: float,
) -> float:

    denominator = (
        precision
        + recall
    )

    if denominator == 0.0:

        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )


# ======================================================================
# Final-evaluation protocol parser
# ======================================================================

def load_final_evaluation_contract(
    *,
    evaluation_cfg: Mapping[str, Any],
) -> FinalEvaluationContract:

    if evaluation_cfg.get(
        "schema_version"
    ) != 1:

        raise RuntimeError(
            "Final-evaluation protocol requires schema_version=1."
        )

    if evaluation_cfg.get(
        "artifact_type"
    ) != "resnet18_final_detection_protocol":

        raise RuntimeError(
            "Unexpected final-evaluation protocol artifact type."
        )

    if evaluation_cfg.get(
        "status"
    ) != "frozen_before_held_out_test_access":

        raise RuntimeError(
            "Final detection protocol is not frozen."
        )

    class_cfg = _require_mapping(
        evaluation_cfg[
            "class_contract"
        ],
        "class_contract",
    )

    index_by_name = _require_mapping(
        class_cfg[
            "index_by_name"
        ],
        "class_contract.index_by_name",
    )

    if dict(
        index_by_name
    ) != {
        "bonafide":
            0,

        "attack":
            1,
    }:

        raise RuntimeError(
            "Frozen class mapping must be bonafide=0 / attack=1."
        )

    if class_cfg[
        "positive_class"
    ] != "attack":

        raise RuntimeError(
            "Frozen positive class must be attack."
        )

    if int(
        class_cfg[
            "positive_class_index"
        ]
    ) != 1:

        raise RuntimeError(
            "Frozen positive class index must equal 1."
        )

    score_cfg = _require_mapping(
        evaluation_cfg[
            "score"
        ],
        "score",
    )

    threshold_score_cfg = _require_mapping(
        score_cfg[
            "threshold_decisions"
        ],
        "score.threshold_decisions",
    )

    if (
        threshold_score_cfg[
            "representation"
        ]
        != "attack_softmax_probability"
    ):

        raise RuntimeError(
            "Threshold score must remain p_attack."
        )

    if (
        threshold_score_cfg[
            "computation_dtype"
        ]
        != "float64"
    ):

        raise RuntimeError(
            "Threshold score must be computed in float64."
        )

    ranking_cfg = _require_mapping(
        score_cfg[
            "threshold_independent_ranking"
        ],
        "score.threshold_independent_ranking",
    )

    if (
        ranking_cfg[
            "authoritative_development_score"
        ]
        != "z1 - z0"
    ):

        raise RuntimeError(
            "Frozen AUROC ranking score changed."
        )

    operating_points = _require_mapping(
        evaluation_cfg[
            "operating_points"
        ],
        "operating_points",
    )

    fixed_cfg = _require_mapping(
        operating_points[
            "fixed_0_5"
        ],
        "operating_points.fixed_0_5",
    )

    fixed_threshold = float(
        fixed_cfg[
            "threshold"
        ]
    )

    if fixed_threshold != 0.5:

        raise RuntimeError(
            "Frozen fixed threshold must equal 0.5."
        )

    if (
        fixed_cfg[
            "prediction_rule"
        ]
        != "attack iff p_attack >= 0.5"
    ):

        raise RuntimeError(
            "Frozen fixed-threshold comparison changed."
        )

    if (
        fixed_cfg[
            "equality_at_threshold"
        ]
        != "attack"
    ):

        raise RuntimeError(
            "Exact threshold equality must classify as attack."
        )

    required_metrics = tuple(
        str(
            value
        )
        for value
        in fixed_cfg[
            "required_metrics"
        ]
    )

    required_set = {
        "confusion_matrix",
        "support_weighted_f1",
        "attack_class_f1",
        "balanced_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "half_total_error_rate",
    }

    if not required_set.issubset(
        set(
            required_metrics
        )
    ):

        raise RuntimeError(
            "Final-evaluation protocol lost a required fixed-threshold "
            "metric."
        )

    fpr_cfg = _require_mapping(
        operating_points[
            "controlled_fpr10"
        ],
        "operating_points.controlled_fpr10",
    )

    if (
        fpr_cfg[
            "report_same_threshold_dependent_metrics_as_fixed_0_5"
        ]
        is not True
    ):

        raise RuntimeError(
            "FPR10 must report the same operating metrics as 0.5."
        )

    auroc_cfg = _require_mapping(
        evaluation_cfg[
            "threshold_independent_metrics"
        ][
            "auroc"
        ],
        "threshold_independent_metrics.auroc",
    )

    if auroc_cfg[
        "enabled"
    ] is not True:

        raise RuntimeError(
            "Final AUROC reporting must remain enabled."
        )

    if auroc_cfg[
        "positive_class"
    ] != "attack":

        raise RuntimeError(
            "AUROC positive class must remain attack."
        )

    if auroc_cfg[
        "score"
    ] != "z1 - z0":

        raise RuntimeError(
            "AUROC score must remain z1-z0."
        )

    held_out_cfg = _require_mapping(
        evaluation_cfg[
            "held_out_test"
        ],
        "held_out_test",
    )

    forbidden_true = (
        "threshold_derivation_from_test",
        "threshold_reselection_on_test",
        "threshold_modification_after_test_access",
        "inspect_test_attack_scores_before_fpr10_derivation",
        "best_seed_selection_on_test",
    )

    for key in forbidden_true:

        if held_out_cfg[
            key
        ] is not False:

            raise RuntimeError(
                "Held-out-test protection changed:\n"
                f"  {key}"
            )

    if float(
        held_out_cfg[
            "fixed_0_5"
        ][
            "threshold"
        ]
    ) != 0.5:

        raise RuntimeError(
            "Held-out fixed threshold must remain 0.5."
        )

    if (
        held_out_cfg[
            "fixed_0_5"
        ][
            "apply_unchanged"
        ]
        is not True
    ):

        raise RuntimeError(
            "Held-out fixed threshold must be applied unchanged."
        )

    if (
        held_out_cfg[
            "controlled_fpr10"
        ][
            "source"
        ]
        != "dev_val"
    ):

        raise RuntimeError(
            "Held-out FPR10 threshold source must remain dev_val."
        )

    if (
        held_out_cfg[
            "controlled_fpr10"
        ][
            "apply_dev_derived_threshold_unchanged"
        ]
        is not True
    ):

        raise RuntimeError(
            "Frozen dev FPR10 threshold must be applied unchanged."
        )

    return FinalEvaluationContract(
        bonafide_index=0,
        attack_index=1,
        positive_class="attack",

        fixed_threshold=0.5,

        threshold_score=(
            "attack_softmax_probability"
        ),

        threshold_score_dtype=(
            "float64"
        ),

        auroc_score=(
            "z1 - z0"
        ),

        required_metrics=(
            required_metrics
        ),
    )


# ======================================================================
# Canonical held-out manifest metadata loader
# ======================================================================

def load_canonical_held_out_test_manifest(
    *,
    repo_root: Path,
    metadata_path: Path,
    expected_metadata_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[
    HeldOutManifestContract,
    tuple[
        dict[str, Any],
        ...,
    ],
]:
    """
    Validate canonical held-out metadata + CSV.

    No dataset_root is accepted by this function.
    Therefore test image paths cannot be followed here.
    """

    repo_root = (
        Path(
            repo_root
        )
        .expanduser()
        .resolve()
    )

    metadata_path = (
        Path(
            metadata_path
        )
        .expanduser()
        .resolve()
    )

    actual_metadata_sha = (
        _sha256_file(
            metadata_path
        )
    )

    if (
        actual_metadata_sha
        != expected_metadata_sha256
    ):

        raise RuntimeError(
            "Held-out manifest metadata SHA mismatch:\n"
            f"  expected={expected_metadata_sha256}\n"
            f"  actual={actual_metadata_sha}"
        )

    metadata = _load_yaml(
        metadata_path
    )

    if (
        metadata.get(
            "artifact_type"
        )
        != "fantasyid_held_out_test_manifest"
    ):

        raise RuntimeError(
            "Unexpected held-out manifest metadata artifact type."
        )

    if (
        metadata.get(
            "status"
        )
        != "FROZEN_BEFORE_MODEL_TEST_ACCESS"
    ):

        raise RuntimeError(
            "Held-out manifest was not frozen before model test access."
        )

    access_boundary = _require_mapping(
        metadata[
            "access_boundary"
        ],
        "held_out_manifest.access_boundary",
    )

    required_false = (
        "test_image_paths_followed",
        "test_image_files_opened",
        "test_image_pixels_decoded",
        "test_image_hashes_recomputed",
        "test_json_files_opened",
        "test_regions_inspected",
        "model_loaded",
        "checkpoint_loaded",
        "model_inference_performed",
        "metric_calculation_performed",
        "threshold_derived_or_modified",
    )

    for key in required_false:

        if access_boundary[
            key
        ] is not False:

            raise RuntimeError(
                "Pre-test access boundary was already crossed:\n"
                f"  {key}"
            )

    manifest_cfg = _require_mapping(
        metadata[
            "manifest"
        ],
        "held_out_manifest.manifest",
    )

    manifest_path = (
        repo_root
        / str(
            manifest_cfg[
                "path"
            ]
        )
    ).resolve()

    try:

        manifest_path.relative_to(
            repo_root
        )

    except ValueError as exc:

        raise RuntimeError(
            "Canonical held-out manifest path escaped repository root."
        ) from exc

    metadata_manifest_sha = str(
        manifest_cfg[
            "sha256"
        ]
    )

    if (
        metadata_manifest_sha
        != expected_manifest_sha256
    ):

        raise RuntimeError(
            "Held-out metadata records an unexpected manifest SHA."
        )

    actual_manifest_sha = _sha256_file(
        manifest_path
    )

    if (
        actual_manifest_sha
        != expected_manifest_sha256
    ):

        raise RuntimeError(
            "Canonical held-out manifest SHA mismatch:\n"
            f"  expected={expected_manifest_sha256}\n"
            f"  actual={actual_manifest_sha}"
        )

    if int(
        manifest_cfg[
            "rows"
        ]
    ) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Held-out metadata row count must equal 1385."
        )

    recorded_columns = tuple(
        str(
            value
        )
        for value
        in manifest_cfg[
            "columns"
        ]
    )

    if (
        recorded_columns
        != HELD_OUT_MANIFEST_COLUMNS
    ):

        raise RuntimeError(
            "Held-out manifest column contract changed."
        )

    if (
        manifest_cfg[
            "sorting_performed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Canonical held-out manifest must preserve frozen order."
        )

    held_out_cfg = _require_mapping(
        metadata[
            "held_out_test"
        ],
        "held_out_manifest.held_out_test",
    )

    if int(
        held_out_cfg[
            "rows"
        ]
    ) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Held-out metadata count changed."
        )

    if dict(
        held_out_cfg[
            "class_mapping"
        ]
    ) != {
        "bonafide":
            0,

        "attack":
            1,
    }:

        raise RuntimeError(
            "Held-out class mapping changed."
        )

    if {
        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value,
        ) in held_out_cfg[
            "class_counts"
        ].items()
    } != EXPECTED_TEST_CLASS_COUNTS:

        raise RuntimeError(
            "Held-out class counts changed."
        )

    # ------------------------------------------------------------------
    # CSV metadata validation only.
    # No dataset_root exists in this function.
    # ------------------------------------------------------------------

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:

            raise RuntimeError(
                "Held-out manifest has no header."
            )

        if tuple(
            reader.fieldnames
        ) != HELD_OUT_MANIFEST_COLUMNS:

            raise RuntimeError(
                "Canonical held-out CSV column order changed."
            )

        rows = list(
            reader
        )

    if len(
        rows
    ) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Canonical held-out CSV must contain 1385 rows."
        )

    class_counts: Counter[
        str
    ] = Counter()

    group_counts: Counter[
        tuple[
            str,
            str,
        ]
    ] = Counter()

    seen_paths: set[
        str
    ] = set()

    seen_hashes: set[
        str
    ] = set()

    previous_workbook_row: int | None = None

    prepared_rows: list[
        dict[str, Any]
    ] = []

    for row_index, row in enumerate(
        rows
    ):

        traffic_type = str(
            row[
                "traffic_type"
            ]
        )

        if traffic_type not in {
            "bonafide",
            "attack",
        }:

            raise RuntimeError(
                "Unexpected held-out traffic_type."
            )

        label = int(
            row[
                "label"
            ]
        )

        expected_label = (
            0
            if traffic_type == "bonafide"
            else 1
        )

        if label != expected_label:

            raise RuntimeError(
                "Held-out semantic/numerical label mismatch."
            )

        if (
            row[
                "project_role"
            ]
            != "held_out_test"
        ):

            raise RuntimeError(
                "Held-out project_role changed."
            )

        image_path = str(
            row[
                "image_path"
            ]
        )

        pure_path = PurePosixPath(
            image_path
        )

        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or pure_path.parts[
                0
            ] != "test"
            or any(
                part in {
                    ".",
                    "..",
                }
                for part
                in pure_path.parts
            )
        ):

            raise RuntimeError(
                "Unsafe canonical held-out image path:\n"
                f"  {image_path}"
            )

        if "\\" in image_path:

            raise RuntimeError(
                "Canonical held-out image path contains backslash."
            )

        if image_path in seen_paths:

            raise RuntimeError(
                "Duplicate held-out image path."
            )

        seen_paths.add(
            image_path
        )

        image_sha = str(
            row[
                "image_sha256"
            ]
        ).lower()

        if (
            len(
                image_sha
            )
            != 64
            or any(
                character
                not in "0123456789abcdef"
                for character
                in image_sha
            )
        ):

            raise RuntimeError(
                "Malformed canonical held-out image SHA-256."
            )

        if image_sha in seen_hashes:

            raise RuntimeError(
                "Duplicate canonical held-out image SHA-256."
            )

        seen_hashes.add(
            image_sha
        )

        workbook_row = int(
            row[
                "source_workbook_row"
            ]
        )

        if workbook_row < 2:

            raise RuntimeError(
                "Invalid source workbook row."
            )

        if (
            previous_workbook_row
            is not None
            and workbook_row
            <= previous_workbook_row
        ):

            raise RuntimeError(
                "Held-out manifest no longer preserves workbook order."
            )

        previous_workbook_row = (
            workbook_row
        )

        variant = str(
            row[
                "variant"
            ]
        )

        class_counts[
            traffic_type
        ] += 1

        group_counts[
            (
                traffic_type,
                variant,
            )
        ] += 1

        prepared_rows.append(
            {
                "source_workbook_row":
                    workbook_row,

                "image_path":
                    image_path,

                "image_sha256":
                    image_sha,

                "file_stem":
                    str(
                        row[
                            "file_stem"
                        ]
                    ),

                "traffic_type":
                    traffic_type,

                "label":
                    label,

                "variant":
                    variant,

                "hardware_source":
                    str(
                        row[
                            "hardware_source"
                        ]
                    ),

                "face_db":
                    str(
                        row[
                            "face_db"
                        ]
                    ),

                "face_id":
                    str(
                        row[
                            "face_id"
                        ]
                    ),

                "gender":
                    str(
                        row[
                            "gender"
                        ]
                    ),

                "project_role":
                    "held_out_test",
            }
        )

    if dict(
        class_counts
    ) != EXPECTED_TEST_CLASS_COUNTS:

        raise RuntimeError(
            "Canonical held-out class counts do not reconcile."
        )

    if dict(
        group_counts
    ) != EXPECTED_TEST_GROUP_COUNTS:

        raise RuntimeError(
            "Canonical held-out traffic/variant counts changed."
        )

    contract = HeldOutManifestContract(
        metadata_path=str(
            metadata_path
        ),

        metadata_sha256=(
            actual_metadata_sha
        ),

        manifest_path=str(
            manifest_path
        ),

        manifest_sha256=(
            actual_manifest_sha
        ),

        rows=(
            EXPECTED_TEST_ROWS
        ),

        bonafide_count=(
            EXPECTED_TEST_CLASS_COUNTS[
                "bonafide"
            ]
        ),

        attack_count=(
            EXPECTED_TEST_CLASS_COUNTS[
                "attack"
            ]
        ),

        project_role=(
            "held_out_test"
        ),
    )

    return (
        contract,
        tuple(
            prepared_rows
        ),
    )


# ======================================================================
# Actual held-out Dataset
# ======================================================================

class HeldOutTestManifestDataset(
    Dataset[
        dict[str, Any]
    ]
):
    """
    Test Dataset with mandatory per-access frozen file SHA verification.

    Constructor:
        metadata only

    __getitem__:
        FIRST actual held-out file access
    """

    def __init__(
        self,
        *,
        rows: Sequence[
            Mapping[str, Any]
        ],
        dataset_root: Path,
        preprocessor: DeterministicDocumentPreprocessor,
        expected_rows: int,
        expected_class_counts: Mapping[
            str,
            int,
        ],
        access_token: str,
    ) -> None:

        super().__init__()

        if (
            access_token
            != HELD_OUT_TEST_ACCESS_TOKEN
        ):

            raise PermissionError(
                "Held-out-test Dataset requires explicit final-"
                "evaluation access authorization."
            )

        dataset_root = (
            Path(
                dataset_root
            )
            .expanduser()
            .resolve()
        )

        if not dataset_root.is_dir():

            raise FileNotFoundError(
                f"Dataset root not found: {dataset_root}"
            )

        if len(
            rows
        ) != expected_rows:

            raise RuntimeError(
                "Held-out Dataset row-count mismatch."
            )

        observed_counts = Counter(
            str(
                row[
                    "traffic_type"
                ]
            )

            for row
            in rows
        )

        normalized_expected_counts = {
            str(
                key
            ):
                int(
                    value
                )

            for (
                key,
                value,
            ) in expected_class_counts.items()
        }

        if (
            dict(
                observed_counts
            )
            != normalized_expected_counts
        ):

            raise RuntimeError(
                "Held-out Dataset class-count mismatch."
            )

        prepared: list[
            dict[str, Any]
        ] = []

        for row in rows:

            traffic_type = str(
                row[
                    "traffic_type"
                ]
            )

            label = int(
                row[
                    "label"
                ]
            )

            if (
                traffic_type == "bonafide"
                and label != 0
            ):

                raise RuntimeError(
                    "Held-out bona-fide label must equal 0."
                )

            if (
                traffic_type == "attack"
                and label != 1
            ):

                raise RuntimeError(
                    "Held-out attack label must equal 1."
                )

            if traffic_type not in {
                "bonafide",
                "attack",
            }:

                raise RuntimeError(
                    "Unexpected held-out traffic type."
                )

            if (
                str(
                    row[
                        "project_role"
                    ]
                )
                != "held_out_test"
            ):

                raise RuntimeError(
                    "Held-out Dataset requires project_role=held_out_test."
                )

            relative_text = str(
                row[
                    "image_path"
                ]
            )

            pure_path = PurePosixPath(
                relative_text
            )

            if (
                pure_path.is_absolute()
                or not pure_path.parts
                or pure_path.parts[
                    0
                ] != "test"
                or any(
                    part in {
                        ".",
                        "..",
                    }
                    for part
                    in pure_path.parts
                )
            ):

                raise RuntimeError(
                    "Unsafe held-out image path."
                )

            image_sha = str(
                row[
                    "image_sha256"
                ]
            ).lower()

            if (
                len(
                    image_sha
                )
                != 64
                or any(
                    character
                    not in "0123456789abcdef"
                    for character
                    in image_sha
                )
            ):

                raise RuntimeError(
                    "Malformed held-out image SHA-256."
                )

            prepared.append(
                dict(
                    row
                )
            )

        self.rows = tuple(
            prepared
        )

        self.dataset_root = (
            dataset_root
        )

        self.preprocessor = (
            preprocessor
        )

        self.expected_rows = int(
            expected_rows
        )

        self.expected_class_counts = (
            normalized_expected_counts
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = self.rows[
            index
        ]

        pure_path = PurePosixPath(
            str(
                row[
                    "image_path"
                ]
            )
        )

        image_path = (
            self.dataset_root
            / Path(
                *pure_path.parts
            )
        ).resolve()

        try:

            image_path.relative_to(
                self.dataset_root
            )

        except ValueError as exc:

            raise RuntimeError(
                "Held-out path escaped configured dataset_root."
            ) from exc

        if not image_path.is_file():

            raise FileNotFoundError(
                "Held-out image does not exist:\n"
                f"  {image_path}"
            )

        # --------------------------------------------------------------
        # Exact-file integrity verification occurs BEFORE decode.
        # --------------------------------------------------------------

        expected_sha = str(
            row[
                "image_sha256"
            ]
        )

        actual_sha = _sha256_file(
            image_path
        )

        if actual_sha != expected_sha:

            raise RuntimeError(
                "Held-out image SHA-256 mismatch:\n"
                f"  path={image_path}\n"
                f"  expected={expected_sha}\n"
                f"  actual={actual_sha}"
            )

        with Image.open(
            image_path
        ) as image:

            tensor, geometry = (
                self.preprocessor(
                    image
                )
            )

        return {
            "image":
                tensor,

            "label":
                int(
                    row[
                        "label"
                    ]
                ),

            "source_workbook_row":
                int(
                    row[
                        "source_workbook_row"
                    ]
                ),

            "image_path":
                str(
                    row[
                        "image_path"
                    ]
                ),

            "image_sha256":
                expected_sha,

            "verified_image_sha256":
                actual_sha,

            "file_stem":
                str(
                    row[
                        "file_stem"
                    ]
                ),

            "traffic_type":
                str(
                    row[
                        "traffic_type"
                    ]
                ),

            "variant":
                str(
                    row[
                        "variant"
                    ]
                ),

            "hardware_source":
                str(
                    row[
                        "hardware_source"
                    ]
                ),

            "face_db":
                str(
                    row[
                        "face_db"
                    ]
                ),

            "face_id":
                str(
                    row[
                        "face_id"
                    ]
                ),

            "gender":
                str(
                    row[
                        "gender"
                    ]
                ),

            "project_role":
                "held_out_test",

            "geometry":
                geometry.as_dict(),
        }


def build_held_out_test_dataset(
    *,
    experiment_cfg: Mapping[str, Any],
    machine_cfg: Mapping[str, Any],
    repo_root: Path,
    metadata_path: Path,
    expected_metadata_sha256: str,
    expected_manifest_sha256: str,
    resolution_name: str,
    access_token: str,
) -> tuple[
    HeldOutManifestContract,
    HeldOutTestManifestDataset,
]:
    """
    Production held-out Dataset builder.

    Passing the explicit access_token is the intentional transition from
    metadata-only pre-test state to actual final held-out access.
    """

    if resolution_name != "r512":

        raise RuntimeError(
            "Frozen final evaluation resolution must equal r512."
        )

    (
        manifest_contract,
        rows,
    ) = load_canonical_held_out_test_manifest(
        repo_root=(
            repo_root
        ),
        metadata_path=(
            metadata_path
        ),
        expected_metadata_sha256=(
            expected_metadata_sha256
        ),
        expected_manifest_sha256=(
            expected_manifest_sha256
        ),
    )

    paths_cfg = _require_mapping(
        machine_cfg[
            "paths"
        ],
        "machine_config.paths",
    )

    dataset_root = Path(
        paths_cfg[
            "dataset_root"
        ]
    ).expanduser().resolve()

    preprocessor = (
        build_preprocessor_from_config(
            experiment_cfg=(
                experiment_cfg
            ),
            resolution_name=(
                resolution_name
            ),
        )
    )

    dataset = HeldOutTestManifestDataset(
        rows=rows,

        dataset_root=(
            dataset_root
        ),

        preprocessor=(
            preprocessor
        ),

        expected_rows=(
            manifest_contract.rows
        ),

        expected_class_counts={
            "bonafide":
                manifest_contract
                .bonafide_count,

            "attack":
                manifest_contract
                .attack_count,
        },

        access_token=(
            access_token
        ),
    )

    return (
        manifest_contract,
        dataset,
    )


# ======================================================================
# Final metric input validation
# ======================================================================

def _validate_final_tensors(
    *,
    logits: Tensor,
    targets: Tensor,
) -> tuple[
    int,
    int,
    int,
]:

    if not isinstance(
        logits,
        Tensor,
    ):

        raise TypeError(
            "logits must be a torch.Tensor."
        )

    if not isinstance(
        targets,
        Tensor,
    ):

        raise TypeError(
            "targets must be a torch.Tensor."
        )

    if (
        logits.ndim != 2
        or logits.shape[
            1
        ] != 2
    ):

        raise RuntimeError(
            "Final logits must have shape [N,2]."
        )

    if targets.ndim != 1:

        raise RuntimeError(
            "Final targets must have shape [N]."
        )

    if (
        logits.shape[
            0
        ]
        != targets.shape[
            0
        ]
    ):

        raise RuntimeError(
            "Final logits/targets sample counts differ."
        )

    if logits.shape[
        0
    ] <= 0:

        raise RuntimeError(
            "Final evaluation cannot be empty."
        )

    if logits.dtype != torch.float32:

        raise RuntimeError(
            "Final stored logits must be float32."
        )

    if targets.dtype != torch.int64:

        raise RuntimeError(
            "Final targets must be int64."
        )

    if logits.device.type != "cpu":

        raise RuntimeError(
            "Final stored logits must be CPU resident."
        )

    if targets.device.type != "cpu":

        raise RuntimeError(
            "Final targets must be CPU resident."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):

        raise RuntimeError(
            "Final logits contain NaN or Inf."
        )

    if not bool(
        (
            (
                targets
                == 0
            )
            |
            (
                targets
                == 1
            )
        ).all()
    ):

        raise RuntimeError(
            "Final targets contain label outside frozen {0,1}."
        )

    bonafide_count = int(
        (
            targets
            == 0
        )
        .sum()
        .item()
    )

    attack_count = int(
        (
            targets
            == 1
        )
        .sum()
        .item()
    )

    if bonafide_count <= 0:

        raise RuntimeError(
            "Final binary evaluation requires bona-fide samples."
        )

    if attack_count <= 0:

        raise RuntimeError(
            "Final binary evaluation requires attack samples."
        )

    return (
        int(
            targets.numel()
        ),
        bonafide_count,
        attack_count,
    )


# ======================================================================
# One operating point
# ======================================================================

def compute_operating_point_metrics(
    *,
    scores: Tensor,
    targets: Tensor,
    threshold: float,
    name: str,
) -> OperatingPointMetrics:

    if not isinstance(
        scores,
        Tensor,
    ):

        raise TypeError(
            "scores must be a torch.Tensor."
        )

    if scores.ndim != 1:

        raise RuntimeError(
            "scores must have shape [N]."
        )

    if scores.dtype != torch.float64:

        raise RuntimeError(
            "Threshold scores must be float64."
        )

    if scores.device.type != "cpu":

        raise RuntimeError(
            "Threshold scores must be CPU resident."
        )

    if targets.ndim != 1:

        raise RuntimeError(
            "targets must have shape [N]."
        )

    if scores.shape != targets.shape:

        raise RuntimeError(
            "scores/targets shape mismatch."
        )

    if targets.dtype != torch.int64:

        raise RuntimeError(
            "targets must be int64."
        )

    if targets.device.type != "cpu":

        raise RuntimeError(
            "targets must be CPU resident."
        )

    if not bool(
        torch.isfinite(
            scores
        ).all()
    ):

        raise RuntimeError(
            "Threshold scores contain NaN or Inf."
        )

    if not bool(
        (
            (
                targets
                == 0
            )
            |
            (
                targets
                == 1
            )
        ).all()
    ):

        raise RuntimeError(
            "Targets outside frozen {0,1}."
        )

    predictions = apply_attack_threshold(
        scores=scores,
        threshold=threshold,
    )

    attack_truth = (
        targets
        == 1
    )

    bonafide_truth = (
        targets
        == 0
    )

    attack_prediction = (
        predictions
        == 1
    )

    bonafide_prediction = (
        predictions
        == 0
    )

    true_positive = int(
        (
            attack_truth
            & attack_prediction
        )
        .sum()
        .item()
    )

    false_negative = int(
        (
            attack_truth
            & bonafide_prediction
        )
        .sum()
        .item()
    )

    false_positive = int(
        (
            bonafide_truth
            & attack_prediction
        )
        .sum()
        .item()
    )

    true_negative = int(
        (
            bonafide_truth
            & bonafide_prediction
        )
        .sum()
        .item()
    )

    attack_support = (
        true_positive
        + false_negative
    )

    bonafide_support = (
        true_negative
        + false_positive
    )

    sample_count = (
        attack_support
        + bonafide_support
    )

    if attack_support <= 0:

        raise RuntimeError(
            "Operating metrics require attack support."
        )

    if bonafide_support <= 0:

        raise RuntimeError(
            "Operating metrics require bona-fide support."
        )

    # ------------------------------------------------------------------
    # Attack-class metrics.
    # ------------------------------------------------------------------

    attack_precision = (
        _precision_or_zero(
            true_positive,
            (
                true_positive
                + false_positive
            ),
        )
    )

    attack_recall = _safe_ratio(
        true_positive,
        attack_support,
    )

    attack_f1 = (
        _f1_from_precision_recall(
            precision=(
                attack_precision
            ),
            recall=(
                attack_recall
            ),
        )
    )

    # ------------------------------------------------------------------
    # Bona-fide class metrics.
    #
    # Treat class 0 as the positive class for this per-class calculation:
    #
    #     TP_0 = TN
    #     FP_0 = FN
    #     FN_0 = FP
    # ------------------------------------------------------------------

    bonafide_precision = (
        _precision_or_zero(
            true_negative,
            (
                true_negative
                + false_negative
            ),
        )
    )

    bonafide_recall = _safe_ratio(
        true_negative,
        bonafide_support,
    )

    bonafide_f1 = (
        _f1_from_precision_recall(
            precision=(
                bonafide_precision
            ),
            recall=(
                bonafide_recall
            ),
        )
    )

    support_weighted_f1 = (
        (
            bonafide_support
            * bonafide_f1
        )
        +
        (
            attack_support
            * attack_f1
        )
    ) / sample_count

    true_positive_rate = (
        attack_recall
    )

    true_negative_rate = (
        bonafide_recall
    )

    false_positive_rate = _safe_ratio(
        false_positive,
        bonafide_support,
    )

    false_negative_rate = _safe_ratio(
        false_negative,
        attack_support,
    )

    hter = (
        false_positive_rate
        + false_negative_rate
    ) / 2.0

    balanced_accuracy = (
        true_positive_rate
        + true_negative_rate
    ) / 2.0

    if not math.isclose(
        balanced_accuracy,
        (
            1.0
            - hter
        ),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):

        raise RuntimeError(
            "Balanced accuracy / HTER identity failed."
        )

    accuracy = _safe_ratio(
        (
            true_positive
            + true_negative
        ),
        sample_count,
    )

    # ------------------------------------------------------------------
    # MCC is supplemental rather than part of the minimum T2 contract.
    #
    # Degenerate prediction cases have denominator zero. Preserve that
    # fact as None rather than silently inventing a numeric value.
    # ------------------------------------------------------------------

    mcc_denominator_product = (
        (
            true_positive
            + false_positive
        )
        *
        (
            true_positive
            + false_negative
        )
        *
        (
            true_negative
            + false_positive
        )
        *
        (
            true_negative
            + false_negative
        )
    )

    if mcc_denominator_product == 0:

        mcc = None

    else:

        mcc = (
            (
                true_positive
                * true_negative
            )
            -
            (
                false_positive
                * false_negative
            )
        ) / math.sqrt(
            mcc_denominator_product
        )

    return OperatingPointMetrics(
        name=str(
            name
        ),

        threshold=float(
            threshold
        ),

        sample_count=(
            sample_count
        ),

        bonafide_support=(
            bonafide_support
        ),

        attack_support=(
            attack_support
        ),

        true_positive=(
            true_positive
        ),

        false_positive=(
            false_positive
        ),

        true_negative=(
            true_negative
        ),

        false_negative=(
            false_negative
        ),

        accuracy=(
            accuracy
        ),

        support_weighted_f1=(
            support_weighted_f1
        ),

        bonafide=PerClassMetrics(
            support=(
                bonafide_support
            ),

            precision=(
                bonafide_precision
            ),

            recall=(
                bonafide_recall
            ),

            f1=(
                bonafide_f1
            ),
        ),

        attack=PerClassMetrics(
            support=(
                attack_support
            ),

            precision=(
                attack_precision
            ),

            recall=(
                attack_recall
            ),

            f1=(
                attack_f1
            ),
        ),

        true_positive_rate=(
            true_positive_rate
        ),

        true_negative_rate=(
            true_negative_rate
        ),

        false_positive_rate=(
            false_positive_rate
        ),

        false_negative_rate=(
            false_negative_rate
        ),

        half_total_error_rate=(
            hter
        ),

        balanced_accuracy=(
            balanced_accuracy
        ),

        # Frozen binary attack-positive PAD-style equivalents.
        apcer=(
            false_negative_rate
        ),

        bpcer=(
            false_positive_rate
        ),

        matthews_correlation_coefficient=(
            mcc
        ),
    )


# ======================================================================
# Complete final binary evaluation
# ======================================================================

def compute_final_binary_evaluation(
    *,
    logits: Tensor,
    targets: Tensor,
    fpr10_threshold: float,
    contract: FinalEvaluationContract,
) -> FinalBinaryEvaluation:

    if not isinstance(
        contract,
        FinalEvaluationContract,
    ):

        raise TypeError(
            "contract must be FinalEvaluationContract."
        )

    (
        sample_count,
        bonafide_count,
        attack_count,
    ) = _validate_final_tensors(
        logits=logits,
        targets=targets,
    )

    threshold = float(
        fpr10_threshold
    )

    if not math.isfinite(
        threshold
    ):

        raise RuntimeError(
            "Frozen FPR10 threshold must be finite."
        )

    # ------------------------------------------------------------------
    # Threshold-dependent score.
    # ------------------------------------------------------------------

    p_attack = attack_probability_from_logits(
        logits
    )

    fixed_metrics = (
        compute_operating_point_metrics(
            scores=(
                p_attack
            ),
            targets=(
                targets
            ),
            threshold=(
                contract
                .fixed_threshold
            ),
            name=(
                "fixed_0_5"
            ),
        )
    )

    fpr10_metrics = (
        compute_operating_point_metrics(
            scores=(
                p_attack
            ),
            targets=(
                targets
            ),
            threshold=(
                threshold
            ),
            name=(
                "controlled_fpr10"
            ),
        )
    )

    # ------------------------------------------------------------------
    # Threshold-independent ranking.
    # ------------------------------------------------------------------

    ranking_scores = (
        attack_ranking_score_from_logits(
            logits
        )
    )

    (
        auroc,
        pair_count,
        strict_wins,
        tied_pairs,
    ) = binary_auroc_pairwise(
        scores=(
            ranking_scores
        ),
        targets=(
            targets
        ),
        positive_class_index=(
            contract
            .attack_index
        ),
    )

    expected_pairs = (
        bonafide_count
        * attack_count
    )

    if pair_count != expected_pairs:

        raise RuntimeError(
            "Final AUROC pair count does not reconcile."
        )

    return FinalBinaryEvaluation(
        sample_count=(
            sample_count
        ),

        bonafide_count=(
            bonafide_count
        ),

        attack_count=(
            attack_count
        ),

        auroc=(
            auroc
        ),

        auroc_positive_negative_pair_count=(
            pair_count
        ),

        auroc_strict_attack_wins=(
            strict_wins
        ),

        auroc_tied_pairs=(
            tied_pairs
        ),

        auroc_score_definition=(
            "attack_logit_minus_bonafide_logit"
        ),

        fixed_0_5=(
            fixed_metrics
        ),

        controlled_fpr10=(
            fpr10_metrics
        ),
    )