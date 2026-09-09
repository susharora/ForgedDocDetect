"""
FantasyID Dataset and deterministic ResNet-18 preprocessing.

This module implements the frozen Tech-2 primary-input contract.

Implemented here
----------------
- frozen project_train / dev_val manifests only;
- manifest SHA-256 verification;
- frozen class mapping;
- Pillow decode;
- explicit RGB conversion;
- NO EXIF orientation correction;
- aspect-ratio-preserving bilinear resize;
- antialias=True;
- NO crop;
- fixed landscape canvas;
- horizontal centre padding;
- ImageNet-mean padding;
- ImageNet normalization;
- geometry metadata required for later localization back-mapping.

Not implemented here
--------------------
- DataLoader construction / worker seeding;
- stochastic augmentation;
- region dropout;
- masked diagnostics;
- model construction;
- training;
- Grad-CAM;
- held-out test access.

The held-out test is deliberately unsupported by this module.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


# ======================================================================
# Development-only split policy
# ======================================================================

ALLOWED_DEVELOPMENT_SPLITS = {
    "project_train",
    "dev_val",
}


REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "image_sha256",
    "file_stem",
    "traffic_type",
    "variant",
    "hardware_source",
    "face_db",
    "face_id",
    "gender",
    "project_role",
}


# ======================================================================
# Geometry records
# ======================================================================

@dataclass(
    frozen=True
)
class CanvasSpec:
    """
    Fixed preprocessing canvas taken from the frozen scientific config.
    """

    name: str

    content_height: int
    canvas_height: int
    canvas_width: int


@dataclass(
    frozen=True
)
class DocumentGeometry:
    """
    Exact source -> model-input geometry for one image.

    Keeping this record now is important for later Grad-CAM mapping.

    Source coordinates:
        original_width x original_height

    Content coordinates:
        resized_width x resized_height

    Model-input coordinates:
        canvas_width x canvas_height
    """

    original_width: int
    original_height: int

    resized_width: int
    resized_height: int

    canvas_width: int
    canvas_height: int

    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int

    def as_dict(
        self,
    ) -> dict[str, int]:

        return {
            "original_width":
                self.original_width,

            "original_height":
                self.original_height,

            "resized_width":
                self.resized_width,

            "resized_height":
                self.resized_height,

            "canvas_width":
                self.canvas_width,

            "canvas_height":
                self.canvas_height,

            "pad_left":
                self.pad_left,

            "pad_right":
                self.pad_right,

            "pad_top":
                self.pad_top,

            "pad_bottom":
                self.pad_bottom,
        }


# ======================================================================
# File helpers
# ======================================================================

def sha256_file(
    path: Path,
) -> str:
    """
    Compute SHA-256 without loading an entire file into memory.
    """

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
    repo_root: Path,
    value: str | Path,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():

        path = (
            repo_root
            / path
        )

    return path.resolve()


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
# Deterministic preprocessing
# ======================================================================

class DeterministicDocumentPreprocessor:
    """
    Frozen deterministic ResNet-18 document preprocessing.

    Pipeline
    --------
    Pillow decode
        ->
    explicit RGB
        ->
    float32 [0, 1] tensor
        ->
    aspect-preserving bilinear resize
        ->
    horizontal centre padding with ImageNet mean
        ->
    ImageNet normalization

    There is intentionally:
    - no crop;
    - no stochastic augmentation;
    - no EXIF transpose;
    - no geometric jitter.
    """

    def __init__(
        self,
        *,
        canvas: CanvasSpec,
        mean: tuple[
            float,
            float,
            float,
        ],
        std: tuple[
            float,
            float,
            float,
        ],
    ) -> None:

        if canvas.content_height <= 0:

            raise ValueError(
                "content_height must be positive."
            )

        if canvas.canvas_height <= 0:

            raise ValueError(
                "canvas_height must be positive."
            )

        if canvas.canvas_width <= 0:

            raise ValueError(
                "canvas_width must be positive."
            )

        # Frozen protocol currently requires the resized document to
        # occupy the full canvas height, with horizontal padding only.
        if (
            canvas.content_height
            != canvas.canvas_height
        ):

            raise ValueError(
                "Frozen preprocessing requires "
                "content_height == canvas_height."
            )

        if len(
            mean
        ) != 3:

            raise ValueError(
                "Normalization mean must contain 3 channels."
            )

        if len(
            std
        ) != 3:

            raise ValueError(
                "Normalization std must contain 3 channels."
            )

        if any(
            value <= 0.0
            for value
            in std
        ):

            raise ValueError(
                "Normalization standard deviations "
                "must be positive."
            )

        self.canvas = canvas

        self.mean = tuple(
            float(
                value
            )
            for value
            in mean
        )

        self.std = tuple(
            float(
                value
            )
            for value
            in std
        )

        # Reuse precisely the same float32 values for:
        #   1. canvas fill
        #   2. normalization
        #
        # Therefore padded pixels become exactly normalized zero.
        self._mean_tensor = torch.tensor(
            self.mean,
            dtype=torch.float32,
        )

        self._std_tensor = torch.tensor(
            self.std,
            dtype=torch.float32,
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> tuple[
        Tensor,
        DocumentGeometry,
    ]:

        # --------------------------------------------------------------
        # Explicit RGB conversion.
        #
        # Deliberately DO NOT call ImageOps.exif_transpose().
        # --------------------------------------------------------------

        rgb_image = image.convert(
            "RGB"
        )

        original_width, original_height = (
            rgb_image.size
        )

        if (
            original_width <= 0
            or
            original_height <= 0
        ):

            raise RuntimeError(
                "Decoded image has invalid dimensions: "
                f"{original_width}x{original_height}"
            )

        # Frozen Phase-1 geometry established this project subset as
        # landscape. Enforce rather than silently changing resize
        # interpretation if unexpected data appears later.
        if (
            original_width
            <= original_height
        ):

            raise RuntimeError(
                "Frozen preprocessing expects a landscape document, "
                "but received "
                f"{original_width}x{original_height}."
            )

        # --------------------------------------------------------------
        # PIL -> float32 RGB tensor in [0,1].
        #
        # Explicit conversion is used instead of a Compose chain so the
        # numerical contract is easy to inspect and audit.
        # --------------------------------------------------------------

        tensor = TF.pil_to_tensor(
            rgb_image
        )

        if tensor.dtype != torch.uint8:

            raise RuntimeError(
                "Frozen FantasyID decode contract expected "
                f"uint8 JPEG data, got {tensor.dtype}."
            )

        if tuple(
            tensor.shape
        ) != (
            3,
            original_height,
            original_width,
        ):

            raise RuntimeError(
                "Unexpected decoded tensor shape:\n"
                f"  expected="
                f"{(3, original_height, original_width)}\n"
                f"  actual={tuple(tensor.shape)}"
            )

        tensor = (
            tensor
            .to(
                dtype=torch.float32
            )
            .div(
                255.0
            )
        )

        # --------------------------------------------------------------
        # Deterministic aspect-preserving resize.
        #
        # torchvision integer `size` sets the shorter spatial dimension.
        # Because landscape orientation is enforced above, the shorter
        # dimension is height.
        # --------------------------------------------------------------

        resized = TF.resize(
            tensor,
            size=self.canvas.content_height,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        if resized.ndim != 3:

            raise RuntimeError(
                "Resize produced an unexpected tensor rank: "
                f"{resized.ndim}"
            )

        channels, resized_height, resized_width = (
            resized.shape
        )

        if channels != 3:

            raise RuntimeError(
                "Resize changed channel count unexpectedly: "
                f"{channels}"
            )

        if (
            resized_height
            != self.canvas.content_height
        ):

            raise RuntimeError(
                "Aspect-preserving resize did not produce "
                "the frozen content height:\n"
                f"  expected={self.canvas.content_height}\n"
                f"  actual={resized_height}"
            )

        if (
            resized_height
            != self.canvas.canvas_height
        ):

            raise RuntimeError(
                "Frozen protocol permits horizontal padding only:\n"
                f"  resized_height={resized_height}\n"
                f"  canvas_height={self.canvas.canvas_height}"
            )

        if (
            resized_width
            > self.canvas.canvas_width
        ):

            raise RuntimeError(
                "Resized document exceeds frozen canvas width. "
                "The protocol requires failure rather than cropping:\n"
                f"  resolution={self.canvas.name}\n"
                f"  resized_width={resized_width}\n"
                f"  canvas_width={self.canvas.canvas_width}\n"
                f"  source={original_width}x{original_height}"
            )

        # --------------------------------------------------------------
        # Horizontal centre padding.
        # --------------------------------------------------------------

        horizontal_padding = (
            self.canvas.canvas_width
            - resized_width
        )

        pad_left = (
            horizontal_padding
            // 2
        )

        pad_right = (
            horizontal_padding
            - pad_left
        )

        pad_top = 0
        pad_bottom = 0

        # Create the entire canvas from the ImageNet mean in [0,1]
        # RGB space.
        #
        # A direct tensor canvas permits channel-specific fill values,
        # avoiding an implicit scalar-padding convention.
        canvas = (
            self._mean_tensor[
                :,
                None,
                None,
            ]
            .expand(
                3,
                self.canvas.canvas_height,
                self.canvas.canvas_width,
            )
            .clone()
        )

        canvas[
            :,
            :,
            pad_left:
            (
                pad_left
                + resized_width
            ),
        ] = resized

        # --------------------------------------------------------------
        # ImageNet normalization.
        # --------------------------------------------------------------

        normalized = TF.normalize(
            canvas,
            mean=self.mean,
            std=self.std,
        )

        expected_shape = (
            3,
            self.canvas.canvas_height,
            self.canvas.canvas_width,
        )

        if tuple(
            normalized.shape
        ) != expected_shape:

            raise RuntimeError(
                "Preprocessed tensor shape mismatch:\n"
                f"  expected={expected_shape}\n"
                f"  actual={tuple(normalized.shape)}"
            )

        if normalized.dtype != torch.float32:

            raise RuntimeError(
                "Preprocessed tensor must be float32, got "
                f"{normalized.dtype}."
            )

        if not torch.isfinite(
            normalized
        ).all():

            raise RuntimeError(
                "Preprocessed tensor contains NaN or Inf."
            )

        geometry = DocumentGeometry(
            original_width=original_width,
            original_height=original_height,
            resized_width=int(
                resized_width
            ),
            resized_height=int(
                resized_height
            ),
            canvas_width=self.canvas.canvas_width,
            canvas_height=self.canvas.canvas_height,
            pad_left=int(
                pad_left
            ),
            pad_right=int(
                pad_right
            ),
            pad_top=pad_top,
            pad_bottom=pad_bottom,
        )

        return (
            normalized,
            geometry,
        )


# ======================================================================
# Frozen-manifest Dataset
# ======================================================================

class FantasyIDManifestDataset(
    Dataset[
        dict[str, Any]
    ]
):
    """
    Dataset backed directly by one frozen Tech-1 development manifest.

    The manifest order is preserved exactly.

    This class deliberately supports only:
        project_train
        dev_val

    It cannot be pointed at the held-out test set through the public
    constructor.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_manifest_sha256: str,
        expected_rows: int,
        expected_role: str,
        dataset_root: Path,
        class_index_by_name: Mapping[
            str,
            int,
        ],
        expected_class_counts: Mapping[
            str,
            int,
        ],
        preprocessor: DeterministicDocumentPreprocessor,
    ) -> None:

        super().__init__()

        if (
            expected_role
            not in ALLOWED_DEVELOPMENT_SPLITS
        ):

            raise ValueError(
                "Dataset access is restricted to frozen development "
                "partitions. Unsupported role: "
                f"{expected_role!r}"
            )

        manifest_path = (
            manifest_path
            .expanduser()
            .resolve()
        )

        dataset_root = (
            dataset_root
            .expanduser()
            .resolve()
        )

        if not manifest_path.is_file():

            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        if not dataset_root.is_dir():

            raise FileNotFoundError(
                f"Dataset root not found: {dataset_root}"
            )

        actual_manifest_sha256 = (
            sha256_file(
                manifest_path
            )
        )

        if (
            actual_manifest_sha256
            != expected_manifest_sha256
        ):

            raise RuntimeError(
                "Frozen manifest SHA-256 mismatch:\n"
                f"  manifest={manifest_path}\n"
                f"  expected={expected_manifest_sha256}\n"
                f"  actual={actual_manifest_sha256}"
            )

        # --------------------------------------------------------------
        # Class contract.
        # --------------------------------------------------------------

        class_index_by_name = dict(
            class_index_by_name
        )

        expected_mapping = {
            "bonafide": 0,
            "attack": 1,
        }

        if (
            class_index_by_name
            != expected_mapping
        ):

            raise RuntimeError(
                "Unexpected binary class mapping:\n"
                f"  expected={expected_mapping}\n"
                f"  actual={class_index_by_name}"
            )

        expected_class_counts = {
            str(
                name
            ):
                int(
                    count
                )
            for (
                name,
                count,
            ) in expected_class_counts.items()
        }

        if (
            set(
                expected_class_counts
            )
            != set(
                expected_mapping
            )
        ):

            raise RuntimeError(
                "Expected class-count mapping must contain "
                "exactly bonafide and attack."
            )

        # --------------------------------------------------------------
        # Load frozen rows.
        # --------------------------------------------------------------

        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if reader.fieldnames is None:

                raise RuntimeError(
                    "Manifest has no header."
                )

            missing_columns = (
                REQUIRED_MANIFEST_COLUMNS
                - set(
                    reader.fieldnames
                )
            )

            if missing_columns:

                raise RuntimeError(
                    "Manifest is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            rows = list(
                reader
            )

        if len(
            rows
        ) != expected_rows:

            raise RuntimeError(
                "Frozen manifest row-count mismatch:\n"
                f"  expected={expected_rows}\n"
                f"  actual={len(rows)}"
            )

        # --------------------------------------------------------------
        # Row-level structural checks.
        # --------------------------------------------------------------

        observed_class_counts = {
            "bonafide": 0,
            "attack": 0,
        }

        seen_image_paths: set[
            str
        ] = set()

        seen_image_hashes: set[
            str
        ] = set()

        prepared_rows: list[
            dict[str, Any]
        ] = []

        for (
            row_index,
            row,
        ) in enumerate(
            rows
        ):

            project_role = row[
                "project_role"
            ]

            if (
                project_role
                != expected_role
            ):

                raise RuntimeError(
                    "Manifest project_role mismatch:\n"
                    f"  row={row_index}\n"
                    f"  expected={expected_role!r}\n"
                    f"  actual={project_role!r}"
                )

            traffic_type = row[
                "traffic_type"
            ]

            if (
                traffic_type
                not in class_index_by_name
            ):

                raise RuntimeError(
                    "Unexpected traffic_type:\n"
                    f"  row={row_index}\n"
                    f"  value={traffic_type!r}"
                )

            observed_class_counts[
                traffic_type
            ] += 1

            relative_image_path = Path(
                row[
                    "image_path"
                ]
            )

            if relative_image_path.is_absolute():

                raise RuntimeError(
                    "Manifest image_path must be relative to "
                    "dataset_root:\n"
                    f"  row={row_index}\n"
                    f"  path={relative_image_path}"
                )

            image_path = (
                dataset_root
                / relative_image_path
            ).resolve()

            # Prevent accidental path traversal outside the configured
            # FantasyID root.
            try:

                image_path.relative_to(
                    dataset_root
                )

            except ValueError as exc:

                raise RuntimeError(
                    "Manifest image path escapes dataset_root:\n"
                    f"  row={row_index}\n"
                    f"  path={image_path}"
                ) from exc

            if not image_path.is_file():

                raise FileNotFoundError(
                    "Manifest image does not exist:\n"
                    f"  row={row_index}\n"
                    f"  path={image_path}"
                )

            image_path_text = row[
                "image_path"
            ]

            if (
                image_path_text
                in seen_image_paths
            ):

                raise RuntimeError(
                    "Duplicate image_path in manifest:\n"
                    f"  {image_path_text}"
                )

            seen_image_paths.add(
                image_path_text
            )

            image_sha256 = (
                row[
                    "image_sha256"
                ]
                .strip()
                .lower()
            )

            if (
                len(
                    image_sha256
                )
                != 64
                or any(
                    character
                    not in "0123456789abcdef"
                    for character
                    in image_sha256
                )
            ):

                raise RuntimeError(
                    "Invalid image SHA-256 in manifest:\n"
                    f"  row={row_index}\n"
                    f"  value={image_sha256!r}"
                )

            if (
                image_sha256
                in seen_image_hashes
            ):

                raise RuntimeError(
                    "Duplicate image SHA-256 in development manifest:\n"
                    f"  row={row_index}\n"
                    f"  sha256={image_sha256}"
                )

            seen_image_hashes.add(
                image_sha256
            )

            prepared_rows.append(
                {
                    "image_path":
                        image_path,

                    "image_path_relative":
                        image_path_text,

                    "image_sha256":
                        image_sha256,

                    "file_stem":
                        row[
                            "file_stem"
                        ],

                    "traffic_type":
                        traffic_type,

                    "label":
                        int(
                            class_index_by_name[
                                traffic_type
                            ]
                        ),

                    "variant":
                        (
                            row.get(
                                "variant"
                            )
                            or ""
                        ),

                    "hardware_source":
                        row[
                            "hardware_source"
                        ],

                    "face_db":
                        row[
                            "face_db"
                        ],

                    "face_id":
                        row[
                            "face_id"
                        ],

                    "gender":
                        row[
                            "gender"
                        ],

                    "project_role":
                        project_role,
                }
            )

        if (
            observed_class_counts
            != expected_class_counts
        ):

            raise RuntimeError(
                "Frozen manifest class-count mismatch:\n"
                f"  expected={expected_class_counts}\n"
                f"  actual={observed_class_counts}"
            )

        self.manifest_path = (
            manifest_path
        )

        self.manifest_sha256 = (
            actual_manifest_sha256
        )

        self.dataset_root = (
            dataset_root
        )

        self.expected_role = (
            expected_role
        )

        self.preprocessor = (
            preprocessor
        )

        self.rows = tuple(
            prepared_rows
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

        image_path: Path = row[
            "image_path"
        ]

        # --------------------------------------------------------------
        # Force actual image decode on every sample access.
        #
        # Image hash verification is intentionally NOT repeated here
        # every epoch. Frozen-image hashes will be checked exhaustively
        # by the preprocessing audit before training.
        # --------------------------------------------------------------

        with Image.open(
            image_path
        ) as image:

            tensor, geometry = (
                self.preprocessor(
                    image
                )
            )

        return {
            # ----------------------------------------------------------
            # Model inputs
            # ----------------------------------------------------------

            "image":
                tensor,

            "label":
                int(
                    row[
                        "label"
                    ]
                ),

            # ----------------------------------------------------------
            # Stable scientific identity / traceability
            # ----------------------------------------------------------

            "image_path":
                row[
                    "image_path_relative"
                ],

            "image_sha256":
                row[
                    "image_sha256"
                ],

            "file_stem":
                row[
                    "file_stem"
                ],

            "traffic_type":
                row[
                    "traffic_type"
                ],

            "variant":
                row[
                    "variant"
                ],

            "hardware_source":
                row[
                    "hardware_source"
                ],

            "face_db":
                row[
                    "face_db"
                ],

            "face_id":
                row[
                    "face_id"
                ],

            "gender":
                row[
                    "gender"
                ],

            "project_role":
                row[
                    "project_role"
                ],

            # ----------------------------------------------------------
            # Geometry trace needed later for localization mapping
            # ----------------------------------------------------------

            "geometry":
                geometry.as_dict(),
        }


# ======================================================================
# Frozen-config constructors
# ======================================================================

def build_preprocessor_from_config(
    *,
    experiment_cfg: Mapping[
        str,
        Any,
    ],
    resolution_name: str,
) -> DeterministicDocumentPreprocessor:
    """
    Construct preprocessing directly from the frozen scientific YAML.

    This function intentionally rejects unsupported policy changes
    instead of silently approximating them.
    """

    preprocessing = require_mapping(
        require_key(
            experiment_cfg,
            "preprocessing",
            "experiment_config",
        ),
        "preprocessing",
    )

    decode = require_mapping(
        require_key(
            preprocessing,
            "decode",
            "preprocessing",
        ),
        "preprocessing.decode",
    )

    if (
        require_key(
            decode,
            "decoder",
            "preprocessing.decode",
        )
        != "Pillow"
    ):

        raise ValueError(
            "Only the frozen Pillow decoder is supported."
        )

    if (
        require_key(
            decode,
            "force_rgb",
            "preprocessing.decode",
        )
        is not True
    ):

        raise ValueError(
            "Frozen preprocessing requires force_rgb=true."
        )

    exif_policy = require_mapping(
        require_key(
            decode,
            "exif_orientation_policy",
            "preprocessing.decode",
        ),
        "preprocessing.decode.exif_orientation_policy",
    )

    if (
        require_key(
            exif_policy,
            "action",
            "preprocessing.decode.exif_orientation_policy",
        )
        != "none"
    ):

        raise ValueError(
            "Frozen preprocessing requires no EXIF "
            "orientation correction."
        )

    canvases = require_mapping(
        require_key(
            preprocessing,
            "candidate_canvases",
            "preprocessing",
        ),
        "preprocessing.candidate_canvases",
    )

    if (
        resolution_name
        not in canvases
    ):

        raise KeyError(
            "Unknown frozen candidate resolution: "
            f"{resolution_name!r}. "
            f"Available: {sorted(canvases)}"
        )

    canvas_cfg = require_mapping(
        canvases[
            resolution_name
        ],
        (
            "preprocessing."
            "candidate_canvases."
            f"{resolution_name}"
        ),
    )

    canvas = CanvasSpec(
        name=resolution_name,

        content_height=int(
            require_key(
                canvas_cfg,
                "content_height",
                (
                    "preprocessing."
                    "candidate_canvases."
                    f"{resolution_name}"
                ),
            )
        ),

        canvas_height=int(
            require_key(
                canvas_cfg,
                "canvas_height",
                (
                    "preprocessing."
                    "candidate_canvases."
                    f"{resolution_name}"
                ),
            )
        ),

        canvas_width=int(
            require_key(
                canvas_cfg,
                "canvas_width",
                (
                    "preprocessing."
                    "candidate_canvases."
                    f"{resolution_name}"
                ),
            )
        ),
    )

    resize = require_mapping(
        require_key(
            preprocessing,
            "resize",
            "preprocessing",
        ),
        "preprocessing.resize",
    )

    frozen_resize_values = {
        "preserve_aspect_ratio":
            True,

        "crop":
            "none",

        "size_argument":
            "content_height_as_integer_short_side",

        "interpolation":
            "bilinear",

        "antialias":
            True,

        "width_overflow_policy":
            "fail",
    }

    for (
        key,
        expected,
    ) in frozen_resize_values.items():

        actual = require_key(
            resize,
            key,
            "preprocessing.resize",
        )

        if actual != expected:

            raise ValueError(
                "Unsupported preprocessing policy drift:\n"
                f"  preprocessing.resize.{key}\n"
                f"  expected={expected!r}\n"
                f"  actual={actual!r}"
            )

    implementation = require_mapping(
        require_key(
            resize,
            "implementation",
            "preprocessing.resize",
        ),
        "preprocessing.resize.implementation",
    )

    if (
        require_key(
            implementation,
            "library",
            "preprocessing.resize.implementation",
        )
        != "torchvision"
    ):

        raise ValueError(
            "Frozen resize implementation must use torchvision."
        )

    if (
        require_key(
            implementation,
            "operation",
            "preprocessing.resize.implementation",
        )
        != "transforms.functional.resize"
    ):

        raise ValueError(
            "Frozen resize operation must use "
            "transforms.functional.resize."
        )

    padding = require_mapping(
        require_key(
            preprocessing,
            "padding",
            "preprocessing",
        ),
        "preprocessing.padding",
    )

    if (
        require_key(
            padding,
            "required_dimension",
            "preprocessing.padding",
        )
        != "horizontal_only"
    ):

        raise ValueError(
            "Frozen protocol permits horizontal padding only."
        )

    if (
        require_key(
            padding,
            "alignment",
            "preprocessing.padding",
        )
        != "center"
    ):

        raise ValueError(
            "Frozen padding alignment must be center."
        )

    if int(
        require_key(
            padding,
            "top_pixels",
            "preprocessing.padding",
        )
    ) != 0:

        raise ValueError(
            "Frozen protocol requires pad_top=0."
        )

    if int(
        require_key(
            padding,
            "bottom_pixels",
            "preprocessing.padding",
        )
    ) != 0:

        raise ValueError(
            "Frozen protocol requires pad_bottom=0."
        )

    normalization = require_mapping(
        require_key(
            preprocessing,
            "normalization",
            "preprocessing",
        ),
        "preprocessing.normalization",
    )

    mean = tuple(
        float(
            value
        )
        for value
        in require_key(
            normalization,
            "mean",
            "preprocessing.normalization",
        )
    )

    std = tuple(
        float(
            value
        )
        for value
        in require_key(
            normalization,
            "std",
            "preprocessing.normalization",
        )
    )

    fill = tuple(
        float(
            value
        )
        for value
        in require_key(
            padding,
            "fill_rgb_0_to_1",
            "preprocessing.padding",
        )
    )

    if fill != mean:

        raise ValueError(
            "Frozen padding fill must exactly equal "
            "the normalization mean."
        )

    stochastic = require_mapping(
        require_key(
            preprocessing,
            "stochastic_augmentation",
            "preprocessing",
        ),
        "preprocessing.stochastic_augmentation",
    )

    for (
        name,
        enabled,
    ) in stochastic.items():

        if enabled is not False:

            raise ValueError(
                "Primary preprocessing does not permit "
                "stochastic augmentation:\n"
                f"  {name}={enabled!r}"
            )

    return DeterministicDocumentPreprocessor(
        canvas=canvas,
        mean=mean,  # type: ignore[arg-type]
        std=std,    # type: ignore[arg-type]
    )


def build_fantasyid_dataset(
    *,
    experiment_cfg: Mapping[
        str,
        Any,
    ],
    machine_cfg: Mapping[
        str,
        Any,
    ],
    repo_root: Path,
    split_name: str,
    resolution_name: str,
) -> FantasyIDManifestDataset:
    """
    Build project_train or dev_val directly from frozen configs.

    The held-out test is intentionally impossible to request here.
    """

    if (
        split_name
        not in ALLOWED_DEVELOPMENT_SPLITS
    ):

        raise ValueError(
            "Only project_train and dev_val are available "
            "during development. Requested: "
            f"{split_name!r}"
        )

    data_cfg = require_mapping(
        require_key(
            experiment_cfg,
            "data",
            "experiment_config",
        ),
        "data",
    )

    frozen_split = require_mapping(
        require_key(
            data_cfg,
            "frozen_split",
            "data",
        ),
        "data.frozen_split",
    )

    split_cfg = require_mapping(
        require_key(
            frozen_split,
            split_name,
            "data.frozen_split",
        ),
        f"data.frozen_split.{split_name}",
    )

    manifest_path = resolve_repo_path(
        repo_root,
        require_key(
            split_cfg,
            "path",
            f"data.frozen_split.{split_name}",
        ),
    )

    expected_manifest_sha256 = str(
        require_key(
            split_cfg,
            "sha256",
            f"data.frozen_split.{split_name}",
        )
    )

    expected_rows = int(
        require_key(
            split_cfg,
            "images",
            f"data.frozen_split.{split_name}",
        )
    )

    machine_paths = require_mapping(
        require_key(
            machine_cfg,
            "paths",
            "machine_config",
        ),
        "machine_config.paths",
    )

    dataset_root = Path(
        require_key(
            machine_paths,
            "dataset_root",
            "machine_config.paths",
        )
    ).expanduser().resolve()

    class_contract = require_mapping(
        require_key(
            experiment_cfg,
            "class_contract",
            "experiment_config",
        ),
        "class_contract",
    )

    class_index_by_name = require_mapping(
        require_key(
            class_contract,
            "index_by_name",
            "class_contract",
        ),
        "class_contract.index_by_name",
    )

    counts_key = (
        "project_train_counts"
        if split_name == "project_train"
        else "dev_val_counts"
    )

    counts_cfg = require_mapping(
        require_key(
            class_contract,
            counts_key,
            "class_contract",
        ),
        f"class_contract.{counts_key}",
    )

    expected_class_counts = {
        "bonafide":
            int(
                require_key(
                    counts_cfg,
                    "bonafide",
                    f"class_contract.{counts_key}",
                )
            ),

        "attack":
            int(
                require_key(
                    counts_cfg,
                    "attack",
                    f"class_contract.{counts_key}",
                )
            ),
    }

    if (
        sum(
            expected_class_counts.values()
        )
        != expected_rows
    ):

        raise RuntimeError(
            "Configured class counts do not reconcile "
            "to configured manifest image count:\n"
            f"  split={split_name}\n"
            f"  class_total="
            f"{sum(expected_class_counts.values())}\n"
            f"  expected_rows={expected_rows}"
        )

    preprocessor = (
        build_preprocessor_from_config(
            experiment_cfg=experiment_cfg,
            resolution_name=resolution_name,
        )
    )

    return FantasyIDManifestDataset(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows=expected_rows,
        expected_role=split_name,
        dataset_root=dataset_root,
        class_index_by_name={
            str(
                name
            ):
                int(
                    index
                )
            for (
                name,
                index,
            ) in class_index_by_name.items()
        },
        expected_class_counts=expected_class_counts,
        preprocessor=preprocessor,
    )