"""
Development-only counterfactual region restoration.

Purpose
-------
Create controlled diagnostic variants of a FantasyID attack image by
restoring selected manipulated semantic regions from the aligned
bona-fide capture.

For a normal FantasyID development attack containing manipulated face
and text:

    restore_mode="face"
        restore altered face region(s)
        -> manipulated text remains

    restore_mode="text"
        restore altered non-face region(s)
        -> manipulated face remains

    restore_mode="both"
        restore all known altered regions

These outputs are COUNTERFACTUAL DIAGNOSTICS.

They must NOT be described as genuine naturally generated:
- face-only attacks;
- text-only attacks.

The aligned attack and bona-fide captures are also NOT assumed to be
pixel-identical or perfectly registered.

Matching therefore uses:
    semantic field_name
        +
    normalized spatial geometry

before copying pixels.

No model loading, inference, training, thresholding, Grad-CAM or
held-out-test access exists in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from PIL import Image


# ======================================================================
# Frozen annotation semantics
# ======================================================================

FACE_FIELD = "face"

ALTERED_PROVENANCE = "altered"

# Narrow empirically established source-field alias.
#
# Evidence:
# logs/probe_counterfactual_semantic_parent_map_2026-09-11_153502_981144Z.yaml
# SHA-256:
# f48acba66bf17d96e49ae69f120cf431b6aa0465862b4ffe15eb1bd637c5a757
#
# Exact same-field correspondence remains primary. This alias is used
# only when the aligned bona-fide annotation lacks the attack field.
SEMANTIC_SOURCE_FIELD_ALIASES: dict[str, str] = {
    "name_other": "name",
}


RestorationMode = Literal[
    "face",
    "text",
    "both",
]


# ======================================================================
# Errors
# ======================================================================

class CounterfactualRestorationError(
    RuntimeError
):
    pass


# ======================================================================
# Region records
# ======================================================================

@dataclass(
    frozen=True
)
class RegionBox:

    x: int
    y: int

    width: int
    height: int

    @property
    def right(
        self,
    ) -> int:

        return (
            self.x
            + self.width
        )

    @property
    def bottom(
        self,
    ) -> int:

        return (
            self.y
            + self.height
        )

    @property
    def center_x(
        self,
    ) -> float:

        return (
            self.x
            + self.width / 2.0
        )

    @property
    def center_y(
        self,
    ) -> float:

        return (
            self.y
            + self.height / 2.0
        )


@dataclass(
    frozen=True
)
class AnnotationRegion:

    region_index: int

    field_name: str
    provenance: str

    box: RegionBox


@dataclass(
    frozen=True
)
class RegionMatch:

    # Destination semantic field in the attack annotation.
    field_name: str

    # Semantic field actually used from the bona-fide annotation.
    # Normally identical to field_name; differs only for an explicitly
    # allowed semantic-parent alias.
    source_field_name: str

    attack_region_index: int
    bonafide_region_index: int

    attack_box: RegionBox
    bonafide_box: RegionBox

    normalized_iou: float
    normalized_center_distance: float


@dataclass(
    frozen=True
)
class AppliedRestoration:

    field_name: str
    source_field_name: str

    attack_region_index: int
    bonafide_region_index: int

    normalized_iou: float

    source_crop_box: tuple[
        int,
        int,
        int,
        int,
    ]

    destination_box: tuple[
        int,
        int,
        int,
        int,
    ]

    common_visible_fraction: float

    source_patch_size: tuple[
        int,
        int,
    ]

    destination_patch_size: tuple[
        int,
        int,
    ]

    resize_required: bool


@dataclass(
    frozen=True
)
class RestorationResult:

    image: Image.Image

    mode: RestorationMode

    matches: tuple[
        RegionMatch,
        ...
    ]

    applied: tuple[
        AppliedRestoration,
        ...
    ]


# ======================================================================
# Annotation normalization
# ======================================================================

def normalize_text(
    value: Any,
) -> str:

    if value is None:

        return ""

    text = str(
        value
    ).strip()

    if text.casefold() in {
        "",
        "none",
        "<none>",
        "nan",
        "<nan>",
        "null",
        "<null>",
    }:

        return ""

    return text.casefold()


def require_integer_like(
    *,
    value: Any,
    label: str,
) -> int:

    try:

        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise CounterfactualRestorationError(
            f"{label} is not numeric: {value!r}"
        ) from exc

    if not math.isfinite(
        numeric
    ):

        raise CounterfactualRestorationError(
            f"{label} is non-finite: {numeric!r}"
        )

    rounded = round(
        numeric
    )

    if not math.isclose(
        numeric,
        rounded,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):

        raise CounterfactualRestorationError(
            f"{label} is not integer-like: {numeric!r}"
        )

    return int(
        rounded
    )


def annotation_region_from_row(
    row: Mapping[
        str,
        Any,
    ],
) -> AnnotationRegion:

    required = {
        "region_index",
        "field_name",
        "region_provenance_raw",
        "x",
        "y",
        "width",
        "height",
    }

    missing = (
        required
        - set(
            row
        )
    )

    if missing:

        raise CounterfactualRestorationError(
            "Region annotation is missing required fields:\n"
            f"  {sorted(missing)}"
        )

    region_index = require_integer_like(
        value=row[
            "region_index"
        ],
        label="region_index",
    )

    x = require_integer_like(
        value=row[
            "x"
        ],
        label=(
            f"region[{region_index}].x"
        ),
    )

    y = require_integer_like(
        value=row[
            "y"
        ],
        label=(
            f"region[{region_index}].y"
        ),
    )

    width = require_integer_like(
        value=row[
            "width"
        ],
        label=(
            f"region[{region_index}].width"
        ),
    )

    height = require_integer_like(
        value=row[
            "height"
        ],
        label=(
            f"region[{region_index}].height"
        ),
    )

    if width <= 0:

        raise CounterfactualRestorationError(
            "Region width must be positive:\n"
            f"  region_index={region_index}\n"
            f"  width={width}"
        )

    if height <= 0:

        raise CounterfactualRestorationError(
            "Region height must be positive:\n"
            f"  region_index={region_index}\n"
            f"  height={height}"
        )

    return AnnotationRegion(
        region_index=(
            region_index
        ),

        field_name=(
            normalize_text(
                row[
                    "field_name"
                ]
            )
        ),

        provenance=(
            normalize_text(
                row[
                    "region_provenance_raw"
                ]
            )
        ),

        box=RegionBox(
            x=x,
            y=y,
            width=width,
            height=height,
        ),
    )


# ======================================================================
# Normalized geometry
# ======================================================================

def normalized_box(
    *,
    box: RegionBox,
    image_size: tuple[
        int,
        int,
    ],
) -> tuple[
    float,
    float,
    float,
    float,
]:

    image_width, image_height = (
        image_size
    )

    if (
        image_width <= 0
        or image_height <= 0
    ):

        raise ValueError(
            "Image dimensions must be positive."
        )

    return (
        box.x / image_width,
        box.y / image_height,
        box.right / image_width,
        box.bottom / image_height,
    )


def rectangle_iou(
    first: tuple[
        float,
        float,
        float,
        float,
    ],
    second: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    ax0, ay0, ax1, ay1 = (
        first
    )

    bx0, by0, bx1, by1 = (
        second
    )

    intersection_width = max(
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

    intersection_height = max(
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

    intersection_area = (
        intersection_width
        * intersection_height
    )

    first_area = max(
        0.0,
        (
            ax1 - ax0
        )
        * (
            ay1 - ay0
        ),
    )

    second_area = max(
        0.0,
        (
            bx1 - bx0
        )
        * (
            by1 - by0
        ),
    )

    union_area = (
        first_area
        + second_area
        - intersection_area
    )

    if union_area <= 0.0:

        return 0.0

    return (
        intersection_area
        / union_area
    )


def normalized_center_distance(
    *,
    attack_box: RegionBox,
    attack_size: tuple[
        int,
        int,
    ],
    bonafide_box: RegionBox,
    bonafide_size: tuple[
        int,
        int,
    ],
) -> float:

    attack_width, attack_height = (
        attack_size
    )

    bonafide_width, bonafide_height = (
        bonafide_size
    )

    attack_x = (
        attack_box.center_x
        / attack_width
    )

    attack_y = (
        attack_box.center_y
        / attack_height
    )

    bonafide_x = (
        bonafide_box.center_x
        / bonafide_width
    )

    bonafide_y = (
        bonafide_box.center_y
        / bonafide_height
    )

    return math.hypot(
        attack_x - bonafide_x,
        attack_y - bonafide_y,
    )


# ======================================================================
# Region selection
# ======================================================================

def select_altered_attack_regions(
    *,
    attack_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    mode: RestorationMode,
) -> tuple[
    AnnotationRegion,
    ...
]:

    regions = tuple(
        annotation_region_from_row(
            row
        )

        for row
        in attack_rows
    )

    altered = tuple(
        region

        for region
        in regions

        if (
            region.provenance
            == ALTERED_PROVENANCE
        )
    )

    if not altered:

        raise CounterfactualRestorationError(
            "Attack image contains no altered regions."
        )

    blank_fields = tuple(
        region

        for region
        in altered

        if not region.field_name
    )

    if blank_fields:

        raise CounterfactualRestorationError(
            "Altered region has blank field_name:\n"
            f"  region_indices="
            f"{[r.region_index for r in blank_fields]}"
        )

    if mode == "face":

        selected = tuple(
            region

            for region
            in altered

            if (
                region.field_name
                == FACE_FIELD
            )
        )

    elif mode == "text":

        selected = tuple(
            region

            for region
            in altered

            if (
                region.field_name
                != FACE_FIELD
            )
        )

    elif mode == "both":

        selected = (
            altered
        )

    else:

        raise ValueError(
            "Unsupported restoration mode:\n"
            f"  {mode!r}"
        )

    if not selected:

        raise CounterfactualRestorationError(
            "Requested restoration mode selected no regions:\n"
            f"  mode={mode!r}"
        )

    return tuple(
        sorted(
            selected,
            key=lambda region: (
                region.field_name,
                region.region_index,
            ),
        )
    )


# ======================================================================
# Semantic + spatial matching
# ======================================================================

def match_restoration_regions(
    *,
    attack_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    bonafide_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    attack_size: tuple[
        int,
        int,
    ],
    bonafide_size: tuple[
        int,
        int,
    ],
    mode: RestorationMode,
    minimum_normalized_iou: float = 0.25,
) -> tuple[
    RegionMatch,
    ...
]:
    """
    Match selected altered attack regions to corresponding regions in
    the aligned bona-fide image.

    Frozen diagnostic matching rule:

        exact normalized field_name
            ->
        highest normalized-box IoU
            ->
        smallest normalized centre distance
            ->
        lowest bona-fide region_index

    A bona-fide source region cannot be reused for another destination
    region of the same semantic field.
    """

    if not (
        0.0
        <= minimum_normalized_iou
        <= 1.0
    ):

        raise ValueError(
            "minimum_normalized_iou must be in [0,1]."
        )

    destinations = (
        select_altered_attack_regions(
            attack_rows=(
                attack_rows
            ),
            mode=(
                mode
            ),
        )
    )

    bonafide_regions = tuple(
        annotation_region_from_row(
            row
        )

        for row
        in bonafide_rows
    )

    # Bona-fide source must never itself be annotated as altered.
    unexpected_altered = tuple(
        region

        for region
        in bonafide_regions

        if (
            region.provenance
            == ALTERED_PROVENANCE
        )
    )

    if unexpected_altered:

        raise CounterfactualRestorationError(
            "Aligned bona-fide image unexpectedly contains "
            "altered regions:\n"
            f"  region_indices="
            f"{[r.region_index for r in unexpected_altered]}"
        )

    by_field: dict[
        str,
        list[
            AnnotationRegion
        ],
    ] = {}

    for region in (
        bonafide_regions
    ):

        if not region.field_name:

            continue

        by_field.setdefault(
            region.field_name,
            [],
        ).append(
            region
        )

    used_by_field: dict[
        str,
        set[int],
    ] = {}

    matches: list[
        RegionMatch
    ] = []

    for destination in (
    destinations
    ):

        # Exact semantic correspondence always has priority.
        if destination.field_name in by_field:

            source_field_name = (
                destination.field_name
            )

        else:

            source_field_name = (
                SEMANTIC_SOURCE_FIELD_ALIASES.get(
                    destination.field_name,
                    destination.field_name,
                )
            )

        candidates = (
            by_field.get(
                source_field_name,
                [],
            )
        )

        if not candidates:

            raise CounterfactualRestorationError(
                "No permitted semantic source region exists in "
                "aligned bona-fide image:\n"
                f"  attack_field={destination.field_name!r}\n"
                f"  attempted_source_field={source_field_name!r}\n"
                f"  attack_region_index="
                f"{destination.region_index}"
            )

        # Preserve the original uniqueness contract: a source region cannot
        # be reused for another destination belonging to the same ATTACK
        # semantic field. Distinct parent/child attack fields remain separate.
        used = (
            used_by_field.setdefault(
                destination.field_name,
                set(),
            )
        )

        destination_normalized = (
            normalized_box(
                box=(
                    destination.box
                ),
                image_size=(
                    attack_size
                ),
            )
        )

        scored: list[
            tuple[
                float,
                float,
                int,
                AnnotationRegion,
            ]
        ] = []

        for candidate in (
            candidates
        ):

            if (
                candidate.region_index
                in used
            ):

                continue

            candidate_normalized = (
                normalized_box(
                    box=(
                        candidate.box
                    ),
                    image_size=(
                        bonafide_size
                    ),
                )
            )

            iou = rectangle_iou(
                destination_normalized,
                candidate_normalized,
            )

            center_distance = (
                normalized_center_distance(
                    attack_box=(
                        destination.box
                    ),
                    attack_size=(
                        attack_size
                    ),
                    bonafide_box=(
                        candidate.box
                    ),
                    bonafide_size=(
                        bonafide_size
                    ),
                )
            )

            # Negative IoU so normal ascending tuple sort gives
            # highest IoU first.
            scored.append(
                (
                    -iou,
                    center_distance,
                    candidate.region_index,
                    candidate,
                )
            )

        if not scored:

            raise CounterfactualRestorationError(
                "Not enough unique bona-fide regions for "
                "repeated semantic field:\n"
                f"  field={destination.field_name!r}"
            )

        scored.sort(
            key=lambda item: (
                item[
                    0
                ],
                item[
                    1
                ],
                item[
                    2
                ],
            )
        )

        (
            negative_iou,
            center_distance,
            _region_index,
            source,
        ) = scored[
            0
        ]

        selected_iou = (
            -negative_iou
        )

        if (
            selected_iou
            < minimum_normalized_iou
        ):

            raise CounterfactualRestorationError(
                "Best semantic-region match is spatially too weak:\n"
                f"  field={destination.field_name!r}\n"
                f"  attack_region_index="
                f"{destination.region_index}\n"
                f"  bonafide_region_index="
                f"{source.region_index}\n"
                f"  normalized_iou="
                f"{selected_iou:.6f}\n"
                f"  minimum="
                f"{minimum_normalized_iou:.6f}"
            )

        used.add(
            source.region_index
        )

        matches.append(
            RegionMatch(
                field_name=(
                    destination.field_name
                ),

                source_field_name=(
                    source.field_name
                ),

                attack_region_index=(
                    destination.region_index
                ),

                bonafide_region_index=(
                    source.region_index
                ),

                attack_box=(
                    destination.box
                ),

                bonafide_box=(
                    source.box
                ),

                normalized_iou=(
                    selected_iou
                ),

                normalized_center_distance=(
                    center_distance
                ),
            )
        )

    return tuple(
        matches
    )


# ======================================================================
# Partially-visible-region handling
# ======================================================================

def visible_unit_interval(
    *,
    origin: int,
    extent: int,
    image_extent: int,
) -> tuple[
    float,
    float,
]:

    if extent <= 0:

        raise ValueError(
            "Region extent must be positive."
        )

    lower = max(
        0.0,
        (
            -origin
            / extent
        ),
    )

    upper = min(
        1.0,
        (
            (
                image_extent
                - origin
            )
            / extent
        ),
    )

    return (
        lower,
        upper,
    )


def common_visible_unit_rectangle(
    *,
    source_box: RegionBox,
    source_size: tuple[
        int,
        int,
    ],
    destination_box: RegionBox,
    destination_size: tuple[
        int,
        int,
    ],
) -> tuple[
    float,
    float,
    float,
    float,
]:

    source_width, source_height = (
        source_size
    )

    destination_width, destination_height = (
        destination_size
    )

    source_u = visible_unit_interval(
        origin=(
            source_box.x
        ),
        extent=(
            source_box.width
        ),
        image_extent=(
            source_width
        ),
    )

    source_v = visible_unit_interval(
        origin=(
            source_box.y
        ),
        extent=(
            source_box.height
        ),
        image_extent=(
            source_height
        ),
    )

    destination_u = visible_unit_interval(
        origin=(
            destination_box.x
        ),
        extent=(
            destination_box.width
        ),
        image_extent=(
            destination_width
        ),
    )

    destination_v = visible_unit_interval(
        origin=(
            destination_box.y
        ),
        extent=(
            destination_box.height
        ),
        image_extent=(
            destination_height
        ),
    )

    u0 = max(
        source_u[
            0
        ],
        destination_u[
            0
        ],
    )

    u1 = min(
        source_u[
            1
        ],
        destination_u[
            1
        ],
    )

    v0 = max(
        source_v[
            0
        ],
        destination_v[
            0
        ],
    )

    v1 = min(
        source_v[
            1
        ],
        destination_v[
            1
        ],
    )

    if (
        u1 <= u0
        or v1 <= v0
    ):

        raise CounterfactualRestorationError(
            "Matched regions have no mutually visible area."
        )

    return (
        u0,
        v0,
        u1,
        v1,
    )


def unit_rectangle_to_pixels(
    *,
    box: RegionBox,
    unit_rectangle: tuple[
        float,
        float,
        float,
        float,
    ],
    image_size: tuple[
        int,
        int,
    ],
) -> tuple[
    int,
    int,
    int,
    int,
]:

    image_width, image_height = (
        image_size
    )

    u0, v0, u1, v1 = (
        unit_rectangle
    )

    left = math.floor(
        box.x
        + u0
        * box.width
    )

    top = math.floor(
        box.y
        + v0
        * box.height
    )

    right = math.ceil(
        box.x
        + u1
        * box.width
    )

    bottom = math.ceil(
        box.y
        + v1
        * box.height
    )

    left = min(
        image_width,
        max(
            0,
            left,
        ),
    )

    right = min(
        image_width,
        max(
            0,
            right,
        ),
    )

    top = min(
        image_height,
        max(
            0,
            top,
        ),
    )

    bottom = min(
        image_height,
        max(
            0,
            bottom,
        ),
    )

    if (
        right <= left
        or bottom <= top
    ):

        raise CounterfactualRestorationError(
            "Visible region collapsed during integer "
            "pixel conversion."
        )

    return (
        left,
        top,
        right,
        bottom,
    )


# ======================================================================
# Counterfactual restoration
# ======================================================================

def restore_counterfactual(
    *,
    attack_image: Image.Image,
    bonafide_image: Image.Image,
    attack_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    bonafide_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    mode: RestorationMode,
    minimum_normalized_iou: float = 0.25,
) -> RestorationResult:

    attack_rgb = (
        attack_image.convert(
            "RGB"
        )
    )

    bonafide_rgb = (
        bonafide_image.convert(
            "RGB"
        )
    )

    matches = (
        match_restoration_regions(
            attack_rows=(
                attack_rows
            ),
            bonafide_rows=(
                bonafide_rows
            ),
            attack_size=(
                attack_rgb.size
            ),
            bonafide_size=(
                bonafide_rgb.size
            ),
            mode=(
                mode
            ),
            minimum_normalized_iou=(
                minimum_normalized_iou
            ),
        )
    )

    restored = (
        attack_rgb.copy()
    )

    applied: list[
        AppliedRestoration
    ] = []

    for match in (
        matches
    ):

        common_rectangle = (
            common_visible_unit_rectangle(
                source_box=(
                    match.bonafide_box
                ),
                source_size=(
                    bonafide_rgb.size
                ),
                destination_box=(
                    match.attack_box
                ),
                destination_size=(
                    attack_rgb.size
                ),
            )
        )

        source_crop_box = (
            unit_rectangle_to_pixels(
                box=(
                    match.bonafide_box
                ),
                unit_rectangle=(
                    common_rectangle
                ),
                image_size=(
                    bonafide_rgb.size
                ),
            )
        )

        destination_box = (
            unit_rectangle_to_pixels(
                box=(
                    match.attack_box
                ),
                unit_rectangle=(
                    common_rectangle
                ),
                image_size=(
                    attack_rgb.size
                ),
            )
        )

        source_patch = (
            bonafide_rgb.crop(
                source_crop_box
            )
        )

        destination_size = (
            (
                destination_box[
                    2
                ]
                - destination_box[
                    0
                ]
            ),
            (
                destination_box[
                    3
                ]
                - destination_box[
                    1
                ]
            ),
        )

        source_patch_size = (
            source_patch.size
        )

        resize_required = (
            source_patch_size
            != destination_size
        )

        if resize_required:

            source_patch = (
                source_patch.resize(
                    destination_size,
                    resample=(
                        Image.Resampling.BILINEAR
                    ),
                )
            )

        restored.paste(
            source_patch,
            destination_box,
        )

        u0, v0, u1, v1 = (
            common_rectangle
        )

        common_visible_fraction = (
            (
                u1 - u0
            )
            * (
                v1 - v0
            )
        )

        applied.append(
            AppliedRestoration(
                field_name=(
                    match.field_name
                ),

                source_field_name=(
                    match.source_field_name
                ),

                attack_region_index=(
                    match.attack_region_index
                ),

                bonafide_region_index=(
                    match.bonafide_region_index
                ),

                normalized_iou=(
                    match.normalized_iou
                ),

                source_crop_box=(
                    source_crop_box
                ),

                destination_box=(
                    destination_box
                ),

                common_visible_fraction=(
                    common_visible_fraction
                ),

                source_patch_size=(
                    source_patch_size
                ),

                destination_patch_size=(
                    destination_size
                ),

                resize_required=(
                    resize_required
                ),
            )
        )

    if (
        restored.size
        != attack_rgb.size
    ):

        raise CounterfactualRestorationError(
            "Counterfactual restoration changed "
            "overall image geometry."
        )

    return RestorationResult(
        image=(
            restored
        ),

        mode=(
            mode
        ),

        matches=(
            matches
        ),

        applied=tuple(
            applied
        ),
    )