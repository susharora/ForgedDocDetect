# purpose:
# Build the project train/dev split from the frozen FantasyID
# discovery workbook.
#
# This program must never silently consume a modified discovery
# workbook. Before reading any dataset rows, it verifies the
# workbook against its frozen SHA-256 sidecar.

from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd
import yaml

import numpy as np
import scipy

import os

from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds,
)


EXPECTED_IMAGE_ROWS = 3284

REQUIRED_IMAGE_COLUMNS = {
    "split",
    "traffic_type",
    "variant",
    "hardware_source",
    "file_stem",
    "image_path",
    "image_sha256",
    "json_path",
    "json_sha256",
    "face_db",
    "face_id",
    "gender",
}

# ------------------------------------------------------------
# Frozen source-train structure
# ------------------------------------------------------------

EXPECTED_SOURCE_TRAIN_ROWS = 1899
EXPECTED_SOURCE_TRAIN_CARDS = 211
EXPECTED_IMAGES_PER_TRAIN_CARD = 9

EXPECTED_TRAIN_COMPOSITION = {
    ("bonafide", ""): 3,
    ("attack", "digital_1"): 3,
    ("attack", "digital_2"): 3,
}

EXPECTED_TRAIN_HARDWARE = {
    "huawei",
    "iphone15pro",
    "scan",
}

EXPECTED_TRAIN_FACE_DATABASES = {
    "AMFD_Faces_Final",
    "facelab_london",
}

EXPECTED_REGION_ROWS = 33924

NULL_MARKER = "<NULL>"

REQUIRED_REGION_AUDIT_COLUMNS = [
    "split",
    "file_stem",
    "image_path",
    "region_index",
    "field_name",
    "language",
]

# ------------------------------------------------------------
# Script-local paths
# ------------------------------------------------------------

SCRIPT_DIR = Path(
    __file__
).resolve().parent

CONFIG_FILE = (
    SCRIPT_DIR
    / "buildconfig.yaml"
)


# One identifier for this entire split-construction run.
RUN_TIMESTAMP = datetime.now().strftime(
    "%Y-%m-%d_%H%M%S"
)


# ------------------------------------------------------------
# FROZEN DISCOVERY INPUT
#
# Replace this with the FINAL frozen workbook filename.
# ------------------------------------------------------------

DISCOVERY_WORKBOOK = Path(
    "../output/fantasyid_inventory_2026-09-05_023126.xlsx"
)

# Interpret the workbook path relative to scripts/.
DISCOVERY_WORKBOOK = (
    SCRIPT_DIR
    / DISCOVERY_WORKBOOK
).resolve()

DISCOVERY_SIDECAR = Path(
    str(DISCOVERY_WORKBOOK)
    + ".sha256"
)

def load_build_config(
    config_path: Path,
) -> dict:

    if not config_path.is_file():

        raise FileNotFoundError(
            "Build configuration file "
            f"not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):

        raise ValueError(
            "buildconfig.yaml did not "
            "produce a dictionary"
        )

    if "logging" not in config:

        raise KeyError(
            "buildconfig.yaml is missing "
            "the 'logging' section"
        )

    return config


def build_log_path(
    config: dict,
) -> Path:

    logging_config = config[
        "logging"
    ]

    log_directory = Path(
        logging_config[
            "directory"
        ]
    )

    # Paths in buildconfig.yaml are interpreted relative
    # to the directory containing this script/config.
    if not log_directory.is_absolute():

        log_directory = (
            SCRIPT_DIR
            / log_directory
        ).resolve()

    base_log = Path(
        logging_config[
            "filename"
        ]
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        log_directory
        / (
            f"{base_log.stem}_"
            f"{RUN_TIMESTAMP}"
            f"{base_log.suffix}"
        )
    )


def write_log(
    log_path: Path,
    log_entries: list[str],
) -> None:

    output = (
        "\n".join(
            log_entries
        )
        + "\n"
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log_file:

        log_file.write(
            output
        )


def calculate_sha256(
    file_path: Path,
) -> str:

    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def verify_frozen_discovery_artifact(
    workbook_path: Path,
    sidecar_path: Path,
    log_path: Path,
) -> str:

    # --------------------------------------------------------
    # Both frozen files must exist.
    # --------------------------------------------------------

    if not workbook_path.is_file():

        raise FileNotFoundError(
            "Frozen discovery workbook "
            f"not found: {workbook_path}"
        )

    if not sidecar_path.is_file():

        raise FileNotFoundError(
            "Frozen discovery SHA sidecar "
            f"not found: {sidecar_path}"
        )

    # --------------------------------------------------------
    # Expected sidecar format:
    #
    # <sha256>  <workbook_filename>
    # --------------------------------------------------------

    sidecar_text = (
        sidecar_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    parts = sidecar_text.split()

    if len(parts) != 2:

        raise ValueError(
            "Unexpected SHA sidecar format: "
            f"{sidecar_text!r}"
        )

    expected_sha256 = parts[0]
    expected_filename = parts[1]

    # --------------------------------------------------------
    # The sidecar must refer to this exact workbook filename.
    # --------------------------------------------------------

    if expected_filename != workbook_path.name:

        raise ValueError(
            "SHA sidecar refers to a different workbook: "
            f"{expected_filename} "
            f"!= {workbook_path.name}"
        )

    # --------------------------------------------------------
    # SHA-256 itself should be exactly 64 hexadecimal chars.
    # --------------------------------------------------------

    if (
        len(expected_sha256) != 64
        or
        any(
            character
            not in "0123456789abcdef"
            for character
            in expected_sha256
        )
    ):

        raise ValueError(
            "Invalid SHA-256 value in sidecar: "
            f"{expected_sha256}"
        )

    # --------------------------------------------------------
    # Calculate the workbook hash independently.
    # --------------------------------------------------------

    actual_sha256 = calculate_sha256(
        workbook_path
    )

    if actual_sha256 != expected_sha256:

        raise RuntimeError(
            "Frozen discovery workbook SHA-256 mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    write_log(
        log_path,
        [
            "Frozen discovery artifact verified:",
            (
                f"  workbook: "
                f"{workbook_path.name}"
            ),
            (
                f"  SHA-256:  "
                f"{actual_sha256}"
            ),
        ],
    )

    return actual_sha256

#Function to load the images.
def load_discovery_images(
    workbook_path: Path,
    log_path: Path,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Check the workbook contains the expected sheet before
    # attempting to load the dataset rows.
    # --------------------------------------------------------

    excel_file = pd.ExcelFile(
        workbook_path
    )

    if "Images" not in excel_file.sheet_names:

        raise ValueError(
            "Frozen discovery workbook does not contain "
            "the required 'Images' sheet"
        )

    # --------------------------------------------------------
    # Read only the Images sheet.
    #
    # Regions and QA_Summary remain untouched for now.
    # --------------------------------------------------------

    images_df = pd.read_excel(
        workbook_path,
        sheet_name="Images",
    )

    # --------------------------------------------------------
    # Frozen source row-count contract.
    # --------------------------------------------------------

    if len(images_df) != EXPECTED_IMAGE_ROWS:

        raise RuntimeError(
            "Unexpected Images row count.\n"
            f"Expected: {EXPECTED_IMAGE_ROWS}\n"
            f"Actual:   {len(images_df)}"
        )

    # --------------------------------------------------------
    # Required-column contract.
    # --------------------------------------------------------

    missing_columns = (
        REQUIRED_IMAGE_COLUMNS
        - set(images_df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Required Images columns are missing: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Excel represents the source bona-fide variant="" cells as
    # blank cells. pandas therefore reads them as NaN.
    #
    # Restore the discovery-program convention:
    #
    #   bona fide -> variant == ""
    #
    # This is a read-time representation fix only.
    # --------------------------------------------------------

    images_df["variant"] = (
        images_df["variant"]
        .fillna("")
    )

    # --------------------------------------------------------
    # The frozen artifact contains exactly train and test.
    # --------------------------------------------------------

    discovered_splits = set(
        images_df[
            "split"
        ]
        .dropna()
        .unique()
    )

    expected_splits = {
        "train",
        "test",
    }

    if discovered_splits != expected_splits:

        raise RuntimeError(
            "Unexpected source split values.\n"
            f"Expected: {sorted(expected_splits)}\n"
            f"Actual:   {sorted(discovered_splits)}"
        )

    # --------------------------------------------------------
    # These fields must identify every source image uniquely
    # enough for downstream split construction.
    # --------------------------------------------------------

    critical_columns = [
        "file_stem",
        "image_path",
        "image_sha256",
        "face_db",
        "face_id",
    ]

    missing_critical_values = (
        images_df[
            critical_columns
        ]
        .isna()
        .sum()
    )

    bad_columns = {
        column: int(count)
        for column, count
        in missing_critical_values.items()
        if count > 0
    }

    if bad_columns:

        raise RuntimeError(
            "Critical source fields contain missing values: "
            f"{bad_columns}"
        )

    write_log(
        log_path,
        [
            "Frozen Images sheet loaded:",
            (
                f"  rows:    "
                f"{len(images_df)}"
            ),
            (
                f"  columns: "
                f"{len(images_df.columns)}"
            ),
            (
                "  splits:  "
                f"{sorted(discovered_splits)}"
            ),
        ],
    )

    return images_df

#Function to validate the 211 cards
def validate_source_train_structure(
    images_df: pd.DataFrame,
    log_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    log_entries = [
        (
            "********************"
            "validate_source_train_structure"
            "********************"
        )
    ]

    # --------------------------------------------------------
    # Isolate only the frozen source train partition.
    # --------------------------------------------------------

    train_df = (
        images_df.loc[
            images_df[
                "split"
            ].eq("train")
        ]
        .copy()
    )

    violations = []

    # --------------------------------------------------------
    # Overall source-train image count.
    # --------------------------------------------------------

    train_row_count = len(
        train_df
    )

    log_entries.append(
        "Source-train image rows: "
        f"{train_row_count}"
    )

    if (
        train_row_count
        != EXPECTED_SOURCE_TRAIN_ROWS
    ):

        violations.append(
            (
                "Unexpected source-train row count: "
                f"{train_row_count} "
                f"!= {EXPECTED_SOURCE_TRAIN_ROWS}"
            )
        )

    # --------------------------------------------------------
    # Required fields for card-level validation.
    # --------------------------------------------------------

    required_columns = [
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "face_db",
        "face_id",
        "gender",
    ]

    missing_values = (
        train_df[
            required_columns
        ]
        .isna()
        .sum()
    )

    columns_with_missing_values = {
        column: int(count)
        for column, count
        in missing_values.items()
        if count > 0
    }

    if columns_with_missing_values:

        violations.append(
            (
                "Source-train card fields contain "
                "missing values: "
                f"{columns_with_missing_values}"
            )
        )

    # --------------------------------------------------------
    # Group all nine source images belonging to each card.
    # --------------------------------------------------------

    train_groups = (
        train_df.groupby(
            "file_stem",
            sort=True,
        )
    )

    train_card_count = (
        train_df[
            "file_stem"
        ]
        .nunique()
    )

    log_entries.append(
        "Source-train unique cards: "
        f"{train_card_count}"
    )

    if (
        train_card_count
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        violations.append(
            (
                "Unexpected number of source-train cards: "
                f"{train_card_count} "
                f"!= {EXPECTED_SOURCE_TRAIN_CARDS}"
            )
        )

    # --------------------------------------------------------
    # Audit each card independently.
    # --------------------------------------------------------

    bad_card_sizes = []
    bad_card_metadata = []
    bad_compositions = []
    bad_hardware_groups = []

    for (
        file_stem,
        card_df,
    ) in train_groups:

        # ----------------------------------------------------
        # Exactly 9 images per source-train card.
        # ----------------------------------------------------

        if (
            len(card_df)
            != EXPECTED_IMAGES_PER_TRAIN_CARD
        ):

            bad_card_sizes.append(
                (
                    file_stem,
                    len(card_df),
                )
            )

        # ----------------------------------------------------
        # A card must have one stable identity record.
        # ----------------------------------------------------

        card_metadata = (
            card_df[
                [
                    "face_db",
                    "face_id",
                    "gender",
                ]
            ]
            .drop_duplicates()
        )

        if len(card_metadata) != 1:

            bad_card_metadata.append(
                (
                    file_stem,
                    len(card_metadata),
                )
            )

        # ----------------------------------------------------
        # Every card must contribute:
        #
        #   3 bonafide
        #   3 digital_1
        #   3 digital_2
        # ----------------------------------------------------

        composition = (
            card_df.groupby(
                [
                    "traffic_type",
                    "variant",
                ]
            )
            .size()
            .to_dict()
        )

        if (
            composition
            != EXPECTED_TRAIN_COMPOSITION
        ):

            bad_compositions.append(
                (
                    file_stem,
                    composition,
                )
            )

        # ----------------------------------------------------
        # Each version must contain exactly one capture from:
        #
        #   huawei
        #   iphone15pro
        #   scan
        # ----------------------------------------------------

        for (
            traffic_type,
            variant,
        ) in EXPECTED_TRAIN_COMPOSITION:

            group_df = card_df[
                card_df[
                    "traffic_type"
                ].eq(
                    traffic_type
                )
                &
                card_df[
                    "variant"
                ].eq(
                    variant
                )
            ]

            hardware = set(
                group_df[
                    "hardware_source"
                ]
            )

            if (
                len(group_df) != 3
                or
                hardware
                != EXPECTED_TRAIN_HARDWARE
            ):

                bad_hardware_groups.append(
                    (
                        file_stem,
                        traffic_type,
                        variant,
                        len(group_df),
                        sorted(
                            hardware
                        ),
                    )
                )

    # --------------------------------------------------------
    # Log card-level audit results.
    # --------------------------------------------------------

    log_entries.append(
        "Cards with unexpected image count: "
        f"{len(bad_card_sizes)}"
    )

    log_entries.append(
        "Cards with inconsistent identity metadata: "
        f"{len(bad_card_metadata)}"
    )

    log_entries.append(
        "Cards with unexpected variant composition: "
        f"{len(bad_compositions)}"
    )

    log_entries.append(
        "Card/version groups with unexpected "
        "hardware coverage: "
        f"{len(bad_hardware_groups)}"
    )

    # --------------------------------------------------------
    # Show a few examples if anything is wrong.
    # --------------------------------------------------------

    for (
        title,
        problems,
    ) in (
        (
            "Bad card sizes",
            bad_card_sizes,
        ),
        (
            "Bad card metadata",
            bad_card_metadata,
        ),
        (
            "Bad compositions",
            bad_compositions,
        ),
        (
            "Bad hardware groups",
            bad_hardware_groups,
        ),
    ):

        if problems:

            log_entries.append(
                f"{title}:"
            )

            for problem in problems[:10]:

                log_entries.append(
                    f"  {problem}"
                )

            if len(problems) > 10:

                log_entries.append(
                    f"  ... "
                    f"{len(problems) - 10} "
                    "more not shown"
                )

    # --------------------------------------------------------
    # Convert all structural findings into one failure decision.
    # --------------------------------------------------------

    if bad_card_sizes:

        violations.append(
            (
                f"{len(bad_card_sizes)} "
                "card(s) do not contain exactly "
                f"{EXPECTED_IMAGES_PER_TRAIN_CARD} images"
            )
        )

    if bad_card_metadata:

        violations.append(
            (
                f"{len(bad_card_metadata)} "
                "card(s) contain inconsistent "
                "identity metadata"
            )
        )

    if bad_compositions:

        violations.append(
            (
                f"{len(bad_compositions)} "
                "card(s) have unexpected "
                "variant composition"
            )
        )

    if bad_hardware_groups:

        violations.append(
            (
                f"{len(bad_hardware_groups)} "
                "card/version group(s) have "
                "unexpected hardware coverage"
            )
        )

    # --------------------------------------------------------
    # Build one deterministic row per training card.
    #
    # This becomes the card-level table used by the later
    # stratification / 51-card selection logic.
    # --------------------------------------------------------

    train_cards_df = (
        train_df[
            [
                "file_stem",
                "face_db",
                "face_id",
                "gender",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "file_stem"
        )
        .reset_index(
            drop=True
        )
    )

    train_cards_df[
        "source_image_count"
    ] = (
        train_cards_df[
            "file_stem"
        ]
        .map(
            train_df.groupby(
                "file_stem"
            )
            .size()
        )
    )

    log_entries.append(
        "Card-level source-train table rows: "
        f"{len(train_cards_df)}"
    )

    # --------------------------------------------------------
    # Final reconciliation.
    # --------------------------------------------------------

    expected_rows_from_cards = (
        EXPECTED_SOURCE_TRAIN_CARDS
        * EXPECTED_IMAGES_PER_TRAIN_CARD
    )

    log_entries.append(
        "Card/image reconciliation: "
        f"{EXPECTED_SOURCE_TRAIN_CARDS} "
        "cards x "
        f"{EXPECTED_IMAGES_PER_TRAIN_CARD} "
        "images = "
        f"{expected_rows_from_cards}"
    )

    if violations:

        log_entries.append(
            "SOURCE-TRAIN STRUCTURE: FAIL"
        )

        for violation in violations:

            log_entries.append(
                f"  {violation}"
            )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Frozen source-train structure "
            "failed validation"
        )

    log_entries.append(
        "SOURCE-TRAIN STRUCTURE: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return (
        train_df,
        train_cards_df,
    )

# ------------------------------------------------------------
#Function for Audit source-train card strata
#
# This is descriptive discovery for split design.
# It does NOT select any validation cards.
#
# At this stage we inspect only metadata already available at the
# card level:
#
#   face_db
#   gender
#   face_db x gender
#
# Template/language structure will be inspected separately before
# the 51-card selection policy is fixed.
# ------------------------------------------------------------
def audit_train_card_strata(
    train_cards_df: pd.DataFrame,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "audit_train_card_strata"
            "********************"
        )
    ]

    violations = []

    # --------------------------------------------------------
    # Card table should still contain one row per file_stem.
    # --------------------------------------------------------

    card_count = len(
        train_cards_df
    )

    unique_card_count = (
        train_cards_df[
            "file_stem"
        ]
        .nunique()
    )

    log_entries.append(
        f"Card-level rows: {card_count}"
    )

    log_entries.append(
        f"Unique file_stems: {unique_card_count}"
    )

    if (
        card_count
        != EXPECTED_SOURCE_TRAIN_CARDS
        or
        unique_card_count
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        violations.append(
            (
                "Card-level table does not contain "
                f"exactly {EXPECTED_SOURCE_TRAIN_CARDS} "
                "unique cards"
            )
        )

    # --------------------------------------------------------
    # Metadata required for basic stratification must exist.
    # --------------------------------------------------------

    required_columns = {
        "file_stem",
        "face_db",
        "face_id",
        "gender",
    }

    missing_columns = (
        required_columns
        - set(
            train_cards_df.columns
        )
    )

    if missing_columns:

        violations.append(
            (
                "Card-strata columns missing: "
                f"{sorted(missing_columns)}"
            )
        )

    else:

        missing_values = (
            train_cards_df[
                [
                    "face_db",
                    "face_id",
                    "gender",
                ]
            ]
            .isna()
            .sum()
        )

        bad_missing = {
            column: int(count)
            for column, count
            in missing_values.items()
            if count > 0
        }

        if bad_missing:

            violations.append(
                (
                    "Card-strata metadata contains "
                    f"missing values: {bad_missing}"
                )
            )

    # --------------------------------------------------------
    # face_db distribution
    # --------------------------------------------------------

    face_db_counts = (
        train_cards_df[
            "face_db"
        ]
        .value_counts()
        .sort_index()
    )

    discovered_face_databases = set(
        face_db_counts.index
    )

    log_entries.append(
        "Source-train cards by face_db:"
    )

    for (
        face_db,
        count,
    ) in face_db_counts.items():

        percentage = (
            count
            / card_count
        )

        log_entries.append(
            f"  {face_db}: "
            f"{int(count)} "
            f"({percentage:.2%})"
        )

    if (
        discovered_face_databases
        != EXPECTED_TRAIN_FACE_DATABASES
    ):

        violations.append(
            (
                "Unexpected training face_db set: "
                f"{sorted(discovered_face_databases)}"
            )
        )

    # --------------------------------------------------------
    # Gender distribution
    #
    # Descriptive only for now.
    # We are not yet deciding whether gender becomes a formal
    # split-stratification variable.
    # --------------------------------------------------------

    gender_counts = (
        train_cards_df[
            "gender"
        ]
        .value_counts()
        .sort_index()
    )

    log_entries.append(
        "Source-train cards by gender:"
    )

    for (
        gender,
        count,
    ) in gender_counts.items():

        percentage = (
            count
            / card_count
        )

        log_entries.append(
            f"  {gender}: "
            f"{int(count)} "
            f"({percentage:.2%})"
        )

    # --------------------------------------------------------
    # Joint face_db x gender structure.
    #
    # This is more informative than looking at either variable
    # independently because the two source databases may have
    # different gender composition.
    # --------------------------------------------------------

    strata_df = (
        train_cards_df
        .groupby(
            [
                "face_db",
                "gender",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="card_count"
        )
        .sort_values(
            [
                "face_db",
                "gender",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    log_entries.append(
        "Source-train cards by face_db x gender:"
    )

    for _, row in (
        strata_df.iterrows()
    ):

        log_entries.append(
            f"  {row['face_db']} / "
            f"{row['gender']}: "
            f"{int(row['card_count'])}"
        )

    # --------------------------------------------------------
    # Reconciliation
    # --------------------------------------------------------

    strata_total = int(
        strata_df[
            "card_count"
        ]
        .sum()
    )

    log_entries.append(
        "Strata reconciliation: "
        f"{strata_total} cards"
    )

    if (
        strata_total
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        violations.append(
            (
                "face_db x gender strata do not "
                "reconcile to 211 cards"
            )
        )

    if violations:

        log_entries.append(
            "SOURCE-TRAIN CARD STRATA: FAIL"
        )

        for violation in violations:

            log_entries.append(
                f"  {violation}"
            )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Source-train card-strata "
            "audit failed"
        )

    log_entries.append(
        "SOURCE-TRAIN CARD STRATA: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return strata_df

# ------------------------------------------------------------
#Function to Load the frozen Regions sheet for TRAIN-ONLY language audit.
#
# We read only the columns needed for D-24.
# No split selection is performed here.
# ------------------------------------------------------------
def load_discovery_regions_for_audit(
    workbook_path: Path,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "load_discovery_regions_for_audit"
            "********************"
        )
    ]

    # --------------------------------------------------------
    # Confirm Regions exists.
    # --------------------------------------------------------

    excel_file = pd.ExcelFile(
        workbook_path
    )

    if "Regions" not in excel_file.sheet_names:

        write_log(
            log_path,
            log_entries
            + [
                "REGIONS LOAD: FAIL",
                "  Required 'Regions' sheet not found",
            ],
        )

        raise RuntimeError(
            "Frozen discovery workbook does not "
            "contain the Regions sheet"
        )

    # --------------------------------------------------------
    # Read only fields needed for the current audit.
    #
    # keep_default_na=False is deliberate:
    #
    #   <NULL> stays the explicit source-missing marker
    #   ""     stays an empty string
    #
    # rather than pandas converting blank text cells to NaN.
    # --------------------------------------------------------

    regions_df = pd.read_excel(
        workbook_path,
        sheet_name="Regions",
        usecols=REQUIRED_REGION_AUDIT_COLUMNS,
        keep_default_na=False,
    )

    log_entries.append(
        f"Frozen Regions rows loaded: "
        f"{len(regions_df)}"
    )

    if (
        len(regions_df)
        != EXPECTED_REGION_ROWS
    ):

        log_entries.append(
            "REGIONS LOAD: FAIL"
        )

        log_entries.append(
            "  Unexpected region row count: "
            f"{len(regions_df)} "
            f"!= {EXPECTED_REGION_ROWS}"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Unexpected frozen Regions row count"
        )

    missing_columns = (
        set(
            REQUIRED_REGION_AUDIT_COLUMNS
        )
        - set(
            regions_df.columns
        )
    )

    if missing_columns:

        log_entries.append(
            "REGIONS LOAD: FAIL"
        )

        log_entries.append(
            "  Missing columns: "
            f"{sorted(missing_columns)}"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Frozen Regions sheet is missing "
            "required audit columns"
        )

    log_entries.append(
        "REGIONS LOAD: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return regions_df


# ------------------------------------------------------------
# TRAIN CARD LANGUAGE STRUCTURE
#
# We derive one language signature per source-training card.
#
# Important:
#   - raw language labels are preserved
#   - <NULL> is not treated as a language
#   - no normalization or language taxonomy is invented here
# ------------------------------------------------------------

def audit_train_language_structure(
    regions_df: pd.DataFrame,
    train_cards_df: pd.DataFrame,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "audit_train_language_structure"
            "********************"
        )
    ]

    violations = []

    # --------------------------------------------------------
    # Source-train regions only.
    # --------------------------------------------------------

    train_regions_df = (
        regions_df.loc[
            regions_df[
                "split"
            ].eq("train")
        ]
        .copy()
    )

    log_entries.append(
        "Source-train region rows: "
        f"{len(train_regions_df)}"
    )

    # --------------------------------------------------------
    # Region/card parent reconciliation.
    # --------------------------------------------------------

    expected_train_stems = set(
        train_cards_df[
            "file_stem"
        ]
    )

    region_train_stems = set(
        train_regions_df[
            "file_stem"
        ]
    )

    unknown_region_stems = (
        region_train_stems
        - expected_train_stems
    )

    cards_without_regions = (
        expected_train_stems
        - region_train_stems
    )

    log_entries.append(
        "Train region stems not in card table: "
        f"{len(unknown_region_stems)}"
    )

    log_entries.append(
        "Train cards with no region rows: "
        f"{len(cards_without_regions)}"
    )

    if unknown_region_stems:

        violations.append(
            (
                "Region table contains "
                f"{len(unknown_region_stems)} "
                "unknown train card stem(s)"
            )
        )

    if cards_without_regions:

        violations.append(
            (
                f"{len(cards_without_regions)} "
                "source-train card(s) have no "
                "region rows"
            )
        )

    # --------------------------------------------------------
    # Raw language values.
    #
    # We preserve labels exactly as they occur in the frozen
    # artifact. This is discovery, not normalization.
    # --------------------------------------------------------

    raw_language_counts = (
        train_regions_df[
            "language"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index(
            key=lambda index:
                index.astype(str)
        )
    )

    log_entries.append(
        "Raw source-train region language values:"
    )

    for (
        language,
        count,
    ) in raw_language_counts.items():

        if language == NULL_MARKER:

            label = "<NULL>"

        elif language == "":

            label = "(empty string)"

        else:

            label = str(
                language
            )

        log_entries.append(
            f"  {label}: "
            f"{int(count)}"
        )

    # --------------------------------------------------------
    # Build one deterministic language set per card.
    #
    # <NULL> means the language field was absent in source and
    # therefore contributes no language label.
    # --------------------------------------------------------

    card_language_rows = []

    for (
        file_stem,
        card_regions,
    ) in train_regions_df.groupby(
        "file_stem",
        sort=True,
    ):

        languages = {
            str(language)
            for language
            in card_regions[
                "language"
            ]
            if (
                language != NULL_MARKER
                and language != ""
            )
        }

        ordered_languages = sorted(
            languages,
            key=str.casefold,
        )

        if ordered_languages:

            language_signature = (
                " | ".join(
                    ordered_languages
                )
            )

        else:

            language_signature = (
                "(no explicit language)"
            )

        card_language_rows.append({
            "file_stem":
                file_stem,

            "language_count":
                len(
                    ordered_languages
                ),

            "language_signature":
                language_signature,
        })

    train_card_language_df = (
        pd.DataFrame(
            card_language_rows
        )
        .sort_values(
            "file_stem"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Every train card must produce exactly one language record.
    # --------------------------------------------------------

    log_entries.append(
        "Card-level language rows: "
        f"{len(train_card_language_df)}"
    )

    if (
        len(train_card_language_df)
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        violations.append(
            (
                "Language audit did not produce "
                f"exactly {EXPECTED_SOURCE_TRAIN_CARDS} "
                "card rows"
            )
        )

    # --------------------------------------------------------
    # How many explicit languages occur per card?
    # --------------------------------------------------------

    language_count_distribution = (
        train_card_language_df[
            "language_count"
        ]
        .value_counts()
        .sort_index()
    )

    log_entries.append(
        "Number of explicit language labels per card:"
    )

    for (
        language_count,
        card_count,
    ) in (
        language_count_distribution.items()
    ):

        log_entries.append(
            f"  {int(language_count)} "
            "language(s): "
            f"{int(card_count)} card(s)"
        )

    # --------------------------------------------------------
    # Card-level language signatures.
    #
    # This is the important view for future stratification.
    # --------------------------------------------------------

    signature_counts = (
        train_card_language_df[
            "language_signature"
        ]
        .value_counts()
        .sort_index()
    )

    log_entries.append(
        "Source-train cards by language signature:"
    )

    for (
        signature,
        count,
    ) in signature_counts.items():

        log_entries.append(
            f"  {signature}: "
            f"{int(count)}"
        )

    # --------------------------------------------------------
    # Join with face_db/gender so we can see whether language is
    # already strongly tied to an existing stratum.
    # --------------------------------------------------------

    language_strata_df = (
        train_cards_df[
            [
                "file_stem",
                "face_db",
                "face_id",
                "gender",
                "source_image_count",
            ]
        ]
        .merge(
            train_card_language_df,
            on="file_stem",
            how="left",
            validate="one_to_one",
        )
    )

    if (
        language_strata_df[
            "language_signature"
        ]
        .isna()
        .any()
    ):

        violations.append(
            "Some train cards failed to receive a language signature"
        )

    cross_counts = (
        language_strata_df
        .groupby(
            [
                "face_db",
                "gender",
                "language_signature",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="card_count"
        )
        .sort_values(
            [
                "face_db",
                "gender",
                "language_signature",
            ]
        )
    )

    log_entries.append(
        "Cards by face_db x gender x language signature:"
    )

    for _, row in (
        cross_counts.iterrows()
    ):

        log_entries.append(
            f"  {row['face_db']} / "
            f"{row['gender']} / "
            f"{row['language_signature']}: "
            f"{int(row['card_count'])}"
        )

    # --------------------------------------------------------
    # Final reconciliation.
    # --------------------------------------------------------

    if violations:

        log_entries.append(
            "SOURCE-TRAIN LANGUAGE AUDIT: FAIL"
        )

        for violation in violations:

            log_entries.append(
                f"  {violation}"
            )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Source-train language audit failed"
        )

    log_entries.append(
        "SOURCE-TRAIN LANGUAGE AUDIT: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return language_strata_df

# ------------------------------------------------------------
# Audit raw file_stem naming families.
#
# We do NOT call these "templates" yet.
#
# For discovery only, define a conservative candidate family as
# the text before the first "-" in file_stem.
#
# Examples:
#
#   arabic-003_03       -> arabic
#   portugal-NF-1060    -> portugal
#
# This is a DERIVED downstream field. The original file_stem is
# preserved unchanged.
# ------------------------------------------------------------

def audit_train_stem_family_structure(
    train_card_language_df: pd.DataFrame,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "audit_train_stem_family_structure"
            "********************"
        )
    ]

    audit_df = (
        train_card_language_df
        .copy()
    )

    # --------------------------------------------------------
    # Derive raw and normalized prefix forms.
    # --------------------------------------------------------

    audit_df[
        "stem_family_raw"
    ] = (
        audit_df[
            "file_stem"
        ]
        .astype(str)
        .str.split(
            "-",
            n=1,
        )
        .str[0]
    )

    audit_df[
        "stem_family"
    ] = (
        audit_df[
            "stem_family_raw"
        ]
        .str.casefold()
    )

    empty_family_rows = audit_df[
        audit_df[
            "stem_family"
        ]
        .eq("")
    ]

    if not empty_family_rows.empty:

        write_log(
            log_path,
            log_entries
            + [
                "STEM-FAMILY AUDIT: FAIL",
                (
                    "  Empty derived family for "
                    f"{len(empty_family_rows)} card(s)"
                ),
            ],
        )

        raise RuntimeError(
            "Unable to derive stem family "
            "for every source-train card"
        )

    # --------------------------------------------------------
    # Basic family distribution.
    # --------------------------------------------------------

    family_counts = (
        audit_df[
            "stem_family"
        ]
        .value_counts()
        .sort_index()
    )

    log_entries.append(
        "Distinct derived stem families: "
        f"{len(family_counts)}"
    )

    log_entries.append(
        "Source-train cards by derived stem family:"
    )

    for (
        family,
        count,
    ) in family_counts.items():

        log_entries.append(
            f"  {family}: "
            f"{int(count)}"
        )

    # --------------------------------------------------------
    # Check whether case-folding merged different raw forms.
    # --------------------------------------------------------

    raw_forms_per_family = (
        audit_df
        .groupby(
            "stem_family"
        )[
            "stem_family_raw"
        ]
        .nunique()
    )

    case_collisions = (
        raw_forms_per_family[
            raw_forms_per_family > 1
        ]
    )

    log_entries.append(
        "Families with >1 raw casing/spelling form: "
        f"{len(case_collisions)}"
    )

    # --------------------------------------------------------
    # Inspect each family against the metadata already audited.
    #
    # This tells us whether the prefix carries information that is
    # independent of face_db, gender and language signature.
    # --------------------------------------------------------

    log_entries.append(
        "Stem-family composition:"
    )

    for (
        family,
        family_df,
    ) in audit_df.groupby(
        "stem_family",
        sort=True,
    ):

        face_db_counts = (
            family_df[
                "face_db"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        gender_counts = (
            family_df[
                "gender"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        language_counts = (
            family_df[
                "language_signature"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        log_entries.append(
            f"  {family}: "
            f"{len(family_df)} card(s)"
        )

        log_entries.append(
            f"    face_db: "
            f"{face_db_counts}"
        )

        log_entries.append(
            f"    gender: "
            f"{gender_counts}"
        )

        log_entries.append(
            f"    language: "
            f"{language_counts}"
        )

    # --------------------------------------------------------
    # Summarize whether one family maps to multiple metadata
    # categories.
    # --------------------------------------------------------

    family_language_counts = (
        audit_df
        .groupby(
            "stem_family"
        )[
            "language_signature"
        ]
        .nunique()
    )

    family_database_counts = (
        audit_df
        .groupby(
            "stem_family"
        )[
            "face_db"
        ]
        .nunique()
    )

    families_with_multiple_languages = int(
        family_language_counts
        .gt(1)
        .sum()
    )

    families_with_multiple_databases = int(
        family_database_counts
        .gt(1)
        .sum()
    )

    log_entries.append(
        "Families spanning >1 language signature: "
        f"{families_with_multiple_languages}"
    )

    log_entries.append(
        "Families spanning >1 face_db: "
        f"{families_with_multiple_databases}"
    )

    # --------------------------------------------------------
    # Reconciliation.
    # --------------------------------------------------------

    log_entries.append(
        "Stem-family reconciliation: "
        f"{len(audit_df)} cards"
    )

    if (
        len(audit_df)
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        log_entries.append(
            "STEM-FAMILY AUDIT: FAIL"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Stem-family audit does not "
            "reconcile to 211 cards"
        )

    log_entries.append(
        "STEM-FAMILY AUDIT: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return audit_df

# ------------------------------------------------------------
#Function Calculate proportional internal-dev quota targets.
#
# No cards are selected here.
#
# The purpose of this stage is only to derive deterministic target
# counts from the observed 211-card source-train population.
#
# Allocation method:
#
#   Largest Remainder / Hamilton apportionment
#
# For each stratum:
#
#   ideal target = source_cards * 51 / 211
#
# We first allocate the integer floor, then distribute the
# remaining cards to the largest fractional remainders.
#
# Ties are resolved deterministically by the stratum labels.
# ------------------------------------------------------------

DEV_CARD_TARGET = 51
EXPECTED_DEV_IMAGE_ROWS = 459


def calculate_proportional_quota(
    cards_df: pd.DataFrame,
    group_columns: list[str],
    target_cards: int,
) -> pd.DataFrame:

    source_total = len(
        cards_df
    )

    if source_total <= 0:

        raise ValueError(
            "Cannot calculate quota from "
            "an empty card table"
        )

    if (
        target_cards <= 0
        or target_cards > source_total
    ):

        raise ValueError(
            "Invalid target-card count: "
            f"{target_cards}"
        )

    # --------------------------------------------------------
    # Count source cards in each requested stratum.
    # --------------------------------------------------------

    quota_df = (
        cards_df
        .groupby(
            group_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="source_cards"
        )
    )

    # --------------------------------------------------------
    # Proportional ideal allocation.
    # --------------------------------------------------------

    quota_df[
        "ideal_dev_cards"
    ] = (
        quota_df[
            "source_cards"
        ]
        * target_cards
        / source_total
    )

    # All values are positive, so integer conversion is
    # equivalent to floor().
    quota_df[
        "floor_dev_cards"
    ] = (
        quota_df[
            "ideal_dev_cards"
        ]
        .astype(int)
    )

    quota_df[
        "remainder"
    ] = (
        quota_df[
            "ideal_dev_cards"
        ]
        - quota_df[
            "floor_dev_cards"
        ]
    )

    quota_df[
        "dev_target"
    ] = quota_df[
        "floor_dev_cards"
    ].copy()

    cards_remaining = (
        target_cards
        - int(
            quota_df[
                "dev_target"
            ]
            .sum()
        )
    )

    # --------------------------------------------------------
    # Deterministic largest-remainder ordering.
    #
    # Primary:
    #   largest fractional remainder first
    #
    # Tie:
    #   lexical ordering of the group columns
    # --------------------------------------------------------

    sort_columns = (
        ["remainder"]
        + group_columns
    )

    ascending = (
        [False]
        + [True] * len(
            group_columns
        )
    )

    allocation_order = (
        quota_df
        .sort_values(
            sort_columns,
            ascending=ascending,
            kind="mergesort",
        )
        .index
        .tolist()
    )

    for row_index in (
        allocation_order[
            :cards_remaining
        ]
    ):

        quota_df.loc[
            row_index,
            "dev_target",
        ] += 1

    # --------------------------------------------------------
    # Final deterministic presentation order.
    # --------------------------------------------------------

    quota_df = (
        quota_df
        .sort_values(
            group_columns
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Internal reconciliation.
    # --------------------------------------------------------

    if (
        int(
            quota_df[
                "source_cards"
            ]
            .sum()
        )
        != source_total
    ):

        raise RuntimeError(
            "Quota source-card counts "
            "do not reconcile"
        )

    if (
        int(
            quota_df[
                "dev_target"
            ]
            .sum()
        )
        != target_cards
    ):

        raise RuntimeError(
            "Quota allocation does not "
            "reconcile to target-card count"
        )

    if (
        quota_df[
            "dev_target"
        ]
        .gt(
            quota_df[
                "source_cards"
            ]
        )
        .any()
    ):

        raise RuntimeError(
            "A dev quota exceeds the available "
            "source cards in its stratum"
        )

    return quota_df


def log_quota_table(
    title: str,
    quota_df: pd.DataFrame,
    group_columns: list[str],
    log_entries: list[str],
) -> None:

    log_entries.append(
        title
    )

    for _, row in (
        quota_df.iterrows()
    ):

        label = " / ".join(
            str(
                row[column]
            )
            for column
            in group_columns
        )

        log_entries.append(
            f"  {label}: "
            f"source={int(row['source_cards'])} "
            f"ideal={row['ideal_dev_cards']:.3f} "
            f"target={int(row['dev_target'])}"
        )


def audit_dev_quota_targets(
    train_card_template_audit_df: pd.DataFrame,
    log_path: Path,
) -> dict[str, pd.DataFrame]:

    log_entries = [
        (
            "********************"
            "audit_dev_quota_targets"
            "********************"
        )
    ]

    cards_df = (
        train_card_template_audit_df
        .copy()
    )

    violations = []

    # --------------------------------------------------------
    # The quota calculation operates on exactly one row/card.
    # --------------------------------------------------------

    if (
        len(cards_df)
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        violations.append(
            (
                "Quota input does not contain "
                f"{EXPECTED_SOURCE_TRAIN_CARDS} cards"
            )
        )

    if (
        cards_df[
            "file_stem"
        ]
        .nunique()
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        violations.append(
            "Quota input does not contain 211 unique file_stems"
        )

    required_columns = {
        "file_stem",
        "stem_family",
        "face_db",
        "gender",
        "language_signature",
    }

    missing_columns = (
        required_columns
        - set(
            cards_df.columns
        )
    )

    if missing_columns:

        violations.append(
            (
                "Quota input columns missing: "
                f"{sorted(missing_columns)}"
            )
        )

    if violations:

        log_entries.append(
            "DEV QUOTA AUDIT: FAIL"
        )

        for violation in violations:

            log_entries.append(
                f"  {violation}"
            )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Cannot calculate development quotas"
        )

    # --------------------------------------------------------
    # Family quota.
    #
    # This is our primary document-structure stratification.
    # --------------------------------------------------------

    family_quota_df = (
        calculate_proportional_quota(
            cards_df,
            [
                "stem_family",
            ],
            DEV_CARD_TARGET,
        )
    )

    # --------------------------------------------------------
    # Source face database.
    # --------------------------------------------------------

    face_db_quota_df = (
        calculate_proportional_quota(
            cards_df,
            [
                "face_db",
            ],
            DEV_CARD_TARGET,
        )
    )

    # --------------------------------------------------------
    # Gender.
    # --------------------------------------------------------

    gender_quota_df = (
        calculate_proportional_quota(
            cards_df,
            [
                "gender",
            ],
            DEV_CARD_TARGET,
        )
    )

    # --------------------------------------------------------
    # Joint source-database x gender quota.
    #
    # This is more informative than preserving the two marginals
    # independently.
    # --------------------------------------------------------

    face_db_gender_quota_df = (
        calculate_proportional_quota(
            cards_df,
            [
                "face_db",
                "gender",
            ],
            DEV_CARD_TARGET,
        )
    )

    # --------------------------------------------------------
    # Language-signature quota.
    # --------------------------------------------------------

    language_quota_df = (
        calculate_proportional_quota(
            cards_df,
            [
                "language_signature",
            ],
            DEV_CARD_TARGET,
        )
    )

    # --------------------------------------------------------
    # Log every calculated target.
    # --------------------------------------------------------

    log_entries.append(
        "Internal development-set target:"
    )

    log_entries.append(
        f"  cards:  {DEV_CARD_TARGET}"
    )

    log_entries.append(
        f"  images: {EXPECTED_DEV_IMAGE_ROWS}"
    )

    log_entries.append(
        ""
    )

    log_quota_table(
        "Derived stem-family quotas:",
        family_quota_df,
        [
            "stem_family",
        ],
        log_entries,
    )

    log_entries.append(
        ""
    )

    log_quota_table(
        "face_db quotas:",
        face_db_quota_df,
        [
            "face_db",
        ],
        log_entries,
    )

    log_entries.append(
        ""
    )

    log_quota_table(
        "Gender quotas:",
        gender_quota_df,
        [
            "gender",
        ],
        log_entries,
    )

    log_entries.append(
        ""
    )

    log_quota_table(
        "face_db x gender quotas:",
        face_db_gender_quota_df,
        [
            "face_db",
            "gender",
        ],
        log_entries,
    )

    log_entries.append(
        ""
    )

    log_quota_table(
        "Language-signature quotas:",
        language_quota_df,
        [
            "language_signature",
        ],
        log_entries,
    )

    log_entries.append(
        "DEV QUOTA AUDIT: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return {
        "family":
            family_quota_df,

        "face_db":
            face_db_quota_df,

        "gender":
            gender_quota_df,

        "face_db_gender":
            face_db_gender_quota_df,

        "language":
            language_quota_df,
    }


# ------------------------------------------------------------
# Build the complete joint support table.
#
# This still does NOT select cards.
#
# Each row describes one actually-observed combination of:
#
#   stem_family
#   face_db
#   gender
#   language_signature
#
# This table is the input to the next exact-feasibility stage.
# ------------------------------------------------------------

def build_joint_split_support_table(
    train_card_template_audit_df: pd.DataFrame,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "build_joint_split_support_table"
            "********************"
        )
    ]

    joint_columns = [
        "stem_family",
        "face_db",
        "gender",
        "language_signature",
    ]

    cards_df = (
        train_card_template_audit_df
        .copy()
    )

    missing_columns = (
        set(
            joint_columns
        )
        - set(
            cards_df.columns
        )
    )

    if missing_columns:

        write_log(
            log_path,
            log_entries
            + [
                "JOINT SUPPORT AUDIT: FAIL",
                (
                    "  Missing columns: "
                    f"{sorted(missing_columns)}"
                ),
            ],
        )

        raise RuntimeError(
            "Cannot build joint split support table"
        )

    # --------------------------------------------------------
    # Count the number of actual source cards in every observed
    # four-dimensional cell.
    # --------------------------------------------------------

    joint_df = (
        cards_df
        .groupby(
            joint_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="source_cards"
        )
        .sort_values(
            joint_columns
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Proportional ideal share of the 51-card development set.
    #
    # This is descriptive only.
    #
    # We are NOT allocating independent quotas to every joint
    # cell because doing so could conflict with the marginal
    # targets calculated above.
    # --------------------------------------------------------

    joint_df[
        "ideal_dev_cards"
    ] = (
        joint_df[
            "source_cards"
        ]
        * DEV_CARD_TARGET
        / EXPECTED_SOURCE_TRAIN_CARDS
    )

    # --------------------------------------------------------
    # Reconciliation.
    # --------------------------------------------------------

    joint_source_total = int(
        joint_df[
            "source_cards"
        ]
        .sum()
    )

    log_entries.append(
        "Observed joint strata: "
        f"{len(joint_df)}"
    )

    log_entries.append(
        "Joint source-card total: "
        f"{joint_source_total}"
    )

    if (
        joint_source_total
        != EXPECTED_SOURCE_TRAIN_CARDS
    ):

        log_entries.append(
            "JOINT SUPPORT AUDIT: FAIL"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Joint support table does not "
            "reconcile to 211 cards"
        )

    # --------------------------------------------------------
    # Useful sparsity diagnostics.
    #
    # Sparse cells matter because they determine whether all the
    # requested marginals can be satisfied simultaneously.
    # --------------------------------------------------------

    singleton_cells = int(
        joint_df[
            "source_cards"
        ]
        .eq(1)
        .sum()
    )

    cells_with_two = int(
        joint_df[
            "source_cards"
        ]
        .eq(2)
        .sum()
    )

    minimum_cell_size = int(
        joint_df[
            "source_cards"
        ]
        .min()
    )

    maximum_cell_size = int(
        joint_df[
            "source_cards"
        ]
        .max()
    )

    log_entries.append(
        "Joint-cell source-card sizes:"
    )

    log_entries.append(
        f"  minimum: {minimum_cell_size}"
    )

    log_entries.append(
        f"  maximum: {maximum_cell_size}"
    )

    log_entries.append(
        f"  singleton cells: {singleton_cells}"
    )

    log_entries.append(
        f"  two-card cells: {cells_with_two}"
    )

    # --------------------------------------------------------
    # Log every actually observed cell.
    # --------------------------------------------------------

    log_entries.append(
        "Observed stem_family x face_db x gender "
        "x language_signature cells:"
    )

    for _, row in (
        joint_df.iterrows()
    ):

        log_entries.append(
            f"  {row['stem_family']} / "
            f"{row['face_db']} / "
            f"{row['gender']} / "
            f"{row['language_signature']}: "
            f"source={int(row['source_cards'])} "
            f"ideal_dev={row['ideal_dev_cards']:.3f}"
        )

    log_entries.append(
        "JOINT SUPPORT AUDIT: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return joint_df

# ------------------------------------------------------------
#Function Exact 51-card marginal-feasibility audit
#
# We now ask:
#
# Does there exist an integer allocation across the 48 observed
# joint cells that simultaneously satisfies:
#
#   1. stem_family quotas
#   2. face_db x gender quotas
#   3. language_signature quotas
#   4. total dev cards = 51
#
# Each decision variable represents:
#
#   number of dev cards taken from one observed joint cell
#
# and is constrained:
#
#   0 <= selected <= number of source cards in that cell
#
# This proves feasibility only.
# It does NOT yet select actual file_stems.
# ------------------------------------------------------------

def audit_exact_dev_feasibility(
    joint_df: pd.DataFrame,
    quota_tables: dict[str, pd.DataFrame],
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "audit_exact_dev_feasibility"
            "********************"
        )
    ]

    required_joint_columns = {
        "stem_family",
        "face_db",
        "gender",
        "language_signature",
        "source_cards",
    }

    missing_columns = (
        required_joint_columns
        - set(
            joint_df.columns
        )
    )

    if missing_columns:

        write_log(
            log_path,
            log_entries
            + [
                "EXACT DEV FEASIBILITY: FAIL",
                (
                    "  Joint support columns missing: "
                    f"{sorted(missing_columns)}"
                ),
            ],
        )

        raise RuntimeError(
            "Cannot run exact dev feasibility audit"
        )

    # --------------------------------------------------------
    # Recover the previously calculated marginal targets.
    # --------------------------------------------------------

    family_targets = {
        row["stem_family"]:
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "family"
        ].iterrows()
    }

    face_db_gender_targets = {
        (
            row["face_db"],
            row["gender"],
        ):
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "face_db_gender"
        ].iterrows()
    }

    language_targets = {
        row["language_signature"]:
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "language"
        ].iterrows()
    }

    # --------------------------------------------------------
    # Every independent target table should reconcile to 51.
    # --------------------------------------------------------

    target_totals = {
        "family":
            sum(
                family_targets.values()
            ),

        "face_db_gender":
            sum(
                face_db_gender_targets.values()
            ),

        "language":
            sum(
                language_targets.values()
            ),
    }

    log_entries.append(
        "Quota totals entering feasibility solver:"
    )

    for (
        target_name,
        target_total,
    ) in target_totals.items():

        log_entries.append(
            f"  {target_name}: "
            f"{target_total}"
        )

    if any(
        target_total != DEV_CARD_TARGET
        for target_total
        in target_totals.values()
    ):

        log_entries.append(
            "EXACT DEV FEASIBILITY: FAIL"
        )

        log_entries.append(
            "  One or more marginal quota tables "
            "do not reconcile to 51 cards"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Marginal quota totals do not reconcile"
        )

    # --------------------------------------------------------
    # One integer decision variable per observed joint cell.
    # --------------------------------------------------------

    number_of_cells = len(
        joint_df
    )

    constraint_rows = []
    constraint_targets = []
    constraint_labels = []

    # --------------------------------------------------------
    # Helper:
    # append an exact equality constraint.
    # --------------------------------------------------------

    def add_constraint(
        mask: pd.Series,
        target: int,
        label: str,
    ) -> None:

        constraint_rows.append(
            mask
            .astype(int)
            .to_numpy()
        )

        constraint_targets.append(
            int(
                target
            )
        )

        constraint_labels.append(
            label
        )

    # --------------------------------------------------------
    # 1. Exact stem-family targets.
    # --------------------------------------------------------

    for (
        stem_family,
        target,
    ) in sorted(
        family_targets.items()
    ):

        add_constraint(
            joint_df[
                "stem_family"
            ].eq(
                stem_family
            ),
            target,
            (
                "stem_family = "
                f"{stem_family}"
            ),
        )

    # --------------------------------------------------------
    # 2. Exact face_db x gender targets.
    # --------------------------------------------------------

    for (
        face_db,
        gender,
    ), target in sorted(
        face_db_gender_targets.items()
    ):

        add_constraint(
            (
                joint_df[
                    "face_db"
                ].eq(
                    face_db
                )
                &
                joint_df[
                    "gender"
                ].eq(
                    gender
                )
            ),
            target,
            (
                "face_db x gender = "
                f"{face_db} / {gender}"
            ),
        )

    # --------------------------------------------------------
    # 3. Exact language-signature targets.
    # --------------------------------------------------------

    for (
        language_signature,
        target,
    ) in sorted(
        language_targets.items()
    ):

        add_constraint(
            joint_df[
                "language_signature"
            ].eq(
                language_signature
            ),
            target,
            (
                "language_signature = "
                f"{language_signature}"
            ),
        )

    # --------------------------------------------------------
    # 4. Explicit total-card constraint.
    #
    # This is mathematically redundant with the family targets,
    # but keeping it explicit makes the solver contract and log
    # easier to audit.
    # --------------------------------------------------------

    constraint_rows.append(
        np.ones(
            number_of_cells,
            dtype=int,
        )
    )

    constraint_targets.append(
        DEV_CARD_TARGET
    )

    constraint_labels.append(
        "total development cards"
    )

    constraint_matrix = np.vstack(
        constraint_rows
    )

    target_vector = np.array(
        constraint_targets,
        dtype=float,
    )

    # --------------------------------------------------------
    # Integer bounds:
    #
    # 0 <= selected joint-cell cards <= source cards available
    # --------------------------------------------------------

    lower_bounds = np.zeros(
        number_of_cells,
        dtype=float,
    )

    upper_bounds = (
        joint_df[
            "source_cards"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Feasibility problem.
    #
    # Objective is all zero because we are NOT yet choosing an
    # optimal allocation. We only need proof that at least one
    # exact integer solution exists.
    # --------------------------------------------------------

    objective = np.zeros(
        number_of_cells,
        dtype=float,
    )

    integrality = np.ones(
        number_of_cells,
        dtype=int,
    )

    equality_constraints = (
        LinearConstraint(
            constraint_matrix,
            target_vector,
            target_vector,
        )
    )

    variable_bounds = Bounds(
        lower_bounds,
        upper_bounds,
    )

    log_entries.append(
        "Exact feasibility problem:"
    )

    log_entries.append(
        f"  joint-cell variables: "
        f"{number_of_cells}"
    )

    log_entries.append(
        f"  equality constraints: "
        f"{len(constraint_targets)}"
    )

    log_entries.append(
        f"  SciPy version: "
        f"{scipy.__version__}"
    )

    # --------------------------------------------------------
    # Solve the mixed-integer feasibility problem.
    # --------------------------------------------------------

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=equality_constraints,
        options={
            "disp": False,
        },
    )

    log_entries.append(
        f"Solver status: "
        f"{result.status}"
    )

    log_entries.append(
        f"Solver message: "
        f"{result.message}"
    )

    if (
        not result.success
        or result.x is None
    ):

        log_entries.append(
            "EXACT DEV FEASIBILITY: FAIL"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "No exact integer allocation satisfies "
            "all requested development-set margins"
        )

    # --------------------------------------------------------
    # MILP solvers return floating-point representations of
    # integer solutions.
    #
    # Round them, but only after verifying they are numerically
    # extremely close to integers.
    # --------------------------------------------------------

    raw_solution = (
        result.x
    )

    integer_solution = (
        np.rint(
            raw_solution
        )
        .astype(int)
    )

    maximum_integer_error = float(
        np.max(
            np.abs(
                raw_solution
                - integer_solution
            )
        )
    )

    log_entries.append(
        "Maximum solver integer-rounding error: "
        f"{maximum_integer_error:.12g}"
    )

    if (
        maximum_integer_error
        > 1e-6
    ):

        log_entries.append(
            "EXACT DEV FEASIBILITY: FAIL"
        )

        log_entries.append(
            "  Solver result is not sufficiently "
            "close to an integer solution"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "MILP result failed integer verification"
        )

    # --------------------------------------------------------
    # Independently verify variable bounds after rounding.
    # --------------------------------------------------------

    source_capacities = (
        joint_df[
            "source_cards"
        ]
        .to_numpy(
            dtype=int
        )
    )

    if (
        (integer_solution < 0).any()
        or
        (
            integer_solution
            > source_capacities
        ).any()
    ):

        log_entries.append(
            "EXACT DEV FEASIBILITY: FAIL"
        )

        log_entries.append(
            "  Rounded solution violates "
            "joint-cell source capacity"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Feasibility witness exceeds source support"
        )

    # --------------------------------------------------------
    # Independently recompute every exact equality.
    # --------------------------------------------------------

    achieved_vector = (
        constraint_matrix
        @ integer_solution
    )

    expected_integer_vector = (
        np.array(
            constraint_targets,
            dtype=int,
        )
    )

    if not np.array_equal(
        achieved_vector,
        expected_integer_vector,
    ):

        log_entries.append(
            "EXACT DEV FEASIBILITY: FAIL"
        )

        log_entries.append(
            "  Independently recomputed margins "
            "do not equal requested targets"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Feasibility witness failed "
            "independent marginal verification"
        )

    # --------------------------------------------------------
    # Log each requested margin and its achieved value.
    # --------------------------------------------------------

    log_entries.append(
        "Exact marginal reconciliation:"
    )

    for (
        label,
        expected,
        achieved,
    ) in zip(
        constraint_labels,
        expected_integer_vector,
        achieved_vector,
    ):

        log_entries.append(
            f"  {label}: "
            f"expected={int(expected)} "
            f"achieved={int(achieved)}"
        )

    # --------------------------------------------------------
    # Store the solver's feasible witness.
    #
    # IMPORTANT:
    # witness_dev_cards is NOT yet the frozen joint-cell
    # allocation policy.
    #
    # It only proves that at least one exact solution exists.
    # --------------------------------------------------------

    feasibility_witness_df = (
        joint_df
        .copy()
    )

    feasibility_witness_df[
        "witness_dev_cards"
    ] = (
        integer_solution
    )

    nonzero_witness = (
        feasibility_witness_df[
            feasibility_witness_df[
                "witness_dev_cards"
            ]
            .gt(0)
        ]
    )

    log_entries.append(
        "Non-zero cells in feasibility witness: "
        f"{len(nonzero_witness)}"
    )

    log_entries.append(
        "Feasibility witness allocation:"
    )

    for _, row in (
        nonzero_witness.iterrows()
    ):

        log_entries.append(
            f"  {row['stem_family']} / "
            f"{row['face_db']} / "
            f"{row['gender']} / "
            f"{row['language_signature']}: "
            f"selected="
            f"{int(row['witness_dev_cards'])} "
            f"of "
            f"{int(row['source_cards'])}"
        )

    witness_total = int(
        feasibility_witness_df[
            "witness_dev_cards"
        ]
        .sum()
    )

    log_entries.append(
        "Feasibility witness total: "
        f"{witness_total} cards"
    )

    if (
        witness_total
        != DEV_CARD_TARGET
    ):

        log_entries.append(
            "EXACT DEV FEASIBILITY: FAIL"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Feasibility witness does not "
            "contain exactly 51 cards"
        )

    log_entries.append(
        "EXACT DEV FEASIBILITY: PASS"
    )

    log_entries.append(
        (
            "  At least one exact 51-card allocation "
            "simultaneously satisfies stem-family, "
            "face_db x gender, and language targets."
        )
    )

    write_log(
        log_path,
        log_entries,
    )

    return feasibility_witness_df

# ------------------------------------------------------------
#Function to Optimize the joint-cell development allocation.
#
# HARD constraints:
#
#   exact stem_family quotas
#   exact face_db x gender quotas
#   exact language_signature quotas
#   exactly 51 development cards
#
# SOFT objective:
#
#   stay as close as possible to proportional representation
#   inside every observed 4-way joint cell.
#
# For cell i:
#
#   ideal_i = source_cards_i * 51 / 211
#
# We minimize:
#
#   sum |selected_i - ideal_i|
#
# To avoid floating-point ambiguity in the objective, multiply
# every deviation by 211:
#
#   |211 * selected_i - 51 * source_cards_i|
#
# This produces an exactly integer-scaled objective.
#
# No actual file_stems are selected in this function.
# ------------------------------------------------------------

def optimize_joint_dev_allocation(
    joint_df: pd.DataFrame,
    quota_tables: dict[str, pd.DataFrame],
    feasibility_witness_df: pd.DataFrame,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "optimize_joint_dev_allocation"
            "********************"
        )
    ]

    joint_columns = [
        "stem_family",
        "face_db",
        "gender",
        "language_signature",
    ]

    required_columns = (
        set(joint_columns)
        | {
            "source_cards",
        }
    )

    missing_columns = (
        required_columns
        - set(
            joint_df.columns
        )
    )

    if missing_columns:

        write_log(
            log_path,
            log_entries
            + [
                "JOINT ALLOCATION OPTIMIZATION: FAIL",
                (
                    "  Missing joint columns: "
                    f"{sorted(missing_columns)}"
                ),
            ],
        )

        raise RuntimeError(
            "Cannot optimize joint allocation"
        )

    # --------------------------------------------------------
    # Use an explicit deterministic joint-cell ordering.
    #
    # This ordering is also used only as the FINAL tie-break
    # among allocations with identical proportional objective.
    # --------------------------------------------------------

    allocation_df = (
        joint_df
        .sort_values(
            joint_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    number_of_cells = len(
        allocation_df
    )

    source_capacities = (
        allocation_df[
            "source_cards"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------
    # Recover exact marginal targets.
    # --------------------------------------------------------

    family_targets = {
        row["stem_family"]:
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "family"
        ].iterrows()
    }

    face_db_gender_targets = {
        (
            row["face_db"],
            row["gender"],
        ):
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "face_db_gender"
        ].iterrows()
    }

    language_targets = {
        row["language_signature"]:
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "language"
        ].iterrows()
    }

    # --------------------------------------------------------
    # Decision variables:
    #
    # first N variables:
    #     x_i = integer number of dev cards selected from cell i
    #
    # next N variables:
    #     d_i = scaled absolute deviation from proportional ideal
    #
    # Therefore there are 2N optimization variables.
    # --------------------------------------------------------

    total_variables = (
        2
        * number_of_cells
    )

    # --------------------------------------------------------
    # Exact hard-margin constraints.
    #
    # d_i variables do not participate in these rows, so their
    # coefficients are zero.
    # --------------------------------------------------------

    equality_rows = []
    equality_targets = []
    equality_labels = []

    def add_exact_margin(
        cell_mask: pd.Series,
        target: int,
        label: str,
    ) -> None:

        row = np.zeros(
            total_variables,
            dtype=float,
        )

        row[
            :number_of_cells
        ] = (
            cell_mask
            .astype(int)
            .to_numpy()
        )

        equality_rows.append(
            row
        )

        equality_targets.append(
            int(
                target
            )
        )

        equality_labels.append(
            label
        )

    # --------------------------------------------------------
    # Stem-family margins.
    # --------------------------------------------------------

    for (
        stem_family,
        target,
    ) in sorted(
        family_targets.items()
    ):

        add_exact_margin(
            allocation_df[
                "stem_family"
            ].eq(
                stem_family
            ),
            target,
            (
                "stem_family = "
                f"{stem_family}"
            ),
        )

    # --------------------------------------------------------
    # face_db x gender margins.
    # --------------------------------------------------------

    for (
        face_db,
        gender,
    ), target in sorted(
        face_db_gender_targets.items()
    ):

        add_exact_margin(
            (
                allocation_df[
                    "face_db"
                ].eq(
                    face_db
                )
                &
                allocation_df[
                    "gender"
                ].eq(
                    gender
                )
            ),
            target,
            (
                "face_db x gender = "
                f"{face_db} / {gender}"
            ),
        )

    # --------------------------------------------------------
    # Language-signature margins.
    # --------------------------------------------------------

    for (
        language_signature,
        target,
    ) in sorted(
        language_targets.items()
    ):

        add_exact_margin(
            allocation_df[
                "language_signature"
            ].eq(
                language_signature
            ),
            target,
            (
                "language_signature = "
                f"{language_signature}"
            ),
        )

    # --------------------------------------------------------
    # Explicit total = 51.
    # --------------------------------------------------------

    total_row = np.zeros(
        total_variables,
        dtype=float,
    )

    total_row[
        :number_of_cells
    ] = 1

    equality_rows.append(
        total_row
    )

    equality_targets.append(
        DEV_CARD_TARGET
    )

    equality_labels.append(
        "total development cards"
    )

    equality_matrix = np.vstack(
        equality_rows
    )

    equality_target_vector = np.array(
        equality_targets,
        dtype=float,
    )

    hard_margin_constraint = (
        LinearConstraint(
            equality_matrix,
            equality_target_vector,
            equality_target_vector,
        )
    )

    # --------------------------------------------------------
    # Absolute-deviation constraints.
    #
    # For every cell:
    #
    #   d_i >= 211*x_i - 51*source_i
    #
    #   d_i >= -(211*x_i - 51*source_i)
    #
    # Rearranged into upper-bound form:
    #
    #   211*x_i - d_i <= 51*source_i
    #
    #  -211*x_i - d_i <= -51*source_i
    # --------------------------------------------------------

    deviation_rows = []
    deviation_upper_bounds = []

    for cell_index in range(
        number_of_cells
    ):

        source_cards = int(
            source_capacities[
                cell_index
            ]
        )

        # ----------------------------------------------------
        # Positive deviation side.
        # ----------------------------------------------------

        row = np.zeros(
            total_variables,
            dtype=float,
        )

        row[
            cell_index
        ] = 211

        row[
            number_of_cells
            + cell_index
        ] = -1

        deviation_rows.append(
            row
        )

        deviation_upper_bounds.append(
            51
            * source_cards
        )

        # ----------------------------------------------------
        # Negative deviation side.
        # ----------------------------------------------------

        row = np.zeros(
            total_variables,
            dtype=float,
        )

        row[
            cell_index
        ] = -211

        row[
            number_of_cells
            + cell_index
        ] = -1

        deviation_rows.append(
            row
        )

        deviation_upper_bounds.append(
            -51
            * source_cards
        )

    deviation_constraint = (
        LinearConstraint(
            np.vstack(
                deviation_rows
            ),
            -np.inf,
            np.array(
                deviation_upper_bounds,
                dtype=float,
            ),
        )
    )

    # --------------------------------------------------------
    # Variable bounds.
    #
    # x_i:
    #     0 <= x_i <= source_cards_i
    #
    # d_i:
    #     d_i >= 0
    # --------------------------------------------------------

    lower_bounds = np.zeros(
        total_variables,
        dtype=float,
    )

    upper_bounds = np.concatenate(
        [
            source_capacities.astype(
                float
            ),
            np.full(
                number_of_cells,
                np.inf,
            ),
        ]
    )

    variable_bounds = Bounds(
        lower_bounds,
        upper_bounds,
    )

    # --------------------------------------------------------
    # Only x_i variables are integer.
    #
    # d_i can remain continuous because, once x_i is integral,
    # the exact lower bound on d_i is itself integer-scaled.
    # --------------------------------------------------------

    integrality = np.concatenate(
        [
            np.ones(
                number_of_cells,
                dtype=int,
            ),
            np.zeros(
                number_of_cells,
                dtype=int,
            ),
        ]
    )

    # --------------------------------------------------------
    # PRIMARY OBJECTIVE:
    #
    # Minimize sum of all scaled absolute deviations.
    # --------------------------------------------------------

    primary_objective = np.concatenate(
        [
            np.zeros(
                number_of_cells,
                dtype=float,
            ),
            np.ones(
                number_of_cells,
                dtype=float,
            ),
        ]
    )

    log_entries.append(
        "Primary optimization problem:"
    )

    log_entries.append(
        f"  joint cells: "
        f"{number_of_cells}"
    )

    log_entries.append(
        f"  optimization variables: "
        f"{total_variables}"
    )

    log_entries.append(
        f"  hard equality constraints: "
        f"{len(equality_targets)}"
    )

    log_entries.append(
        f"  SciPy version: "
        f"{scipy.__version__}"
    )

    # --------------------------------------------------------
    # Solve primary proportionality objective.
    # --------------------------------------------------------

    primary_result = milp(
        c=primary_objective,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=[
            hard_margin_constraint,
            deviation_constraint,
        ],
        options={
            "disp": False,
        },
    )

    log_entries.append(
        f"Primary solver status: "
        f"{primary_result.status}"
    )

    log_entries.append(
        f"Primary solver message: "
        f"{primary_result.message}"
    )

    if (
        not primary_result.success
        or primary_result.x is None
    ):

        log_entries.append(
            "JOINT ALLOCATION OPTIMIZATION: FAIL"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Unable to optimize joint "
            "development allocation"
        )

    # --------------------------------------------------------
    # The scaled objective should be effectively an integer.
    # --------------------------------------------------------

    raw_primary_objective = float(
        primary_result.fun
    )

    primary_scaled_optimum = int(
        round(
            raw_primary_objective
        )
    )

    objective_rounding_error = abs(
        raw_primary_objective
        - primary_scaled_optimum
    )

    log_entries.append(
        "Primary scaled objective: "
        f"{raw_primary_objective:.12g}"
    )

    log_entries.append(
        "Primary objective rounding error: "
        f"{objective_rounding_error:.12g}"
    )

    if (
        objective_rounding_error
        > 1e-6
    ):

        log_entries.append(
            "JOINT ALLOCATION OPTIMIZATION: FAIL"
        )

        write_log(
            log_path,
            log_entries,
        )

        raise RuntimeError(
            "Primary proportional objective "
            "is not numerically stable"
        )

    # --------------------------------------------------------
    # Compare against the earlier arbitrary feasibility witness.
    # --------------------------------------------------------

    witness_comparison_available = (
        "witness_dev_cards"
        in feasibility_witness_df.columns
    )

    if witness_comparison_available:

        witness_df = (
            feasibility_witness_df
            .sort_values(
                joint_columns,
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        if not (
            witness_df[
                joint_columns
                + [
                    "source_cards",
                ]
            ]
            .equals(
                allocation_df[
                    joint_columns
                    + [
                        "source_cards",
                    ]
                ]
            )
        ):

            raise RuntimeError(
                "Feasibility witness joint-cell "
                "ordering/content does not reconcile"
            )

        witness_selected = (
            witness_df[
                "witness_dev_cards"
            ]
            .to_numpy(
                dtype=int
            )
        )

        witness_scaled_deviation = int(
            np.abs(
                (
                    211
                    * witness_selected
                )
                -
                (
                    51
                    * source_capacities
                )
            )
            .sum()
        )

        log_entries.append(
            "Previous feasibility-witness "
            "scaled deviation: "
            f"{witness_scaled_deviation}"
        )

        log_entries.append(
            "Optimized scaled deviation: "
            f"{primary_scaled_optimum}"
        )

    # --------------------------------------------------------
    # DETERMINISTIC TIE-BREAK
    #
    # The proportional objective is the scientific criterion.
    #
    # If multiple allocations achieve exactly the same minimum
    # deviation, choose the lexicographically smallest x-vector
    # according to the deterministic joint-cell ordering above.
    #
    # This is only a reproducibility tie-break. It cannot worsen
    # the proportional objective.
    # --------------------------------------------------------

    primary_objective_row = np.zeros(
        total_variables,
        dtype=float,
    )

    primary_objective_row[
        number_of_cells:
    ] = 1

    primary_optimum_constraint = (
        LinearConstraint(
            primary_objective_row,
            -np.inf,
            float(
                primary_scaled_optimum
            ),
        )
    )

    tie_constraints = [
        hard_margin_constraint,
        deviation_constraint,
        primary_optimum_constraint,
    ]

    deterministic_solution = np.zeros(
        number_of_cells,
        dtype=int,
    )

    # --------------------------------------------------------
    # Sequentially minimize x_0, then x_1, ..., while fixing all
    # earlier values.
    #
    # This defines a true lexicographic tie-break without relying
    # on arbitrary floating-point "tiny weights".
    # --------------------------------------------------------

    for cell_index in range(
        number_of_cells
    ):

        tie_objective = np.zeros(
            total_variables,
            dtype=float,
        )

        tie_objective[
            cell_index
        ] = 1

        tie_result = milp(
            c=tie_objective,
            integrality=integrality,
            bounds=variable_bounds,
            constraints=tie_constraints,
            options={
                "disp": False,
            },
        )

        if (
            not tie_result.success
            or tie_result.x is None
        ):

            log_entries.append(
                "JOINT ALLOCATION OPTIMIZATION: FAIL"
            )

            log_entries.append(
                "  Deterministic tie-break failed "
                f"at joint cell {cell_index}"
            )

            write_log(
                log_path,
                log_entries,
            )

            raise RuntimeError(
                "Deterministic joint-allocation "
                "tie-break failed"
            )

        raw_value = float(
            tie_result.x[
                cell_index
            ]
        )

        integer_value = int(
            round(
                raw_value
            )
        )

        if (
            abs(
                raw_value
                - integer_value
            )
            > 1e-6
        ):

            raise RuntimeError(
                "Tie-break result is not "
                "sufficiently integral"
            )

        deterministic_solution[
            cell_index
        ] = integer_value

        # ----------------------------------------------------
        # Freeze this cell's selected-card count before moving
        # to the next cell.
        # ----------------------------------------------------

        fixed_row = np.zeros(
            total_variables,
            dtype=float,
        )

        fixed_row[
            cell_index
        ] = 1

        tie_constraints.append(
            LinearConstraint(
                fixed_row,
                integer_value,
                integer_value,
            )
        )

    # --------------------------------------------------------
    # Independent exact verification of the final solution.
    # --------------------------------------------------------

    if (
        (deterministic_solution < 0).any()
        or
        (
            deterministic_solution
            > source_capacities
        ).any()
    ):

        raise RuntimeError(
            "Optimized allocation exceeds "
            "joint-cell source support"
        )

    achieved_margins = (
        equality_matrix[
            :,
            :number_of_cells
        ]
        @ deterministic_solution
    )

    expected_margins = np.array(
        equality_targets,
        dtype=int,
    )

    if not np.array_equal(
        achieved_margins,
        expected_margins,
    ):

        raise RuntimeError(
            "Optimized allocation failed "
            "hard-margin reconciliation"
        )

    # --------------------------------------------------------
    # Recompute proportional objective entirely independently
    # from solver d_i variables.
    # --------------------------------------------------------

    exact_scaled_deviations = np.abs(
        (
            211
            * deterministic_solution
        )
        -
        (
            51
            * source_capacities
        )
    )

    final_scaled_objective = int(
        exact_scaled_deviations.sum()
    )

    if (
        final_scaled_objective
        != primary_scaled_optimum
    ):

        raise RuntimeError(
            "Deterministic tie-break changed "
            "the optimal proportional objective"
        )

    # --------------------------------------------------------
    # Build the final JOINT-CELL allocation table.
    # --------------------------------------------------------

    allocation_df[
        "ideal_dev_cards"
    ] = (
        allocation_df[
            "source_cards"
        ]
        * DEV_CARD_TARGET
        / EXPECTED_SOURCE_TRAIN_CARDS
    )

    allocation_df[
        "optimized_dev_cards"
    ] = deterministic_solution

    allocation_df[
        "signed_deviation_cards"
    ] = (
        allocation_df[
            "optimized_dev_cards"
        ]
        -
        allocation_df[
            "ideal_dev_cards"
        ]
    )

    allocation_df[
        "absolute_deviation_cards"
    ] = (
        allocation_df[
            "signed_deviation_cards"
        ]
        .abs()
    )

    # --------------------------------------------------------
    # Log final hard-margin reconciliation.
    # --------------------------------------------------------

    log_entries.append(
        "Optimized hard-margin reconciliation:"
    )

    for (
        label,
        expected,
        achieved,
    ) in zip(
        equality_labels,
        expected_margins,
        achieved_margins,
    ):

        log_entries.append(
            f"  {label}: "
            f"expected={int(expected)} "
            f"achieved={int(achieved)}"
        )

    # --------------------------------------------------------
    # Proportional-fit summary.
    # --------------------------------------------------------

    total_absolute_deviation = float(
        allocation_df[
            "absolute_deviation_cards"
        ]
        .sum()
    )

    maximum_absolute_deviation = float(
        allocation_df[
            "absolute_deviation_cards"
        ]
        .max()
    )

    log_entries.append(
        "Joint proportional-fit summary:"
    )

    log_entries.append(
        "  exact scaled L1 objective: "
        f"{final_scaled_objective}"
    )

    log_entries.append(
        "  total absolute deviation: "
        f"{total_absolute_deviation:.6f} cards"
    )

    log_entries.append(
        "  maximum single-cell deviation: "
        f"{maximum_absolute_deviation:.6f} cards"
    )

    nonzero_cells = allocation_df[
        allocation_df[
            "optimized_dev_cards"
        ]
        .gt(0)
    ]

    log_entries.append(
        "Non-zero optimized joint cells: "
        f"{len(nonzero_cells)}"
    )

    # --------------------------------------------------------
    # Log the complete final allocation.
    # --------------------------------------------------------

    log_entries.append(
        "Optimized joint-cell allocation:"
    )

    for _, row in (
        allocation_df.iterrows()
    ):

        log_entries.append(
            f"  {row['stem_family']} / "
            f"{row['face_db']} / "
            f"{row['gender']} / "
            f"{row['language_signature']}: "
            f"source={int(row['source_cards'])} "
            f"ideal={row['ideal_dev_cards']:.3f} "
            f"selected="
            f"{int(row['optimized_dev_cards'])} "
            f"abs_dev="
            f"{row['absolute_deviation_cards']:.3f}"
        )

    optimized_total = int(
        allocation_df[
            "optimized_dev_cards"
        ]
        .sum()
    )

    log_entries.append(
        "Optimized allocation total: "
        f"{optimized_total} cards"
    )

    if (
        optimized_total
        != DEV_CARD_TARGET
    ):

        raise RuntimeError(
            "Optimized allocation does not "
            "contain exactly 51 cards"
        )

    log_entries.append(
        "JOINT ALLOCATION OPTIMIZATION: PASS"
    )

    log_entries.append(
        (
            "  Exact hard margins retained while "
            "minimizing deviation from proportional "
            "representation across all observed "
            "joint cells."
        )
    )

    write_log(
        log_path,
        log_entries,
    )

    return allocation_df

# ------------------------------------------------------------
#Function Deterministically select the actual 51 development cards.
#
# The optimized joint-cell allocation has already decided HOW MANY
# cards must come from each observed cell.
#
# This function now decides WHICH cards.
#
# Within each joint cell, cards are ranked by:
#
#   SHA256(
#       selection_seed
#       + frozen discovery SHA256
#       + file_stem
#   )
#
# The frozen discovery SHA binds the selection to the exact source
# artifact.
#
# We then take the first N hashes in each cell, where N is the
# previously optimized_dev_cards value.
#
# No Python random generator is used.
# No filename-order preference is used.
# ------------------------------------------------------------

def select_dev_cards_deterministically(
    train_cards_df: pd.DataFrame,
    optimized_allocation_df: pd.DataFrame,
    quota_tables: dict[str, pd.DataFrame],
    discovery_sha256: str,
    config: dict,
    log_path: Path,
) -> pd.DataFrame:

    log_entries = [
        (
            "********************"
            "select_dev_cards_deterministically"
            "********************"
        )
    ]

    joint_columns = [
        "stem_family",
        "face_db",
        "gender",
        "language_signature",
    ]

    # --------------------------------------------------------
    # Read and validate the frozen selection seed.
    # --------------------------------------------------------

    split_policy = config.get(
        "split_policy"
    )

    if not isinstance(
        split_policy,
        dict,
    ):

        write_log(
            log_path,
            log_entries
            + [
                "DETERMINISTIC CARD SELECTION: FAIL",
                (
                    "  buildconfig.yaml is missing "
                    "split_policy"
                ),
            ],
        )

        raise RuntimeError(
            "Split policy configuration missing"
        )

    selection_seed = split_policy.get(
        "selection_seed"
    )

    if (
        not isinstance(
            selection_seed,
            str,
        )
        or
        not selection_seed.strip()
    ):

        write_log(
            log_path,
            log_entries
            + [
                "DETERMINISTIC CARD SELECTION: FAIL",
                (
                    "  split_policy.selection_seed "
                    "must be a non-empty string"
                ),
            ],
        )

        raise RuntimeError(
            "Invalid split selection seed"
        )

    selection_seed = (
        selection_seed.strip()
    )

    log_entries.append(
        "Selection policy:"
    )

    log_entries.append(
        "  algorithm: "
        "SHA-256 deterministic within-cell ranking"
    )

    log_entries.append(
        f"  seed: {selection_seed}"
    )

    log_entries.append(
        "  discovery SHA-256: "
        f"{discovery_sha256}"
    )

    # --------------------------------------------------------
    # Required card-level fields.
    # --------------------------------------------------------

    required_card_columns = (
        set(
            joint_columns
        )
        | {
            "file_stem",
            "source_image_count",
        }
    )

    missing_card_columns = (
        required_card_columns
        - set(
            train_cards_df.columns
        )
    )

    if missing_card_columns:

        raise RuntimeError(
            "Card-selection input is missing columns: "
            f"{sorted(missing_card_columns)}"
        )

    required_allocation_columns = (
        set(
            joint_columns
        )
        | {
            "source_cards",
            "optimized_dev_cards",
        }
    )

    missing_allocation_columns = (
        required_allocation_columns
        - set(
            optimized_allocation_df.columns
        )
    )

    if missing_allocation_columns:

        raise RuntimeError(
            "Optimized allocation is missing columns: "
            f"{sorted(missing_allocation_columns)}"
        )

    # --------------------------------------------------------
    # Every allocation row must describe one unique joint cell.
    # --------------------------------------------------------

    duplicate_allocation_cells = (
        optimized_allocation_df
        .duplicated(
            subset=joint_columns,
            keep=False,
        )
    )

    if duplicate_allocation_cells.any():

        raise RuntimeError(
            "Optimized allocation contains "
            "duplicate joint-cell rows"
        )

    # --------------------------------------------------------
    # Join the optimized cell quota onto every individual card.
    # --------------------------------------------------------

    allocation_columns = (
        joint_columns
        + [
            "source_cards",
            "optimized_dev_cards",
        ]
    )

    split_cards_df = (
        train_cards_df
        .merge(
            optimized_allocation_df[
                allocation_columns
            ],
            on=joint_columns,
            how="left",
            validate="many_to_one",
        )
    )

    if (
        split_cards_df[
            "optimized_dev_cards"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "One or more training cards did not "
            "match an optimized joint cell"
        )

    # --------------------------------------------------------
    # Independently check that actual card support agrees with
    # source_cards recorded by the optimization table.
    # --------------------------------------------------------

    observed_cell_sizes = (
        split_cards_df
        .groupby(
            joint_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="observed_source_cards"
        )
    )

    support_check_df = (
        optimized_allocation_df[
            joint_columns
            + [
                "source_cards",
            ]
        ]
        .merge(
            observed_cell_sizes,
            on=joint_columns,
            how="outer",
            validate="one_to_one",
        )
    )

    bad_support = support_check_df[
        support_check_df[
            "source_cards"
        ].ne(
            support_check_df[
                "observed_source_cards"
            ]
        )
    ]

    if not bad_support.empty:

        raise RuntimeError(
            "Individual card support does not "
            "reconcile with optimized joint allocation"
        )

    # --------------------------------------------------------
    # Deterministic SHA-256 ranking key.
    #
    # Newline separators make the payload unambiguous.
    # --------------------------------------------------------

    def calculate_selection_hash(
        file_stem: str,
    ) -> str:

        payload = (
            f"{selection_seed}\n"
            f"{discovery_sha256}\n"
            f"{file_stem}"
        )

        return hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

    split_cards_df[
        "selection_hash"
    ] = (
        split_cards_df[
            "file_stem"
        ]
        .map(
            calculate_selection_hash
        )
    )

    # --------------------------------------------------------
    # Hashes should be unique for all 211 cards.
    #
    # A lexical file_stem tie-break is still retained below even
    # though a SHA-256 collision is practically implausible.
    # --------------------------------------------------------

    if (
        split_cards_df[
            "selection_hash"
        ]
        .nunique()
        != len(
            split_cards_df
        )
    ):

        raise RuntimeError(
            "Selection-hash collision detected"
        )

    # --------------------------------------------------------
    # Sort deterministically within every joint cell.
    # --------------------------------------------------------

    split_cards_df = (
        split_cards_df
        .sort_values(
            joint_columns
            + [
                "selection_hash",
                "file_stem",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    split_cards_df[
        "selection_rank_within_cell"
    ] = (
        split_cards_df
        .groupby(
            joint_columns,
            dropna=False,
            sort=False,
        )
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Actual project role.
    #
    # First N ranked cards in each cell -> dev_val.
    # Remaining cards                    -> project_train.
    # --------------------------------------------------------

    split_cards_df[
        "project_role"
    ] = np.where(
        split_cards_df[
            "selection_rank_within_cell"
        ]
        <=
        split_cards_df[
            "optimized_dev_cards"
        ],
        "dev_val",
        "project_train",
    )

    # --------------------------------------------------------
    # Card-level totals.
    # --------------------------------------------------------

    role_card_counts = (
        split_cards_df[
            "project_role"
        ]
        .value_counts()
        .to_dict()
    )

    dev_card_count = int(
        role_card_counts.get(
            "dev_val",
            0,
        )
    )

    project_train_card_count = int(
        role_card_counts.get(
            "project_train",
            0,
        )
    )

    log_entries.append(
        "Selected card totals:"
    )

    log_entries.append(
        f"  dev_val: "
        f"{dev_card_count}"
    )

    log_entries.append(
        f"  project_train: "
        f"{project_train_card_count}"
    )

    if (
        dev_card_count
        != DEV_CARD_TARGET
    ):

        raise RuntimeError(
            "Deterministic selection did not "
            "produce exactly 51 dev cards"
        )

    expected_project_train_cards = (
        EXPECTED_SOURCE_TRAIN_CARDS
        - DEV_CARD_TARGET
    )

    if (
        project_train_card_count
        != expected_project_train_cards
    ):

        raise RuntimeError(
            "Unexpected project-train "
            "card count"
        )

    # --------------------------------------------------------
    # Image-level totals implied by whole-card assignment.
    # --------------------------------------------------------

    dev_image_count = int(
        split_cards_df.loc[
            split_cards_df[
                "project_role"
            ].eq(
                "dev_val"
            ),
            "source_image_count",
        ]
        .sum()
    )

    project_train_image_count = int(
        split_cards_df.loc[
            split_cards_df[
                "project_role"
            ].eq(
                "project_train"
            ),
            "source_image_count",
        ]
        .sum()
    )

    log_entries.append(
        "Whole-card image totals:"
    )

    log_entries.append(
        f"  dev_val: "
        f"{dev_image_count}"
    )

    log_entries.append(
        f"  project_train: "
        f"{project_train_image_count}"
    )

    if (
        dev_image_count
        != EXPECTED_DEV_IMAGE_ROWS
    ):

        raise RuntimeError(
            "Selected development cards do not "
            "represent exactly 459 images"
        )

    expected_project_train_images = (
        EXPECTED_SOURCE_TRAIN_ROWS
        - EXPECTED_DEV_IMAGE_ROWS
    )

    if (
        project_train_image_count
        != expected_project_train_images
    ):

        raise RuntimeError(
            "Selected project-train cards do not "
            "represent exactly 1440 images"
        )

    # --------------------------------------------------------
    # Independent joint-cell selection reconciliation.
    #
    # The number of actual dev cards selected from each cell must
    # exactly equal optimized_dev_cards.
    # --------------------------------------------------------

    selected_cell_counts = (
        split_cards_df.loc[
            split_cards_df[
                "project_role"
            ].eq(
                "dev_val"
            )
        ]
        .groupby(
            joint_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="actual_dev_cards"
        )
    )

    cell_reconciliation_df = (
        optimized_allocation_df[
            joint_columns
            + [
                "source_cards",
                "optimized_dev_cards",
            ]
        ]
        .merge(
            selected_cell_counts,
            on=joint_columns,
            how="left",
            validate="one_to_one",
        )
    )

    cell_reconciliation_df[
        "actual_dev_cards"
    ] = (
        cell_reconciliation_df[
            "actual_dev_cards"
        ]
        .fillna(0)
        .astype(int)
    )

    bad_cell_counts = (
        cell_reconciliation_df[
            cell_reconciliation_df[
                "actual_dev_cards"
            ]
            .ne(
                cell_reconciliation_df[
                    "optimized_dev_cards"
                ]
            )
        ]
    )

    if not bad_cell_counts.empty:

        raise RuntimeError(
            "Actual deterministic card selection "
            "does not match optimized joint-cell allocation"
        )

    # --------------------------------------------------------
    # Directly verify the three frozen marginal target families.
    # --------------------------------------------------------

    dev_cards_df = (
        split_cards_df.loc[
            split_cards_df[
                "project_role"
            ].eq(
                "dev_val"
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Family targets.
    # --------------------------------------------------------

    actual_family_counts = (
        dev_cards_df[
            "stem_family"
        ]
        .value_counts()
        .to_dict()
    )

    expected_family_counts = {
        row["stem_family"]:
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "family"
        ].iterrows()
    }

    if (
        actual_family_counts
        != expected_family_counts
    ):

        raise RuntimeError(
            "Actual dev family margins do not "
            "match frozen quota targets"
        )

    # --------------------------------------------------------
    # face_db x gender targets.
    # --------------------------------------------------------

    actual_face_gender_counts = (
        dev_cards_df
        .groupby(
            [
                "face_db",
                "gender",
            ]
        )
        .size()
        .to_dict()
    )

    expected_face_gender_counts = {
        (
            row["face_db"],
            row["gender"],
        ):
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "face_db_gender"
        ].iterrows()
    }

    if (
        actual_face_gender_counts
        != expected_face_gender_counts
    ):

        raise RuntimeError(
            "Actual dev face_db x gender margins "
            "do not match frozen quota targets"
        )

    # --------------------------------------------------------
    # Language targets.
    # --------------------------------------------------------

    actual_language_counts = (
        dev_cards_df[
            "language_signature"
        ]
        .value_counts()
        .to_dict()
    )

    expected_language_counts = {
        row["language_signature"]:
            int(
                row["dev_target"]
            )

        for _, row
        in quota_tables[
            "language"
        ].iterrows()
    }

    if (
        actual_language_counts
        != expected_language_counts
    ):

        raise RuntimeError(
            "Actual dev language margins do not "
            "match frozen quota targets"
        )

    # --------------------------------------------------------
    # Card disjointness.
    # --------------------------------------------------------

    dev_stems = set(
        dev_cards_df[
            "file_stem"
        ]
    )

    project_train_stems = set(
        split_cards_df.loc[
            split_cards_df[
                "project_role"
            ].eq(
                "project_train"
            ),
            "file_stem",
        ]
    )

    overlap = (
        dev_stems
        & project_train_stems
    )

    log_entries.append(
        "Project train/dev card overlap: "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Project train and dev_val are "
            "not card-disjoint"
        )

    # --------------------------------------------------------
    # Log exact development-card membership.
    #
    # This is now reproducibility evidence.
    # --------------------------------------------------------

    log_entries.append(
        "Selected dev_val cards:"
    )

    for _, row in (
        dev_cards_df
        .sort_values(
            "file_stem"
        )
        .iterrows()
    ):

        log_entries.append(
            f"  {row['file_stem']} "
            f"| family={row['stem_family']} "
            f"| face_db={row['face_db']} "
            f"| gender={row['gender']} "
            f"| language={row['language_signature']} "
            f"| rank={int(row['selection_rank_within_cell'])} "
            f"| SHA256={row['selection_hash']}"
        )

    log_entries.append(
        "DETERMINISTIC CARD SELECTION: PASS"
    )

    log_entries.append(
        (
            "  51 dev_val cards and "
            "160 project_train cards selected "
            "with exact optimized joint-cell "
            "and marginal quotas."
        )
    )

    write_log(
        log_path,
        log_entries,
    )

    # --------------------------------------------------------
    # Return in a convenient stable order for downstream work.
    # --------------------------------------------------------

    return (
        split_cards_df
        .sort_values(
            "file_stem"
        )
        .reset_index(
            drop=True
        )
    )

# ------------------------------------------------------------
#Function Build image-level project train / dev_val manifests.
#
# Card selection is already frozen in split_cards_df.
#
# This function propagates each card's project_role onto all
# nine source images belonging to that card.
#
# Nothing is written to disk yet.
# ------------------------------------------------------------

EXPECTED_PROJECT_TRAIN_ROWS = 1440
EXPECTED_PROJECT_TRAIN_CARDS = 160


def build_image_level_split_manifests(
    source_train_df: pd.DataFrame,
    split_cards_df: pd.DataFrame,
    log_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    log_entries = [
        (
            "********************"
            "build_image_level_split_manifests"
            "********************"
        )
    ]

    # --------------------------------------------------------
    # Card-level fields to propagate onto every image.
    #
    # Keep source image metadata untouched.
    # --------------------------------------------------------

    card_assignment_columns = [
        "file_stem",
        "stem_family",
        "language_signature",
        "selection_hash",
        "selection_rank_within_cell",
        "optimized_dev_cards",
        "project_role",
    ]

    missing_columns = (
        set(card_assignment_columns)
        - set(split_cards_df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Card assignment table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Every source-train card must occur exactly once in the
    # card assignment table.
    # --------------------------------------------------------

    if (
        split_cards_df[
            "file_stem"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Card assignment table contains "
            "duplicate file_stem rows"
        )

    # --------------------------------------------------------
    # Propagate card role onto all 1,899 source images.
    # --------------------------------------------------------

    project_split_df = (
        source_train_df
        .merge(
            split_cards_df[
                card_assignment_columns
            ],
            on="file_stem",
            how="left",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # No source-training image may remain unassigned.
    # --------------------------------------------------------

    unassigned_rows = int(
        project_split_df[
            "project_role"
        ]
        .isna()
        .sum()
    )

    log_entries.append(
        "Source-train images without project role: "
        f"{unassigned_rows}"
    )

    if unassigned_rows:

        raise RuntimeError(
            "One or more source-train images "
            "did not receive a project role"
        )

    # --------------------------------------------------------
    # The merge itself must preserve all source-train rows.
    # --------------------------------------------------------

    log_entries.append(
        "Image-level split rows: "
        f"{len(project_split_df)}"
    )

    if (
        len(project_split_df)
        != EXPECTED_SOURCE_TRAIN_ROWS
    ):

        raise RuntimeError(
            "Image-level split table does not "
            "contain exactly 1899 rows"
        )

    # --------------------------------------------------------
    # Only the two intended project roles are legal.
    # --------------------------------------------------------

    discovered_roles = set(
        project_split_df[
            "project_role"
        ]
        .unique()
    )

    expected_roles = {
        "project_train",
        "dev_val",
    }

    if (
        discovered_roles
        != expected_roles
    ):

        raise RuntimeError(
            "Unexpected project roles: "
            f"{sorted(discovered_roles)}"
        )

    # --------------------------------------------------------
    # Create the two image-level manifests.
    # --------------------------------------------------------

    project_train_manifest_df = (
        project_split_df.loc[
            project_split_df[
                "project_role"
            ].eq(
                "project_train"
            )
        ]
        .copy()
    )

    dev_val_manifest_df = (
        project_split_df.loc[
            project_split_df[
                "project_role"
            ].eq(
                "dev_val"
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Image counts.
    # --------------------------------------------------------

    project_train_rows = len(
        project_train_manifest_df
    )

    dev_val_rows = len(
        dev_val_manifest_df
    )

    log_entries.append(
        "Image-level project roles:"
    )

    log_entries.append(
        f"  project_train: "
        f"{project_train_rows}"
    )

    log_entries.append(
        f"  dev_val: "
        f"{dev_val_rows}"
    )

    if (
        project_train_rows
        != EXPECTED_PROJECT_TRAIN_ROWS
    ):

        raise RuntimeError(
            "project_train does not contain "
            "exactly 1440 images"
        )

    if (
        dev_val_rows
        != EXPECTED_DEV_IMAGE_ROWS
    ):

        raise RuntimeError(
            "dev_val does not contain "
            "exactly 459 images"
        )

    # --------------------------------------------------------
    # Card counts.
    # --------------------------------------------------------

    project_train_cards = int(
        project_train_manifest_df[
            "file_stem"
        ]
        .nunique()
    )

    dev_val_cards = int(
        dev_val_manifest_df[
            "file_stem"
        ]
        .nunique()
    )

    log_entries.append(
        "Image-manifest card counts:"
    )

    log_entries.append(
        f"  project_train: "
        f"{project_train_cards}"
    )

    log_entries.append(
        f"  dev_val: "
        f"{dev_val_cards}"
    )

    if (
        project_train_cards
        != EXPECTED_PROJECT_TRAIN_CARDS
    ):

        raise RuntimeError(
            "project_train does not contain "
            "exactly 160 cards"
        )

    if (
        dev_val_cards
        != DEV_CARD_TARGET
    ):

        raise RuntimeError(
            "dev_val does not contain "
            "exactly 51 cards"
        )

    # --------------------------------------------------------
    # Verify every card remains whole.
    #
    # Every file_stem in either role must still contain exactly
    # nine images.
    # --------------------------------------------------------

    card_sizes = (
        project_split_df
        .groupby(
            [
                "project_role",
                "file_stem",
            ]
        )
        .size()
    )

    bad_card_sizes = (
        card_sizes[
            card_sizes.ne(
                EXPECTED_IMAGES_PER_TRAIN_CARD
            )
        ]
    )

    log_entries.append(
        "Cards split incompletely across image manifests: "
        f"{len(bad_card_sizes)}"
    )

    if not bad_card_sizes.empty:

        raise RuntimeError(
            "One or more project cards do not "
            "contain exactly nine images"
        )

    # --------------------------------------------------------
    # Explicit train/dev card disjointness.
    # --------------------------------------------------------

    project_train_stems = set(
        project_train_manifest_df[
            "file_stem"
        ]
    )

    dev_val_stems = set(
        dev_val_manifest_df[
            "file_stem"
        ]
    )

    card_overlap = (
        project_train_stems
        & dev_val_stems
    )

    log_entries.append(
        "Image-manifest train/dev card overlap: "
        f"{len(card_overlap)}"
    )

    if card_overlap:

        raise RuntimeError(
            "project_train and dev_val "
            "share one or more cards"
        )

    # --------------------------------------------------------
    # Image identity must also be disjoint.
    # --------------------------------------------------------

    train_hashes = set(
        project_train_manifest_df[
            "image_sha256"
        ]
    )

    dev_hashes = set(
        dev_val_manifest_df[
            "image_sha256"
        ]
    )

    image_hash_overlap = (
        train_hashes
        & dev_hashes
    )

    log_entries.append(
        "Image SHA-256 overlap between "
        "project_train and dev_val: "
        f"{len(image_hash_overlap)}"
    )

    if image_hash_overlap:

        raise RuntimeError(
            "project_train and dev_val contain "
            "overlapping source images"
        )

    # --------------------------------------------------------
    # Every source image must occur exactly once across the two
    # project roles.
    # --------------------------------------------------------

    combined_rows = (
        project_train_rows
        + dev_val_rows
    )

    log_entries.append(
        "Role-row reconciliation: "
        f"{project_train_rows} + "
        f"{dev_val_rows} = "
        f"{combined_rows}"
    )

    if (
        combined_rows
        != EXPECTED_SOURCE_TRAIN_ROWS
    ):

        raise RuntimeError(
            "Project-role rows do not reconcile "
            "to source train"
        )

    # --------------------------------------------------------
    # Verify image paths and hashes remain unique.
    # --------------------------------------------------------

    duplicate_image_paths = int(
        project_split_df[
            "image_path"
        ]
        .duplicated()
        .sum()
    )

    duplicate_image_hashes = int(
        project_split_df[
            "image_sha256"
        ]
        .duplicated()
        .sum()
    )

    log_entries.append(
        "Duplicate image paths in project split: "
        f"{duplicate_image_paths}"
    )

    log_entries.append(
        "Duplicate image SHA-256 values in project split: "
        f"{duplicate_image_hashes}"
    )

    if (
        duplicate_image_paths
        or duplicate_image_hashes
    ):

        raise RuntimeError(
            "Project split contains duplicate "
            "source-image identities"
        )

    # --------------------------------------------------------
    # Recheck the expected three-version composition separately
    # inside both project roles.
    # --------------------------------------------------------

    for (
        role_name,
        role_df,
        expected_cards,
    ) in (
        (
            "project_train",
            project_train_manifest_df,
            EXPECTED_PROJECT_TRAIN_CARDS,
        ),
        (
            "dev_val",
            dev_val_manifest_df,
            DEV_CARD_TARGET,
        ),
    ):

        role_composition = (
            role_df.groupby(
                [
                    "traffic_type",
                    "variant",
                ]
            )
            .size()
            .to_dict()
        )

        expected_role_composition = {
            (
                "bonafide",
                "",
            ):
                expected_cards * 3,

            (
                "attack",
                "digital_1",
            ):
                expected_cards * 3,

            (
                "attack",
                "digital_2",
            ):
                expected_cards * 3,
        }

        log_entries.append(
            f"{role_name} composition:"
        )

        for (
            traffic_type,
            variant,
        ), count in sorted(
            role_composition.items()
        ):

            variant_label = (
                variant
                if variant
                else "(empty)"
            )

            log_entries.append(
                f"  {traffic_type} / "
                f"{variant_label}: "
                f"{int(count)}"
            )

        if (
            role_composition
            != expected_role_composition
        ):

            raise RuntimeError(
                f"{role_name} has unexpected "
                "traffic/variant composition"
            )

    # --------------------------------------------------------
    # Stable row ordering.
    #
    # This will also make later CSV output deterministic.
    # --------------------------------------------------------

    manifest_sort_columns = [
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "image_path",
    ]

    project_split_df = (
        project_split_df
        .sort_values(
            [
                "project_role",
            ]
            + manifest_sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    project_train_manifest_df = (
        project_train_manifest_df
        .sort_values(
            manifest_sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    dev_val_manifest_df = (
        dev_val_manifest_df
        .sort_values(
            manifest_sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    log_entries.append(
        "IMAGE-LEVEL SPLIT MANIFESTS: PASS"
    )

    log_entries.append(
        (
            "  Source train partitioned into "
            "160 complete project_train cards "
            "(1440 images) and 51 complete "
            "dev_val cards (459 images)."
        )
    )

    write_log(
        log_path,
        log_entries,
    )

    return (
        project_split_df,
        project_train_manifest_df,
        dev_val_manifest_df,
    )

# ------------------------------------------------------------
#Function Frozen split-artifact export.
#
# Outputs from one run:
#
#   <prefix>_<timestamp>_cards.csv
#   <prefix>_<timestamp>_project_train.csv
#   <prefix>_<timestamp>_dev_val.csv
#   <prefix>_<timestamp>_bundle.yaml
#
# The YAML bundle records:
#
#   frozen discovery lineage
#   split-selection seed
#   build script/config hashes
#   output filenames and SHA-256 hashes
#
# Files are first written as .part files and only promoted to
# their final names after all validation and hashing succeeds.
# ------------------------------------------------------------

CARD_SPLIT_COLUMNS = [
    "file_stem",
    "face_db",
    "face_id",
    "gender",
    "stem_family",
    "language_signature",
    "source_image_count",
    "selection_hash",
    "selection_rank_within_cell",
    "project_role",
]


IMAGE_MANIFEST_COLUMNS = [
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
]


def resolve_config_path(
    path_value: str,
) -> Path:

    path = Path(
        path_value
    )

    if not path.is_absolute():

        path = (
            SCRIPT_DIR
            / path
        )

    return path.resolve()


def write_atomic_csv(
    dataframe: pd.DataFrame,
    final_path: Path,
) -> str:

    # --------------------------------------------------------
    # Never overwrite a frozen artifact.
    # --------------------------------------------------------

    if final_path.exists():

        raise FileExistsError(
            f"Final split artifact already exists: "
            f"{final_path}"
        )

    temp_path = Path(
        str(final_path)
        + ".part"
    )

    if temp_path.exists():

        temp_path.unlink()

    # --------------------------------------------------------
    # Deterministic CSV representation.
    #
    # UTF-8, Unix newline and no dataframe index.
    # --------------------------------------------------------

    dataframe.to_csv(
        temp_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if not temp_path.is_file():

        raise RuntimeError(
            "Temporary CSV was not created: "
            f"{temp_path}"
        )

    file_sha256 = calculate_sha256(
        temp_path
    )

    os.replace(
        temp_path,
        final_path,
    )

    # --------------------------------------------------------
    # Verify atomic promotion did not change bytes.
    # --------------------------------------------------------

    promoted_sha256 = calculate_sha256(
        final_path
    )

    if (
        promoted_sha256
        != file_sha256
    ):

        try:
            final_path.unlink()
        except OSError:
            pass

        raise RuntimeError(
            "CSV SHA-256 changed after promotion: "
            f"{final_path}"
        )

    return promoted_sha256


def export_frozen_split_artifacts(
    split_cards_df: pd.DataFrame,
    project_train_manifest_df: pd.DataFrame,
    dev_val_manifest_df: pd.DataFrame,
    discovery_sha256: str,
    config: dict,
    log_path: Path,
) -> dict:

    log_entries = [
        (
            "********************"
            "export_frozen_split_artifacts"
            "********************"
        )
    ]

    # --------------------------------------------------------
    # Output configuration.
    # --------------------------------------------------------

    output_config = config.get(
        "output"
    )

    if not isinstance(
        output_config,
        dict,
    ):

        raise RuntimeError(
            "buildconfig.yaml is missing "
            "the output section"
        )

    output_directory = resolve_config_path(
        output_config[
            "directory"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = output_config.get(
        "prefix"
    )

    if (
        not isinstance(
            prefix,
            str,
        )
        or
        not prefix.strip()
    ):

        raise RuntimeError(
            "output.prefix must be "
            "a non-empty string"
        )

    prefix = prefix.strip()

    # --------------------------------------------------------
    # Validate frozen output schemas before writing.
    # --------------------------------------------------------

    for (
        dataframe,
        required_columns,
        artifact_name,
    ) in (
        (
            split_cards_df,
            CARD_SPLIT_COLUMNS,
            "card split",
        ),
        (
            project_train_manifest_df,
            IMAGE_MANIFEST_COLUMNS,
            "project_train manifest",
        ),
        (
            dev_val_manifest_df,
            IMAGE_MANIFEST_COLUMNS,
            "dev_val manifest",
        ),
    ):

        missing_columns = (
            set(
                required_columns
            )
            - set(
                dataframe.columns
            )
        )

        if missing_columns:

            raise RuntimeError(
                f"{artifact_name} missing "
                "required columns: "
                f"{sorted(missing_columns)}"
            )

    # --------------------------------------------------------
    # Build minimal frozen copies.
    #
    # This intentionally avoids carrying all 53 discovery
    # columns into model-facing manifests.
    # --------------------------------------------------------

    cards_export_df = (
        split_cards_df[
            CARD_SPLIT_COLUMNS
        ]
        .sort_values(
            "file_stem",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    project_train_export_df = (
        project_train_manifest_df[
            IMAGE_MANIFEST_COLUMNS
        ]
        .sort_values(
            [
                "file_stem",
                "traffic_type",
                "variant",
                "hardware_source",
                "image_path",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    dev_val_export_df = (
        dev_val_manifest_df[
            IMAGE_MANIFEST_COLUMNS
        ]
        .sort_values(
            [
                "file_stem",
                "traffic_type",
                "variant",
                "hardware_source",
                "image_path",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Final pre-export row contracts.
    # --------------------------------------------------------

    expected_counts = {
        "cards":
            EXPECTED_SOURCE_TRAIN_CARDS,

        "project_train":
            EXPECTED_PROJECT_TRAIN_ROWS,

        "dev_val":
            EXPECTED_DEV_IMAGE_ROWS,
    }

    actual_counts = {
        "cards":
            len(
                cards_export_df
            ),

        "project_train":
            len(
                project_train_export_df
            ),

        "dev_val":
            len(
                dev_val_export_df
            ),
    }

    if (
        actual_counts
        != expected_counts
    ):

        raise RuntimeError(
            "Frozen split export counts "
            "do not reconcile: "
            f"{actual_counts}"
        )

    # --------------------------------------------------------
    # Output filenames share this run's immutable timestamp.
    # --------------------------------------------------------

    cards_path = (
        output_directory
        / (
            f"{prefix}_"
            f"{RUN_TIMESTAMP}_cards.csv"
        )
    )

    project_train_path = (
        output_directory
        / (
            f"{prefix}_"
            f"{RUN_TIMESTAMP}_project_train.csv"
        )
    )

    dev_val_path = (
        output_directory
        / (
            f"{prefix}_"
            f"{RUN_TIMESTAMP}_dev_val.csv"
        )
    )

    bundle_path = (
        output_directory
        / (
            f"{prefix}_"
            f"{RUN_TIMESTAMP}_bundle.yaml"
        )
    )

    # --------------------------------------------------------
    # Never begin if any final artifact already exists.
    # --------------------------------------------------------

    for final_path in (
        cards_path,
        project_train_path,
        dev_val_path,
        bundle_path,
    ):

        if final_path.exists():

            raise FileExistsError(
                "Final split artifact already exists: "
                f"{final_path}"
            )

    # --------------------------------------------------------
    # Write and hash CSV artifacts.
    # --------------------------------------------------------

    created_files = []

    try:

        cards_sha256 = write_atomic_csv(
            cards_export_df,
            cards_path,
        )

        created_files.append(
            cards_path
        )

        project_train_sha256 = (
            write_atomic_csv(
                project_train_export_df,
                project_train_path,
            )
        )

        created_files.append(
            project_train_path
        )

        dev_val_sha256 = (
            write_atomic_csv(
                dev_val_export_df,
                dev_val_path,
            )
        )

        created_files.append(
            dev_val_path
        )

        # ----------------------------------------------------
        # Provenance of the program/config that produced them.
        # ----------------------------------------------------

        script_sha256 = calculate_sha256(
            Path(
                __file__
            ).resolve()
        )

        build_config_sha256 = (
            calculate_sha256(
                CONFIG_FILE.resolve()
            )
        )

        selection_seed = (
            config[
                "split_policy"
            ][
                "selection_seed"
            ]
        )

        # ----------------------------------------------------
        # Machine-readable bundle manifest.
        # ----------------------------------------------------

        bundle = {
            "artifact_type":
                "FantasyID project split",

            "artifact_version":
                1,

            "run_timestamp":
                RUN_TIMESTAMP,

            "source_discovery": {
                "workbook":
                    DISCOVERY_WORKBOOK.name,

                "workbook_sha256":
                    discovery_sha256,
            },

            "build_provenance": {
                "script":
                    Path(
                        __file__
                    ).name,

                "script_sha256":
                    script_sha256,

                "config":
                    CONFIG_FILE.name,

                "config_sha256":
                    build_config_sha256,

                "selection_seed":
                    selection_seed,

                "selection_method":
                    (
                        "exact marginal quotas + "
                        "minimum joint-cell L1 deviation + "
                        "SHA-256 within-cell ranking"
                    ),
            },

            "counts": {
                "source_train_cards":
                    EXPECTED_SOURCE_TRAIN_CARDS,

                "source_train_images":
                    EXPECTED_SOURCE_TRAIN_ROWS,

                "project_train_cards":
                    EXPECTED_PROJECT_TRAIN_CARDS,

                "project_train_images":
                    EXPECTED_PROJECT_TRAIN_ROWS,

                "dev_val_cards":
                    DEV_CARD_TARGET,

                "dev_val_images":
                    EXPECTED_DEV_IMAGE_ROWS,
            },

            "artifacts": {
                "cards": {
                    "filename":
                        cards_path.name,

                    "sha256":
                        cards_sha256,

                    "rows":
                        len(
                            cards_export_df
                        ),
                },

                "project_train": {
                    "filename":
                        project_train_path.name,

                    "sha256":
                        project_train_sha256,

                    "rows":
                        len(
                            project_train_export_df
                        ),
                },

                "dev_val": {
                    "filename":
                        dev_val_path.name,

                    "sha256":
                        dev_val_sha256,

                    "rows":
                        len(
                            dev_val_export_df
                        ),
                },
            },
        }

        # ----------------------------------------------------
        # Write bundle YAML through a temporary file.
        # ----------------------------------------------------

        temp_bundle_path = Path(
            str(
                bundle_path
            )
            + ".part"
        )

        if temp_bundle_path.exists():

            temp_bundle_path.unlink()

        with temp_bundle_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:

            yaml.safe_dump(
                bundle,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        bundle_sha256 = (
            calculate_sha256(
                temp_bundle_path
            )
        )

        os.replace(
            temp_bundle_path,
            bundle_path,
        )

        created_files.append(
            bundle_path
        )

        if (
            calculate_sha256(
                bundle_path
            )
            != bundle_sha256
        ):

            raise RuntimeError(
                "Bundle YAML hash changed "
                "after promotion"
            )

    except Exception:

        # ----------------------------------------------------
        # Do not leave a partial frozen bundle behind.
        # ----------------------------------------------------

        for created_path in (
            created_files
        ):

            if created_path.exists():

                try:
                    created_path.unlink()
                except OSError:
                    pass

        raise

    # --------------------------------------------------------
    # Final log evidence.
    # --------------------------------------------------------

    log_entries.append(
        "Frozen split artifacts written:"
    )

    log_entries.append(
        f"  cards: "
        f"{cards_path.name}"
    )

    log_entries.append(
        f"    rows: "
        f"{len(cards_export_df)}"
    )

    log_entries.append(
        f"    SHA-256: "
        f"{cards_sha256}"
    )

    log_entries.append(
        f"  project_train: "
        f"{project_train_path.name}"
    )

    log_entries.append(
        f"    rows: "
        f"{len(project_train_export_df)}"
    )

    log_entries.append(
        f"    SHA-256: "
        f"{project_train_sha256}"
    )

    log_entries.append(
        f"  dev_val: "
        f"{dev_val_path.name}"
    )

    log_entries.append(
        f"    rows: "
        f"{len(dev_val_export_df)}"
    )

    log_entries.append(
        f"    SHA-256: "
        f"{dev_val_sha256}"
    )

    log_entries.append(
        f"  bundle: "
        f"{bundle_path.name}"
    )

    log_entries.append(
        f"    SHA-256: "
        f"{bundle_sha256}"
    )

    log_entries.append(
        "FROZEN SPLIT EXPORT: PASS"
    )

    write_log(
        log_path,
        log_entries,
    )

    return {
        "cards_path":
            cards_path,

        "cards_sha256":
            cards_sha256,

        "project_train_path":
            project_train_path,

        "project_train_sha256":
            project_train_sha256,

        "dev_val_path":
            dev_val_path,

        "dev_val_sha256":
            dev_val_sha256,

        "bundle_path":
            bundle_path,

        "bundle_sha256":
            bundle_sha256,
    }

# ------------------------------------------------------------
#Function Independent read-back verification of frozen split artifacts.
#
# This is the final release gate.
#
# We reopen the files that were actually written to disk and
# independently verify:
#
#   - bundle YAML structure and provenance
#   - bundle/file SHA-256 values
#   - exact output schemas
#   - 211 card rows
#   - 160 / 51 card assignment
#   - 1440 / 459 image assignment
#   - complete 9-image cards
#   - train/dev card disjointness
#   - train/dev image-hash disjointness
#   - expected traffic/variant composition
#   - expected three-hardware structure per card/version
#
# No split decisions are made here.
# ------------------------------------------------------------

def verify_frozen_split_artifacts(
    split_artifacts: dict,
    discovery_sha256: str,
    config: dict,
    log_path: Path,
) -> None:

    log_entries = [
        (
            "********************"
            "verify_frozen_split_artifacts"
            "********************"
        )
    ]

    try:

        # --------------------------------------------------------
        # Recover artifact paths produced by the exporter.
        # --------------------------------------------------------

        cards_path = Path(
            split_artifacts[
                "cards_path"
            ]
        )

        project_train_path = Path(
            split_artifacts[
                "project_train_path"
            ]
        )

        dev_val_path = Path(
            split_artifacts[
                "dev_val_path"
            ]
        )

        bundle_path = Path(
            split_artifacts[
                "bundle_path"
            ]
        )

        artifact_paths = {
            "cards":
                cards_path,

            "project_train":
                project_train_path,

            "dev_val":
                dev_val_path,

            "bundle":
                bundle_path,
        }

        # --------------------------------------------------------
        # Every final file must physically exist.
        # --------------------------------------------------------

        for (
            artifact_name,
            artifact_path,
        ) in artifact_paths.items():

            if not artifact_path.is_file():

                raise FileNotFoundError(
                    f"Frozen {artifact_name} artifact "
                    f"not found: {artifact_path}"
                )

        # ========================================================
        # 1. RE-HASH EVERY FINAL FILE FROM DISK
        # ========================================================

        actual_cards_sha256 = (
            calculate_sha256(
                cards_path
            )
        )

        actual_project_train_sha256 = (
            calculate_sha256(
                project_train_path
            )
        )

        actual_dev_val_sha256 = (
            calculate_sha256(
                dev_val_path
            )
        )

        actual_bundle_sha256 = (
            calculate_sha256(
                bundle_path
            )
        )

        actual_hashes = {
            "cards":
                actual_cards_sha256,

            "project_train":
                actual_project_train_sha256,

            "dev_val":
                actual_dev_val_sha256,

            "bundle":
                actual_bundle_sha256,
        }

        expected_export_hashes = {
            "cards":
                split_artifacts[
                    "cards_sha256"
                ],

            "project_train":
                split_artifacts[
                    "project_train_sha256"
                ],

            "dev_val":
                split_artifacts[
                    "dev_val_sha256"
                ],

            "bundle":
                split_artifacts[
                    "bundle_sha256"
                ],
        }

        log_entries.append(
            "Disk SHA-256 verification:"
        )

        for artifact_name in (
            "cards",
            "project_train",
            "dev_val",
            "bundle",
        ):

            actual = actual_hashes[
                artifact_name
            ]

            expected = expected_export_hashes[
                artifact_name
            ]

            log_entries.append(
                f"  {artifact_name}: "
                f"{actual}"
            )

            if actual != expected:

                raise RuntimeError(
                    f"{artifact_name} SHA-256 "
                    "does not match exporter result"
                )

        # ========================================================
        # 2. READ BUNDLE YAML FROM DISK
        # ========================================================

        with bundle_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            bundle = yaml.safe_load(
                file
            )

        if not isinstance(
            bundle,
            dict,
        ):

            raise RuntimeError(
                "Frozen bundle YAML is not "
                "a dictionary"
            )

        if (
            bundle.get(
                "artifact_type"
            )
            !=
            "FantasyID project split"
        ):

            raise RuntimeError(
                "Unexpected bundle artifact_type"
            )

        if (
            bundle.get(
                "artifact_version"
            )
            != 1
        ):

            raise RuntimeError(
                "Unexpected split artifact version"
            )

        if (
            bundle.get(
                "run_timestamp"
            )
            != RUN_TIMESTAMP
        ):

            raise RuntimeError(
                "Bundle run timestamp does not "
                "match current split-build run"
            )

        # ========================================================
        # 3. VERIFY FROZEN DISCOVERY LINEAGE
        # ========================================================

        source_discovery = bundle.get(
            "source_discovery"
        )

        if not isinstance(
            source_discovery,
            dict,
        ):

            raise RuntimeError(
                "Bundle source_discovery section "
                "is missing or invalid"
            )

        if (
            source_discovery.get(
                "workbook"
            )
            != DISCOVERY_WORKBOOK.name
        ):

            raise RuntimeError(
                "Bundle refers to an unexpected "
                "discovery workbook"
            )

        if (
            source_discovery.get(
                "workbook_sha256"
            )
            != discovery_sha256
        ):

            raise RuntimeError(
                "Bundle discovery SHA-256 "
                "does not match verified input"
            )

        log_entries.append(
            "Discovery lineage:"
        )

        log_entries.append(
            f"  workbook: "
            f"{DISCOVERY_WORKBOOK.name}"
        )

        log_entries.append(
            f"  SHA-256: "
            f"{discovery_sha256}"
        )

        # ========================================================
        # 4. VERIFY BUILD PROVENANCE
        # ========================================================

        build_provenance = bundle.get(
            "build_provenance"
        )

        if not isinstance(
            build_provenance,
            dict,
        ):

            raise RuntimeError(
                "Bundle build_provenance section "
                "is missing or invalid"
            )

        current_script_sha256 = (
            calculate_sha256(
                Path(
                    __file__
                ).resolve()
            )
        )

        current_config_sha256 = (
            calculate_sha256(
                CONFIG_FILE.resolve()
            )
        )

        if (
            build_provenance.get(
                "script_sha256"
            )
            != current_script_sha256
        ):

            raise RuntimeError(
                "Bundle script SHA-256 does not "
                "match current build script"
            )

        if (
            build_provenance.get(
                "config_sha256"
            )
            != current_config_sha256
        ):

            raise RuntimeError(
                "Bundle config SHA-256 does not "
                "match current build config"
            )

        expected_seed = (
            config[
                "split_policy"
            ][
                "selection_seed"
            ]
        )

        if (
            build_provenance.get(
                "selection_seed"
            )
            != expected_seed
        ):

            raise RuntimeError(
                "Bundle split seed does not "
                "match build configuration"
            )

        log_entries.append(
            "Build provenance:"
        )

        log_entries.append(
            f"  script SHA-256: "
            f"{current_script_sha256}"
        )

        log_entries.append(
            f"  config SHA-256: "
            f"{current_config_sha256}"
        )

        log_entries.append(
            f"  selection seed: "
            f"{expected_seed}"
        )

        # ========================================================
        # 5. VERIFY BUNDLE ARTIFACT RECORDS
        # ========================================================

        bundle_artifacts = bundle.get(
            "artifacts"
        )

        if not isinstance(
            bundle_artifacts,
            dict,
        ):

            raise RuntimeError(
                "Bundle artifacts section "
                "is missing or invalid"
            )

        expected_artifact_info = {
            "cards": {
                "path":
                    cards_path,

                "sha256":
                    actual_cards_sha256,

                "rows":
                    EXPECTED_SOURCE_TRAIN_CARDS,
            },

            "project_train": {
                "path":
                    project_train_path,

                "sha256":
                    actual_project_train_sha256,

                "rows":
                    EXPECTED_PROJECT_TRAIN_ROWS,
            },

            "dev_val": {
                "path":
                    dev_val_path,

                "sha256":
                    actual_dev_val_sha256,

                "rows":
                    EXPECTED_DEV_IMAGE_ROWS,
            },
        }

        for (
            artifact_name,
            expected_info,
        ) in expected_artifact_info.items():

            bundle_info = (
                bundle_artifacts.get(
                    artifact_name
                )
            )

            if not isinstance(
                bundle_info,
                dict,
            ):

                raise RuntimeError(
                    "Bundle missing artifact entry: "
                    f"{artifact_name}"
                )

            if (
                bundle_info.get(
                    "filename"
                )
                != expected_info[
                    "path"
                ].name
            ):

                raise RuntimeError(
                    "Bundle filename mismatch for "
                    f"{artifact_name}"
                )

            if (
                bundle_info.get(
                    "sha256"
                )
                != expected_info[
                    "sha256"
                ]
            ):

                raise RuntimeError(
                    "Bundle SHA-256 mismatch for "
                    f"{artifact_name}"
                )

            if (
                bundle_info.get(
                    "rows"
                )
                != expected_info[
                    "rows"
                ]
            ):

                raise RuntimeError(
                    "Bundle row-count mismatch for "
                    f"{artifact_name}"
                )

        # ========================================================
        # 6. READ THE THREE CSV FILES FROM DISK
        #
        # keep_default_na=False is important because bona-fide
        # variant="" must remain an empty string.
        # ========================================================

        cards_df = pd.read_csv(
            cards_path,
            keep_default_na=False,
        )

        project_train_df = pd.read_csv(
            project_train_path,
            keep_default_na=False,
        )

        dev_val_df = pd.read_csv(
            dev_val_path,
            keep_default_na=False,
        )

        log_entries.append(
            "Read-back CSV rows:"
        )

        log_entries.append(
            f"  cards: "
            f"{len(cards_df)}"
        )

        log_entries.append(
            f"  project_train: "
            f"{len(project_train_df)}"
        )

        log_entries.append(
            f"  dev_val: "
            f"{len(dev_val_df)}"
        )

        # ========================================================
        # 7. EXACT OUTPUT SCHEMA
        # ========================================================

        if (
            list(
                cards_df.columns
            )
            != CARD_SPLIT_COLUMNS
        ):

            raise RuntimeError(
                "Read-back card CSV schema "
                "differs from frozen schema"
            )

        if (
            list(
                project_train_df.columns
            )
            != IMAGE_MANIFEST_COLUMNS
        ):

            raise RuntimeError(
                "Read-back project_train CSV schema "
                "differs from frozen schema"
            )

        if (
            list(
                dev_val_df.columns
            )
            != IMAGE_MANIFEST_COLUMNS
        ):

            raise RuntimeError(
                "Read-back dev_val CSV schema "
                "differs from frozen schema"
            )

        # ========================================================
        # 8. EXACT ROW COUNTS
        # ========================================================

        if (
            len(cards_df)
            != EXPECTED_SOURCE_TRAIN_CARDS
        ):

            raise RuntimeError(
                "Read-back card CSV does not "
                "contain exactly 211 rows"
            )

        if (
            len(project_train_df)
            != EXPECTED_PROJECT_TRAIN_ROWS
        ):

            raise RuntimeError(
                "Read-back project_train CSV does not "
                "contain exactly 1440 rows"
            )

        if (
            len(dev_val_df)
            != EXPECTED_DEV_IMAGE_ROWS
        ):

            raise RuntimeError(
                "Read-back dev_val CSV does not "
                "contain exactly 459 rows"
            )

        # ========================================================
        # 9. CARD TABLE STRUCTURE
        # ========================================================

        if (
            cards_df[
                "file_stem"
            ]
            .nunique()
            != EXPECTED_SOURCE_TRAIN_CARDS
        ):

            raise RuntimeError(
                "Read-back card CSV does not "
                "contain 211 unique file_stems"
            )

        if (
            cards_df[
                "file_stem"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                "Read-back card CSV contains "
                "duplicate file_stems"
            )

        card_role_counts = (
            cards_df[
                "project_role"
            ]
            .value_counts()
            .to_dict()
        )

        expected_card_role_counts = {
            "project_train":
                EXPECTED_PROJECT_TRAIN_CARDS,

            "dev_val":
                DEV_CARD_TARGET,
        }

        log_entries.append(
            "Read-back card roles:"
        )

        log_entries.append(
            f"  project_train: "
            f"{card_role_counts.get('project_train', 0)}"
        )

        log_entries.append(
            f"  dev_val: "
            f"{card_role_counts.get('dev_val', 0)}"
        )

        if (
            card_role_counts
            != expected_card_role_counts
        ):

            raise RuntimeError(
                "Read-back card-role counts "
                "are incorrect"
            )

        if not (
            cards_df[
                "source_image_count"
            ]
            .eq(
                EXPECTED_IMAGES_PER_TRAIN_CARD
            )
            .all()
        ):

            raise RuntimeError(
                "Read-back card CSV contains "
                "a card with source_image_count != 9"
            )

        # ========================================================
        # 10. MANIFEST ROLE VALUES
        # ========================================================

        if not (
            project_train_df[
                "project_role"
            ]
            .eq(
                "project_train"
            )
            .all()
        ):

            raise RuntimeError(
                "project_train CSV contains "
                "a non-project_train role"
            )

        if not (
            dev_val_df[
                "project_role"
            ]
            .eq(
                "dev_val"
            )
            .all()
        ):

            raise RuntimeError(
                "dev_val CSV contains "
                "a non-dev_val role"
            )

        # ========================================================
        # 11. MANIFEST STEMS MUST MATCH CARD TABLE ASSIGNMENT
        # ========================================================

        card_role_lookup = dict(
            zip(
                cards_df[
                    "file_stem"
                ],
                cards_df[
                    "project_role"
                ],
            )
        )

        for (
            manifest_name,
            manifest_df,
            expected_role,
        ) in (
            (
                "project_train",
                project_train_df,
                "project_train",
            ),
            (
                "dev_val",
                dev_val_df,
                "dev_val",
            ),
        ):

            assigned_roles = (
                manifest_df[
                    "file_stem"
                ]
                .map(
                    card_role_lookup
                )
            )

            if (
                assigned_roles
                .isna()
                .any()
            ):

                raise RuntimeError(
                    f"{manifest_name} contains "
                    "a file_stem absent from card CSV"
                )

            if not (
                assigned_roles
                .eq(
                    expected_role
                )
                .all()
            ):

                raise RuntimeError(
                    f"{manifest_name} contains "
                    "a card assigned to the wrong role"
                )

        # ========================================================
        # 12. CARD COUNTS AND DISJOINTNESS
        # ========================================================

        project_train_stems = set(
            project_train_df[
                "file_stem"
            ]
        )

        dev_val_stems = set(
            dev_val_df[
                "file_stem"
            ]
        )

        if (
            len(
                project_train_stems
            )
            != EXPECTED_PROJECT_TRAIN_CARDS
        ):

            raise RuntimeError(
                "Read-back project_train does not "
                "contain 160 unique cards"
            )

        if (
            len(
                dev_val_stems
            )
            != DEV_CARD_TARGET
        ):

            raise RuntimeError(
                "Read-back dev_val does not "
                "contain 51 unique cards"
            )

        card_overlap = (
            project_train_stems
            & dev_val_stems
        )

        log_entries.append(
            "Read-back train/dev card overlap: "
            f"{len(card_overlap)}"
        )

        if card_overlap:

            raise RuntimeError(
                "Read-back manifests are "
                "not card-disjoint"
            )

        all_manifest_stems = (
            project_train_stems
            | dev_val_stems
        )

        card_table_stems = set(
            cards_df[
                "file_stem"
            ]
        )

        if (
            all_manifest_stems
            != card_table_stems
        ):

            raise RuntimeError(
                "Manifest-card union does not "
                "equal the 211-row card table"
            )

        # ========================================================
        # 13. EVERY CARD MUST STILL HAVE EXACTLY NINE IMAGES
        # ========================================================

        for (
            manifest_name,
            manifest_df,
        ) in (
            (
                "project_train",
                project_train_df,
            ),
            (
                "dev_val",
                dev_val_df,
            ),
        ):

            card_sizes = (
                manifest_df
                .groupby(
                    "file_stem"
                )
                .size()
            )

            bad_card_sizes = int(
                card_sizes
                .ne(
                    EXPECTED_IMAGES_PER_TRAIN_CARD
                )
                .sum()
            )

            log_entries.append(
                f"{manifest_name} cards "
                "with image count != 9: "
                f"{bad_card_sizes}"
            )

            if bad_card_sizes:

                raise RuntimeError(
                    f"{manifest_name} contains "
                    "an incomplete card"
                )

        # ========================================================
        # 14. IMAGE PATH/HASH UNIQUENESS AND DISJOINTNESS
        # ========================================================

        combined_images_df = pd.concat(
            [
                project_train_df,
                dev_val_df,
            ],
            ignore_index=True,
        )

        if (
            len(
                combined_images_df
            )
            != EXPECTED_SOURCE_TRAIN_ROWS
        ):

            raise RuntimeError(
                "Read-back image manifests do not "
                "reconcile to 1899 rows"
            )

        duplicate_paths = int(
            combined_images_df[
                "image_path"
            ]
            .duplicated()
            .sum()
        )

        duplicate_hashes = int(
            combined_images_df[
                "image_sha256"
            ]
            .duplicated()
            .sum()
        )

        log_entries.append(
            "Read-back duplicate image paths: "
            f"{duplicate_paths}"
        )

        log_entries.append(
            "Read-back duplicate image hashes: "
            f"{duplicate_hashes}"
        )

        if (
            duplicate_paths
            or duplicate_hashes
        ):

            raise RuntimeError(
                "Read-back manifests contain "
                "duplicate source images"
            )

        project_train_hashes = set(
            project_train_df[
                "image_sha256"
            ]
        )

        dev_val_hashes = set(
            dev_val_df[
                "image_sha256"
            ]
        )

        image_overlap = (
            project_train_hashes
            & dev_val_hashes
        )

        log_entries.append(
            "Read-back train/dev image SHA-256 overlap: "
            f"{len(image_overlap)}"
        )

        if image_overlap:

            raise RuntimeError(
                "Read-back train/dev manifests "
                "share source-image hashes"
            )

        # ========================================================
        # 15. TRAFFIC / VARIANT COMPOSITION
        # ========================================================

        for (
            manifest_name,
            manifest_df,
            expected_cards,
        ) in (
            (
                "project_train",
                project_train_df,
                EXPECTED_PROJECT_TRAIN_CARDS,
            ),
            (
                "dev_val",
                dev_val_df,
                DEV_CARD_TARGET,
            ),
        ):

            actual_composition = (
                manifest_df
                .groupby(
                    [
                        "traffic_type",
                        "variant",
                    ]
                )
                .size()
                .to_dict()
            )

            expected_composition = {
                (
                    "bonafide",
                    "",
                ):
                    expected_cards * 3,

                (
                    "attack",
                    "digital_1",
                ):
                    expected_cards * 3,

                (
                    "attack",
                    "digital_2",
                ):
                    expected_cards * 3,
            }

            log_entries.append(
                f"Read-back {manifest_name} composition:"
            )

            for (
                traffic_type,
                variant,
            ), count in sorted(
                actual_composition.items()
            ):

                variant_label = (
                    variant
                    if variant
                    else "(empty)"
                )

                log_entries.append(
                    f"  {traffic_type} / "
                    f"{variant_label}: "
                    f"{int(count)}"
                )

            if (
                actual_composition
                != expected_composition
            ):

                raise RuntimeError(
                    f"Read-back {manifest_name} "
                    "composition is incorrect"
                )

        # ========================================================
        # 16. THREE-HARDWARE STRUCTURE PER CARD/VERSION
        # ========================================================

        expected_hardware = {
            "huawei",
            "iphone15pro",
            "scan",
        }

        bad_hardware_groups = []

        for (
            manifest_name,
            manifest_df,
        ) in (
            (
                "project_train",
                project_train_df,
            ),
            (
                "dev_val",
                dev_val_df,
            ),
        ):

            for (
                file_stem,
                traffic_type,
                variant,
            ), group_df in (
                manifest_df
                .groupby(
                    [
                        "file_stem",
                        "traffic_type",
                        "variant",
                    ],
                    dropna=False,
                )
            ):

                hardware = set(
                    group_df[
                        "hardware_source"
                    ]
                )

                if (
                    len(
                        group_df
                    )
                    != 3
                    or
                    hardware
                    != expected_hardware
                ):

                    bad_hardware_groups.append(
                        (
                            manifest_name,
                            file_stem,
                            traffic_type,
                            variant,
                            len(
                                group_df
                            ),
                            sorted(
                                hardware
                            ),
                        )
                    )

        log_entries.append(
            "Read-back card/version hardware "
            "mismatches: "
            f"{len(bad_hardware_groups)}"
        )

        if bad_hardware_groups:

            for problem in (
                bad_hardware_groups[
                    :10
                ]
            ):

                log_entries.append(
                    f"  {problem}"
                )

            raise RuntimeError(
                "Read-back manifests contain "
                "unexpected hardware coverage"
            )

        # ========================================================
        # 17. BUNDLE COUNT RECORD
        # ========================================================

        bundle_counts = bundle.get(
            "counts"
        )

        expected_bundle_counts = {
            "source_train_cards":
                EXPECTED_SOURCE_TRAIN_CARDS,

            "source_train_images":
                EXPECTED_SOURCE_TRAIN_ROWS,

            "project_train_cards":
                EXPECTED_PROJECT_TRAIN_CARDS,

            "project_train_images":
                EXPECTED_PROJECT_TRAIN_ROWS,

            "dev_val_cards":
                DEV_CARD_TARGET,

            "dev_val_images":
                EXPECTED_DEV_IMAGE_ROWS,
        }

        if (
            bundle_counts
            != expected_bundle_counts
        ):

            raise RuntimeError(
                "Bundle count metadata does not "
                "match frozen split contract"
            )

        # ========================================================
        # FINAL RELEASE RESULT
        # ========================================================

        log_entries.append(
            "FROZEN SPLIT READ-BACK VERIFICATION: PASS"
        )

        log_entries.append(
            (
                "  Disk artifacts independently "
                "reconcile to 211 source-train cards, "
                "160 project_train cards / 1440 images, "
                "and 51 dev_val cards / 459 images."
            )
        )

        write_log(
            log_path,
            log_entries,
        )

    except Exception as error:

        log_entries.append(
            "FROZEN SPLIT READ-BACK VERIFICATION: FAIL"
        )

        log_entries.append(
            (
                "  "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        write_log(
            log_path,
            log_entries,
        )

        raise

#Main function
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load split-construction configuration and establish this
    # run's unique log before doing any dataset processing.
    # ------------------------------------------------------------

    config = load_build_config(
        CONFIG_FILE
    )

    final_log_path = build_log_path(
        config
    )

    write_log(
        final_log_path,
        [
            (
                "********************"
                "PROJECT SPLIT BUILD"
                "********************"
            ),
            (
                f"Run timestamp: "
                f"{RUN_TIMESTAMP}"
            ),
            (
                f"Build config: "
                f"{CONFIG_FILE.name}"
            ),
            (
                f"Discovery workbook: "
                f"{DISCOVERY_WORKBOOK.name}"
            ),
        ],
    )


    # ------------------------------------------------------------
    # The frozen workbook must pass its SHA-256 verification
    # before any dataset rows are consumed.
    # ------------------------------------------------------------

    discovery_sha256 = (
        verify_frozen_discovery_artifact(
            DISCOVERY_WORKBOOK,
            DISCOVERY_SIDECAR,
            final_log_path,
        )
    )


    # ------------------------------------------------------------
    # Only after verification do we load the Images sheet.
    # ------------------------------------------------------------

    images_df = load_discovery_images(
        DISCOVERY_WORKBOOK,
        final_log_path,
    )

    # ------------------------------------------------------------
    # Validation of source train 211 cards.
    # ------------------------------------------------------------

    source_train_df, train_cards_df = (
        validate_source_train_structure(
            images_df,
            final_log_path,
        )
    )

    # ------------------------------------------------------------
    # audit of source train 211 cards.
    # ------------------------------------------------------------

    train_card_strata_df = (
        audit_train_card_strata(
            train_cards_df,
            final_log_path,
        )
    )

    regions_audit_df = (
        load_discovery_regions_for_audit(
            DISCOVERY_WORKBOOK,
            final_log_path,
        )
    )

    train_card_language_df = (
        audit_train_language_structure(
            regions_audit_df,
            train_cards_df,
            final_log_path,
        )
    )

    train_card_template_audit_df = (
        audit_train_stem_family_structure(
            train_card_language_df,
            final_log_path,
        )
    )

    dev_quota_tables = (
        audit_dev_quota_targets(
            train_card_template_audit_df,
            final_log_path,
        )
    )

    joint_split_support_df = (
        build_joint_split_support_table(
            train_card_template_audit_df,
            final_log_path,
        )
    )

    feasibility_witness_df = (
        audit_exact_dev_feasibility(
            joint_split_support_df,
            dev_quota_tables,
            final_log_path,
        )
    )

    optimized_joint_allocation_df = (
        optimize_joint_dev_allocation(
            joint_split_support_df,
            dev_quota_tables,
            feasibility_witness_df,
            final_log_path,
        )
    )

    split_cards_df = (
        select_dev_cards_deterministically(
            train_card_template_audit_df,
            optimized_joint_allocation_df,
            dev_quota_tables,
            discovery_sha256,
            config,
            final_log_path,
        )
    )

    project_split_df, project_train_manifest_df, dev_val_manifest_df = (
        build_image_level_split_manifests(
            source_train_df,
            split_cards_df,
            final_log_path,
        )
    )

    split_artifacts = (
        export_frozen_split_artifacts(
            split_cards_df,
            project_train_manifest_df,
            dev_val_manifest_df,
            discovery_sha256,
            config,
            final_log_path,
        )
    )

    verify_frozen_split_artifacts(
        split_artifacts,
        discovery_sha256,
        config,
        final_log_path,
    )