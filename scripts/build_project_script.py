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
                "gender",
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