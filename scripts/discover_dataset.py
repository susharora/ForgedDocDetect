#purpose: scan through directory containing image dataset and discover:
#   * data split , traffic type, attack variant, hardware source 
# also match images to their json descriptive files and validation transformation
# information available in JSON against recorded trnsformed height width. 

#Block: Imports
from pathlib import Path 
from datetime import datetime
from collections import Counter
from collections import defaultdict
import hashlib 
import json
from PIL import Image
import pandas as pd
import os
import math
import warnings


import yaml
import inspect

#Block: Global constants

CONFIG_FILE = Path("dataconfig.yaml")

# One identifier for the entire execution.
# The log, workbook and workbook provenance all use the same
# timestamp, so they can be associated unambiguously with one run.
RUN_TIMESTAMP = datetime.now().strftime(
    "%Y-%m-%d_%H%M%S"
)

ORIENTATION_LABELS = {
    None: "no EXIF orientation tag",
    1: "normal",
    2: "mirrored horizontally",
    3: "rotated 180",
    4: "mirrored vertically",
    5: "mirrored horizontally then rotated 270 CW",
    6: "rotated 90 CW",
    7: "mirrored horizontally then rotated 90 CW",
    8: "rotated 270 CW",
}

ZONE_IDENTIFIER_SUFFIX = ":Zone.Identifier"

#Block: Function definitions

#function to load config file as a dictionary of key value pairs
def load_config(config_path: Path) -> dict:
    func_name = inspect.currentframe().f_code.co_name
    #load the yaml file as dictionary     
    with config_path.open("r",encoding="utf-8") as file:
        config = yaml.safe_load(file) # yaml safe load function returns a dict as dataconfig file is key values pairs

    #debug print("config-",config) 

    dataset_config = config["dataset"]
    root = Path(dataset_config["root"])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    # Build the log path using the single process-wide run timestamp.
    log_path = build_log_path(config)

    
    

    #debug print (f"{log_file_path} , {type(config)}")
    
    #add log entry
    log_entries = [f"********************{func_name}: ********************"]    
    write_log(log_path,log_entries)
    log_entries = [f"{func_name}: Configuration loaded successfully from {CONFIG_FILE}", 
                       f"   Dataset name:   {config['dataset']['name']}",
                        f"   Dataset root:   {config['dataset']['root']}",
                        f"   Allowed splits: {config['dataset']['allowed_splits']}"]
    
    write_log(log_path,log_entries)
    
    return config

# build log path TBD
def build_log_path(
    config: dict
) -> Path:

    base_log = Path(
        config[
            "dataset"
        ][
            "log"
        ]
    )

    base_log.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        base_log.parent
        / (
            f"{base_log.stem}_"
            f"{RUN_TIMESTAMP}"
            f"{base_log.suffix}"
        )
    )

#function to write logs for data ingest

def write_log(log_path: Path, log_entries: list[str]) -> None:
    full_log_output = "\n".join(log_entries) + "\n"
     
    with log_path.open("a",encoding="utf-8") as log_file:
        log_file.write(full_log_output)
    
    #debug     print(f"\n[info] Output successfully written to log file: {log_path}")

#function to traverse through dataset split defined in yaml file
def find_dataset_splits(config: dict) -> dict[str, Path]:
    func_name = inspect.currentframe().f_code.co_name
    dataset_config = config["dataset"]
    #final_log_path = build_log_path(config)
    log_entries = [f"********************{func_name}: ********************"]      
    write_log(final_log_path,log_entries)

    root = Path(dataset_config["root"])
    allowed_splits = {
        split.lower()
        for split in dataset_config["allowed_splits"]
    }
    #debug print(allowed_splits)
    found_splits = {} # dictinoary to store found splits in the folder. 

    #Using sorted builtin function, we return a list of all folders/directories
    #below implementation is only suitable when number of folders is small as its in memory sorting. 
    for item in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir():
            log_entries =[f"{func_name}: Ignoring top-level file: {item.name}"]
            write_log(final_log_path,log_entries)
            continue    # a gaurd to skip processing files which are not directories. 

        #debug print (item,type(item))    
        folder_name = item.name.lower()

        # assigning folder name to its path in the dictionary found_splits
        if folder_name in allowed_splits:
            found_splits[folder_name] = item
            log_entries = [f"{func_name}: Using dataset split: {item.name}"]
            write_log(final_log_path,log_entries)
        else:
            log_entries = [f"{func_name}: Ignoring top-level folder: {item.name}"]    
            write_log(final_log_path,log_entries)

    missing_splits = allowed_splits - set(found_splits.keys())

    if missing_splits:
        log_entries = [f"{func_name}: Configured dataset splits not found: {sorted(missing_splits)}"]  
        write_log(final_log_path,log_entries)  
        raise FileNotFoundError(
            f"Configured dataset splits not found: {sorted(missing_splits)}"
        )
    else:
        log_entries = [f"{func_name}: All Configured dataset splits found: {sorted(found_splits)}"]    
        write_log(final_log_path,log_entries)

    return found_splits

#function to traverse through traffic type as identified in YAML dict
def find_traffic_types(
    split_paths: dict[str, Path],
    config: dict
) -> dict[str, dict[str, Path]]:

    func_name = inspect.currentframe().f_code.co_name
    #final_log_path = build_log_path(config)
    log_entries = [f"********************{func_name}: ********************"]        
    write_log(final_log_path,log_entries)


    allowed_traffic_types = {
        traffic_type.lower()
        for traffic_type in config["dataset"]["allowed_traffic_types"]
    }

    traffic_paths = {}

    for split_name, split_path in split_paths.items():
        traffic_paths[split_name] = {}
        log_entries = [f"{func_name}: Inspecting traffic split: {split_name}"]    
        write_log(final_log_path,log_entries)

        for item in sorted(
            split_path.iterdir(),
            key=lambda path: path.name.lower()
        ):
            if not item.is_dir():
                log_entries = [f"{func_name}: Ignoring file: {item.name}"]    
                write_log(final_log_path,log_entries)
                continue

            folder_name = item.name.lower()

            if folder_name in allowed_traffic_types:
                traffic_paths[split_name][folder_name] = item
                log_entries = [f"{func_name}: Using traffic type: {item.name}"]    
                write_log(final_log_path,log_entries)
            else:
                log_entries = [f"{func_name}: Ignoring folder: {item.name}"]    
                write_log(final_log_path,log_entries)

    log_entries = [f"{func_name}: Directory structure found:"]    
    write_log(final_log_path,log_entries)

    for split_name, traffic_types in traffic_paths.items():
        log_entries = [f"{func_name}: {split_name}"]    
        write_log(final_log_path,log_entries)
    #    print(split_name,type(split_name))
    #    print(traffic_types,type(traffic_types))
    

        for traffic_type, path in traffic_types.items():
            log_entries = [f"{func_name}: {traffic_type}: {path}"]    
            write_log(final_log_path,log_entries)
    
    return traffic_paths

#Function to traverse traffic type. folder structure in traffic type is different hence code caters to that.

def find_data_folders(
    traffic_paths: dict[
        str,
        dict[str, Path]
    ]
) -> list[dict]:

    func_name = (
        inspect.currentframe()
        .f_code.co_name
    )

    log_entries = [
        f"********************"
        f"{func_name}: "
        f"********************"
    ]

    data_folders = []

    ignored_zone_identifiers = 0

    for (
        split_name,
        traffic_types,
    ) in traffic_paths.items():

        for (
            traffic_type,
            traffic_path,
        ) in traffic_types.items():

            # ====================================================
            # BONAFIDE:
            #
            # split / bonafide / hardware
            # ====================================================

            if traffic_type == "bonafide":

                for hardware_path in sorted(
                    traffic_path.iterdir(),
                    key=lambda path:
                        path.name.lower(),
                ):

                    if is_zone_identifier(
                        hardware_path
                    ):

                        ignored_zone_identifiers += 1
                        continue

                    if not hardware_path.is_dir():

                        log_entries.append(
                            f"{func_name}: "
                            f"Ignoring file: "
                            f"{hardware_path}"
                        )

                        continue

                    data_folders.append({
                        "split":
                            split_name,

                        "traffic_type":
                            traffic_type,

                        "variant":
                            "",

                        "hardware_source":
                            hardware_path.name,

                        "path":
                            hardware_path,
                    })

            # ====================================================
            # ATTACK:
            #
            # split / attack / variant / hardware
            # ====================================================

            elif traffic_type == "attack":

                for variant_path in sorted(
                    traffic_path.iterdir(),
                    key=lambda path:
                        path.name.lower(),
                ):

                    if is_zone_identifier(
                        variant_path
                    ):

                        ignored_zone_identifiers += 1
                        continue

                    if not variant_path.is_dir():

                        # Correct variable is variant_path.
                        log_entries.append(
                            f"{func_name}: "
                            f"Ignoring file: "
                            f"{variant_path}"
                        )

                        continue

                    for hardware_path in sorted(
                        variant_path.iterdir(),
                        key=lambda path:
                            path.name.lower(),
                    ):

                        if is_zone_identifier(
                            hardware_path
                        ):

                            ignored_zone_identifiers += 1
                            continue

                        if not hardware_path.is_dir():

                            log_entries.append(
                                f"{func_name}: "
                                f"Ignoring file: "
                                f"{hardware_path}"
                            )

                            continue

                        data_folders.append({
                            "split":
                                split_name,

                            "traffic_type":
                                traffic_type,

                            "variant":
                                variant_path.name,

                            "hardware_source":
                                hardware_path.name,

                            "path":
                                hardware_path,
                        })

    log_entries.append(
        f"{func_name}: "
        "Ignored Zone.Identifier "
        f"metadata files: "
        f"{ignored_zone_identifiers}"
    )

    log_entries.append(
        f"{func_name}: "
        "Final data folders:"
    )

    for folder in data_folders:

        log_entries.append(
            f"{func_name}:  "
            f"{folder['split']},"
            f"{folder['traffic_type']},"
            f"{folder['variant']},"
            f"{folder['hardware_source']} "
            f"----> "
            f"{folder['path']}"
        )

    write_log(
        final_log_path,
        log_entries,
    )

    return data_folders

#Function to discover all images and their JSON description files from discovered folders
def discover_files(
    data_folders: list[dict],
    config: dict,
) -> list[dict]:

    func_name = (
        inspect.currentframe()
        .f_code.co_name
    )

    log_entries = [
        f"********************"
        f"{func_name}: "
        f"********************"
    ]

    image_extensions = {
        extension.lower()
        for extension
        in config[
            "dataset"
        ][
            "image_extensions"
        ]
    }

    json_extension = (
        config[
            "dataset"
        ][
            "json_extension"
        ]
        .lower()
    )

    discovered_files = []

    ignored_zone_identifiers = 0
    ignored_unsupported_files = 0
    ignored_nested_folders = 0

    for folder in data_folders:

        folder_path = folder[
            "path"
        ]

        for file_path in sorted(
            folder_path.iterdir(),
            key=lambda path:
                path.name.lower(),
        ):

            # ----------------------------------------------------
            # Windows-origin metadata files are not dataset files.
            #
            # Count them, but do not emit thousands of log lines.
            # ----------------------------------------------------

            if is_zone_identifier(
                file_path
            ):

                ignored_zone_identifiers += 1
                continue

            if not file_path.is_file():

                ignored_nested_folders += 1

                log_entries.append(
                    f"{func_name}: "
                    "Ignoring nested folder: "
                    f"{file_path}"
                )

                continue

            suffix = (
                file_path
                .suffix
                .lower()
            )

            if suffix in image_extensions:

                file_type = "image"

            elif suffix == json_extension:

                file_type = "json"

            else:

                ignored_unsupported_files += 1

                log_entries.append(
                    f"{func_name}: "
                    "Ignoring unsupported file: "
                    f"{file_path}"
                )

                continue

            discovered_files.append({
                "split":
                    folder["split"],

                "traffic_type":
                    folder[
                        "traffic_type"
                    ],

                "variant":
                    folder["variant"],

                "hardware_source":
                    folder[
                        "hardware_source"
                    ],

                "file_type":
                    file_type,

                "file_name":
                    file_path.name,

                "file_stem":
                    file_path.stem,

                "file_extension":
                    file_path.suffix,

                "file_path":
                    file_path,
            })

    counts = Counter(
        file_record[
            "file_type"
        ]
        for file_record
        in discovered_files
    )

    log_entries.append(
        f"{func_name}: "
        "Discovered files totals:"
    )

    for (
        file_type,
        count,
    ) in sorted(
        counts.items()
    ):

        log_entries.append(
            f"   {file_type}: "
            f"{count}"
        )

    log_entries.append(
        f"{func_name}: "
        "Ignored Zone.Identifier "
        f"metadata files: "
        f"{ignored_zone_identifiers}"
    )

    log_entries.append(
        f"{func_name}: "
        "Ignored other unsupported files: "
        f"{ignored_unsupported_files}"
    )

    log_entries.append(
        f"{func_name}: "
        "Ignored nested folders: "
        f"{ignored_nested_folders}"
    )

    write_log(
        final_log_path,
        log_entries,
    )

    return discovered_files

#Function to perform file level hashing but in blocks as thats a good practice
def calculate_sha256(file_path: Path) -> str:

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()

# this function updates the existing discovered_files dict and adds another key value pair of hash. 
def add_file_hashes(
    discovered_files: list[dict]
) -> None:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ****Calculating SHA-256 hashes*********"]        
    write_log(final_log_path,log_entries)

    for file_record in discovered_files:
        file_path = file_record["file_path"]

        file_record["sha256"] = calculate_sha256(
            file_path
        )

# Function to match images to JSON by creating groups of image/JSON as a pair and giving status to group
def match_images_and_jsons(
    discovered_files: list[dict]
) -> list[dict]:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]        
    write_log(final_log_path,log_entries)

    file_groups = defaultdict(
        lambda: {
            "images": [],
            "jsons": [],
        }
    )
  
    for file_record in discovered_files:

        file_path = file_record["file_path"]

        # key for file_group is single tuple containing both folder and stem of the file. 
        key = (
            file_path.parent,                       # path includes folder as well. 
            file_record["file_stem"].casefold(),    # stem is just the file name without extension.
        )

        if file_record["file_type"] == "image":
            file_groups[key]["images"].append(file_record)

        elif file_record["file_type"] == "json":
            file_groups[key]["jsons"].append(file_record)

        else:
            log_entries = [f"{func_name}: ***Error*** file other than image/json found"]
            write_log(final_log_path,log_entries)

    matched_records = []

    for key in sorted(
        file_groups,
        key=lambda item: (
            str(item[0]).casefold(),
            item[1],
        )
    ):
        group = file_groups[key]

        images = group["images"]
        jsons = group["jsons"]

        source_record = images[0] if images else jsons[0]

        if len(images) == 1 and len(jsons) == 1:
            match_status = "matched"

        elif len(images) == 1 and len(jsons) == 0:
            match_status = "missing_json"

        elif len(images) == 0 and len(jsons) == 1:
            match_status = "orphan_json"

        elif len(images) > 1:
            match_status = "multiple_images"

        else:
            match_status = "multiple_jsons"

        matched_records.append({
            "split": source_record["split"],
            "traffic_type": source_record["traffic_type"],
            "variant": source_record["variant"],
            "hardware_source": source_record["hardware_source"],
            "file_stem": source_record["file_stem"],

            "image_count": len(images),
            "json_count": len(jsons),

            "image_path"    : (images[0]["file_path"] if len(images) == 1 else None ),
            "json_path"     : (jsons[0]["file_path"] if len(jsons) == 1 else None ),
            "image_sha256"  : (images[0]["sha256"] if len(images) == 1 else "" ),
            "json_sha256"   : (jsons[0]["sha256"] if len(jsons) == 1 else "" ),

            "match_status": match_status,
        })

    log_entries = [f"{func_name}: Image / JSON matching summary:-----------------------------"]

    ALL_STATUSES = ("matched", "missing_json", "orphan_json", "multiple_images", "multiple_jsons")
    match_counts = Counter({s: 0 for s in ALL_STATUSES})
    match_counts.update(r["match_status"] for r in matched_records)
    
    for status, count in sorted(match_counts.items()):
        log_entries.append(f"{func_name}: {status}: {count}")

    #debug : uncomment the following code if numbers dont tally above    
    # log_entries.append(f"{func_name}: Image / JSON matching detailed:-----------------------------")

    # for record in matched_records:
    #     log_entries.append (f"{func_name}: {record['match_status']} {record['split']} {record['traffic_type']}" 
    #                        f"{record['variant']} {record['hardware_source']} {record['file_stem']}"
    #                        f" {record['image_sha256']} {record['json_sha256']}")
        
    write_log(final_log_path,log_entries)

    return matched_records

#Function to establish structure of JSON files before we extract information from them.
#This function does update the dictionary passed as parameter but returns none. 
def parse_json_files(
    matched_records: list[dict]
) -> None:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]        
    write_log(final_log_path,log_entries)
    
    for record in matched_records:

        json_path = record["json_path"]

        record["json_parse_ok"] = False
        record["json_top_level_type"] = ""
        record["json_top_level_keys"] = ""
        record["json_error"] = ""
        record["_json_data"] = None

        if json_path is None:
            continue

        try:
            with json_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError
        ) as error:

            record["json_error"] = str(error)
            continue

        record["json_parse_ok"] = True
        record["json_top_level_type"] = type(data).__name__
        record["_json_data"] = data

        if isinstance(data, dict):
            record["json_top_level_keys"] = " | ".join(
                sorted(data.keys())
            )

    log_entries = ["JSON structure - first 5 parsed files:-------------------"]
    shown = 0
    for record in matched_records:
        if not record["json_parse_ok"]:
            continue

        log_entries.append(f" {record['file_stem']}  'type:' {record['json_top_level_type']}")
        log_entries.append(f" keys: {record['json_top_level_keys']}" )
        shown += 1

        if shown == 5:
            break
    log_entries.append("JSON parse failures:------------------------------")

    failure_count = 0

    for record in matched_records:
        if (
            record["json_path"] is not None
            and not record["json_parse_ok"]
        ):
            log_entries.append(f"{record['json_path']} -> {record['json_error']}" )
            failure_count += 1
    log_entries.append(f"Total JSON parse failures: {failure_count}")

    key_patterns = set()
    for record in matched_records:
        if record["json_parse_ok"]:
            key_patterns.add(
                record["json_top_level_keys"]
            )
    log_entries.append("Distinct JSON top-level structures:")
    for pattern in sorted(key_patterns):
        log_entries.append(f"  {pattern}")
    write_log(final_log_path,log_entries)

# Function to extract cropping information from JSON files. note the structure varies slighlt in naming conv.
# Available Distinct JSON top-level structures:
#       cropping_info | person_info | regions
#       cropping_info-altered-recaptured | person_info | regions
#The difference is ignored by using only first few characters of the key cropping-info. 
def find_crop_information(
    matched_records: list[dict]
) -> None:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]        
    write_log(final_log_path,log_entries)

    for record in matched_records:

        record["crop_info_key"] = ""
        record["crop_info_found"] = False
        record["crop_info_error"] = ""
        record["_crop_data"] = None

        data = record["_json_data"]

        if not isinstance(data, dict):
            continue

        crop_keys = [
            key
            for key in data.keys()
            if key.startswith("cropping_info")
        ]

        if len(crop_keys) == 0:
            record["crop_info_error"] = "No cropping_info key found"
            continue

        if len(crop_keys) > 1:
            record["crop_info_error"] = (
                f"Multiple cropping_info keys found: {crop_keys}"
            )
            continue

        crop_key = crop_keys[0]
        crop_data = data[crop_key]

        if not isinstance(crop_data, dict):
            record["crop_info_error"] = (
                f"{crop_key} is not a dictionary"
            )
            continue

        record["crop_info_key"] = crop_key
        record["crop_info_found"] = True
        record["_crop_data"] = crop_data

    #Diagnostics
    structures_by_key = defaultdict(set)
    for record in matched_records:
        if record["crop_info_found"]:
            structures_by_key[record["crop_info_key"]].add(
                tuple(sorted(record["_crop_data"].keys()))
            )

    log_entries = [f"{func_name}: Distinct crop-information structures:"]
    for crop_key in sorted(structures_by_key):
        variants = structures_by_key[crop_key]
        log_entries.append(f"{func_name}:  {crop_key}  ({len(variants)} variant(s)):")
        for i, structure in enumerate(sorted(variants), 1):
            log_entries.append(f"{func_name}:    variant {i} ({len(structure)} fields):")
            for field in structure:
                log_entries.append(f"{func_name}:      {field}")
    write_log(final_log_path, log_entries)

        # Crop-information key counts
    crop_key_counts = Counter(
        record["crop_info_key"]
        for record in matched_records
        if record["crop_info_found"]
    )

    log_entries = ["Crop-information key counts:"]
    for crop_key, count in sorted(crop_key_counts.items()):
        log_entries.append(f"  {crop_key}: {count}")

    # Records that parsed but yielded no crop information
    crop_error_count = 0
    error_entries = []
    for record in matched_records:
        if record["json_parse_ok"] and not record["crop_info_found"]:
            error_entries.append(
                f"  {record['file_stem']} -> {record['crop_info_error']}"
            )
            crop_error_count += 1

    log_entries.append(f"Crop-information problems: {crop_error_count}")
    log_entries.extend(error_entries)

    # Reconciliation — every record should land in exactly one bucket
    n_total = len(matched_records)
    n_no_json = sum(1 for r in matched_records if r["json_path"] is None)
    n_parse_fail = sum(
        1 for r in matched_records
        if r["json_path"] is not None and not r["json_parse_ok"]
    )
    n_found = sum(crop_key_counts.values())

    log_entries.append(
        f"Reconciliation: total={n_total} no_json={n_no_json} "
        f"parse_failed={n_parse_fail} crop_found={n_found} "
        f"crop_missing={crop_error_count}"
    )
    if n_no_json + n_parse_fail + n_found + crop_error_count != n_total:
        log_entries.append("  WARNING: buckets do not sum to total")

    write_log(final_log_path, log_entries)

#helper functions for extracting crop field true to source into immutable tuples. 
def nested_list_to_tuple(value):
    if isinstance(value, list):
        return tuple(
            nested_list_to_tuple(item)
            for item in value
        )

    return value

def is_zone_identifier(
    path: Path,
) -> bool:

    return path.name.endswith(
        ZONE_IDENTIFIER_SUFFIX
    )

def has_shape(value, rows: int, columns: int) -> bool:
    if not isinstance(value, (list, tuple)):
        return False

    if len(value) != rows:
        return False

    for row in value:
        if not isinstance(row, (list, tuple)):
            return False

        if len(row) != columns:
            return False

    return True

#Function to extract the crop fields. 
def extract_crop_fields(
    matched_records: list[dict]
) -> None:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]
    for record in matched_records:

        record["original_image_width"] = None
        record["original_image_height"] = None

        record["resulted_cropped_image_width"] = None
        record["resulted_cropped_image_height"] = None

        record["transformation_matrix"] = None
        record["original_rectangle"] = None

        record["matrix_shape_ok"] = False
        record["rectangle_shape_ok"] = False

        record["crop_field_error"] = ""

        crop_data = record["_crop_data"]

        if not isinstance(crop_data, dict):
            continue

        record["original_image_width"] = (
            crop_data.get("original_image_width")
        )

        record["original_image_height"] = (
            crop_data.get("original_image_height")
        )

        record["resulted_cropped_image_width"] = (
            crop_data.get("resulted_cropped_image_width")
        )

        record["resulted_cropped_image_height"] = (
            crop_data.get("resulted_cropped_image_height")
        )

        matrix = crop_data.get(
            "transformation_matrix_into_cropped"
        )

        rectangle = crop_data.get(
            "original_rectangle_tl_tr_br_bl"
        )

        record["transformation_matrix"] = (
            nested_list_to_tuple(matrix)
        )

        record["original_rectangle"] = (
            nested_list_to_tuple(rectangle)
        )

        record["matrix_shape_ok"] = has_shape(
            matrix,
            rows=3,
            columns=3,
        )

        record["rectangle_shape_ok"] = has_shape(
            rectangle,
            rows=4,
            columns=2,
        )

        errors = []

        if not record["matrix_shape_ok"]:
            errors.append(
                "transformation matrix is not 3x3"
            )

        if not record["rectangle_shape_ok"]:
            errors.append(
                "original rectangle is not 4x2"
            )

        if (
            record["resulted_cropped_image_width"]
            is None
        ):
            errors.append(
                "missing resulted cropped width"
            )

        if (
            record["resulted_cropped_image_height"]
            is None
        ):
            errors.append(
                "missing resulted cropped height"
            )

        record["crop_field_error"] = " | ".join(
            errors
        )

    #diagnostic code
    # Records with field-level problems
    crop_field_error_count = 0
    error_entries = []
    for record in matched_records:
        if record["crop_field_error"]:
            error_entries.append(
                f"  {record['file_stem']} -> {record['crop_field_error']}"
            )
            crop_field_error_count += 1

    log_entries.append(f"Crop field problems: {crop_field_error_count}")
    log_entries.extend(error_entries)

    # One example per distinct crop_info_key
    log_entries.append("Example extracted crop metadata:")
    shown_crop_types = set()
    for record in matched_records:
        crop_key = record["crop_info_key"]
        if crop_key in shown_crop_types:
            continue
        shown_crop_types.add(crop_key)
        log_entries.extend([
            f"  Crop type: {crop_key}  (example: {record['file_stem']})",
            f"    original image size:  {record['original_image_width']}"
            f" x {record['original_image_height']}",
            f"    resulted crop size:   {record['resulted_cropped_image_width']}"
            f" x {record['resulted_cropped_image_height']}",
            f"    matrix shape OK:      {record['matrix_shape_ok']}",
            f"    rectangle shape OK:   {record['rectangle_shape_ok']}",
            f"    matrix:               {record['transformation_matrix']}",
            f"    rectangle:            {record['original_rectangle']}",
        ])

    # Reconciliation
    n_total = len(matched_records)
    n_crop_found = sum(1 for r in matched_records if r["crop_info_found"])
    n_clean = sum(
        1 for r in matched_records
        if r["crop_info_found"] and not r["crop_field_error"]
    )
    log_entries.append(
        f"Reconciliation: total={n_total} crop_found={n_crop_found} "
        f"clean={n_clean} with_field_errors={crop_field_error_count}"
    )

    write_log(final_log_path, log_entries)

#Function for transformation of x,y point via homography
def transform_point(
    point: tuple,
    matrix: tuple
) -> tuple[float, float]:

    x, y = point

    denominator = (
        matrix[2][0] * x
        + matrix[2][1] * y
        + matrix[2][2]
    )

    if abs(denominator) < 1e-12:
        raise ValueError(
            "Homography denominator is too close to zero"
        )

    transformed_x = (
        matrix[0][0] * x
        + matrix[0][1] * y
        + matrix[0][2]
    ) / denominator

    transformed_y = (
        matrix[1][0] * x
        + matrix[1][1] * y
        + matrix[1][2]
    ) / denominator

    return transformed_x, transformed_y

# Function to Transform full rectangle 

def transform_rectangles(
    matched_records: list[dict]
) -> None:

    func_name = inspect.currentframe().f_code.co_name

    log_entries = [
        f"********************{func_name}: ********************"
    ]

    for record in matched_records:

        record["transformed_rectangle"] = None
        record["transform_ok"] = False
        record["transform_error"] = ""

        # New audit field:
        # maximum Euclidean corner residual for this image.
        record["max_corner_error_px"] = None

        if not (
            record["matrix_shape_ok"]
            and record["rectangle_shape_ok"]
        ):
            continue

        matrix = record["transformation_matrix"]
        rectangle = record["original_rectangle"]

        try:
            transformed_rectangle = tuple(
                transform_point(
                    point,
                    matrix
                )
                for point in rectangle
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            record["transform_error"] = str(error)
            continue

        record["transformed_rectangle"] = (
            transformed_rectangle
        )

        record["transform_ok"] = True

        # --------------------------------------------------------
        # Full-dataset corner residual check
        #
        # A crop of W x H pixels has corner pixel coordinates:
        #
        #   TL = (0, 0)
        #   TR = (W-1, 0)
        #   BR = (W-1, H-1)
        #   BL = (0, H-1)
        # --------------------------------------------------------

        width = record[
            "resulted_cropped_image_width"
        ]

        height = record[
            "resulted_cropped_image_height"
        ]

        if (
            width is None
            or height is None
            or width <= 0
            or height <= 0
        ):
            record["transform_ok"] = False
            record["transform_error"] = (
                "Invalid resulted crop dimensions "
                "for corner residual calculation"
            )
            continue

        expected_corners = (
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (
                float(width - 1),
                float(height - 1),
            ),
            (0.0, float(height - 1)),
        )

        corner_errors = []

        for transformed, expected in zip(
            transformed_rectangle,
            expected_corners,
        ):

            dx = (
                transformed[0]
                - expected[0]
            )

            dy = (
                transformed[1]
                - expected[1]
            )

            error = math.hypot(
                dx,
                dy
            )

            corner_errors.append(
                error
            )

        record["max_corner_error_px"] = max(
            corner_errors
        )

    # ============================================================
    # Diagnostics
    # ============================================================

    transform_error_records = [
        record
        for record in matched_records
        if not record["transform_ok"]
    ]

    log_entries.append(
        f"Transformation problems: "
        f"{len(transform_error_records)}"
    )

    for record in transform_error_records[:20]:

        log_entries.append(
            f"  {record['file_stem']} -> "
            f"{record['transform_error']}"
        )

    if len(transform_error_records) > 20:

        log_entries.append(
            f"  ... "
            f"{len(transform_error_records) - 20} "
            f"more not shown"
        )

    # ------------------------------------------------------------
    # Full-dataset residual distribution
    # ------------------------------------------------------------

    corner_residuals = sorted(
        record["max_corner_error_px"]
        for record in matched_records
        if (
            record["transform_ok"]
            and record["max_corner_error_px"]
            is not None
        )
    )

    if corner_residuals:

        n = len(
            corner_residuals
        )

        if n % 2:

            median = (
                corner_residuals[
                    n // 2
                ]
            )

        else:

            median = (
                corner_residuals[
                    n // 2 - 1
                ]
                + corner_residuals[
                    n // 2
                ]
            ) / 2

        log_entries.append(
            "Maximum corner residual per image:"
        )

        log_entries.append(
            f"  n={n}"
        )

        log_entries.append(
            f"  min="
            f"{corner_residuals[0]:.12g} px"
        )

        log_entries.append(
            f"  median="
            f"{median:.12g} px"
        )

        log_entries.append(
            f"  max="
            f"{corner_residuals[-1]:.12g} px"
        )

    # ------------------------------------------------------------
    # Explicit tolerance diagnostic
    # ------------------------------------------------------------

    CORNER_ERROR_TOLERANCE_PX = 0.5

    above_tolerance = [
        record
        for record in matched_records
        if (
            record["max_corner_error_px"]
            is not None
            and record["max_corner_error_px"]
            > CORNER_ERROR_TOLERANCE_PX
        )
    ]

    log_entries.append(
        "Images with max corner residual "
        f"> {CORNER_ERROR_TOLERANCE_PX} px: "
        f"{len(above_tolerance)}"
    )

    for record in sorted(
        above_tolerance,
        key=lambda r: r[
            "max_corner_error_px"
        ],
        reverse=True,
    )[:20]:

        log_entries.append(
            f"  {record['split']}/"
            f"{record['traffic_type']}/"
            f"{record['variant']} "
            f"{record['hardware_source']} "
            f"{record['file_stem']} "
            f"error="
            f"{record['max_corner_error_px']:.6f} px"
        )

    # ------------------------------------------------------------
    # Keep one worked example per crop schema because it is useful
    # for a human reviewer alongside the dataset-wide statistics.
    # ------------------------------------------------------------

    log_entries.append(
        "Example transformed rectangles:"
    )

    shown_crop_types = set()

    for record in matched_records:

        if not record["transform_ok"]:
            continue

        crop_key = record[
            "crop_info_key"
        ]

        if crop_key in shown_crop_types:
            continue

        shown_crop_types.add(
            crop_key
        )

        width = record[
            "resulted_cropped_image_width"
        ]

        height = record[
            "resulted_cropped_image_height"
        ]

        expected_corners = (
            (0, 0),
            (width - 1, 0),
            (
                width - 1,
                height - 1,
            ),
            (0, height - 1),
        )

        log_entries.append(
            f"  Crop type: {crop_key} "
            f"(example: "
            f"{record['file_stem']})"
        )

        log_entries.append(
            f"    Declared crop size: "
            f"{width} x {height}"
        )

        log_entries.append(
            "    original -> transformed "
            "(expected corner)"
        )

        for (
            original,
            transformed,
            expected,
        ) in zip(
            record["original_rectangle"],
            record["transformed_rectangle"],
            expected_corners,
        ):

            log_entries.append(
                f"      "
                f"({original[0]:8.3f}, "
                f"{original[1]:8.3f})"
                f"  ->  "
                f"({transformed[0]:9.3f}, "
                f"{transformed[1]:9.3f})"
                f"   "
                f"(expect "
                f"{expected[0]}, "
                f"{expected[1]})"
            )

        log_entries.append(
            f"    max corner residual: "
            f"{record['max_corner_error_px']:.12g} px"
        )

    # ------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------

    n_eligible = sum(
        1
        for record in matched_records
        if (
            record["matrix_shape_ok"]
            and record["rectangle_shape_ok"]
        )
    )

    n_transformed = sum(
        1
        for record in matched_records
        if record["transform_ok"]
    )

    n_residuals = sum(
        1
        for record in matched_records
        if record[
            "max_corner_error_px"
        ] is not None
    )

    log_entries.append(
        "Reconciliation: "
        f"total={len(matched_records)} "
        f"eligible={n_eligible} "
        f"transformed_ok={n_transformed} "
        f"corner_residuals={n_residuals} "
        f"failed="
        f"{len(transform_error_records)}"
    )

    write_log(
        final_log_path,
        log_entries
    )

#Function to extract image width height from header and exif orientation and check for errors
def extract_image_metadata(
    matched_records: list[dict]
) -> None:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]
    dimension_matches = 0
    dimension_mismatches = []
    for record in matched_records:

        record["image_width"] = None
        record["image_height"] = None
        record["exif_orientation"] = None
        record["image_metadata_ok"] = False
        record["image_metadata_error"] = ""
        record["dims_match_declared_crop"] = False

        image_path = record["image_path"]

        if image_path is None:
            continue

        try:
            with Image.open(image_path) as image:

                record["image_width"] = image.width
                record["image_height"] = image.height

                exif = image.getexif()

                record["exif_orientation"] = (
                    exif.get(274)
                )

                record["image_metadata_ok"] = True

                record["dims_match_declared_crop"] = (
                    record["image_width"] == record["resulted_cropped_image_width"]
                    and record["image_height"] == record["resulted_cropped_image_height"]
                )
                            
                if record["dims_match_declared_crop"]:
                    dimension_matches += 1
                else:
                    dimension_mismatches.append({
                        "file_stem": record["file_stem"],
                        "hardware_source": record["hardware_source"],
                        "disk": (record["image_width"], record["image_height"]),
                        "declared": (record["resulted_cropped_image_width"],
                                     record["resulted_cropped_image_height"]),
                        "transformed_rectangle": record["transformed_rectangle"],
                    })

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:

            record["image_metadata_error"] = str(
                error
            )       

#Diagnostics
    n_no_image = sum(1 for r in matched_records if r["image_path"] is None)
    n_failed = sum(
        1 for r in matched_records
        if r["image_path"] is not None and not r["image_metadata_ok"]
    )
    n_ok = sum(1 for r in matched_records if r["image_metadata_ok"])
    log_entries.append(
        f"{func_name}: Reconciliation: \n total={len(matched_records)} \n Images with no path(no_image)={n_no_image} "
        f"\n {func_name}: \n read_ok(image_meta_data_ok)={n_ok} \n read_failed(Bad path and metadata incorrect)={n_failed}"
    )
    log_entries.append(
    f"On-disk / resulted-crop dimension matches: "
    f"{dimension_matches}"
    )

    log_entries.append(f"On-disk / declared-crop mismatches: {len(dimension_mismatches)}")
    if len(dimension_mismatches) < 10:
        for m in dimension_mismatches:
            log_entries.append(
                f"  {m['file_stem']} ({m['hardware_source']}): "
                f"disk {m['disk'][0]}x{m['disk'][1]} vs declared "
                f"{m['declared'][0]}x{m['declared'][1]}"
            )
#additional disgnostics

    # ---- EXIF orientation ----
    orientation_counts = Counter(
        record["exif_orientation"]
        for record in matched_records
        if record["image_metadata_ok"]
    )

    log_entries.append("EXIF orientation counts:")
    for orientation, count in sorted(
        orientation_counts.items(),
        key=lambda item: str(item[0])
    ):
        label = ORIENTATION_LABELS.get(orientation, "unrecognised value")
        log_entries.append(f"  {orientation}: {count}   ({label})")

    n_needs_transform = sum(
        count for value, count in orientation_counts.items()
        if value not in (None, 1)
    )
    log_entries.append(
        f"  images needing EXIF transform before use: {n_needs_transform}"
    )
    if n_needs_transform:
        log_entries.append(
            "  WARNING: annotation coordinates may not match displayed pixels"
        )

    # ---- Orientation by device ----
    by_device = defaultdict(Counter)
    for r in matched_records:
        if r["image_metadata_ok"]:
            by_device[r["hardware_source"]][r["exif_orientation"]] += 1
    log_entries.append("EXIF orientation by device:")
    for device in sorted(by_device):
        log_entries.append(f"  {device}: {dict(by_device[device])}")

# Final writing of the logs.        
    write_log(final_log_path, log_entries)

def validate_full_image_decode(
    matched_records: list[dict]
) -> None:

    func_name = inspect.currentframe().f_code.co_name

    log_entries = [
        f"********************{func_name}: ********************"
    ]

    for record in matched_records:

        record["image_decode_ok"] = False
        record["image_decode_error"] = ""

        record["image_decode_warning_count"] = 0
        record["image_decode_warnings"] = ""

        record["decoded_pixel_sha256"] = ""

        image_path = record["image_path"]

        if image_path is None:
            record["image_decode_error"] = (
                "No unique image path available"
            )
            continue

        try:

            # ----------------------------------------------------
            # Warnings are captured as evidence.
            #
            # They are NOT converted into exceptions.
            # ----------------------------------------------------

            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:

                warnings.simplefilter(
                    "always"
                )

                with Image.open(
                    image_path
                ) as image:

                    # --------------------------------------------
                    # Force a complete image decode.
                    #
                    # convert("RGB") also gives both pipelines a
                    # useful common decoded representation for
                    # duplicate checking.
                    # --------------------------------------------

                    rgb_image = image.convert(
                        "RGB"
                    )

                    rgb_image.load()

                    # --------------------------------------------
                    # Build a decoded-pixel hash.
                    #
                    # Include dimensions and mode before the pixel
                    # bytes so differently-shaped images cannot
                    # accidentally have the same semantic hash
                    # merely because their byte sequence matches.
                    # --------------------------------------------

                    pixel_hash = hashlib.sha256()

                    pixel_hash.update(
                        (
                            f"{rgb_image.width}x"
                            f"{rgb_image.height}|RGB|"
                        ).encode(
                            "ascii"
                        )
                    )

                    pixel_hash.update(
                        rgb_image.tobytes()
                    )

                    record[
                        "decoded_pixel_sha256"
                    ] = pixel_hash.hexdigest()

                    record[
                        "image_decode_ok"
                    ] = True

                # -----------------------------------------------
                # Save warning text after successful/failed decode.
                # -----------------------------------------------

                warning_messages = [
                    str(warning.message)
                    for warning
                    in caught_warnings
                ]

                record[
                    "image_decode_warning_count"
                ] = len(
                    warning_messages
                )

                record[
                    "image_decode_warnings"
                ] = " | ".join(
                    warning_messages
                )

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:

            record[
                "image_decode_error"
            ] = str(
                error
            )

    # ============================================================
    # DECODE DIAGNOSTICS
    # ============================================================

    decode_ok = [
        record
        for record in matched_records
        if record["image_decode_ok"]
    ]

    decode_failed = [
        record
        for record in matched_records
        if not record["image_decode_ok"]
    ]

    warning_records = [
        record
        for record in matched_records
        if (
            record[
                "image_decode_warning_count"
            ] > 0
        )
    ]

    log_entries.append(
        "Full-image decode:"
    )

    log_entries.append(
        f"  total images: "
        f"{len(matched_records)}"
    )

    log_entries.append(
        f"  decode OK: "
        f"{len(decode_ok)}"
    )

    log_entries.append(
        f"  decode failed: "
        f"{len(decode_failed)}"
    )

    log_entries.append(
        f"  images with Pillow warnings: "
        f"{len(warning_records)}"
    )

    total_warnings = sum(
        record[
            "image_decode_warning_count"
        ]
        for record
        in matched_records
    )

    log_entries.append(
        f"  total Pillow warnings: "
        f"{total_warnings}"
    )

    # ------------------------------------------------------------
    # Show failed images
    # ------------------------------------------------------------

    if decode_failed:

        log_entries.append(
            "Decode failures:"
        )

        for record in decode_failed[:20]:

            log_entries.append(
                f"  "
                f"{record['split']}/"
                f"{record['traffic_type']}/"
                f"{record['variant']} "
                f"{record['hardware_source']} "
                f"{record['file_stem']} "
                f"-> "
                f"{record['image_decode_error']}"
            )

        if len(decode_failed) > 20:

            log_entries.append(
                f"  ... "
                f"{len(decode_failed) - 20} "
                f"more not shown"
            )

    # ------------------------------------------------------------
    # Show warning examples
    # ------------------------------------------------------------

    if warning_records:

        log_entries.append(
            "Images with decode warnings:"
        )

        for record in warning_records[:20]:

            log_entries.append(
                f"  "
                f"{record['split']}/"
                f"{record['traffic_type']}/"
                f"{record['variant']} "
                f"{record['hardware_source']} "
                f"{record['file_stem']} "
                f"warnings="
                f"{record['image_decode_warning_count']} "
                f"-> "
                f"{record['image_decode_warnings']}"
            )

        if len(warning_records) > 20:

            log_entries.append(
                f"  ... "
                f"{len(warning_records) - 20} "
                f"more not shown"
            )

    # ============================================================
    # AUTHORITATIVE FILE-HASH DUPLICATES
    # ============================================================

    file_hash_groups = defaultdict(
        list
    )

    for record in matched_records:

        image_hash = record[
            "image_sha256"
        ]

        if image_hash:

            file_hash_groups[
                image_hash
            ].append(
                record
            )

    duplicate_file_hash_groups = {
        image_hash: records
        for image_hash, records
        in file_hash_groups.items()
        if len(records) > 1
    }

    log_entries.append(
        "Exact-file SHA-256 duplicate groups: "
        f"{len(duplicate_file_hash_groups)}"
    )

    # ============================================================
    # DECODED-PIXEL DUPLICATES
    # ============================================================

    pixel_hash_groups = defaultdict(
        list
    )

    for record in matched_records:

        pixel_hash = record[
            "decoded_pixel_sha256"
        ]

        if pixel_hash:

            pixel_hash_groups[
                pixel_hash
            ].append(
                record
            )

    duplicate_pixel_hash_groups = {
        pixel_hash: records
        for pixel_hash, records
        in pixel_hash_groups.items()
        if len(records) > 1
    }

    log_entries.append(
        "Decoded-RGB duplicate groups: "
        f"{len(duplicate_pixel_hash_groups)}"
    )

    # ============================================================
    # TRAIN ↔ TEST FILE-HASH INTERSECTION
    # ============================================================

    train_file_hashes = {
        record["image_sha256"]
        for record
        in matched_records
        if (
            record["split"] == "train"
            and record["image_sha256"]
        )
    }

    test_file_hashes = {
        record["image_sha256"]
        for record
        in matched_records
        if (
            record["split"] == "test"
            and record["image_sha256"]
        )
    }

    shared_file_hashes = (
        train_file_hashes
        & test_file_hashes
    )

    log_entries.append(
        "Train / test exact-file SHA-256 overlap: "
        f"{len(shared_file_hashes)}"
    )

    # ============================================================
    # TRAIN ↔ TEST DECODED-PIXEL INTERSECTION
    # ============================================================

    train_pixel_hashes = {
        record[
            "decoded_pixel_sha256"
        ]
        for record
        in matched_records
        if (
            record["split"] == "train"
            and record[
                "decoded_pixel_sha256"
            ]
        )
    }

    test_pixel_hashes = {
        record[
            "decoded_pixel_sha256"
        ]
        for record
        in matched_records
        if (
            record["split"] == "test"
            and record[
                "decoded_pixel_sha256"
            ]
        )
    }

    shared_pixel_hashes = (
        train_pixel_hashes
        & test_pixel_hashes
    )

    log_entries.append(
        "Train / test decoded-RGB overlap: "
        f"{len(shared_pixel_hashes)}"
    )

    # ------------------------------------------------------------
    # Show examples if overlap exists
    # ------------------------------------------------------------

    if shared_file_hashes:

        log_entries.append(
            "WARNING: exact-byte image overlap "
            "between train and test:"
        )

        for image_hash in sorted(
            shared_file_hashes
        )[:20]:

            records = file_hash_groups[
                image_hash
            ]

            log_entries.append(
                f"  SHA={image_hash}"
            )

            for record in records:

                log_entries.append(
                    f"    "
                    f"{record['split']} "
                    f"{record['image_path']}"
                )

    if shared_pixel_hashes:

        log_entries.append(
            "WARNING: decoded-pixel image overlap "
            "between train and test:"
        )

        for pixel_hash in sorted(
            shared_pixel_hashes
        )[:20]:

            records = pixel_hash_groups[
                pixel_hash
            ]

            log_entries.append(
                f"  pixel_SHA="
                f"{pixel_hash}"
            )

            for record in records:

                log_entries.append(
                    f"    "
                    f"{record['split']} "
                    f"{record['image_path']}"
                )

    # ============================================================
    # RECONCILIATION
    # ============================================================

    pixel_hash_count = sum(
        1
        for record
        in matched_records
        if record[
            "decoded_pixel_sha256"
        ]
    )

    log_entries.append(
        "Reconciliation: "
        f"total={len(matched_records)} "
        f"decode_ok={len(decode_ok)} "
        f"decode_failed={len(decode_failed)} "
        f"pixel_hashes={pixel_hash_count}"
    )

    write_log(
        final_log_path,
        log_entries
    )

#Function to check for errors in regions specified with JSON files to match they exist within the bounded images
def inspect_regions(
    matched_records: list[dict]
) -> None:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]
    for record in matched_records:

        record["regions_found"] = False
        record["regions_structure_ok"] = False
        record["region_count"] = 0
        record["regions_error"] = ""
        record["_regions_data"] = None

        data = record["_json_data"]

        if not isinstance(data, dict):
            continue

        if "regions" not in data:
            record["regions_error"] = (
                "regions key not found"
            )
            continue

        regions = data["regions"]

        if not isinstance(regions, list):
            record["regions_error"] = (
                "regions is not a list"
            )
            continue

        record["regions_found"] = True
        record["regions_structure_ok"] = True
        record["region_count"] = len(regions)
        record["_regions_data"] = regions

# Diagnostics 
    region_key_patterns = Counter()
    shape_key_patterns = Counter()
    attribute_key_patterns = Counter()
    malformed_regions = 0

    for record in matched_records:
        regions = record["_regions_data"]
        if not isinstance(regions, list):
            continue

        for region in regions:
            if not isinstance(region, dict):
                malformed_regions += 1
                continue

            region_key_patterns[tuple(sorted(region.keys()))] += 1
            #debug print("region.keys()-",region.keys())

            shape_attributes = region.get("shape_attributes")
            if isinstance(shape_attributes, dict):
                shape_key_patterns[tuple(sorted(shape_attributes.keys()))] += 1
            #debug print("shape_attributes.keys()-",shape_attributes.keys())

            region_attributes = region.get("region_attributes")
            if isinstance(region_attributes, dict):
                attribute_key_patterns[tuple(sorted(region_attributes.keys()))] += 1
            #debug print("region_attributes.keys()-",region_attributes.keys())

    for title, patterns in (
        ("region top-level", region_key_patterns),
        ("shape_attributes", shape_key_patterns),
        ("region_attributes", attribute_key_patterns),
    ):
        log_entries.append(
            f"Distinct {title} structures ({len(patterns)} distinct, "
            f"{sum(patterns.values())} total):"
        )
        for pattern, count in sorted(patterns.items()):
            log_entries.append(f"  {count:6d}  " + " | ".join(pattern))

    # Region-level reconciliation
    n_regions_found = sum(1 for r in matched_records if r["regions_found"])
    n_regions_missing = sum(
        1 for r in matched_records
        if r["json_parse_ok"] and not r["regions_found"]
    )
    total_regions = sum(r["region_count"] for r in matched_records)
    counted_regions = sum(region_key_patterns.values()) + malformed_regions

    log_entries.append(
        f"Reconciliation: records={len(matched_records)} "
        f"regions_found={n_regions_found} regions_missing={n_regions_missing} "
        f"total_regions={total_regions} counted={counted_regions} "
        f"malformed={malformed_regions}"
    )
    if total_regions != counted_regions:
        log_entries.append("  WARNING: region counts do not reconcile")

    #additional diagnostics
    no_prov = defaultdict(Counter)
    for record in matched_records:
        for region in record["_regions_data"] or []:
            ra = region.get("region_attributes", {})
            if "region_provenance" not in ra:
                no_prov[record["traffic_type"]][record["variant"]] += 1
    log_entries.append("Regions missing region_provenance, by traffic/variant:")
    for tt in sorted(no_prov):
        log_entries.append(f"  {tt}: {dict(no_prov[tt])}")

    shown = 0
    for record in matched_records:
        for region in record["_regions_data"] or []:
            ra = region.get("region_attributes", {})
            if set(ra.keys()) == {"field_name"}:
                log_entries.append(
                    f"  {record['split']}/{record['traffic_type']}/{record['variant']}"
                    f" {record['file_stem']}: {ra} {region['shape_attributes']}"
                )
                shown += 1
                break
        if shown >= 5:
            break


    write_log(final_log_path, log_entries)

#Function to scan through regions in the JSON's for each image
def extract_region_records(
    matched_records: list[dict]
) -> list[dict]:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]
    region_records = []
    skipped_records = 0

    for record in matched_records:

        regions = record["_regions_data"]

        if not isinstance(regions, list):
            skipped_records += 1
            continue

        for region_index, region in enumerate(regions):

            shape = region.get("shape_attributes", {})
            attributes = region.get("region_attributes", {})

            region_records.append({
                # Parent image information
                "split": record["split"],
                "traffic_type": record["traffic_type"],
                "variant": record["variant"],
                "hardware_source": record["hardware_source"],
                "file_stem": record["file_stem"],
                "image_path": record["image_path"],
                "image_sha256": record["image_sha256"],
                "json_path": record["json_path"],
                "json_sha256": record["json_sha256"],

                # Position within the JSON regions list
                "region_index": region_index,

                # Raw shape_attributes values
                "shape_name": shape.get("name"),
                "x": shape.get("x"),
                "y": shape.get("y"),
                "width": shape.get("width"),
                "height": shape.get("height"),

                # Raw region_attributes values
                "field_name": attributes.get("field_name"),
                "region_provenance_raw": attributes.get(
                    "region_provenance"
                ),
                "language": attributes.get("language"),
                "val": attributes.get("val"),
                "org_value": attributes.get("org_value"),
                "new_value": attributes.get("new_value"),
                "source": attributes.get("source"),
                "target": attributes.get("target"),
            })

     #Diagnostics ----
    expected_regions = sum(r["region_count"] for r in matched_records)
    actual_regions = len(region_records)

    log_entries.append(
        f"Reconciliation: images={len(matched_records)} "
        f"skipped_images={skipped_records} "
        f"expected_regions={expected_regions} extracted={actual_regions}"
    )
    if expected_regions != actual_regions:
        log_entries.append("  WARNING: extracted region count does not reconcile")

    # Geometry completeness
    log_entries.append("Missing geometry values:")
    for field in ("x", "y", "width", "height"):
        missing = sum(1 for r in region_records if r[field] is None)
        log_entries.append(f"  {field}: {missing}")

    # Non-positive extents would rasterize to empty masks
    degenerate = sum(
        1 for r in region_records
        if r["width"] is not None and r["height"] is not None
        and (r["width"] <= 0 or r["height"] <= 0)
    )
    log_entries.append(f"  non-positive width or height: {degenerate}")

    # Shape names
    shape_name_counts = Counter(r["shape_name"] for r in region_records)
    log_entries.append("Shape names:")
    for shape_name, count in sorted(
        shape_name_counts.items(), key=lambda item: str(item[0])
    ):
        log_entries.append(f"  {shape_name}: {count}")

    # field_name — raw and case-folded, to expose casing inconsistency
    field_name_counts = Counter(r["field_name"] for r in region_records)
    log_entries.append(f"Region field_name counts ({len(field_name_counts)} distinct):")
    for field_name, count in sorted(
        field_name_counts.items(), key=lambda item: str(item[0])
    ):
        log_entries.append(f"  {field_name}: {count}")

    folded_counts = Counter(
        (r["field_name"] or "").casefold() for r in region_records
    )
    if len(folded_counts) != len(field_name_counts):
        log_entries.append(
            f"  NOTE: {len(field_name_counts)} distinct raw names collapse to "
            f"{len(folded_counts)} when case-folded"
        )

    # Provenance, overall and by traffic type
    provenance_counts = Counter(r["region_provenance_raw"] for r in region_records)
    log_entries.append("Raw region_provenance counts:")
    for provenance, count in sorted(
        provenance_counts.items(), key=lambda item: str(item[0])
    ):
        log_entries.append(f"  {provenance}: {count}")

    prov_by_traffic = defaultdict(Counter)
    for r in region_records:
        prov_by_traffic[r["traffic_type"]][r["region_provenance_raw"]] += 1
    log_entries.append("Provenance by traffic type:")
    for traffic_type in sorted(prov_by_traffic):
        log_entries.append(f"  {traffic_type}: {dict(prov_by_traffic[traffic_type])}")

    write_log(final_log_path, log_entries)
    return region_records

def is_finite_number(
    value,
) -> bool:

    # bool is technically a subclass of int in Python,
    # but True/False are not valid geometry coordinates.
    if isinstance(
        value,
        bool,
    ):
        return False

    if not isinstance(
        value,
        (int, float),
    ):
        return False

    return math.isfinite(
        value
    )

#Function helper to evaluate if altered regions are within the image width x height dimensions
def classify_region_bounds(
    x,
    y,
    width,
    height,
    image_width,
    image_height,
) -> str:

    right = x + width
    bottom = y + height

    fully_inside = (
        x >= 0
        and y >= 0
        and right <= image_width
        and bottom <= image_height
    )

    if fully_inside:
        touches_boundary = (
            x == 0
            or y == 0
            or right == image_width
            or bottom == image_height
        )

        if touches_boundary:
            return "inside_touching_boundary"

        return "fully_inside"

    completely_outside = (
        right <= 0
        or bottom <= 0
        or x >= image_width
        or y >= image_height
    )

    if completely_outside:
        return "completely_outside"

    return "partially_outside"

#Function to evaluate if altered regions are within the image width x height dimensions
def validate_region_bounds(
    region_records: list[dict],
    matched_records: list[dict],
) -> None:

    func_name = (
        inspect.currentframe()
        .f_code.co_name
    )

    log_entries = [
        f"********************"
        f"{func_name}: "
        f"********************"
    ]

    image_lookup = {
        record[
            "image_path"
        ]: record

        for record
        in matched_records

        if record[
            "image_path"
        ] is not None
    }

    for region in region_records:

        # --------------------------------------------------------
        # Always initialise outputs.
        #
        # An empty bounds status now means:
        # "bounds check did not complete"
        # rather than causing an arithmetic crash.
        # --------------------------------------------------------

        region[
            "disk_bounds_status"
        ] = ""

        region[
            "declared_bounds_status"
        ] = ""

        region["right"] = None
        region["bottom"] = None

        region[
            "bounds_check_error"
        ] = ""

        parent = image_lookup.get(
            region[
                "image_path"
            ]
        )

        # ========================================================
        # Validate region geometry before arithmetic.
        # ========================================================

        geometry_values = {
            "x":
                region["x"],

            "y":
                region["y"],

            "width":
                region["width"],

            "height":
                region["height"],
        }

        invalid_geometry = [
            name
            for name, value
            in geometry_values.items()
            if not is_finite_number(
                value
            )
        ]

        if invalid_geometry:

            region[
                "bounds_check_error"
            ] = (
                "Missing/non-numeric/non-finite "
                "region geometry: "
                + ", ".join(
                    invalid_geometry
                )
            )

            continue

        if (
            region["width"] <= 0
            or region["height"] <= 0
        ):

            region[
                "bounds_check_error"
            ] = (
                "Region width/height "
                "must be positive"
            )

            continue

        # Safe only after validation above.
        region["right"] = (
            region["x"]
            + region["width"]
        )

        region["bottom"] = (
            region["y"]
            + region["height"]
        )

        if parent is None:

            region[
                "bounds_check_error"
            ] = (
                "Parent image record "
                "not found"
            )

            continue

        # ========================================================
        # Validate actual image dimensions.
        # ========================================================

        disk_dimensions = (
            parent[
                "image_width"
            ],
            parent[
                "image_height"
            ],
        )

        if not all(
            is_finite_number(
                value
            )
            for value
            in disk_dimensions
        ):

            region[
                "bounds_check_error"
            ] = (
                "Actual image dimensions "
                "missing/non-numeric"
            )

            continue

        if (
            parent[
                "image_width"
            ] <= 0
            or parent[
                "image_height"
            ] <= 0
        ):

            region[
                "bounds_check_error"
            ] = (
                "Actual image dimensions "
                "are non-positive"
            )

            continue

        # ========================================================
        # Validate declared crop dimensions.
        # ========================================================

        declared_dimensions = (
            parent[
                "resulted_cropped_image_width"
            ],
            parent[
                "resulted_cropped_image_height"
            ],
        )

        if not all(
            is_finite_number(
                value
            )
            for value
            in declared_dimensions
        ):

            region[
                "bounds_check_error"
            ] = (
                "Declared crop dimensions "
                "missing/non-numeric"
            )

            continue

        if (
            parent[
                "resulted_cropped_image_width"
            ] <= 0
            or parent[
                "resulted_cropped_image_height"
            ] <= 0
        ):

            region[
                "bounds_check_error"
            ] = (
                "Declared crop dimensions "
                "are non-positive"
            )

            continue

        # ========================================================
        # Actual on-disk image
        # ========================================================

        region[
            "disk_bounds_status"
        ] = classify_region_bounds(
            region["x"],
            region["y"],
            region["width"],
            region["height"],
            parent[
                "image_width"
            ],
            parent[
                "image_height"
            ],
        )

        # ========================================================
        # JSON-declared crop
        # ========================================================

        region[
            "declared_bounds_status"
        ] = classify_region_bounds(
            region["x"],
            region["y"],
            region["width"],
            region["height"],
            parent[
                "resulted_cropped_image_width"
            ],
            parent[
                "resulted_cropped_image_height"
            ],
        )

    # ============================================================
    # Diagnostics
    # ============================================================

    disk_status_counts = Counter(
        region[
            "disk_bounds_status"
        ]
        for region
        in region_records
    )

    declared_status_counts = Counter(
        region[
            "declared_bounds_status"
        ]
        for region
        in region_records
    )

    for (
        title,
        counts,
    ) in (
        (
            "actual on-disk image",
            disk_status_counts,
        ),
        (
            "JSON-declared crop",
            declared_status_counts,
        ),
    ):

        log_entries.append(
            f"Region bounds against "
            f"{title}:"
        )

        for (
            status,
            count,
        ) in sorted(
            counts.items()
        ):

            label = (
                status
                if status
                else "(not checked)"
            )

            log_entries.append(
                f"  {label}: "
                f"{count}"
            )

    # ------------------------------------------------------------
    # Bounds checks that could not run
    # ------------------------------------------------------------

    failed_checks = [
        region
        for region
        in region_records
        if not region[
            "disk_bounds_status"
        ]
    ]

    log_entries.append(
        "Bounds checks not completed: "
        f"{len(failed_checks)}"
    )

    for region in failed_checks[:20]:

        log_entries.append(
            f"  "
            f"{region['split']}/"
            f"{region['traffic_type']}/"
            f"{region['variant']} "
            f"{region['hardware_source']} "
            f"{region['file_stem']} "
            f"region="
            f"{region['region_index']} "
            f"-> "
            f"{region['bounds_check_error']}"
        )

    if len(
        failed_checks
    ) > 20:

        log_entries.append(
            f"  ... "
            f"{len(failed_checks) - 20} "
            f"more not shown"
        )

    # ------------------------------------------------------------
    # Out-of-bounds regions by provenance
    # ------------------------------------------------------------

    oob_by_provenance = defaultdict(
        Counter
    )

    for region in region_records:

        if region[
            "disk_bounds_status"
        ] in (
            "partially_outside",
            "completely_outside",
        ):

            oob_by_provenance[
                region[
                    "region_provenance_raw"
                ]
            ][
                region[
                    "disk_bounds_status"
                ]
            ] += 1

    if oob_by_provenance:

        log_entries.append(
            "Out-of-bounds regions "
            "by provenance:"
        )

        for provenance in sorted(
            oob_by_provenance,
            key=str,
        ):

            log_entries.append(
                f"  {provenance}: "
                f"{dict(oob_by_provenance[provenance])}"
            )

        n_altered_oob = sum(
            oob_by_provenance
            .get(
                "altered",
                Counter(),
            )
            .values()
        )

        if n_altered_oob:

            log_entries.append(
                f"  WARNING: "
                f"{n_altered_oob} "
                "altered regions fall "
                "outside the image"
            )

    # ------------------------------------------------------------
    # Only compare rows where BOTH checks actually ran.
    # ------------------------------------------------------------

    disagreements = [
        region
        for region
        in region_records
        if (
            region[
                "disk_bounds_status"
            ]
            and
            region[
                "declared_bounds_status"
            ]
            and
            region[
                "disk_bounds_status"
            ]
            !=
            region[
                "declared_bounds_status"
            ]
        )
    ]

    log_entries.append(
        "Disk / declared bounds "
        f"disagreements: "
        f"{len(disagreements)}"
    )

    for region in (
        disagreements[:20]
    ):

        parent = image_lookup[
            region[
                "image_path"
            ]
        ]

        log_entries.append(
            f"  {region['file_stem']} "
            f"region="
            f"{region['region_index']} "
            f"field="
            f"{region['field_name']} "
            f"box=("
            f"{region['x']}, "
            f"{region['y']}, "
            f"{region['width']}, "
            f"{region['height']}) "
            f"disk="
            f"{parent['image_width']}x"
            f"{parent['image_height']} "
            f"declared="
            f"{parent['resulted_cropped_image_width']}x"
            f"{parent['resulted_cropped_image_height']} "
            f"{region['disk_bounds_status']} "
            f"vs "
            f"{region['declared_bounds_status']}"
        )

    if len(
        disagreements
    ) > 20:

        log_entries.append(
            f"  ... "
            f"{len(disagreements) - 20} "
            f"more not shown"
        )

    log_entries.append(
        "Reconciliation: "
        f"total_regions="
        f"{len(region_records)} "
        f"checked="
        f"{len(region_records) - len(failed_checks)} "
        f"not_checked="
        f"{len(failed_checks)} "
        f"disagreements="
        f"{len(disagreements)}"
    )

    write_log(
        final_log_path,
        log_entries,
    )

#Function helper to extract information about altered regions which are outside boundary based on results of previous functio
                # ********************validate_region_bounds: ********************
                # Region bounds against actual on-disk image:
                # fully_inside: 31955
                # inside_touching_boundary: 1455
                # partially_outside: 514
                # Region bounds against JSON-declared crop:
                # fully_inside: 31955
                # inside_touching_boundary: 1455
                # partially_outside: 514
                # Out-of-bounds regions by provenance:
                # None: {'partially_outside': 85}
                # altered: {'partially_outside': 171}
                # original: {'partially_outside': 258}
                # WARNING: 171 altered regions fall outside the image
                # Disk / declared bounds disagreements: 0
                # Reconciliation: total_regions=33924 with_parent=33924 no_parent=0 disagreements=0
def calculate_visible_region(
    x,
    y,
    width,
    height,
    image_width,
    image_height,
) -> dict:

    right = x + width
    bottom = y + height

    visible_left = max(0, x)
    visible_top = max(0, y)
    visible_right = min(image_width, right)
    visible_bottom = min(image_height, bottom)

    visible_width = max(
        0,
        visible_right - visible_left
    )

    visible_height = max(
        0,
        visible_bottom - visible_top
    )

    original_area = width * height
    visible_area = visible_width * visible_height

    visible_fraction = (
        visible_area / original_area
        if original_area > 0
        else None
    )

    return {
        "outside_left": max(0, -x),
        "outside_top": max(0, -y),
        "outside_right": max(
            0,
            right - image_width
        ),
        "outside_bottom": max(
            0,
            bottom - image_height
        ),

        "visible_left": visible_left,
        "visible_top": visible_top,
        "visible_right": visible_right,
        "visible_bottom": visible_bottom,

        "visible_width": visible_width,
        "visible_height": visible_height,

        "original_area": original_area,
        "visible_area": visible_area,
        "visible_fraction": visible_fraction,
    }

#Function to calculate the severity of outside bounds regions
def analyse_region_visibility(
    region_records: list[dict],
    matched_records: list[dict],
) -> None:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]

    image_lookup = {
        record["image_path"]: record
        for record in matched_records
        if record["image_path"] is not None
    }

    for region in region_records:

        parent = image_lookup.get(
            region["image_path"]
        )

        region["outside_left"] = None
        region["outside_top"] = None
        region["outside_right"] = None
        region["outside_bottom"] = None

        region["visible_left"] = None
        region["visible_top"] = None
        region["visible_right"] = None
        region["visible_bottom"] = None

        region["visible_width"] = None
        region["visible_height"] = None

        region["original_area"] = None
        region["visible_area"] = None
        region["visible_fraction"] = None

        if parent is None:
            continue

            # validate_region_bounds() is the upstream authority.
            # Empty status means the bounds/geometry check did not
            # complete, so visibility arithmetic must not run.
        if not region["disk_bounds_status"]:
            continue

        visibility = calculate_visible_region(

            region["x"],
            region["y"],
            region["width"],
            region["height"],
            parent["image_width"],
            parent["image_height"],
        )

        for key, value in visibility.items():
            region[key] = value

    # ---- Diagnostics ----
    partially_outside = [
        r for r in region_records
        if r["disk_bounds_status"] == "partially_outside"
    ]
    log_entries.append(f"Partially outside regions: {len(partially_outside)}")

    def fraction_summary(regions, label):
        fractions = sorted(
            r["visible_fraction"] for r in regions
            if r["visible_fraction"] is not None
        )
        if not fractions:
            return
        n = len(fractions)
        median = (
            fractions[n // 2] if n % 2
            else (fractions[n // 2 - 1] + fractions[n // 2]) / 2
        )
        log_entries.append(f"{label} visible-fraction (n={n}):")
        log_entries.append(
            f"  min={fractions[0]:.6f}  median={median:.6f}  max={fractions[-1]:.6f}"
        )

    fraction_summary(partially_outside, "All partially-outside")

    log_entries.append("Partially outside below visible-fraction thresholds:")
    for threshold in (0.99, 0.95, 0.90, 0.70, 0.50):
        count = sum(
            1 for r in partially_outside
            if r["visible_fraction"] is not None and r["visible_fraction"] < threshold
        )
        log_entries.append(f"  < {threshold:.2f}: {count}")

    # Altered regions are the ones that become ground-truth mask
    altered_outside = [
        r for r in partially_outside if r["region_provenance_raw"] == "altered"
    ]
    log_entries.append(f"Partially outside ALTERED regions: {len(altered_outside)}")
    fraction_summary(altered_outside, "  Altered")
    for threshold in (0.99, 0.95, 0.90, 0.70, 0.50):
        count = sum(
            1 for r in altered_outside
            if r["visible_fraction"] is not None and r["visible_fraction"] < threshold
        )
        log_entries.append(f"  altered < {threshold:.2f}: {count}")

    # Which edges
    edge_counts = Counter()
    for r in partially_outside:
        for edge in ("left", "top", "right", "bottom"):
            if r[f"outside_{edge}"] > 0:
                edge_counts[edge] += 1
    log_entries.append("Out-of-bounds edges:")
    for edge, count in sorted(edge_counts.items()):
        log_entries.append(f"  {edge}: {count}")

    # Breakdowns
    for field in ("split", "traffic_type", "variant", "hardware_source", "field_name"):
        counts = Counter(r[field] for r in partially_outside)
        log_entries.append(f"Partially outside by {field}:")
        for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
            log_entries.append(f"  {value}: {count}")

    # Image-level impact
    affected_images = {r["image_path"] for r in partially_outside}
    altered_affected = {r["image_path"] for r in altered_outside}
    log_entries.append(f"Images with any partially-outside region: {len(affected_images)}")
    log_entries.append(
        f"Images with partially-outside ALTERED region: {len(altered_affected)}"
        f"  ({len(altered_affected) / len(matched_records):.2%} of all images)"
    )

    # Worst cases, altered first
    most_truncated = sorted(
        partially_outside,
        key=lambda r: (r["region_provenance_raw"] != "altered", r["visible_fraction"]),
    )
    log_entries.append("Most truncated regions (altered listed first):")
    for r in most_truncated[:20]:
        log_entries.append(
            f"  {r['split']}/{r['traffic_type']}/{r['variant']} "
            f"{r['hardware_source']} {r['file_stem']} "
            f"region={r['region_index']} field={r['field_name']} "
            f"prov={r['region_provenance_raw']} "
            f"box=({r['x']}, {r['y']}, {r['width']}, {r['height']}) "
            f"visible={r['visible_fraction']:.6f}"
        )

    # Reconciliation
    n_no_parent = sum(1 for r in region_records if r["visible_fraction"] is None)
    log_entries.append(
        f"Reconciliation: total_regions={len(region_records)} "
        f"visibility_computed={len(region_records) - n_no_parent} "
        f"skipped={n_no_parent} partially_outside={len(partially_outside)} "
        f"altered_partially_outside={len(altered_outside)}"
    )

    write_log(final_log_path, log_entries)

#Function to analyse further the 514 outliers. 
def analyse_right_edge_truncation(
    region_records: list[dict],
) -> None:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]

    outside_regions = [
        r for r in region_records
        if r["disk_bounds_status"] == "partially_outside"
    ]

    # Add two useful derived values to each region record.
    for region in region_records:
        region["outside_right_fraction"] = 0.0
        if (
            region["outside_right"] is not None
            and region["width"] > 0
        ):
            region["outside_right_fraction"] = (
                region["outside_right"]
                / region["width"]
            )

    # --------------------------------------------------
    # Helper for min / median / max diagnostics
    # --------------------------------------------------
    def summary(values, label):

        values = sorted(values)

        if not values:
            log_entries.append(f"{label}: no values")
            return

        n = len(values)

        if n % 2:
            median = values[n // 2]
        else:
            median = (
                values[n // 2 - 1]
                + values[n // 2]
            ) / 2

        log_entries.append(
            f"{label} (n={n}): "
            f"min={values[0]:.3f} "
            f"median={median:.3f} "
            f"max={values[-1]:.3f}"
        )

    # --------------------------------------------------
    # Absolute number of pixels outside
    # --------------------------------------------------
    summary(
        [r["outside_right"] for r in outside_regions],
        "Right-edge pixels outside"
    )

    # --------------------------------------------------
    # Fraction of the raw region width outside
    # --------------------------------------------------
    summary(
        [
            r["outside_right_fraction"]
            for r in outside_regions
        ],
        "Fraction of region width outside"
    )

    # --------------------------------------------------
    # Altered regions separately
    # --------------------------------------------------
    altered_outside = [
        r for r in outside_regions
        if r["region_provenance_raw"] == "altered"
    ]

    summary(
        [r["outside_right"] for r in altered_outside],
        "ALTERED right-edge pixels outside"
    )

    summary(
        [
            r["outside_right_fraction"]
            for r in altered_outside
        ],
        "ALTERED fraction of region width outside"
    )

    # --------------------------------------------------
    # By hardware source
    # --------------------------------------------------
    by_hardware = defaultdict(list)

    for region in outside_regions:
        by_hardware[
            region["hardware_source"]
        ].append(region)

    log_entries.append(
        "Right-edge truncation by hardware source:"
    )

    for hardware in sorted(by_hardware):

        regions = by_hardware[hardware]

        pixels = sorted(
            r["outside_right"]
            for r in regions
        )

        fractions = sorted(
            r["outside_right_fraction"]
            for r in regions
        )

        log_entries.append(
            f"  {hardware}: "
            f"n={len(regions)} "
            f"outside_px="
            f"{pixels[0]}..{pixels[-1]} "
            f"outside_fraction="
            f"{fractions[0]:.4f}.."
            f"{fractions[-1]:.4f}"
        )

    # --------------------------------------------------
    # Group the same field / stem across hardware
    # --------------------------------------------------
    capture_groups = defaultdict(list)

    for region in outside_regions:

        key = (
            region["split"],
            region["traffic_type"],
            region["variant"],
            region["file_stem"],
            region["field_name"],
            region["region_provenance_raw"],
        )

        capture_groups[key].append(region)

    multi_hardware_groups = []

    for key, regions in capture_groups.items():

        hardware_sources = {
            r["hardware_source"]
            for r in regions
        }

        if len(hardware_sources) >= 2:
            multi_hardware_groups.append(
                (key, regions)
            )

    log_entries.append(
    "Same stem/field/provenance truncated "
    "across >=2 hardware sources: "
    f"{len(multi_hardware_groups)}"
    )   

    # --------------------------------------------------
    # Measure how similar truncation fraction is
    # across those hardware captures
    # --------------------------------------------------
    spreads = []

    for key, regions in multi_hardware_groups:

        fractions = [
            r["outside_right_fraction"]
            for r in regions
        ]

        spread = max(fractions) - min(fractions)

        spreads.append(
            (spread, key, regions)
        )

    if spreads:

        summary(
            [item[0] for item in spreads],
            "Cross-hardware truncation-fraction spread"
        )

        log_entries.append(
            "Largest cross-hardware differences:"
        )

        for spread, key, regions in sorted(
            spreads, key=lambda item: item[0], reverse=True
        )[:10]:

            split, traffic, variant, stem, field, provenance = key

            log_entries.append(
                f"  {split}/{traffic}/{variant} "
                f"{stem} field={field} "
                f"prov={provenance} "
                f"spread={spread:.4f}"
            )

            for region in sorted(
                regions,
                key=lambda r: r["hardware_source"]
            ):
                log_entries.append(
                    f"    {region['hardware_source']}: "
                    f"outside={region['outside_right']} px "
                    f"width={region['width']} "
                    f"fraction="
                    f"{region['outside_right_fraction']:.4f}"
                )

    write_log(final_log_path,log_entries)

#Function to distinguish original field vs altered field in the same image where its out of bound in one JSON
#Function to detect regions sharing the same image/field/provenance identity
def validate_region_identity(
    region_records: list[dict],
) -> None:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]

    region_groups = defaultdict(list)
    for region in region_records:
        key = (
            region["image_path"],
            region["field_name"],
            region["region_provenance_raw"],
        )
        region_groups[key].append(region)

    duplicate_groups = {
        key: regions
        for key, regions in region_groups.items()
        if len(regions) > 1
    }

    # ---- Diagnostics ----
    log_entries.append(f"Unique image/field/provenance groups: {len(region_groups)}")
    log_entries.append(
        f"Groups occurring more than once within an image: {len(duplicate_groups)}"
    )

    # Distribution of group sizes — is duplication occasional or systematic?
    group_sizes = Counter(len(regions) for regions in region_groups.values())
    log_entries.append("Regions per group:")
    for size, count in sorted(group_sizes.items()):
        log_entries.append(f"  {size} region(s): {count} groups")

    # Which fields duplicate, under which provenance
    dup_by_field = defaultdict(Counter)
    for (_, field_name, provenance), regions in duplicate_groups.items():
        dup_by_field[field_name][provenance] += 1
    if dup_by_field:
        log_entries.append("Duplicate groups by field / provenance:")
        for field_name in sorted(dup_by_field, key=str):
            log_entries.append(f"  {field_name}: {dict(dup_by_field[field_name])}")

    # Distinct boxes sharing a label is expected (two face regions per card).
    # Identical boxes sharing a label is an annotation error.
    identical_box_groups = []
    for key, regions in duplicate_groups.items():
        boxes = [(r["x"], r["y"], r["width"], r["height"]) for r in regions]
        if len(set(boxes)) < len(boxes):
            identical_box_groups.append((key, regions))
    log_entries.append(
        f"Duplicate groups containing identical boxes: {len(identical_box_groups)}"
    )
    if identical_box_groups:
        log_entries.append("  WARNING: identical boxes would double-count mask pixels")
        for key, regions in identical_box_groups[:10]:
            first = regions[0]
            log_entries.append(
                f"    {first['split']}/{first['traffic_type']}/{first['variant']} "
                f"{first['hardware_source']} {first['file_stem']} "
                f"field={key[1]} prov={key[2]} count={len(regions)}"
            )

    if duplicate_groups:
        log_entries.append("Examples of duplicate image/field/provenance groups:")
        for key, regions in sorted(
            duplicate_groups.items(), key=lambda item: str(item[0])
        )[:20]:
            _, field_name, provenance = key
            first = regions[0]
            log_entries.append(
                f"  {first['split']}/{first['traffic_type']}/{first['variant']} "
                f"{first['hardware_source']} {first['file_stem']} "
                f"field={field_name} prov={provenance} count={len(regions)}"
            )
            for region in regions:
                log_entries.append(
                    f"    region={region['region_index']} "
                    f"box=({region['x']}, {region['y']}, "
                    f"{region['width']}, {region['height']})"
                )
        if len(duplicate_groups) > 20:
            log_entries.append(
                f"  ... {len(duplicate_groups) - 20} more not shown"
            )

    # Reconciliation
    total_in_groups = sum(len(r) for r in region_groups.values())
    n_in_duplicates = sum(len(r) for r in duplicate_groups.values())
    log_entries.append(
        f"Reconciliation: total_regions={len(region_records)} "
        f"grouped={total_in_groups} groups={len(region_groups)} "
        f"regions_in_duplicate_groups={n_in_duplicates}"
    )
    if total_in_groups != len(region_records):
        log_entries.append("  WARNING: grouped region count does not reconcile")

    write_log(final_log_path, log_entries)

#Function to extract person info from JSON
def inspect_person_info(
    matched_records: list[dict]
) -> None:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [
        f"********************{func_name}: ********************"
    ]

    for record in matched_records:

        record["person_info_found"] = False
        record["person_info_structure_ok"] = False
        record["person_info_error"] = ""
        record["_person_info_data"] = None

        data = record["_json_data"]

        if not isinstance(data, dict):
            continue

        if "person_info" not in data:
            record["person_info_error"] = (
                "person_info key not found"
            )
            continue

        person_info = data["person_info"]

        if not isinstance(person_info, dict):
            record["person_info_error"] = (
                "person_info is not a dictionary"
            )
            continue

        record["person_info_found"] = True
        record["person_info_structure_ok"] = True
        record["_person_info_data"] = person_info

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    key_patterns = Counter()

    for record in matched_records:

        person_info = record["_person_info_data"]

        if not isinstance(person_info, dict):
            continue

        key_patterns[
            tuple(sorted(person_info.keys()))
        ] += 1

    log_entries.append(
        f"Distinct person_info structures: "
        f"{len(key_patterns)}"
    )

    for pattern, count in sorted(
        key_patterns.items()
    ):
        log_entries.append(
            f"  {count:6d}  "
            + " | ".join(pattern)
        )

    n_found = sum(
        1 for record in matched_records
        if record["person_info_found"]
    )

    n_missing = sum(
        1 for record in matched_records
        if (
            record["json_parse_ok"]
            and not record["person_info_found"]
        )
    )

    log_entries.append(
        f"Reconciliation: total={len(matched_records)} "
        f"person_info_found={n_found} "
        f"missing_or_invalid={n_missing}"
    )

    write_log(
        final_log_path,
        log_entries
    )
#Function to extract person info for each person in the JSON
def extract_person_info(
    matched_records: list[dict]
) -> None:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [
        f"********************{func_name}: ********************"
    ]

    for record in matched_records:

        record["face_db"] = None
        record["face_id"] = None
        record["gender"] = None

        person_info = record["_person_info_data"]

        if not isinstance(person_info, dict):
            continue

        record["face_db"] = person_info.get("face_db")
        record["face_id"] = person_info.get("face_id")
        record["gender"] = person_info.get("gender")

    # -------------------------
    # Diagnostics
    # -------------------------
    for field in ("face_db", "face_id", "gender"):

        missing = sum(
            1
            for record in matched_records
            if record[field] is None
        )

        log_entries.append(
            f"Missing {field}: {missing}"
        )

    face_db_counts = Counter(
        record["face_db"]
        for record in matched_records
    )

    log_entries.append("face_db counts:")

    for value, count in sorted(
        face_db_counts.items(),
        key=lambda item: str(item[0])
    ):
        log_entries.append(
            f"  {value}: {count}"
        )

    gender_counts = Counter(
        record["gender"]
        for record in matched_records
    )

    log_entries.append("gender counts:")

    for value, count in sorted(
        gender_counts.items(),
        key=lambda item: str(item[0])
    ):
        log_entries.append(
            f"  {value}: {count}"
        )

    unique_face_ids = {
        (
            record["face_db"],
            record["face_id"],
        )
        for record in matched_records
        if record["face_id"] is not None
    }

    log_entries.append(
        f"Unique face_db + face_id combinations: "
        f"{len(unique_face_ids)}"
    )

    write_log(
        final_log_path,
        log_entries
    )

def audit_base_card_structure(
    matched_records: list[dict],
) -> None:

    func_name = inspect.currentframe().f_code.co_name

    log_entries = [
        f"********************{func_name}: ********************"
    ]

    # ============================================================
    # 1. file_stem -> face identity
    #
    # A valid base-card key should not point to multiple different
    # (face_db, face_id) identities.
    # ============================================================

    stem_to_identities = defaultdict(set)

    for record in matched_records:

        identity = (
            record["face_db"],
            record["face_id"],
        )

        stem_to_identities[
            record["file_stem"]
        ].add(identity)

    stems_with_multiple_identities = {
        stem: identities
        for stem, identities
        in stem_to_identities.items()
        if len(identities) > 1
    }

    log_entries.append(
        "file_stem -> face identity:"
    )

    log_entries.append(
        f"  unique file_stems: "
        f"{len(stem_to_identities)}"
    )

    log_entries.append(
        f"  stems mapping to >1 face identity: "
        f"{len(stems_with_multiple_identities)}"
    )

    for stem, identities in list(
        sorted(
            stems_with_multiple_identities.items()
        )
    )[:20]:

        log_entries.append(
            f"    {stem}: "
            f"{sorted(identities, key=str)}"
        )


    # ============================================================
    # 2. face identity -> file_stem
    #
    # This checks the reverse direction.
    #
    # If one identity maps to multiple stems, then face identity
    # and card identity are not interchangeable.
    # ============================================================

    identity_to_stems = defaultdict(set)

    for record in matched_records:

        identity = (
            record["face_db"],
            record["face_id"],
        )

        identity_to_stems[
            identity
        ].add(
            record["file_stem"]
        )

    identities_with_multiple_stems = {
        identity: stems
        for identity, stems
        in identity_to_stems.items()
        if len(stems) > 1
    }

    log_entries.append(
        "face identity -> file_stem:"
    )

    log_entries.append(
        f"  unique identities: "
        f"{len(identity_to_stems)}"
    )

    log_entries.append(
        f"  identities mapping to >1 file_stem: "
        f"{len(identities_with_multiple_stems)}"
    )

    for identity, stems in list(
        sorted(
            identities_with_multiple_stems.items(),
            key=lambda item: str(item[0]),
        )
    )[:20]:

        log_entries.append(
            f"    {identity}: "
            f"{sorted(stems)}"
        )


    # ============================================================
    # 3. Training-card cardinality
    #
    # We expect the available train partition to contain:
    #
    #   3 bonafide
    #   3 digital_1
    #   3 digital_2
    #
    # = 9 images per underlying card.
    # ============================================================

    train_records = [
        record
        for record in matched_records
        if record["split"] == "train"
    ]

    train_by_stem = defaultdict(list)

    for record in train_records:

        train_by_stem[
            record["file_stem"]
        ].append(record)

    train_stem_sizes = Counter(
        len(records)
        for records in train_by_stem.values()
    )

    log_entries.append(
        "Training images per file_stem:"
    )

    for count, number_of_stems in sorted(
        train_stem_sizes.items()
    ):

        log_entries.append(
            f"  {count} image(s): "
            f"{number_of_stems} stem(s)"
        )

    train_stems_not_nine = {
        stem: records
        for stem, records
        in train_by_stem.items()
        if len(records) != 9
    }

    log_entries.append(
        f"  train stems not containing exactly "
        f"9 images: "
        f"{len(train_stems_not_nine)}"
    )


    # ============================================================
    # 4. Training composition per stem
    #
    # Every training card should contribute exactly:
    #
    #   bonafide  = 3
    #   digital_1 = 3
    #   digital_2 = 3
    # ============================================================

    bad_train_composition = []

    for stem, records in train_by_stem.items():

        composition = Counter()

        for record in records:

            if record["traffic_type"] == "bonafide":
                label = "bonafide"

            else:
                label = record["variant"]

            composition[label] += 1

        expected = {
            "bonafide": 3,
            "digital_1": 3,
            "digital_2": 3,
        }

        if dict(composition) != expected:

            bad_train_composition.append(
                (
                    stem,
                    dict(composition),
                )
            )

    log_entries.append(
        "Training-card composition:"
    )

    log_entries.append(
        f"  stems with expected "
        f"3 bonafide + 3 digital_1 + "
        f"3 digital_2: "
        f"{len(train_by_stem) - len(bad_train_composition)}"
    )

    log_entries.append(
        f"  stems with unexpected composition: "
        f"{len(bad_train_composition)}"
    )

    for stem, composition in (
        bad_train_composition[:20]
    ):

        log_entries.append(
            f"    {stem}: {composition}"
        )


    # ============================================================
    # 5. Hardware coverage per training card
    #
    # Each of the three versions should occur once for each of:
    #
    #   huawei
    #   iphone15pro
    #   scan
    # ============================================================

    expected_train_hardware = {
        "huawei",
        "iphone15pro",
        "scan",
    }

    bad_hardware_groups = []

    for stem, records in train_by_stem.items():

        by_variant = defaultdict(set)

        for record in records:

            if record["traffic_type"] == "bonafide":
                label = "bonafide"

            else:
                label = record["variant"]

            by_variant[label].add(
                record["hardware_source"]
            )

        for label in (
            "bonafide",
            "digital_1",
            "digital_2",
        ):

            hardware = by_variant.get(
                label,
                set(),
            )

            if hardware != expected_train_hardware:

                bad_hardware_groups.append(
                    (
                        stem,
                        label,
                        hardware,
                    )
                )

    log_entries.append(
        "Training hardware coverage:"
    )

    log_entries.append(
        f"  stem/variant groups with "
        f"unexpected hardware set: "
        f"{len(bad_hardware_groups)}"
    )

    for (
        stem,
        label,
        hardware,
    ) in bad_hardware_groups[:20]:

        log_entries.append(
            f"    {stem} / {label}: "
            f"{sorted(hardware)}"
        )


    # ============================================================
    # 6. Train / test stem overlap
    # ============================================================

    train_stems = set(
        train_by_stem.keys()
    )

    test_records = [
        record
        for record in matched_records
        if record["split"] == "test"
    ]

    test_stems = {
        record["file_stem"]
        for record in test_records
    }

    shared_stems = (
        train_stems
        & test_stems
    )

    log_entries.append(
        "Train / test base-card overlap:"
    )

    log_entries.append(
        f"  train stems: "
        f"{len(train_stems)}"
    )

    log_entries.append(
        f"  test stems: "
        f"{len(test_stems)}"
    )

    log_entries.append(
        f"  shared stems: "
        f"{len(shared_stems)}"
    )


    # ============================================================
    # 7. Which test variants contain the shared train stems?
    #
    # This formally checks the earlier inference that train-card
    # overlap occurs through digital_3.
    # ============================================================

    shared_test_records = [
        record
        for record in test_records
        if record["file_stem"] in shared_stems
    ]

    shared_variant_counts = Counter(
        (
            record["traffic_type"],
            record["variant"],
        )
        for record in shared_test_records
    )

    log_entries.append(
        "Test records belonging to train-seen stems:"
    )

    for (
        traffic_type,
        variant,
    ), count in sorted(
        shared_variant_counts.items(),
        key=lambda item: str(item[0]),
    ):

        variant_label = (
            variant
            if variant
            else "(empty)"
        )

        log_entries.append(
            f"  {traffic_type} / "
            f"{variant_label}: "
            f"{count}"
        )


    # ============================================================
    # 8. Shared stem coverage inside digital_3
    #
    # We want to distinguish:
    #
    #   number of images
    #   number of stems
    #   hardware coverage
    # ============================================================

    shared_digital3 = [
        record
        for record in shared_test_records
        if (
            record["traffic_type"] == "attack"
            and record["variant"] == "digital_3"
        )
    ]

    shared_digital3_stems = {
        record["file_stem"]
        for record in shared_digital3
    }

    log_entries.append(
        "Seen-card digital_3 summary:"
    )

    log_entries.append(
        f"  images: "
        f"{len(shared_digital3)}"
    )

    log_entries.append(
        f"  unique stems: "
        f"{len(shared_digital3_stems)}"
    )

    # Expected if every one of the 211 train cards has
    # one Huawei, iPhone15Pro and scan digital_3 image.
    digital3_bad_hardware = []

    d3_by_stem = defaultdict(set)

    for record in shared_digital3:

        d3_by_stem[
            record["file_stem"]
        ].add(
            record["hardware_source"]
        )

    for stem, hardware in (
        d3_by_stem.items()
    ):

        if (
            hardware
            != expected_train_hardware
        ):

            digital3_bad_hardware.append(
                (
                    stem,
                    hardware,
                )
            )

    log_entries.append(
        f"  shared digital_3 stems with "
        f"unexpected hardware coverage: "
        f"{len(digital3_bad_hardware)}"
    )

    for stem, hardware in (
        digital3_bad_hardware[:20]
    ):

        log_entries.append(
            f"    {stem}: "
            f"{sorted(hardware)}"
        )


    # ============================================================
    # 9. Test-only stems
    #
    # These are useful later for the unseen-card reporting stratum.
    # ============================================================

    unseen_test_stems = (
        test_stems
        - train_stems
    )

    log_entries.append(
        "Test-only stems:"
    )

    log_entries.append(
        f"  unique unseen stems: "
        f"{len(unseen_test_stems)}"
    )

    unseen_variant_counts = Counter(
        (
            record["traffic_type"],
            record["variant"],
        )
        for record in test_records
        if (
            record["file_stem"]
            in unseen_test_stems
        )
    )

    for (
        traffic_type,
        variant,
    ), count in sorted(
        unseen_variant_counts.items(),
        key=lambda item: str(item[0]),
    ):

        variant_label = (
            variant
            if variant
            else "(empty)"
        )

        log_entries.append(
            f"  {traffic_type} / "
            f"{variant_label}: "
            f"{count}"
        )


    # ============================================================
    # 10. Reconciliation
    # ============================================================

    total_train_rows = sum(
        len(records)
        for records in train_by_stem.values()
    )

    log_entries.append(
        "Reconciliation:"
    )

    log_entries.append(
        f"  train records="
        f"{len(train_records)} "
        f"grouped train records="
        f"{total_train_rows}"
    )

    log_entries.append(
        f"  all records="
        f"{len(matched_records)} "
        f"train={len(train_records)} "
        f"test={len(test_records)}"
    )

    write_log(
        final_log_path,
        log_entries
    )

# Function to release JSON objects not needed from memory
def release_json_memory(
    matched_records: list[dict]
) -> None:

    for record in matched_records:

        record.pop("_json_data", None)
        record.pop("_crop_data", None)
        record.pop("_regions_data", None)
        record.pop("_person_info_data", None)

#audit 
def audit_image_metadata(
    matched_records: list[dict]
) -> None:
    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]

    # ---- Simple distributions ----
    for field in (
        "split", "traffic_type", "variant", "hardware_source",
        "crop_info_key", "face_db", "gender",
    ):
        counts = Counter(record[field] for record in matched_records)
        total = sum(counts.values())
        log_entries.append(
            f"{field} counts ({len(counts)} distinct, {total} total):"
        )
        for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
            label = value if value not in ("", None) else "(empty)"
            log_entries.append(f"  {label}: {count}")
        if total != len(matched_records):
            log_entries.append(
                f"  WARNING: total {total} != record count {len(matched_records)}"
            )


    # ---- face_db x crop_info_key ----
    crop_face_counts = Counter(
        (record["face_db"], record["crop_info_key"])
        for record in matched_records
    )

    log_entries.append(
        f"face_db x crop_info_key ({len(crop_face_counts)} combinations, "
        f"{sum(crop_face_counts.values())} total):"
    )
    for (face_db, crop_key), count in sorted(
        crop_face_counts.items(), key=lambda item: str(item[0])
    ):
        log_entries.append(
            f"  {face_db or '(empty)'} | {crop_key or '(empty)'}: {count}"
        )

    # ---- Unique identities per database ----
    identities_by_db = defaultdict(set)
    for record in matched_records:
        identities_by_db[record["face_db"]].add(record["face_id"])
    log_entries.append("Unique face identities by face_db:")
    for face_db in sorted(identities_by_db, key=str):
        log_entries.append(
            f"  {face_db or '(empty)'}: {len(identities_by_db[face_db])}"
        )

    # ---- Identity leakage between train and test ----
    identities_by_split = defaultdict(set)
    for record in matched_records:
        identities_by_split[record["split"]].add(
            (record["face_db"], record["face_id"])
        )
    train_identities = identities_by_split.get("train", set())
    test_identities = identities_by_split.get("test", set())
    shared_identities = train_identities & test_identities

    log_entries.append("Identity split summary:")
    log_entries.append(f"  train unique identities: {len(train_identities)}")
    log_entries.append(f"  test unique identities:  {len(test_identities)}")
    log_entries.append(f"  present in BOTH splits:  {len(shared_identities)}")
    if shared_identities:
        log_entries.append(
            "NOTE: face identities appear in both train and test"
        )
        for face_db, face_id in sorted(shared_identities, key=str)[:20]:
            log_entries.append(f"    {face_db}:{face_id}")
        if len(shared_identities) > 20:
            log_entries.append(f"    ... {len(shared_identities) - 20} more not shown")

    # ---- Card-stem leakage, independent of face identity ----
    stems_by_split = defaultdict(set)
    for record in matched_records:
        stems_by_split[record["split"]].add(record["file_stem"])
    shared_stems = stems_by_split.get("train", set()) & stems_by_split.get("test", set())
    log_entries.append(
        f"Card stems in both train and test: {len(shared_stems)}"
    )
    if shared_stems:
        log_entries.append(f"NOTE: same card stem appears in both train and test")

    write_log(final_log_path, log_entries)


# ================================================================
# DISCOVERY WORKBOOK EXPORT ( Excel code review)
# ================================================================

NULL_MARKER = "<NULL>"

# Only source-text fields where we need to preserve the difference:
#
#   None  -> source field/key absent
#   ""    -> genuine empty string
#
# Numeric/general columns keep None as a true blank Excel cell.
NULL_MARKER_COLUMNS = {
    "field_name",
    "region_provenance_raw",
    "language",
    "val",
    "org_value",
    "new_value",
    "source",
    "target",
}


# ================================================================
# PREFERRED COLUMN ORDER
#
# These lists control ordering only.
# Any future/unrecognised columns are automatically appended.
# ================================================================

IMAGE_COLUMN_ORDER = [
    # ---- Source identity ----
    "split",
    "traffic_type",
    "variant",
    "hardware_source",
    "file_stem",

    # ---- Files / pairing ----
    "image_path",
    "image_sha256",
    "json_path",
    "json_sha256",
    "image_count",
    "json_count",
    "match_status",

    # ---- JSON QA ----
    "json_parse_ok",
    "json_top_level_type",
    "json_top_level_keys",
    "json_error",

    # ---- Person metadata ----
    "face_db",
    "face_id",
    "gender",

    # ---- Person QA ----
    "person_info_found",
    "person_info_structure_ok",
    "person_info_error",

    # ---- Crop metadata ----
    "crop_info_key",
    "original_image_width",
    "original_image_height",
    "resulted_cropped_image_width",
    "resulted_cropped_image_height",
    "transformation_matrix",
    "original_rectangle",

    # ---- Crop / transform QA ----
    "crop_info_found",
    "crop_info_error",
    "matrix_shape_ok",
    "rectangle_shape_ok",
    "crop_field_error",
    "transformed_rectangle",
    "max_corner_error_px",
    "transform_ok",
    "transform_error",

    # ---- On-disk image metadata / QA ----
    "image_width",
    "image_height",
    "exif_orientation",
    "image_metadata_ok",
    "image_metadata_error",

    # ---- Full decode / pixel QA ----
    "image_decode_ok",
    "image_decode_error",
    "image_decode_warning_count",
    "image_decode_warnings",
    "decoded_pixel_sha256",

    # ---- Dimension QA ----
    "dims_match_declared_crop",

    # ---- Region summary / QA ----
    "region_count",
    "regions_found",
    "regions_structure_ok",
    "regions_error",
]


REGION_COLUMN_ORDER = [
    # ---- Parent image ----
    "split",
    "traffic_type",
    "variant",
    "hardware_source",
    "file_stem",
    "image_path",
    "image_sha256",
    "json_path",
    "json_sha256",

    # ---- Annotation identity ----
    "region_index",
    "shape_name",
    "field_name",
    "region_provenance_raw",

    # ---- Raw annotation attributes ----
    "language",
    "val",
    "org_value",
    "new_value",
    "source",
    "target",

    # ---- Raw geometry ----
    "x",
    "y",
    "width",
    "height",
    "right",
    "bottom",

    # ---- Boundary QA ----
    "disk_bounds_status",
    "declared_bounds_status",
    "bounds_check_error",
    "outside_left",
    "outside_top",
    "outside_right",
    "outside_bottom",

    # ---- Visible geometry ----
    "visible_left",
    "visible_top",
    "visible_right",
    "visible_bottom",
    "visible_width",
    "visible_height",
    "original_area",
    "visible_area",
    "visible_fraction",
    "outside_right_fraction",
]


# ================================================================
# EXCEL VALUE CONVERSION
# ================================================================

def make_excel_value(
    value,
    dataset_root: Path,
    column_name: str,
):
    """
    Convert a Python value into an Excel-friendly representation.

    IMPORTANT:
    Scientific QA is performed BEFORE this conversion.
    """

    # ------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------

    if value is None:

        if column_name in NULL_MARKER_COLUMNS:
            return NULL_MARKER

        return None

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    if isinstance(value, Path):

        try:
            return value.relative_to(
                dataset_root
            ).as_posix()

        except ValueError:
            return value.as_posix()

    # ------------------------------------------------------------
    # Structured values
    #
    # Stored as valid JSON text.
    # json.loads() reconstructs the nested values as lists/dicts.
    # ------------------------------------------------------------

    if isinstance(
        value,
        (tuple, list, dict),
    ):

        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return value


# ================================================================
# COLUMN ORDERING
# ================================================================

def reorder_columns(
    dataframe: pd.DataFrame,
    preferred_order: list[str],
) -> pd.DataFrame:

    ordered = [
        column
        for column in preferred_order
        if column in dataframe.columns
    ]

    extra = [
        column
        for column in dataframe.columns
        if column not in ordered
    ]

    return dataframe[
        ordered + extra
    ]


# ================================================================
# EXCEL FORMULA PROTECTION
# ================================================================

def count_formula_like_source_cells(
    *dataframes: pd.DataFrame,
) -> int:
    """
    Count raw strings beginning with '='.

    In XLSX output, openpyxl may otherwise interpret these
    as formulas.
    """

    count = 0

    for dataframe in dataframes:

        for column in dataframe.columns:

            for value in dataframe[column]:

                if (
                    isinstance(value, str)
                    and value.startswith("=")
                ):
                    count += 1

    return count


def force_formula_cells_to_text(
    worksheet,
) -> int:
    """
    Convert Excel formula-typed cells back into literal text cells.

    The original source string is preserved exactly.
    """

    converted = 0

    for row in worksheet.iter_rows(
        min_row=2
    ):

        for cell in row:

            if (
                cell.data_type == "f"
                and isinstance(
                    cell.value,
                    str,
                )
            ):
                cell.data_type = "s"
                converted += 1

    return converted


# ================================================================
# QA SUMMARY
# ================================================================

def build_qa_summary(
    images_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    config: dict,
    provenance: dict,
    formula_like_source_cells: int,
) -> pd.DataFrame:
    """
    Build QA from RAW DataFrames.

    Descriptive QA reports values/rates only.

    PASS / FAIL is emitted ONLY by:
        - Release Gate
        - Regression QA

    Regression semantics:
        0     = check ran and found zero
        None  = check did NOT run
    """

    rows = []

    # ============================================================
    # General row helper
    # ============================================================

    def add_row(
        section,
        metric,
        value,
        expected=None,
        total=None,
        notes="",
    ):

        if (
            total is not None
            and total != 0
            and isinstance(
                value,
                (int, float),
            )
        ):
            rate = value / total

        else:
            rate = None

        if expected is None:
            status = None

        else:
            status = (
                "PASS"
                if value == expected
                else "FAIL"
            )

        rows.append({
            "section": section,
            "metric": metric,
            "value": value,
            "expected": expected,
            "total": total,
            "rate": rate,
            "status": status,
            "notes": notes,
        })

    # ============================================================
    # Descriptive boolean QA
    #
    # These rows deliberately DO NOT produce PASS / FAIL.
    # ============================================================

    def add_boolean_check(
        section,
        metric,
        dataframe,
        column,
    ):

        if column not in dataframe.columns:
            return

        passed = int(
            dataframe[column]
            .eq(True)
            .sum()
        )

        total = len(dataframe)

        add_row(
            section=section,
            metric=metric,
            value=passed,
            total=total,
            notes=(
                f"{total - passed} "
                f"record(s) did not satisfy check"
            ),
        )

    # ============================================================
    # Descriptive distribution helper
    # ============================================================

    def add_distribution(
        section,
        column,
        dataframe,
    ):

        if column not in dataframe.columns:
            return

        counts = (
            dataframe[column]
            .value_counts(
                dropna=False
            )
        )

        total = len(dataframe)

        for value, count in counts.items():

            if pd.isna(value):
                label = "(missing)"

            elif value == "":
                label = "(empty)"

            else:
                label = str(value)

            add_row(
                section=section,
                metric=f"{column} = {label}",
                value=int(count),
                total=total,
            )

    # ============================================================
    # PROVENANCE
    # ============================================================

    add_row(
        "Provenance",
        "Run timestamp",
        provenance["run_timestamp"],
    )

    add_row(
        "Provenance",
        "Config file",
        provenance["config_file"],
    )

    add_row(
        "Provenance",
        "Config SHA-256",
        provenance["config_sha256"],
    )

    add_row(
        "Provenance",
        "Discovery script",
        provenance["script_file"],
    )

    add_row(
        "Provenance",
        "Discovery script SHA-256",
        provenance["script_sha256"],
    )

    add_row(
        "Provenance",
        "Workbook filename",
        provenance["workbook_file"],
    )

    # ============================================================
    # DATASET COUNTS
    # ============================================================

    add_row(
        "Dataset",
        "Image records",
        len(images_df),
    )

    add_row(
        "Dataset",
        "Region records",
        len(regions_df),
    )

    # ============================================================
    # FILE INTEGRITY - DESCRIPTIVE
    # ============================================================

    if "match_status" in images_df.columns:

        matched = int(
            images_df[
                "match_status"
            ]
            .eq("matched")
            .sum()
        )

        add_row(
            "File integrity",
            "Image / JSON pairs matched",
            matched,
            total=len(images_df),
            notes=(
                f"{len(images_df) - matched} "
                f"record(s) not uniquely matched"
            ),
        )

    if "image_sha256" in images_df.columns:

        unique_images = int(
            images_df[
                "image_sha256"
            ]
            .nunique(
                dropna=True
            )
        )

        add_row(
            "File integrity",
            "Unique image SHA-256 values",
            unique_images,
            total=len(images_df),
            notes=(
                f"{len(images_df) - unique_images} "
                f"duplicate value(s)"
            ),
        )

    if "json_sha256" in images_df.columns:

        unique_jsons = int(
            images_df[
                "json_sha256"
            ]
            .nunique(
                dropna=True
            )
        )

        add_row(
            "File integrity",
            "Unique JSON SHA-256 values",
            unique_jsons,
            total=len(images_df),
            notes=(
                f"{len(images_df) - unique_jsons} "
                f"duplicate value(s)"
            ),
        )

    # ============================================================
    # JSON / PERSON QA - DESCRIPTIVE
    # ============================================================

    add_boolean_check(
        "JSON QA",
        "JSON parsed successfully",
        images_df,
        "json_parse_ok",
    )

    add_boolean_check(
        "Person-info QA",
        "person_info found",
        images_df,
        "person_info_found",
    )

    add_boolean_check(
        "Person-info QA",
        "person_info structure valid",
        images_df,
        "person_info_structure_ok",
    )

    # ============================================================
    # CROP QA - DESCRIPTIVE
    # ============================================================

    add_boolean_check(
        "Crop QA",
        "Crop information found",
        images_df,
        "crop_info_found",
    )

    add_boolean_check(
        "Crop QA",
        "Transformation matrix is 3 x 3",
        images_df,
        "matrix_shape_ok",
    )

    add_boolean_check(
        "Crop QA",
        "Original rectangle is 4 x 2",
        images_df,
        "rectangle_shape_ok",
    )

    add_boolean_check(
        "Crop QA",
        "Homography transform successful",
        images_df,
        "transform_ok",
    )

    # ============================================================
    # IMAGE QA - DESCRIPTIVE
    # ============================================================

    add_boolean_check(
        "Image QA",
        "Image metadata read successfully",
        images_df,
        "image_metadata_ok",
    )

    add_boolean_check(
        "Image QA",
        "On-disk dimensions match declared crop",
        images_df,
        "dims_match_declared_crop",
    )

    # ============================================================
    # REGION QA - DESCRIPTIVE
    # ============================================================

    add_boolean_check(
        "Region QA",
        "Regions found",
        images_df,
        "regions_found",
    )

    add_boolean_check(
        "Region QA",
        "Region structure valid",
        images_df,
        "regions_structure_ok",
    )

    # ============================================================
    # REGION-BOUND DISTRIBUTION
    # ============================================================

    if "disk_bounds_status" in regions_df.columns:

        counts = (
            regions_df[
                "disk_bounds_status"
            ]
            .value_counts(
                dropna=False
            )
        )

        for status, count in counts.items():

            add_row(
                "Region bounds",
                (
                    "disk_bounds_status = "
                    f"{status}"
                ),
                int(count),
                total=len(regions_df),
            )

    # ============================================================
    # COMPOUND VALUES USED LATER BY REGRESSION QA
    #
    # IMPORTANT:
    # None = check did not run
    # ============================================================

    altered_partially_outside = None
    images_with_partial_region = None
    face_group_total = None
    face_multi_region_groups = None

    # ------------------------------------------------------------
    # Altered partially-outside regions
    # ------------------------------------------------------------

    if {
        "region_provenance_raw",
        "disk_bounds_status",
    }.issubset(
        regions_df.columns
    ):

        altered_mask = (
            regions_df[
                "region_provenance_raw"
            ]
            .eq("altered")
        )

        altered_total = int(
            altered_mask.sum()
        )

        altered_partially_outside = int(
            (
                altered_mask
                & regions_df[
                    "disk_bounds_status"
                ]
                .eq(
                    "partially_outside"
                )
            )
            .sum()
        )

        add_row(
            "Region bounds",
            (
                "Altered regions partially "
                "outside image"
            ),
            altered_partially_outside,
            total=altered_total,
            notes=(
                "Raw annotation retained; "
                "visible intersection stored separately"
            ),
        )

    # ------------------------------------------------------------
    # Images containing any partial region
    # ------------------------------------------------------------

    if {
        "image_path",
        "disk_bounds_status",
    }.issubset(
        regions_df.columns
    ):

        images_with_partial_region = int(
            regions_df.loc[
                regions_df[
                    "disk_bounds_status"
                ]
                .eq(
                    "partially_outside"
                ),
                "image_path",
            ]
            .nunique()
        )

        add_row(
            "Region bounds",
            (
                "Images containing "
                "partially-outside region"
            ),
            images_with_partial_region,
            total=len(images_df),
        )


    # ------------------------------------------------------------
    # Multiple face regions
    #
    # Group by image_path so this check is independent of SHA
    # uniqueness.
    #
    # Important:
    # If the required columns exist but there are zero face rows,
    # the check DID run and the correct values are 0 / 0.
    # ------------------------------------------------------------

    required_face_columns = {
        "image_path",
        "field_name",
        "region_provenance_raw",
    }

    if required_face_columns.issubset(
        regions_df.columns
    ):

        face_regions = regions_df[
            regions_df[
                "field_name"
            ]
            .eq("face")
        ]

        # The check has run once the required columns exist.
        face_group_total = 0
        face_multi_region_groups = 0

        if not face_regions.empty:

            face_group_sizes = (
                face_regions
                .groupby(
                    [
                        "image_path",
                        "field_name",
                        "region_provenance_raw",
                    ],
                    dropna=False,
                )
                .size()
            )

            face_group_total = int(
                len(
                    face_group_sizes
                )
            )

            face_multi_region_groups = int(
                (
                    face_group_sizes > 1
                )
                .sum()
            )

        add_row(
            "Region structure",
            (
                "Face field/provenance "
                "groups with >1 rectangle"
            ),
            face_multi_region_groups,
            total=face_group_total,
            notes=(
                "Legitimate primary and "
                "secondary/hologram portraits"
            ),
        )

    # ============================================================
    # HUMAN-READABLE DISTRIBUTIONS
    # ============================================================

    for column in (
        "split",
        "traffic_type",
        "variant",
        "hardware_source",
        "crop_info_key",
        "face_db",
        "gender",
    ):

        add_distribution(
            "Image distribution",
            column,
            images_df,
        )

    add_distribution(
        "Region distribution",
        "region_provenance_raw",
        regions_df,
    )

    # ============================================================
    # RELEASE GATE
    #
    # Only this section and Regression QA emit PASS / FAIL.
    # ============================================================

    expected_counts = config.get(
        "expected_counts"
    )

    if not isinstance(
        expected_counts,
        dict,
    ):

        add_row(
            "Release Gate",
            "expected_counts configuration present",
            False,
            expected=True,
            notes=(
                "No expected_counts dictionary "
                "found in configuration"
            ),
        )

    else:

        # --------------------------------------------------------
        # Overall totals
        # --------------------------------------------------------

        expected_images = (
            expected_counts.get(
                "images_total"
            )
        )

        if expected_images is None:

            add_row(
                "Release Gate",
                "Expected total image count configured",
                False,
                expected=True,
            )

        else:

            add_row(
                "Release Gate",
                "Expected total images",
                len(images_df),
                expected=expected_images,
            )

        expected_regions = (
            expected_counts.get(
                "regions_total"
            )
        )

        if expected_regions is None:

            add_row(
                "Release Gate",
                "Expected total region count configured",
                False,
                expected=True,
            )

        else:

            add_row(
                "Release Gate",
                "Expected total regions",
                len(regions_df),
                expected=expected_regions,
            )

        # --------------------------------------------------------
        # Partition expectations
        # --------------------------------------------------------

        partition_specs = (
            expected_counts.get(
                "partitions"
            )
        )

        if not isinstance(
            partition_specs,
            dict,
        ):

            add_row(
                "Release Gate",
                "Partition expectations configured",
                False,
                expected=True,
            )

            partition_specs = {}

        # --------------------------------------------------------
        # Missing split column means partition gates cannot run.
        # Every configured partition/group still gets a FAIL row.
        # --------------------------------------------------------

        if "split" not in images_df.columns:

            add_row(
                "Release Gate",
                "Required source column: split",
                False,
                expected=True,
                notes=(
                    "Partition release checks "
                    "cannot run"
                ),
            )

            for (
                partition_name,
                partition_spec,
            ) in partition_specs.items():

                add_row(
                    "Release Gate",
                    f"{partition_name} image count",
                    None,
                    expected=(
                        partition_spec.get(
                            "total"
                        )
                    ),
                    notes=(
                        "Source column split missing - "
                        "check did not run"
                    ),
                )

                for group in partition_spec.get(
                    "groups",
                    [],
                ):

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} / "
                            f"{group.get('name', '(unnamed group)')}"
                        ),
                        None,
                        expected=group.get(
                            "expected"
                        ),
                        notes=(
                            "Source column split missing - "
                            "check did not run"
                        ),
                    )

        else:

            configured_partitions = set(
                partition_specs.keys()
            )

            discovered_partitions = set(
                images_df[
                    "split"
                ]
                .dropna()
                .unique()
            )

            # ====================================================
            # FORWARD PASS:
            # every expected partition/group must run.
            # ====================================================

            for (
                partition_name,
                partition_spec,
            ) in partition_specs.items():

                if not isinstance(
                    partition_spec,
                    dict,
                ):

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} "
                            "partition configuration valid"
                        ),
                        False,
                        expected=True,
                    )

                    continue

                partition_mask = (
                    images_df[
                        "split"
                    ]
                    .eq(
                        partition_name
                    )
                )

                actual_partition_total = int(
                    partition_mask.sum()
                )

                expected_partition_total = (
                    partition_spec.get(
                        "total"
                    )
                )

                if expected_partition_total is None:

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} "
                            "expected total configured"
                        ),
                        False,
                        expected=True,
                    )

                else:

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} "
                            "image count"
                        ),
                        actual_partition_total,
                        expected=(
                            expected_partition_total
                        ),
                    )

                groups = partition_spec.get(
                    "groups",
                    [],
                )

                if not isinstance(
                    groups,
                    list,
                ):

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} "
                            "group configuration valid"
                        ),
                        False,
                        expected=True,
                    )

                    groups = []

                # ------------------------------------------------
                # Validate expectation policy itself.
                # ------------------------------------------------

                if (
                    groups
                    and expected_partition_total
                    is not None
                ):

                    missing_group_expectations = [
                        group.get(
                            "name",
                            "(unnamed group)",
                        )
                        for group in groups
                        if group.get(
                            "expected"
                        ) is None
                    ]

                    if missing_group_expectations:

                        add_row(
                            "Release Gate",
                            (
                                f"{partition_name} "
                                "configured group totals reconcile"
                            ),
                            None,
                            expected=(
                                expected_partition_total
                            ),
                            notes=(
                                "Expected count missing for: "
                                + ", ".join(
                                    missing_group_expectations
                                )
                            ),
                        )

                    else:

                        expected_group_sum = sum(
                            group[
                                "expected"
                            ]
                            for group in groups
                        )

                        add_row(
                            "Release Gate",
                            (
                                f"{partition_name} "
                                "configured group totals reconcile"
                            ),
                            expected_group_sum,
                            expected=(
                                expected_partition_total
                            ),
                            notes=(
                                "Checks expectation "
                                "policy itself"
                            ),
                        )

                # ------------------------------------------------
                # Count how many configured groups cover each row.
                # ------------------------------------------------

                coverage_count = pd.Series(
                    0,
                    index=images_df.index,
                    dtype="int64",
                )

                for group in groups:

                    if not isinstance(
                        group,
                        dict,
                    ):

                        add_row(
                            "Release Gate",
                            (
                                f"{partition_name} / "
                                "invalid group definition"
                            ),
                            False,
                            expected=True,
                        )

                        continue

                    group_name = group.get(
                        "name",
                        "(unnamed group)",
                    )

                    selector = group.get(
                        "selector",
                        {},
                    )

                    expected = group.get(
                        "expected"
                    )

                    if expected is None:

                        add_row(
                            "Release Gate",
                            (
                                f"{partition_name} / "
                                f"{group_name}"
                            ),
                            None,
                            expected=0,
                            notes=(
                                "Expected count missing "
                                "from configuration"
                            ),
                        )

                        continue

                    if not isinstance(
                        selector,
                        dict,
                    ):

                        add_row(
                            "Release Gate",
                            (
                                f"{partition_name} / "
                                f"{group_name}"
                            ),
                            None,
                            expected=expected,
                            notes=(
                                "Selector is not a dictionary - "
                                "check did not run"
                            ),
                        )

                        continue

                    missing_selector_columns = [
                        column
                        for column
                        in selector.keys()
                        if column
                        not in images_df.columns
                    ]

                    if missing_selector_columns:

                        add_row(
                            "Release Gate",
                            (
                                f"{partition_name} / "
                                f"{group_name}"
                            ),
                            None,
                            expected=expected,
                            notes=(
                                "Selector column(s) missing: "
                                + ", ".join(
                                    missing_selector_columns
                                )
                                + " - check did not run"
                            ),
                        )

                        continue

                    group_mask = (
                        partition_mask.copy()
                    )

                    for (
                        column,
                        expected_value,
                    ) in selector.items():

                        if expected_value is None:

                            group_mask &= (
                                images_df[
                                    column
                                ]
                                .isna()
                            )

                        else:

                            group_mask &= (
                                images_df[
                                    column
                                ]
                                .eq(
                                    expected_value
                                )
                            )

                    actual = int(
                        group_mask.sum()
                    )

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} / "
                            f"{group_name}"
                        ),
                        actual,
                        expected=expected,
                    )

                    coverage_count += (
                        group_mask.astype(
                            "int64"
                        )
                    )

                # ------------------------------------------------
                # Reverse gate within configured partition.
                #
                # Every row must be covered by exactly one
                # configured group when groups are supplied.
                # ------------------------------------------------

                if groups:

                    uncovered_mask = (
                        partition_mask
                        & coverage_count.eq(0)
                    )

                    overlapping_mask = (
                        partition_mask
                        & coverage_count.gt(1)
                    )

                    uncovered_count = int(
                        uncovered_mask.sum()
                    )

                    overlapping_count = int(
                        overlapping_mask.sum()
                    )

                    uncovered_notes = ""

                    if uncovered_count:

                        description_columns = [
                            column
                            for column in (
                                "traffic_type",
                                "variant",
                            )
                            if column
                            in images_df.columns
                        ]

                        if description_columns:

                            combinations = (
                                images_df.loc[
                                    uncovered_mask,
                                    description_columns,
                                ]
                                .drop_duplicates()
                                .head(10)
                            )

                            descriptions = []

                            for _, row in (
                                combinations.iterrows()
                            ):

                                values = []

                                for column in (
                                    description_columns
                                ):

                                    value = row[
                                        column
                                    ]

                                    if pd.isna(value):
                                        value = "(missing)"

                                    elif value == "":
                                        value = "(empty)"

                                    values.append(
                                        str(value)
                                    )

                                descriptions.append(
                                    " / ".join(
                                        values
                                    )
                                )

                            uncovered_notes = (
                                "Unexpected discovered "
                                "group(s): "
                                + "; ".join(
                                    descriptions
                                )
                            )

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} "
                            "rows not covered by "
                            "configured groups"
                        ),
                        uncovered_count,
                        expected=0,
                        notes=uncovered_notes,
                    )

                    add_row(
                        "Release Gate",
                        (
                            f"{partition_name} "
                            "rows matched by >1 "
                            "configured group"
                        ),
                        overlapping_count,
                        expected=0,
                        notes=(
                            "Configured group selectors "
                            "must not overlap"
                        ),
                    )

            # ====================================================
            # REVERSE PASS:
            # discovered split absent from config -> FAIL.
            # ====================================================

            unexpected_partitions = (
                discovered_partitions
                - configured_partitions
            )

            for partition_name in sorted(
                unexpected_partitions,
                key=str,
            ):

                count = int(
                    images_df[
                        "split"
                    ]
                    .eq(
                        partition_name
                    )
                    .sum()
                )

                add_row(
                    "Release Gate",
                    (
                        "Unexpected discovered split: "
                        f"{partition_name}"
                    ),
                    count,
                    expected=0,
                    notes=(
                        "Dataset contains a split "
                        "not represented in expected_counts"
                    ),
                )

    # ============================================================
    # REGRESSION QA
    #
    # 0    = check ran and found zero
    # None = check did not run
    # ============================================================

    regression_actuals = {}
    regression_notes = {}

    # ------------------------------------------------------------
    # JSON parse failures
    # ------------------------------------------------------------

    if "json_parse_ok" in images_df.columns:

        regression_actuals[
            "json_parse_failures"
        ] = int(
            images_df[
                "json_parse_ok"
            ]
            .eq(False)
            .sum()
        )

        regression_notes[
            "json_parse_failures"
        ] = ""

    else:

        regression_actuals[
            "json_parse_failures"
        ] = None

        regression_notes[
            "json_parse_failures"
        ] = (
            "Source column json_parse_ok missing - "
            "check did not run"
        )

    # ------------------------------------------------------------
    # Dimension mismatches
    # ------------------------------------------------------------

    if (
        "dims_match_declared_crop"
        in images_df.columns
    ):

        regression_actuals[
            "dimension_mismatches"
        ] = int(
            images_df[
                "dims_match_declared_crop"
            ]
            .eq(False)
            .sum()
        )

        regression_notes[
            "dimension_mismatches"
        ] = ""

    else:

        regression_actuals[
            "dimension_mismatches"
        ] = None

        regression_notes[
            "dimension_mismatches"
        ] = (
            "Source column "
            "dims_match_declared_crop missing - "
            "check did not run"
        )

    # ------------------------------------------------------------
    # Region bounds
    #
    # A bounds regression check is valid only if EVERY region has
    # one of the recognised completed statuses.
    #
    # This prevents an unprocessed/invalid region from being
    # confused with "zero completely-outside regions".
    # ------------------------------------------------------------

    valid_bounds_statuses = {
        "fully_inside",
        "inside_touching_boundary",
        "partially_outside",
        "completely_outside",
    }

    if (
        "disk_bounds_status"
        in regions_df.columns
    ):

        bounds_complete = (
            regions_df[
                "disk_bounds_status"
            ]
            .isin(
                valid_bounds_statuses
            )
            .all()
        )

    else:

        bounds_complete = False

    if bounds_complete:

        regression_actuals[
            "partially_outside_regions"
        ] = int(
            regions_df[
                "disk_bounds_status"
            ]
            .eq(
                "partially_outside"
            )
            .sum()
        )

        regression_actuals[
            "completely_outside_regions"
        ] = int(
            regions_df[
                "disk_bounds_status"
            ]
            .eq(
                "completely_outside"
            )
            .sum()
        )

        regression_notes[
            "partially_outside_regions"
        ] = ""

        regression_notes[
            "completely_outside_regions"
        ] = ""

    else:

        regression_actuals[
            "partially_outside_regions"
        ] = None

        regression_actuals[
            "completely_outside_regions"
        ] = None

        if (
            "disk_bounds_status"
            not in regions_df.columns
        ):

            reason = (
                "Source column "
                "disk_bounds_status missing"
            )

        else:

            incomplete_count = int(
                (
                    ~regions_df[
                        "disk_bounds_status"
                    ]
                    .isin(
                        valid_bounds_statuses
                    )
                )
                .sum()
            )

            reason = (
                f"{incomplete_count} region "
                "bounds check(s) incomplete "
                "or invalid"
            )

        regression_notes[
            "partially_outside_regions"
        ] = (
            f"{reason} - "
            "check did not run"
        )

        regression_notes[
            "completely_outside_regions"
        ] = (
            f"{reason} - "
            "check did not run"
        )

    # ------------------------------------------------------------
    # Compound checks
    # ------------------------------------------------------------

    regression_actuals[
        "altered_partially_outside_regions"
    ] = altered_partially_outside

    regression_actuals[
        "images_with_partially_outside_region"
    ] = images_with_partial_region

    regression_actuals[
        "face_field_provenance_groups_total"
    ] = face_group_total

    regression_actuals[
        "face_multi_region_groups"
    ] = face_multi_region_groups

    if altered_partially_outside is None:

        regression_notes[
            "altered_partially_outside_regions"
        ] = (
            "Required provenance/bounds columns "
            "missing - check did not run"
        )

    else:

        regression_notes[
            "altered_partially_outside_regions"
        ] = ""

    if images_with_partial_region is None:

        regression_notes[
            "images_with_partially_outside_region"
        ] = (
            "Required image_path/bounds columns "
            "missing - check did not run"
        )

    else:

        regression_notes[
            "images_with_partially_outside_region"
        ] = ""

    if face_group_total is None:

        regression_notes[
            "face_field_provenance_groups_total"
        ] = (
            "Required face-group columns "
            "missing - check did not run"
        )

        regression_notes[
            "face_multi_region_groups"
        ] = (
            "Required face-group columns "
            "missing - check did not run"
        )

    else:

        regression_notes[
            "face_field_provenance_groups_total"
        ] = ""

        regression_notes[
            "face_multi_region_groups"
        ] = ""

    # ============================================================
    # NEW FREEZE CHECKS
    #
    # These convert the recently completed discovery audits into
    # actual Regression QA gates.
    #
    # Important invariant:
    #
    #   numeric value = check ran
    #   0             = check ran and found none
    #   None          = check did not run completely
    #
    # None must therefore FAIL any configured expectation.
    # ============================================================

    def regression_ran(
        check_name,
        value,
    ):

        regression_actuals[
            check_name
        ] = value

        regression_notes[
            check_name
        ] = ""


    def regression_not_run(
        check_name,
        reason,
    ):

        regression_actuals[
            check_name
        ] = None

        regression_notes[
            check_name
        ] = (
            f"{reason} - check did not run"
        )


    # ============================================================
    # HOMOGRAPHY AUDIT
    # ============================================================

    # ------------------------------------------------------------
    # Transformation failures
    # ------------------------------------------------------------

    if "transform_ok" not in images_df.columns:

        regression_not_run(
            "homography_transform_failures",
            "Source column transform_ok missing",
        )

    elif not images_df[
        "transform_ok"
    ].isin(
        [True, False]
    ).all():

        regression_not_run(
            "homography_transform_failures",
            (
                "transform_ok contains "
                "missing/non-boolean values"
            ),
        )

    else:

        regression_ran(
            "homography_transform_failures",
            int(
                images_df[
                    "transform_ok"
                ]
                .eq(False)
                .sum()
            ),
        )


    # ------------------------------------------------------------
    # Corner residual > 0.5 px
    #
    # We only allow this check to run if EVERY image has a
    # successfully calculated residual.
    # ------------------------------------------------------------

    if (
        "transform_ok"
        not in images_df.columns
        or
        "max_corner_error_px"
        not in images_df.columns
    ):

        regression_not_run(
            "homography_corner_residual_over_0_5px",
            (
                "Required transform/residual "
                "column missing"
            ),
        )

    elif not images_df[
        "transform_ok"
    ].eq(True).all():

        regression_not_run(
            "homography_corner_residual_over_0_5px",
            (
                "Not every image completed "
                "homography transformation"
            ),
        )

    else:

        residuals = pd.to_numeric(
            images_df[
                "max_corner_error_px"
            ],
            errors="coerce",
        )

        if residuals.isna().any():

            regression_not_run(
                "homography_corner_residual_over_0_5px",
                (
                    "One or more corner residuals "
                    "are missing/non-numeric"
                ),
            )

        else:

            regression_ran(
                "homography_corner_residual_over_0_5px",
                int(
                    residuals.gt(
                        0.5
                    ).sum()
                ),
            )


    # ============================================================
    # FULL IMAGE DECODE AUDIT
    # ============================================================

    # ------------------------------------------------------------
    # Decode failures
    # ------------------------------------------------------------

    if (
        "image_decode_ok"
        not in images_df.columns
    ):

        regression_not_run(
            "image_decode_failures",
            (
                "Source column "
                "image_decode_ok missing"
            ),
        )

    elif not images_df[
        "image_decode_ok"
    ].isin(
        [True, False]
    ).all():

        regression_not_run(
            "image_decode_failures",
            (
                "image_decode_ok contains "
                "missing/non-boolean values"
            ),
        )

    else:

        regression_ran(
            "image_decode_failures",
            int(
                images_df[
                    "image_decode_ok"
                ]
                .eq(False)
                .sum()
            ),
        )


    # ------------------------------------------------------------
    # Pillow warning count
    # ------------------------------------------------------------

    if (
        "image_decode_warning_count"
        not in images_df.columns
    ):

        regression_not_run(
            "images_with_decode_warnings",
            (
                "Source column "
                "image_decode_warning_count missing"
            ),
        )

    else:

        warning_counts = pd.to_numeric(
            images_df[
                "image_decode_warning_count"
            ],
            errors="coerce",
        )

        if warning_counts.isna().any():

            regression_not_run(
                "images_with_decode_warnings",
                (
                    "Decode warning count contains "
                    "missing/non-numeric values"
                ),
            )

        else:

            regression_ran(
                "images_with_decode_warnings",
                int(
                    warning_counts.gt(
                        0
                    ).sum()
                ),
            )


    # ============================================================
    # HASH DUPLICATE AUDIT
    # ============================================================

    def complete_hash_series(
        column_name,
    ):

        if (
            column_name
            not in images_df.columns
        ):
            return None

        values = (
            images_df[
                column_name
            ]
            .replace(
                "",
                pd.NA,
            )
        )

        if values.isna().any():
            return None

        return values


    # ------------------------------------------------------------
    # Exact source-file duplicates
    # ------------------------------------------------------------

    image_hashes = complete_hash_series(
        "image_sha256"
    )

    if image_hashes is None:

        regression_not_run(
            "duplicate_image_sha256_groups",
            (
                "image_sha256 missing or "
                "incomplete"
            ),
        )

    else:

        hash_counts = (
            image_hashes
            .value_counts()
        )

        regression_ran(
            "duplicate_image_sha256_groups",
            int(
                hash_counts.gt(
                    1
                ).sum()
            ),
        )


    # ------------------------------------------------------------
    # Decoded RGB pixel duplicates
    # ------------------------------------------------------------

    decoded_hashes = complete_hash_series(
        "decoded_pixel_sha256"
    )

    if decoded_hashes is None:

        regression_not_run(
            "duplicate_decoded_pixel_sha256_groups",
            (
                "decoded_pixel_sha256 missing "
                "or incomplete"
            ),
        )

    else:

        decoded_hash_counts = (
            decoded_hashes
            .value_counts()
        )

        regression_ran(
            "duplicate_decoded_pixel_sha256_groups",
            int(
                decoded_hash_counts.gt(
                    1
                ).sum()
            ),
        )


    # ============================================================
    # TRAIN / TEST HASH OVERLAP
    # ============================================================

    if (
        "split"
        not in images_df.columns
    ):

        regression_not_run(
            "train_test_exact_file_overlap",
            "Source column split missing",
        )

        regression_not_run(
            "train_test_decoded_pixel_overlap",
            "Source column split missing",
        )

    else:

        # --------------------------------------------------------
        # Exact source bytes
        # --------------------------------------------------------

        if image_hashes is None:

            regression_not_run(
                "train_test_exact_file_overlap",
                (
                    "image_sha256 missing "
                    "or incomplete"
                ),
            )

        else:

            train_hashes = set(
                images_df.loc[
                    images_df[
                        "split"
                    ].eq(
                        "train"
                    ),
                    "image_sha256",
                ]
            )

            test_hashes = set(
                images_df.loc[
                    images_df[
                        "split"
                    ].eq(
                        "test"
                    ),
                    "image_sha256",
                ]
            )

            regression_ran(
                "train_test_exact_file_overlap",
                len(
                    train_hashes
                    & test_hashes
                ),
            )

        # --------------------------------------------------------
        # Decoded RGB pixels
        # --------------------------------------------------------

        if decoded_hashes is None:

            regression_not_run(
                "train_test_decoded_pixel_overlap",
                (
                    "decoded_pixel_sha256 "
                    "missing or incomplete"
                ),
            )

        else:

            train_pixel_hashes = set(
                images_df.loc[
                    images_df[
                        "split"
                    ].eq(
                        "train"
                    ),
                    "decoded_pixel_sha256",
                ]
            )

            test_pixel_hashes = set(
                images_df.loc[
                    images_df[
                        "split"
                    ].eq(
                        "test"
                    ),
                    "decoded_pixel_sha256",
                ]
            )

            regression_ran(
                "train_test_decoded_pixel_overlap",
                len(
                    train_pixel_hashes
                    & test_pixel_hashes
                ),
            )


    # ============================================================
    # BASE-CARD IDENTITY AUDIT
    # ============================================================

    identity_columns = {
        "file_stem",
        "face_db",
        "face_id",
    }

    if not identity_columns.issubset(
        images_df.columns
    ):

        regression_not_run(
            "file_stems_with_multiple_identities",
            (
                "Required card/identity "
                "column missing"
            ),
        )

        regression_not_run(
            "identities_with_multiple_file_stems",
            (
                "Required card/identity "
                "column missing"
            ),
        )

    elif images_df[
        [
            "file_stem",
            "face_db",
            "face_id",
        ]
    ].isna().any().any():

        regression_not_run(
            "file_stems_with_multiple_identities",
            (
                "Card/identity columns "
                "contain missing values"
            ),
        )

        regression_not_run(
            "identities_with_multiple_file_stems",
            (
                "Card/identity columns "
                "contain missing values"
            ),
        )

    else:

        # --------------------------------------------------------
        # file_stem -> identity
        # --------------------------------------------------------

        unique_stem_identity_pairs = (
            images_df[
                [
                    "file_stem",
                    "face_db",
                    "face_id",
                ]
            ]
            .drop_duplicates()
        )

        identities_per_stem = (
            unique_stem_identity_pairs
            .groupby(
                "file_stem"
            )
            .size()
        )

        regression_ran(
            "file_stems_with_multiple_identities",
            int(
                identities_per_stem.gt(
                    1
                ).sum()
            ),
        )

        # --------------------------------------------------------
        # identity -> file_stem
        # --------------------------------------------------------

        stems_per_identity = (
            unique_stem_identity_pairs
            .groupby(
                [
                    "face_db",
                    "face_id",
                ]
            )[
                "file_stem"
            ]
            .nunique()
        )

        regression_ran(
            "identities_with_multiple_file_stems",
            int(
                stems_per_identity.gt(
                    1
                ).sum()
            ),
        )


    # ============================================================
    # TRAINING CARD STRUCTURE
    # ============================================================

    train_structure_columns = {
        "split",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
    }

    if not train_structure_columns.issubset(
        images_df.columns
    ):

        for check_name in (
            "train_stems_not_nine_images",
            "train_stems_bad_composition",
            "train_stem_variant_hardware_mismatches",
        ):

            regression_not_run(
                check_name,
                (
                    "Required training-card "
                    "column missing"
                ),
            )

    else:

        train_df = images_df[
            images_df[
                "split"
            ].eq(
                "train"
            )
        ]

        if (
            train_df.empty
            or train_df[
                [
                    "file_stem",
                    "traffic_type",
                    "variant",
                    "hardware_source",
                ]
            ].isna().any().any()
        ):

            for check_name in (
                "train_stems_not_nine_images",
                "train_stems_bad_composition",
                "train_stem_variant_hardware_mismatches",
            ):

                regression_not_run(
                    check_name,
                    (
                        "Training rows absent or "
                        "required values missing"
                    ),
                )

        else:

            # ----------------------------------------------------
            # Exactly nine images per train stem
            # ----------------------------------------------------

            train_stem_sizes = (
                train_df
                .groupby(
                    "file_stem"
                )
                .size()
            )

            regression_ran(
                "train_stems_not_nine_images",
                int(
                    train_stem_sizes.ne(
                        9
                    ).sum()
                ),
            )

            # ----------------------------------------------------
            # Exactly:
            #
            #   bonafide  = 3
            #   digital_1 = 3
            #   digital_2 = 3
            # ----------------------------------------------------

            expected_composition = Counter({
                "bonafide": 3,
                "digital_1": 3,
                "digital_2": 3,
            })

            bad_composition = 0

            expected_hardware = {
                "huawei",
                "iphone15pro",
                "scan",
            }

            bad_hardware_groups = 0

            for (
                file_stem,
                stem_df,
            ) in train_df.groupby(
                "file_stem"
            ):

                composition = Counter()

                hardware_by_group = defaultdict(
                    set
                )

                for _, row in (
                    stem_df.iterrows()
                ):

                    if (
                        row[
                            "traffic_type"
                        ]
                        == "bonafide"
                    ):

                        group_name = (
                            "bonafide"
                        )

                    else:

                        group_name = row[
                            "variant"
                        ]

                    composition[
                        group_name
                    ] += 1

                    hardware_by_group[
                        group_name
                    ].add(
                        row[
                            "hardware_source"
                        ]
                    )

                if (
                    composition
                    != expected_composition
                ):

                    bad_composition += 1

                for group_name in (
                    "bonafide",
                    "digital_1",
                    "digital_2",
                ):

                    if (
                        hardware_by_group.get(
                            group_name,
                            set(),
                        )
                        != expected_hardware
                    ):

                        bad_hardware_groups += 1

            regression_ran(
                "train_stems_bad_composition",
                bad_composition,
            )

            regression_ran(
                "train_stem_variant_hardware_mismatches",
                bad_hardware_groups,
            )


    # ============================================================
    # INTENTIONAL TRAIN / TEST CARD REUSE
    #
    # This formally freezes the observed FantasyID structure:
    #
    # 211 train card stems recur in test, and those reused cards
    # appear ONLY through digital_3.
    # ============================================================

    overlap_columns = {
        "split",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
    }

    if not overlap_columns.issubset(
        images_df.columns
    ):

        for check_name in (
            "shared_train_test_stems",
            "shared_train_test_non_digital3_records",
            "shared_train_test_digital3_images",
            "shared_train_test_digital3_stems",
            "shared_digital3_bad_hardware_stems",
        ):

            regression_not_run(
                check_name,
                (
                    "Required train/test card "
                    "column missing"
                ),
            )

    elif images_df[
        [
            "split",
            "file_stem",
            "traffic_type",
            "variant",
            "hardware_source",
        ]
    ].isna().any().any():

        for check_name in (
            "shared_train_test_stems",
            "shared_train_test_non_digital3_records",
            "shared_train_test_digital3_images",
            "shared_train_test_digital3_stems",
            "shared_digital3_bad_hardware_stems",
        ):

            regression_not_run(
                check_name,
                (
                    "Train/test card columns "
                    "contain missing values"
                ),
            )

    else:

        train_stems = set(
            images_df.loc[
                images_df[
                    "split"
                ].eq(
                    "train"
                ),
                "file_stem",
            ]
        )

        test_df = images_df[
            images_df[
                "split"
            ].eq(
                "test"
            )
        ]

        test_stems = set(
            test_df[
                "file_stem"
            ]
        )

        shared_stems = (
            train_stems
            & test_stems
        )

        regression_ran(
            "shared_train_test_stems",
            len(
                shared_stems
            ),
        )

        shared_test_df = test_df[
            test_df[
                "file_stem"
            ].isin(
                shared_stems
            )
        ]

        shared_digital3_mask = (
            shared_test_df[
                "traffic_type"
            ].eq(
                "attack"
            )
            &
            shared_test_df[
                "variant"
            ].eq(
                "digital_3"
            )
        )

        # --------------------------------------------------------
        # Any reused-card test record outside digital_3 would
        # contradict the structure we just established.
        # --------------------------------------------------------

        regression_ran(
            "shared_train_test_non_digital3_records",
            int(
                (
                    ~shared_digital3_mask
                ).sum()
            ),
        )

        shared_digital3_df = (
            shared_test_df[
                shared_digital3_mask
            ]
        )

        regression_ran(
            "shared_train_test_digital3_images",
            len(
                shared_digital3_df
            ),
        )

        digital3_shared_stems = set(
            shared_digital3_df[
                "file_stem"
            ]
        )

        regression_ran(
            "shared_train_test_digital3_stems",
            len(
                digital3_shared_stems
            ),
        )

        # --------------------------------------------------------
        # Every reused digital_3 stem should contain exactly
        # Huawei + iPhone15Pro + scan.
        # --------------------------------------------------------

        expected_hardware = {
            "huawei",
            "iphone15pro",
            "scan",
        }

        bad_shared_d3_hardware = 0

        for (
            file_stem,
            stem_df,
        ) in shared_digital3_df.groupby(
            "file_stem"
        ):

            hardware = set(
                stem_df[
                    "hardware_source"
                ]
            )

            if (
                hardware
                != expected_hardware
            ):

                bad_shared_d3_hardware += 1

        regression_ran(
            "shared_digital3_bad_hardware_stems",
            bad_shared_d3_hardware,
        )

    # ============================================================
    # TEST FLICKR HARDWARE STRUCTURE
    #
    # Raw hardware folder names are deliberately preserved.
    #
    # The Flickr component uses:
    #
    #   huawei
    #   iphone15
    #   scan
    #
    # This is intentionally NOT aliased to iphone15pro here.
    #
    # We audit two structures:
    #
    # 1. Test Flickr bonafide:
    #       every card should have exactly three captures,
    #       one from each raw hardware source.
    #
    # 2. Test Flickr attacks, combining facedancer and
    #    textdiffuserft_bfei:
    #       the same three-capture structure is expected except
    #       for one known source-release incomplete stem.
    # ============================================================

    flickr_hardware_columns = {
        "split",
        "traffic_type",
        "variant",
        "face_db",
        "file_stem",
        "hardware_source",
    }

    flickr_check_names = (
        (
            "test_flickr_bonafide_stems_"
            "bad_hardware_structure"
        ),
        (
            "test_flickr_attack_stems_"
            "bad_hardware_structure"
        ),
    )

    if not flickr_hardware_columns.issubset(
        images_df.columns
    ):

        for check_name in flickr_check_names:

            regression_not_run(
                check_name,
                (
                    "Required Flickr hardware "
                    "audit column missing"
                ),
            )

    else:

        expected_flickr_hardware = {
            "huawei",
            "iphone15",
            "scan",
        }

        def count_bad_flickr_hardware_stems(
            component_df: pd.DataFrame,
        ):

            # Empty is NOT equivalent to a clean result.
            if component_df.empty:
                return None

            if component_df[
                [
                    "file_stem",
                    "hardware_source",
                ]
            ].isna().any().any():

                return None

            bad_stems = []

            for (
                file_stem,
                stem_df,
            ) in component_df.groupby(
                "file_stem"
            ):

                hardware = set(
                    stem_df[
                        "hardware_source"
                    ]
                )

                # Require both:
                #
                #   exactly 3 rows
                #   exactly the expected 3 raw hardware names
                #
                # This also catches duplicate-device rows.
                if (
                    len(stem_df) != 3
                    or
                    hardware
                    != expected_flickr_hardware
                ):

                    bad_stems.append(
                        file_stem
                    )

            return bad_stems


        # --------------------------------------------------------
        # Test Flickr bonafide
        # --------------------------------------------------------

        flickr_bonafide_df = images_df[
            images_df[
                "split"
            ].eq(
                "test"
            )
            &
            images_df[
                "traffic_type"
            ].eq(
                "bonafide"
            )
            &
            images_df[
                "face_db"
            ].eq(
                "flickr-cropped"
            )
        ]

        bad_flickr_bonafide_stems = (
            count_bad_flickr_hardware_stems(
                flickr_bonafide_df
            )
        )

        if bad_flickr_bonafide_stems is None:

            regression_not_run(
                (
                    "test_flickr_bonafide_stems_"
                    "bad_hardware_structure"
                ),
                (
                    "Test Flickr bonafide rows "
                    "absent or incomplete"
                ),
            )

        else:

            regression_ran(
                (
                    "test_flickr_bonafide_stems_"
                    "bad_hardware_structure"
                ),
                len(
                    bad_flickr_bonafide_stems
                ),
            )

            regression_notes[
                (
                    "test_flickr_bonafide_stems_"
                    "bad_hardware_structure"
                )
            ] = (
                "Expected raw hardware set: "
                "huawei, iphone15, scan"
            )


        # --------------------------------------------------------
        # Combined Flickr attack component
        #
        # facedancer and textdiffuserft_bfei are combined because
        # the individual methods are not complete three-device
        # triplets for every identity.
        # --------------------------------------------------------

        flickr_attack_df = images_df[
            images_df[
                "split"
            ].eq(
                "test"
            )
            &
            images_df[
                "traffic_type"
            ].eq(
                "attack"
            )
            &
            images_df[
                "face_db"
            ].eq(
                "flickr-cropped"
            )
            &
            images_df[
                "variant"
            ].isin(
                [
                    "facedancer",
                    "textdiffuserft_bfei",
                ]
            )
        ]

        bad_flickr_attack_stems = (
            count_bad_flickr_hardware_stems(
                flickr_attack_df
            )
        )

        if bad_flickr_attack_stems is None:

            regression_not_run(
                (
                    "test_flickr_attack_stems_"
                    "bad_hardware_structure"
                ),
                (
                    "Test Flickr attack rows "
                    "absent or incomplete"
                ),
            )

        else:

            regression_ran(
                (
                    "test_flickr_attack_stems_"
                    "bad_hardware_structure"
                ),
                len(
                    bad_flickr_attack_stems
                ),
            )

            regression_notes[
                (
                    "test_flickr_attack_stems_"
                    "bad_hardware_structure"
                )
            ] = (
                "Raw hardware names preserved. "
                "Known source-release exception: "
                "chinese2-flickr_F_30990251270_6801e68759_c "
                "has iphone15 + scan attack captures "
                "but no Huawei attack capture."
            )


    # ------------------------------------------------------------
    # Formula-like source strings
    # ------------------------------------------------------------

    regression_actuals[
        "formula_like_source_cells"
    ] = formula_like_source_cells

    regression_notes[
        "formula_like_source_cells"
    ] = ""

    # ------------------------------------------------------------
    # Human-readable labels
    # ------------------------------------------------------------

    regression_labels = {
        "json_parse_failures":
            "JSON parse failures",

        "dimension_mismatches":
            (
                "Image / declared-crop "
                "dimension mismatches"
            ),

        (
            "test_flickr_bonafide_stems_"
            "bad_hardware_structure"
        ):
            (
                "Test Flickr bonafide stems with "
                "unexpected 3-capture hardware structure"
            ),

        (
            "test_flickr_attack_stems_"
            "bad_hardware_structure"
        ):
            (
                "Test Flickr attack stems with "
                "unexpected combined hardware structure"
            ),

        "partially_outside_regions":
            "Partially-outside regions",

        "altered_partially_outside_regions":
            (
                "Altered partially-outside "
                "regions"
            ),

        "images_with_partially_outside_region":
            (
                "Images with partially-outside "
                "region"
            ),

        "completely_outside_regions":
            "Completely-outside regions",

        "face_field_provenance_groups_total":
            (
                "Total face field/provenance "
                "groups"
            ),

        "face_multi_region_groups":
            (
                "Face field/provenance groups "
                "with >1 rectangle"
            ),

        "formula_like_source_cells":
            (
                "Source cells requiring "
                "formula-to-text protection"
            ),

        "homography_transform_failures":
            "Homography transformation failures",

        "homography_corner_residual_over_0_5px":
            (
                "Images with homography corner "
                "residual > 0.5 px"
            ),

        "image_decode_failures":
            "Full RGB image decode failures",

        "images_with_decode_warnings":
            "Images producing Pillow decode warnings",

        "duplicate_image_sha256_groups":
            "Duplicate source-image SHA-256 groups",

        "duplicate_decoded_pixel_sha256_groups":
            "Duplicate decoded-RGB pixel-hash groups",

        "train_test_exact_file_overlap":
            (
                "Train/test exact source-image "
                "SHA-256 overlap"
            ),

        "train_test_decoded_pixel_overlap":
            (
                "Train/test decoded-RGB "
                "pixel-hash overlap"
            ),

        "file_stems_with_multiple_identities":
            (
                "file_stems mapping to "
                ">1 face identity"
            ),

        "identities_with_multiple_file_stems":
            (
                "Face identities mapping to "
                ">1 file_stem"
            ),

        "train_stems_not_nine_images":
            (
                "Training file_stems not "
                "containing exactly 9 images"
            ),

        "train_stems_bad_composition":
            (
                "Training file_stems with "
                "unexpected variant composition"
            ),

        "train_stem_variant_hardware_mismatches":
            (
                "Training stem/variant groups "
                "with unexpected hardware coverage"
            ),

        "shared_train_test_stems":
            "file_stems shared between train and test",

        "shared_train_test_non_digital3_records":
            (
                "Test records from train-seen "
                "stems outside digital_3"
            ),

        "shared_train_test_digital3_images":
            (
                "digital_3 images belonging "
                "to train-seen stems"
            ),

        "shared_train_test_digital3_stems":
            (
                "Train-seen stems represented "
                "in test digital_3"
            ),

        "shared_digital3_bad_hardware_stems":
            (
                "Shared digital_3 stems with "
                "unexpected hardware coverage"
            ),
    }

    # ------------------------------------------------------------
    # Frozen regression expectations
    # ------------------------------------------------------------

    expected_regression = config.get(
        "regression_checks"
    )

    if not isinstance(
        expected_regression,
        dict,
    ):

        add_row(
            "Regression QA",
            (
                "regression_checks "
                "configuration present"
            ),
            False,
            expected=True,
        )

        expected_regression = {}

    # ------------------------------------------------------------
    # Forward regression pass:
    # every configured expectation emits a row.
    # ------------------------------------------------------------

    for (
        check_name,
        expected_value,
    ) in expected_regression.items():

        if check_name not in regression_actuals:

            add_row(
                "Regression QA",
                (
                    "Unknown configured "
                    "regression check: "
                    f"{check_name}"
                ),
                None,
                expected=expected_value,
                notes=(
                    "No calculation exists for "
                    "this configured check"
                ),
            )

            continue

        add_row(
            "Regression QA",
            regression_labels[
                check_name
            ],
            regression_actuals[
                check_name
            ],
            expected=expected_value,
            notes=(
                regression_notes.get(
                    check_name,
                    ""
                )
            ),
        )

    # ------------------------------------------------------------
    # Calculated checks that have no frozen expectation
    # remain visible but do not PASS / FAIL.
    # ------------------------------------------------------------

    unconfigured_regressions = (
        set(
            regression_actuals.keys()
        )
        - set(
            expected_regression.keys()
        )
    )

    for check_name in sorted(
        unconfigured_regressions
    ):

        add_row(
            "Regression QA",
            regression_labels[
                check_name
            ],
            regression_actuals[
                check_name
            ],
            notes=(
                regression_notes.get(
                    check_name,
                    ""
                )
                or
                "No frozen expectation configured"
            ),
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# MAIN EXCEL EXPORTER
# ================================================================

def export_inventory_to_excel(
    matched_records: list[dict],
    region_records: list[dict],
    config: dict,
    config_path: Path,
) -> None:

    func_name = (
        inspect.currentframe()
        .f_code.co_name
    )

    log_entries = [
        f"********************"
        f"{func_name}: "
        f"********************"
    ]

    dataset_root = Path(
        config["dataset"]["root"]
    )

    output_dir = Path(
        config["output"]["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = Path(
        config["output"]["workbook_name"]
    )

    # Timestamped name avoids accidental overwrite of a frozen run.
    run_timestamp = RUN_TIMESTAMP

    output_path = (
        output_dir
        / (
            f"{base_name.stem}_"
            f"{run_timestamp}"
            f"{base_name.suffix}"
        )
    )

    # Temporary workbook still ends in .xlsx.
    temp_output_path = (
        output_dir
        / (
            f"{base_name.stem}_"
            f"{run_timestamp}"
            f".part"
            f"{base_name.suffix}"
        )
    )

    sidecar_path = Path(
        str(output_path)
        + ".sha256"
    )

    temp_sidecar_path = Path(
        str(sidecar_path)
        + ".part"
    )

    # ------------------------------------------------------------
    # Never overwrite final artifacts.
    # ------------------------------------------------------------

    if output_path.exists():

        raise FileExistsError(
            f"Final workbook already exists: "
            f"{output_path}"
        )

    if sidecar_path.exists():

        raise FileExistsError(
            f"SHA sidecar already exists: "
            f"{sidecar_path}"
        )

    # Remove temporary files belonging to THIS run identifier.
    #
    # We deliberately do not glob-delete .part files from older
    # runs because those may be useful evidence of a failed run.
    if temp_output_path.exists():
        temp_output_path.unlink()

    if temp_sidecar_path.exists():
        temp_sidecar_path.unlink()

    # ============================================================
    # PROVENANCE
    # ============================================================

    config_path = Path(
        config_path
    ).resolve()

    script_path = Path(
        __file__
    ).resolve()

    config_sha256 = calculate_sha256(
        config_path
    )

    script_sha256 = calculate_sha256(
        script_path
    )

    provenance = {
        "run_timestamp":
            run_timestamp,

        "config_file":
            config_path.name,

        "config_sha256":
            config_sha256,

        "script_file":
            script_path.name,

        "script_sha256":
            script_sha256,

        "workbook_file":
            output_path.name,
    }

    # ============================================================
    # RAW DATAFRAMES
    #
    # These preserve Python values and drive all QA.
    # ============================================================

    def raw_rows(
        records: list[dict],
    ) -> list[dict]:

        return [
            {
                key: value
                for key, value
                in record.items()
                if not key.startswith("_")
            }
            for record in records
        ]

    images_raw_df = pd.DataFrame(
        raw_rows(
            matched_records
        )
    )

    regions_raw_df = pd.DataFrame(
        raw_rows(
            region_records
        )
    )

    formula_like_source_cells = (
        count_formula_like_source_cells(
            images_raw_df,
            regions_raw_df,
        )
    )

    # ============================================================
    # QA FROM RAW VALUES
    # ============================================================

    qa_df = build_qa_summary(
        images_raw_df,
        regions_raw_df,
        config,
        provenance,
        formula_like_source_cells,
    )

    # ============================================================
    # EXCEL-SAFE COPIES
    # ============================================================

    def excel_rows(
        records: list[dict],
    ) -> list[dict]:

        return [
            {
                key: make_excel_value(
                    value=value,
                    dataset_root=dataset_root,
                    column_name=key,
                )
                for key, value
                in record.items()
                if not key.startswith("_")
            }
            for record in records
        ]

    images_excel_df = pd.DataFrame(
        excel_rows(
            matched_records
        )
    )

    regions_excel_df = pd.DataFrame(
        excel_rows(
            region_records
        )
    )

    # ============================================================
    # LOGICAL COLUMN ORDER
    #
    # No data columns are deleted.
    # ============================================================

    images_excel_df = reorder_columns(
        images_excel_df,
        IMAGE_COLUMN_ORDER,
    )

    regions_excel_df = reorder_columns(
        regions_excel_df,
        REGION_COLUMN_ORDER,
    )

    # ============================================================
    # PRE-WRITE RECONCILIATION
    # ============================================================

    if (
        len(images_excel_df)
        != len(matched_records)
    ):

        raise RuntimeError(
            "Image row count does not reconcile "
            "before workbook export"
        )

    if (
        len(regions_excel_df)
        != len(region_records)
    ):

        raise RuntimeError(
            "Region row count does not reconcile "
            "before workbook export"
        )

    # ============================================================
    # WRITE TEMPORARY WORKBOOK
    #
    # A workbook never receives its final name until this entire
    # section completes successfully.
    # ============================================================

    formula_cells_converted = 0

    try:

        with pd.ExcelWriter(
            temp_output_path,
            engine="openpyxl",
        ) as writer:

            # ----------------------------------------------------
            # Worksheets
            # ----------------------------------------------------

            qa_df.to_excel(
                writer,
                sheet_name="QA_Summary",
                index=False,
            )

            images_excel_df.to_excel(
                writer,
                sheet_name="Images",
                index=False,
            )

            regions_excel_df.to_excel(
                writer,
                sheet_name="Regions",
                index=False,
            )

            # ----------------------------------------------------
            # Formula protection.
            #
            # QA_Summary is not source data, so only Images and
            # Regions participate in reconciliation.
            # ----------------------------------------------------

            for sheet_name in (
                "Images",
                "Regions",
            ):

                formula_cells_converted += (
                    force_formula_cells_to_text(
                        writer.sheets[
                            sheet_name
                        ]
                    )
                )

            if (
                formula_cells_converted
                != formula_like_source_cells
            ):

                raise RuntimeError(
                    "Formula-protection count "
                    "does not reconcile: "
                    f"raw="
                    f"{formula_like_source_cells}, "
                    f"converted="
                    f"{formula_cells_converted}"
                )

            # ----------------------------------------------------
            # Freeze headers and add filters.
            # ----------------------------------------------------

            for sheet_name in (
                "QA_Summary",
                "Images",
                "Regions",
            ):

                worksheet = (
                    writer.sheets[
                        sheet_name
                    ]
                )

                worksheet.freeze_panes = (
                    "A2"
                )

                worksheet.auto_filter.ref = (
                    worksheet.dimensions
                )

            # ----------------------------------------------------
            # QA sheet formatting.
            # ----------------------------------------------------

            qa_sheet = writer.sheets[
                "QA_Summary"
            ]

            qa_widths = {
                "A": 24,   # section
                "B": 60,   # metric
                "C": 18,   # value
                "D": 18,   # expected
                "E": 14,   # total
                "F": 14,   # rate
                "G": 12,   # status
                "H": 80,   # notes
            }

            for (
                column_letter,
                width,
            ) in qa_widths.items():

                qa_sheet.column_dimensions[
                    column_letter
                ].width = width

            # Rate column = F.
            for row_number in range(
                2,
                qa_sheet.max_row + 1,
            ):

                qa_sheet.cell(
                    row=row_number,
                    column=6,
                ).number_format = (
                    "0.00%"
                )

    except Exception as error:

        # ExcelWriter may save its temporary file while unwinding.
        # It must never remain as a candidate discovery artifact.
        if temp_output_path.exists():

            try:
                temp_output_path.unlink()

            except OSError:
                pass

        if temp_sidecar_path.exists():

            try:
                temp_sidecar_path.unlink()

            except OSError:
                pass

        log_entries.append(
            "ERROR writing temporary workbook: "
            f"{error}"
        )

        write_log(
            final_log_path,
            log_entries,
        )

        raise

    # ============================================================
    # TEMPORARY WORKBOOK MUST NOW EXIST
    # ============================================================

    if not temp_output_path.exists():

        raise RuntimeError(
            "Temporary workbook was not created"
        )

    # ============================================================
    # HASH THE COMPLETE TEMPORARY WORKBOOK
    # ============================================================

    workbook_sha256 = calculate_sha256(
        temp_output_path
    )

    # Prepare the matching sidecar before promoting the workbook.
    temp_sidecar_path.write_text(
        (
            f"{workbook_sha256}  "
            f"{output_path.name}\n"
        ),
        encoding="utf-8",
    )

    # ============================================================
    # ATOMIC PROMOTION
    #
    # A final-named workbook is never created until the temporary
    # workbook has completely closed and passed export checks.
    # ============================================================

    workbook_promoted = False

    try:

        # Atomic rename on the same filesystem.
        os.replace(
            temp_output_path,
            output_path,
        )

        workbook_promoted = True

        os.replace(
            temp_sidecar_path,
            sidecar_path,
        )

    except Exception:

        # Clean remaining temporary files.
        if temp_output_path.exists():

            try:
                temp_output_path.unlink()

            except OSError:
                pass

        if temp_sidecar_path.exists():

            try:
                temp_sidecar_path.unlink()

            except OSError:
                pass

        # If the workbook was promoted but the sidecar promotion
        # failed, remove the workbook where possible.
        if (
            workbook_promoted
            and output_path.exists()
            and not sidecar_path.exists()
        ):

            try:
                output_path.unlink()

            except OSError:
                pass

        raise

    # ============================================================
    # POST-PROMOTION HASH VERIFICATION
    # ============================================================

    final_sha256 = calculate_sha256(
        output_path
    )

    if (
        final_sha256
        != workbook_sha256
    ):

        # Do not leave a mismatched final artifact behind.
        try:
            output_path.unlink()
        except OSError:
            pass

        try:
            sidecar_path.unlink()
        except OSError:
            pass

        raise RuntimeError(
            "Final workbook hash differs "
            "from temporary workbook hash"
        )

    # ============================================================
    # FINAL EXPORT DIAGNOSTICS
    # ============================================================

    qa_failures = int(
        qa_df[
            "status"
        ]
        .eq("FAIL")
        .sum()
    )

    dropped_image_cols = sorted(
        key
        for key in (
            matched_records[0]
            if matched_records
            else {}
        )
        if key.startswith("_")
    )

    dropped_region_cols = sorted(
        key
        for key in (
            region_records[0]
            if region_records
            else {}
        )
        if key.startswith("_")
    )

    extra_image_cols = [
        column
        for column
        in images_excel_df.columns
        if column
        not in IMAGE_COLUMN_ORDER
    ]

    extra_region_cols = [
        column
        for column
        in regions_excel_df.columns
        if column
        not in REGION_COLUMN_ORDER
    ]

    log_entries.append(
        f"Workbook written to: "
        f"{output_path}"
    )

    log_entries.append(
        f"Workbook SHA-256: "
        f"{final_sha256}"
    )

    log_entries.append(
        f"SHA sidecar: "
        f"{sidecar_path}"
    )

    log_entries.append(
        f"Config SHA-256: "
        f"{config_sha256}"
    )

    log_entries.append(
        f"Discovery script SHA-256: "
        f"{script_sha256}"
    )

    log_entries.append(
        f"QA_Summary: "
        f"{len(qa_df)} rows x "
        f"{len(qa_df.columns)} columns"
    )

    log_entries.append(
        f"Images: "
        f"{len(images_excel_df)} rows x "
        f"{len(images_excel_df.columns)} columns"
    )

    log_entries.append(
        f"Regions: "
        f"{len(regions_excel_df)} rows x "
        f"{len(regions_excel_df.columns)} columns"
    )

    log_entries.append(
        f"QA FAIL rows: "
        f"{qa_failures}"
    )

    if qa_failures:

        log_entries.append(
            "NOTE: workbook exported successfully "
            "but is NOT release-clean because "
            "Release Gate / Regression QA "
            "contains FAIL row(s)"
        )

    else:

        log_entries.append(
            "Release Gate / Regression QA: "
            "all configured checks PASS"
        )

    log_entries.append(
        "Formula protection:"
    )

    log_entries.append(
        f"  Raw '='-leading source cells: "
        f"{formula_like_source_cells}"
    )

    log_entries.append(
        f"  Formula cells converted to text: "
        f"{formula_cells_converted}"
    )

    log_entries.append(
        "Columns dropped from Images "
        "(leading underscore only): "
        f"{dropped_image_cols}"
    )

    log_entries.append(
        "Columns dropped from Regions "
        "(leading underscore only): "
        f"{dropped_region_cols}"
    )

    if extra_image_cols:

        log_entries.append(
            "New/unordered image columns "
            "appended automatically: "
            f"{extra_image_cols}"
        )

    if extra_region_cols:

        log_entries.append(
            "New/unordered region columns "
            "appended automatically: "
            f"{extra_region_cols}"
        )

    log_entries.append(
        "Images sheet columns:"
    )

    log_entries.append(
        "  "
        + " | ".join(
            images_excel_df.columns
        )
    )

    log_entries.append(
        "Regions sheet columns:"
    )

    log_entries.append(
        "  "
        + " | ".join(
            regions_excel_df.columns
        )
    )

    log_entries.append(
        "Reconciliation: "
        f"matched_records="
        f"{len(matched_records)} "
        f"images_rows="
        f"{len(images_excel_df)} "
        f"region_records="
        f"{len(region_records)} "
        f"regions_rows="
        f"{len(regions_excel_df)}"
    )

    write_log(
        final_log_path,
        log_entries,
    )


#Main Execution
if __name__ == "__main__":
    #load the yaml dictionary
    config = load_config(CONFIG_FILE)

    #log path
    final_log_path = build_log_path(config)

    #traverse data splits per yaml dictionary 
    split_paths = find_dataset_splits(config)

    #traverse traffic type split under data split
    traffic_path = find_traffic_types(split_paths,config)

    #get folders that have images and JSON files for each traffic type. 
    data_folders = find_data_folders(traffic_path)

    # Get all discovered files. 
    discovered_files = discover_files(data_folders,config)

    # add hash values to the existing dict discovered_files
    add_file_hashes(discovered_files)

    #Match images to their JSON's using groups and their status
    matched_records = match_images_and_jsons(discovered_files)
    
    #Parse JSON files.
    parse_json_files(matched_records)

    #only inspect person info from JSON. Helps discover legitimate variants, before flattening them.
    inspect_person_info(matched_records)

    #extract person info from JSON
    extract_person_info(matched_records)

    # Based on above result, The 362 identities across 3,284 images also makes intuitive sense.
    # The same underlying person/card identity appears in multiple captures, devices, and/or manipulation versions.

    #Extract cropping info
    find_crop_information(matched_records) 

    #Debug - Diagnostic prints to check fully qualified matched records after adding crop values. 
    # print(type(matched_records))
    # for record in matched_records[0:2]:
    #     crop_data = record["_crop_data"]
    #     print("type of crop data - ", type(crop_data))
    #     print("type of record    - ",  type(record))
    #     record["original_image_width"] = (crop_data.get("original_image_width"))
    #     print(record["original_image_width"])
    #     print("tyep of orig image width" , type(record["original_image_width"]))

    # extract crop fields
    extract_crop_fields(matched_records) 

    # Memory consideration : Images are always streamed from disk. JSONs are temporarily retained only while 
    # we extract all required annotation information. Once extraction is complete, the original parsed JSON objects 
    # are explicitly discarded.

    #Tranform rectangles for each image without changing the source or rounding transformed float to INT.
    transform_rectangles(matched_records)

    #Get each images exif orientation and validate its height and width 
    extract_image_metadata(matched_records) 

    #Validate full image
    validate_full_image_decode(matched_records)
    
    #Extract regions which has labelled information on altered areas in attack images
    inspect_regions(matched_records)

    #Extract data from regions 
    region_records = extract_region_records(matched_records)

    #second review needed.****************************************************

    #validate regions
    validate_region_bounds(region_records,matched_records)

    # Analysis of regions which are out of bound. 
    analyse_region_visibility(region_records,matched_records)

    # Region annotations appear to be expressed in the final cropped-image coordinate frame. 
    # A small but systematic subset—predominantly DOB fields—extends past the final right crop boundary. 
    # This affects 514 regions, including 171 altered regions, with all regions retaining at least 75.6% visibility.

    #Function to analyse the right edge trunction 
    analyse_right_edge_truncation(region_records)

    #validation regions and their identity
    validate_region_identity(region_records)

    #clear fields of JSOn not required. 
    release_json_memory(matched_records)

    #Audit base card 
    audit_base_card_structure(matched_records)

    #Audit function
    audit_image_metadata(matched_records)

    # ------------------------------------------------------------
    # SOURCE TEST STRUCTURE DISCOVERED
    # ------------------------------------------------------------
    #
    # Total test images = 1385
    #
    # digital_3 = 786
    #
    #   211 cards already represented in source train:
    #
    #       211 cards x 3 hardware captures = 633 images
    #
    #   51 test-only HQ-WMCA-HFACE cards:
    #
    #       51 cards x 3 hardware captures = 153 images
    #
    #       633 + 153 = 786 digital_3 images
    #
    # Altered/recaptured Flickr component = 599:
    #
    #   bonafide             = 300
    #   facedancer           = 150
    #   textdiffuserft_bfei  = 149
    #
    #   total                = 599
    #
    # Test structure may be audited for integrity and reporting.
    # Test labels, distributions or model outputs must not be used
    # for fitting, threshold selection, hyperparameter selection,
    # model selection, or model-facing semantic-policy decisions.


    # ------------------------------------------------------------
    # INTERNAL VALIDATION PLAN
    # ------------------------------------------------------------
    #
    # The official 459-image FantasyID validation set is unavailable
    # under the applicable access conditions for this project.
    #
    # Discovery has established that every source-train card contains
    # exactly 9 images:
    #
    #   3 bonafide
    #   3 digital_1
    #   3 digital_2
    #
    # Therefore:
    #
    #   51 whole cards x 9 images = exactly 459 images
    #
    # A deterministic, card-disjoint 51-card internal validation set
    # can therefore reproduce the official validation-set SIZE without
    # splitting an underlying card across project train and dev_val.
    #
    # The 51 cards have NOT yet been selected at discovery stage.
    # Selection policy belongs to the downstream split-construction
    # stage.

    #training identifies:
        # AMFD_Faces_Final: 109 identities
        # facelab_london:   102 identities

    # RAW AVAILABLE DATA

    # SOURCE DATA AVAILABLE TO THE PROJECT
    #
    # source train = 1899 images
    #              = 211 cards x 9 images
    #
    #         │
    #         ├── project train
    #         │
    #         └── internal dev_val
    #             51 whole cards
    #             exactly 459 images
    #             deterministic selection
    #             card-disjoint from project train
    #
    # source test = 1385 images
    #
    #         structurally audited during discovery,
    #         but reserved from all fitting/model-selection decisions.


    # ------------------------------------------------------------
    # IF THE OFFICIAL VALIDATION SET LATER BECOMES AVAILABLE
    # ------------------------------------------------------------
    #
    # source train       = 1899 -> project training
    # official validation = 459 -> development / threshold selection
    # source test         = 1385 -> final held-out evaluation
    #
    # In that case the internally generated dev_val split is retired.

    # ------------------------------------------------------------
    # Final discovery-artifact export
    #
    # Any exporter failure must leave an explicit FATAL record in
    # the run log, including failures that happen before the Excel
    # writer itself starts.
    # ------------------------------------------------------------

    try:

        export_inventory_to_excel(
            matched_records,
            region_records,
            config,
            CONFIG_FILE,
        )

    except Exception as error:

        write_log(
            final_log_path,
            [
                (
                    "FATAL: "
                    "export_inventory_to_excel failed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            ],
        )

        # Preserve normal Python failure behaviour and traceback.
        raise

