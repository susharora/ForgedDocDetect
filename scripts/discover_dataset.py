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

import yaml
import inspect

#Block: Global constants

CONFIG_FILE = Path("dataconfig.yaml")

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

    #Log file creation based on file path mentioned in dataconfig yaml
    #Append current date to log filename and creates it
    log_file_path = dataset_config["log"]
    base_log = Path(log_file_path)
    #extract filename components
    log_dir  = base_log.parent 
    log_name = base_log.stem
    log_ext  = base_log.suffix
    #format date
    current_date = datetime.now().strftime("%Y-%m-%d")
    #construct new filename:
    new_log_name = f"{log_name}_{current_date}{log_ext}"
    final_log_path = log_dir / new_log_name
    log_dir.mkdir(parents=True,exist_ok=True)

    #debug print (f"{log_file_path} , {type(config)}")
    
    #add log entry
    log_entries = [f"********************{func_name}: ********************"]    
    write_log(final_log_path,log_entries)
    log_entries = [f"{func_name}: Configuration loaded successfully from {CONFIG_FILE}", 
                       f"   Dataset name:   {config['dataset']['name']}",
                        f"   Dataset root:   {config['dataset']['root']}",
                        f"   Allowed splits: {config['dataset']['allowed_splits']}"]
    
    write_log(final_log_path,log_entries)
    
    return config

# build log path TBD
def build_log_path(config: dict) -> Path:
    base_log = Path(config["dataset"]["log"])
    current_date = datetime.now().strftime("%Y-%m-%d")
    final_log_path = base_log.parent / f"{base_log.stem}_{current_date}{base_log.suffix}"
    base_log.parent.mkdir(parents=True, exist_ok=True)
    return final_log_path 

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
        traffic_paths: dict[str,dict[str,Path]]
) -> list[dict]:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]        
    write_log(final_log_path,log_entries)
    data_folders = []

    for split_name, traffic_types in traffic_paths.items():

        for traffic_type, traffic_path in traffic_types.items():

            if traffic_type == "bonafide":

                for hardware_path in sorted(
                    traffic_path.iterdir(),
                    key=lambda path: path.name.lower()
                ):
                    if not hardware_path.is_dir():
                        log_entries = [f"{func_name}: Ignoring file: {hardware_path}"]
                        write_log(final_log_path,log_entries)
                        continue

                    data_folders.append( 
                        {
                        "split": split_name,
                        "traffic_type": traffic_type,
                        "variant":"",
                        "hardware_source":hardware_path.name,
                        "path":hardware_path         
                        }
                    )        

            elif traffic_type == "attack":

                for variant_path in sorted(
                    traffic_path.iterdir(),
                    key=lambda path: path.name.lower()
                ):
                    if not variant_path.is_dir():
                        log_entries = [f"{func_name}: Ignoring file: {hardware_path}"]
                        write_log(final_log_path,log_entries)
                        continue

                    for hardware_path in sorted(
                        variant_path.iterdir(),
                        key=lambda path: path.name.lower()
                    ):
                        if not hardware_path.is_dir():
                            log_entries = [f"{func_name}: Ignoring file: {hardware_path}"]
                            write_log(final_log_path,log_entries)
                            continue

                        data_folders.append({
                            "split": split_name,
                            "traffic_type": traffic_type,
                            "variant": variant_path.name,
                            "hardware_source":hardware_path.name,
                            "path":hardware_path
                        })

    log_entries = [f"{func_name}: Final data folders:"]
    
    for folder in data_folders:
        log_entries.append (f"{func_name}:  {folder['split']},{folder['traffic_type']},{folder['variant']}," 
                            f"{folder['hardware_source']} ----> {folder['path']}")

    write_log(final_log_path,log_entries)               
    
    return data_folders

#Function to discover all images and their JSON description files from discovered folders
def discover_files(
    data_folders: list[dict],
    config: dict
) -> list[dict]:

    func_name = inspect.currentframe().f_code.co_name
    log_entries = [f"********************{func_name}: ********************"]        
    write_log(final_log_path,log_entries)
    image_extensions = {
        extension.lower()
        for extension in config["dataset"]["image_extensions"]
    }

    json_extension = config["dataset"]["json_extension"].lower()

    discovered_files = []

    for folder in data_folders:

        folder_path = folder["path"]

        for file_path in sorted(
            folder_path.iterdir(),
            key=lambda path: path.name.lower()
        ):
            if not file_path.is_file():
                log_entries = [f"{func_name}: Ignoring nested folder: {file_path}"]
                write_log(final_log_path,log_entries)
                continue

            suffix = file_path.suffix.lower()

            if suffix in image_extensions:
                file_type = "image"

            elif suffix == json_extension:
                file_type = "json"

            else:
                log_entries = [f"{func_name}: Ignoring unsupported file: {file_path}"]
                write_log(final_log_path,log_entries)
                continue

            discovered_files.append({
                "split": folder["split"],
                "traffic_type": folder["traffic_type"],
                "variant": folder["variant"],
                "hardware_source": folder["hardware_source"],
                "file_type": file_type,
                "file_name": file_path.name,
                "file_stem": file_path.stem,
                "file_extension": file_path.suffix,
                "file_path": file_path,
            })

    log_entries = [f"{func_name}: Discovered files totals:--------------"]
    counts = Counter(f["file_type"] for f in discovered_files)
    for file_type , n in sorted(counts.items()):
        log_entries.append(f"   {file_type} : {n}")
    
    #debug : uncomment the following code if numbers dont tally in above step
    # log_entries.append(f"{func_name}: Discovered files:--------------")
    
    # for file_record in discovered_files:
    #     log_entries.append (f"{func_name}: {file_record['file_type']} {file_record['split']} {file_record['traffic_type']}"
    #                        f"{file_record['variant']} {file_record['hardware_source']} {file_record['file_name']}")
            
    write_log(final_log_path,log_entries)

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
    log_entries = [f"********************{func_name}: ********************"]
    for record in matched_records:

        record["transformed_rectangle"] = None
        record["transform_ok"] = False
        record["transform_error"] = ""

        if not (
            record["matrix_shape_ok"]
            and record["rectangle_shape_ok"]
        ):
            continue

        matrix = record["transformation_matrix"]
        rectangle = record["original_rectangle"]

        try:
            transformed_rectangle = tuple(
                transform_point(point, matrix)
                for point in rectangle
            )

        except (ValueError, TypeError) as error:
            record["transform_error"] = str(error)
            continue

        record["transformed_rectangle"] = (
            transformed_rectangle
        )

        record["transform_ok"] = True

    # ---- Diagnostics ----
    
    # Records that failed to transform
    transform_error_count = 0
    error_entries = []
    for record in matched_records:
        if not record["transform_ok"]:
            error_entries.append(
                f"  {record['file_stem']} -> {record['transform_error']}"
            )
            transform_error_count += 1

    log_entries.append(f"Transformation problems: {transform_error_count}")
    log_entries.extend(error_entries)

    # One worked example per distinct crop_info_key
    log_entries.append("Example transformed rectangles:")
    shown_crop_types = set()
    for record in matched_records:
        if not record["transform_ok"]:
            continue
        crop_key = record["crop_info_key"]
        if crop_key in shown_crop_types:
            continue
        shown_crop_types.add(crop_key)

        width = record["resulted_cropped_image_width"]
        height = record["resulted_cropped_image_height"]

        log_entries.append(
            f"  Crop type: {crop_key}  (example: {record['file_stem']})"
        )
        log_entries.append(f"    Declared crop size: {width} x {height}")
        log_entries.append("    original -> transformed  (expected corner)")

        expected_corners = ((0, 0), (width, 0), (width, height), (0, height))
        for original, transformed, expected in zip(
            record["original_rectangle"],
            record["transformed_rectangle"],
            expected_corners,
        ):
            log_entries.append(
                f"      ({original[0]:8.3f}, {original[1]:8.3f})"
                f"  ->  ({transformed[0]:9.3f}, {transformed[1]:9.3f})"
                f"   (expect {expected[0]}, {expected[1]})"
            )

    # Reconciliation
    n_total = len(matched_records)
    n_eligible = sum(
        1 for r in matched_records
        if r["matrix_shape_ok"] and r["rectangle_shape_ok"]
    )
    n_ok = sum(1 for r in matched_records if r["transform_ok"])
    log_entries.append(
        f"Reconciliation: total={n_total} eligible={n_eligible} "
        f"transformed_ok={n_ok} failed={transform_error_count}"
    )

    write_log(final_log_path, log_entries)

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

        except (OSError, ValueError) as error:
            record["image_metadata_error"] = str(error)

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
        log_entries.append(f"Distinct {title} structures ({len(patterns)}):")
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

    func_name = inspect.currentframe().f_code.co_name

    log_entries = [
        f"********************{func_name}: ********************"
    ]

    image_lookup = {
        record["image_path"]: record
        for record in matched_records
        if record["image_path"] is not None
    }

    for region in region_records:

        parent = image_lookup.get(
            region["image_path"]
        )

        region["disk_bounds_status"] = ""
        region["declared_bounds_status"] = ""

        region["right"] = (
            region["x"] + region["width"]
        )

        region["bottom"] = (
            region["y"] + region["height"]
        )

        if parent is None:
            continue

        # -------------------------
        # Actual on-disk image
        # -------------------------
        region["disk_bounds_status"] = (
            classify_region_bounds(
                region["x"],
                region["y"],
                region["width"],
                region["height"],
                parent["image_width"],
                parent["image_height"],
            )
        )

        # -------------------------
        # JSON-declared crop size
        # -------------------------
        region["declared_bounds_status"] = (
            classify_region_bounds(
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
        )

        # ---- Diagnostics ----
    disk_status_counts = Counter(r["disk_bounds_status"] for r in region_records)
    declared_status_counts = Counter(r["declared_bounds_status"] for r in region_records)

    for title, counts in (
        ("actual on-disk image", disk_status_counts),
        ("JSON-declared crop", declared_status_counts),
    ):
        log_entries.append(f"Region bounds against {title}:")
        for status, count in sorted(counts.items()):
            label = status if status else "(not checked - no parent image)"
            log_entries.append(f"  {label}: {count}")

    # Out-of-bounds regions broken down by provenance — altered boxes are the
    # ones that become ground-truth mask, so a bad box there costs more.
    oob_by_provenance = defaultdict(Counter)
    for r in region_records:
        if r["disk_bounds_status"] in ("partially_outside", "completely_outside"):
            oob_by_provenance[r["region_provenance_raw"]][r["disk_bounds_status"]] += 1
    if oob_by_provenance:
        log_entries.append("Out-of-bounds regions by provenance:")
        for provenance in sorted(oob_by_provenance, key=str):
            log_entries.append(f"  {provenance}: {dict(oob_by_provenance[provenance])}")
        n_altered_oob = sum(oob_by_provenance.get("altered", Counter()).values())
        if n_altered_oob:
            log_entries.append(
                f"  WARNING: {n_altered_oob} altered regions fall outside the image"
            )

    # Where the two frames disagree
    disagreements = [
        r for r in region_records
        if r["disk_bounds_status"] != r["declared_bounds_status"]
    ]
    log_entries.append(f"Disk / declared bounds disagreements: {len(disagreements)}")
    for region in disagreements[:20]:
        parent = image_lookup[region["image_path"]]
        log_entries.append(
            f"  {region['file_stem']} region={region['region_index']} "
            f"field={region['field_name']} "
            f"box=({region['x']}, {region['y']}, {region['width']}, {region['height']}) "
            f"disk={parent['image_width']}x{parent['image_height']} "
            f"declared={parent['resulted_cropped_image_width']}"
            f"x{parent['resulted_cropped_image_height']} "
            f"{region['disk_bounds_status']} vs {region['declared_bounds_status']}"
        )
    if len(disagreements) > 20:
        log_entries.append(f"  ... {len(disagreements) - 20} more not shown")

    # Reconciliation
    n_no_parent = sum(1 for r in region_records if not r["disk_bounds_status"])
    log_entries.append(
        f"Reconciliation: total_regions={len(region_records)} "
        f"with_parent={len(region_records) - n_no_parent} "
        f"no_parent={n_no_parent} "
        f"disagreements={len(disagreements)}"
    )

    write_log(final_log_path, log_entries)

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

#Block: Execution
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
    
    #Extract regions which has labelled information on altered areas in attack images
    inspect_regions(matched_records)

    #Extract data from regions 
    region_records = extract_region_records(matched_records)

    #validate regions
    validate_region_bounds(region_records,matched_records)

    # Analysis of regions which are out of bound. 
    analyse_region_visibility(region_records,matched_records)
