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

import yaml
import inspect

#Block: Global constants

CONFIG_FILE = Path("dataconfig.yaml")

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
    
    
