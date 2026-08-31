#purpose: scan through directory containing image dataset and discover:
#   * data split , traffic type, attack variant, hardware source 
# also match images to their json descriptive files and validation transformation
# information available in JSON against recorded trnsformed height width. 

#Block: Imports
from pathlib import Path 
from datetime import datetime

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

#Block: Execution
if __name__ == "__main__":
    #load the yaml dictionary
    config = load_config(CONFIG_FILE)
    #log path
    final_log_path = build_log_path(config)
    #traverse data splits per yaml dictionary 
    split_paths = find_dataset_splits(config)
    #traverse traffic type split under data split
    traffic_path = find_traffic_types(
        split_paths,
        config
    )
    #get folders that have images and JSON files for each traffic type. 
    data_folders = find_data_folders(traffic_path)

    
    
