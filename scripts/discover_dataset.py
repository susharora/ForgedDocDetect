#purpose: scan through directory containing image dataset and discover:
#   * data split , traffic type, attack variant, hardware source 
# also match images to their json descriptive files and validation transformation
# information available in JSON against recorded trnsformed height width. 
from pathlib import Path 
from datetime import datetime

import yaml

CONFIG_FILE = Path("dataconfig.yaml")

def load_config(config_path: Path) -> dict:
    with config_path.open("r",encoding="utf-8") as file:
        config = yaml.safe_load(file)

#debug   print("config-",config) 

    dataset_config = config["dataset"]

    root = Path(dataset_config["root"])

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    return config

def setup_log_file(yaml_log_path: str) -> Path:
    """Append current date to log filename and creates it"""
    base_log = Path(yaml_log_path)

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

    return final_log_path

config = load_config(CONFIG_FILE)

log_file_path = setup_log_file(config['dataset']['log'])
#debug print (log_file_path)

log_entries = [f"Configuration loaded successfully from {CONFIG_FILE}", 
               f"   Dataset name:   {config['dataset']['name']}",
               f"   Dataset root:   {config['dataset']['root']}",
               f"   Allowed splits: {config['dataset']['allowed_splits']}"]

full_log_output = "\n".join(log_entries) + "\n"

#debug print(full_log_output,end="")

with log_file_path.open("a",encoding="utf-8") as log_file:
    log_file.write(full_log_output)

#debug print(f"\n[info] Output successfully written to log file: {log_file_path}")