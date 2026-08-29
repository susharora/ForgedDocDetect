from pathlib import Path
import yaml

def load_config():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    local_path = root / "configs" / "local.yaml"
    if local_path.exists():
        with open(local_path) as f:
            local = yaml.safe_load(f) or {}
        for section, values in local.items():
            cfg.setdefault(section, {}).update(values)
    return cfg
