import yaml
import os
def load_config(config_path: str = None) -> dict:
    if config_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base, "config", "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)