import json
import os

def load_config(config_path):
    """Safely opens and reads the JSON configuration profiles."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration profile not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)

    return config