import json
import os
from pathlib import Path
from typing import Callable
from bold_dementia import get_config

config = get_config()

def save_run(run_config: str, save_func: Callable, save_mapping: dict) -> Path:
    """Save current run object and parameters

    Args:
        save_func (Callable): dedicated func which takes (obj, path) 
        as paramaters, to save the python objects in save mapping
        save_mapping (dict): Map between fname and python object

    Returns:
        Path: path of the folder containing all the saved objects
    """

    name = f"atlas-{run_config['ATLAS']}_{run_config['NAME']}"

    experience_path = Path(config["connectivity_matrices"]) / name

    if not os.path.exists(experience_path):
        os.makedirs(experience_path)

    with open(experience_path / "parameters.json", "w") as stream:
        json.dump(run_config, stream)

    for fname, obj in save_mapping.items():
        save_func(obj, experience_path / fname)

    return experience_path