from pyaml_env import parse_config
import sys

def init_notebook():
    sys.path.append("..")
    config = parse_config(path="../config.yml")
    return config

