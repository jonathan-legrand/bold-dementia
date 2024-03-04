import yaml

def get_config():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_custom_config(fpath):
    with open(fpath, 'r') as file:
        config = yaml.safe_load(file)
    return config
