import yaml

def get_config(fpath="config.yml"):
    with open(fpath, 'r') as file:
        config = yaml.safe_load(file)
    return config
