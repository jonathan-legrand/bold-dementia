import yaml

def get_config():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    return config
