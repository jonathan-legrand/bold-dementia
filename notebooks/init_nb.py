def init_notebook():
    import sys
    import yaml
    
    sys.path.append("..")
    with open('../config.yml', 'r') as file:
        config = yaml.safe_load(file)
    return config

