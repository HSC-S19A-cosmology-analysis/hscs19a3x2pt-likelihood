
def empty_dict():
    return {}

def setdefault_config(config, config_default):
    for key, val in config_default.items():
        config.setdefault(key, val)
    