import json


configs = {}


def get_configs():
    return configs


def parse_configs(s: str = None):
    if s is None:
        return get_defaults()
    return json.loads(s)


def get_defaults():
    return {}