import json


configs = {}


def get_configs():
    return configs


def parse_configs(s):
    return json.loads(s)
