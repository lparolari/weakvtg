import json
import pickle


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
