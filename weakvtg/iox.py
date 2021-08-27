import json
import pickle
import cv2


def load_txt(filename):
    with open(filename, "r") as f:
        return f.read().split("\n")


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_image(filename):
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im
