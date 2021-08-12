import pickle

import cv2
from PIL import Image

from weakvtg.iox import load_json, load_pickle, load_image


def test_load_json(tmp_path):
    json_path = tmp_path / "foo.json"
    json_path.write_text('{ "foo": "hello", "bar": 2 }')

    assert load_json(json_path) == {"foo": "hello", "bar": 2}


def test_load_pickle(tmp_path):
    pickle_path = tmp_path / "foo.pickle"
    pickle_path.write_bytes(pickle.dumps({"foo": "hello", "bar": 2}))

    assert load_pickle(pickle_path) == {"foo": "hello", "bar": 2}


def test_load_image(tmp_path):
    image_path = tmp_path / "foo.jpg"

    import numpy as np
    cv2.imwrite(str(image_path), np.zeros(10))

    # We do not test for explicit properties, instead we check no exception is raised
    load_image(str(image_path))
