import pickle

from weakvtg.iox import load_json, load_pickle


def test_load_json(tmp_path):
    json_path = tmp_path / "foo.json"
    json_path.write_text('{ "foo": "hello", "bar": 2 }')

    assert load_json(json_path) == {"foo": "hello", "bar": 2}


def test_load_pickle(tmp_path):
    pickle_path = tmp_path / "foo.pickle"
    pickle_path.write_bytes(pickle.dumps({"foo": "hello", "bar": 2}))

    assert load_pickle(pickle_path) == {"foo": "hello", "bar": 2}
