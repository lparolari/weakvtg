from weakvtg.utils import get_batch_size, percent, pivot, identity, map_dict


def test_get_batch_size():
    assert get_batch_size({"id": range(10), "foo": range(100), "bar": "hello"}) == 10


def test_percent():
    assert percent(0) == 0
    assert percent(54) == 5400


def test_pivot():
    out = pivot([{"a": 1, "b": 100},{"a": 2, "b": 200}])

    assert out == {"a": [1, 2], "b": [100, 200]}


def test_identity():
    assert identity(0) == 0
    assert identity("do I really need to test this?") == "do I really need to test this?"
    assert identity(True)


def test_map_dict():
    d = {"foo": 1, "bar": 0.15}
    def k_fn(k): return f"awesome_{k}"
    def v_fn(v): return v + 1

    assert map_dict(d, key_fn=k_fn, value_fn=v_fn) == {"awesome_foo": 2, "awesome_bar": 1.15}
