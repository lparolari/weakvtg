from weakvtg.utils import get_batch_size, percent, pivot


def test_get_batch_size():
    assert get_batch_size({"id": range(10), "foo": range(100), "bar": "hello"}) == 10


def test_percent():
    assert percent(0) == 0
    assert percent(54) == 5400


def test_pivot():
    out = pivot([{"a": 1, "b": 100},{"a": 2, "b": 200}])

    assert out == {"a": [1, 2], "b": [100, 200]}
