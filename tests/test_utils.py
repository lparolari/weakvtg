from weakvtg.utils import get_batch_size, percent


def test_get_batch_size():
    assert get_batch_size({"id": range(10), "foo": range(100), "bar": "hello"}) == 10


def test_percent():
    assert percent(0) == 0
    assert percent(54) == 5400
