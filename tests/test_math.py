from weakvtg.math import get_max, get_argmax


def test_get_max():
    assert get_max([1, 4, 2, 5, 1]) == 5


def test_get_argmax():
    assert get_argmax([1, 4, 2, 5, 1]) == 3
