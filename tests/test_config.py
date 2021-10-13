from unittest import mock

import pytest

from weakvtg.config import get_config, __defaults, make_options


def test_config_defaults():
    assert get_config({}) == __defaults


def test_config():
    d1 = {"batch_size": 32, "num_workers": None}
    d2 = {"batch_size": 64, "num_workers": 0, "prefetch_factor": 2}

    assert get_config(d1, d2) == {"batch_size": 32, "num_workers": 0, "prefetch_factor": 2}


def test_make_options():
    options = {"one": 1, "two": "three"}
    name = "foo"

    make_foo = make_options(name, options)

    assert make_foo("one") == 1
    assert make_foo("two") == "three"

    with pytest.raises(ValueError):
        make_foo("three")


def test_make_options_given_params():
    f = mock.Mock()
    options = {"one": f}
    name = "foo"

    make_foo = make_options(name, options)

    g = make_foo("one", params={"one": {"lam": 1}})
    g()

    f.assert_called_with(lam=1)
