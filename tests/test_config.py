from weakvtg.config import get_config, __defaults


def test_config_defaults():
    assert get_config({}) == __defaults


def test_config():
    d1 = {"batch_size": 32, "num_workers": None}
    d2 = {"batch_size": 64, "num_workers": 0, "prefetch_factor": 2}

    assert get_config(d1, d2) == {"batch_size": 32, "num_workers": 0, "prefetch_factor": 2}
