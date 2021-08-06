import pytest

from weakvtg.timeit import get_delta, get_hms, get_eta, get_fancy_time, get_fancy_eta


@pytest.fixture
def ts(): return 1625727822.981586
@pytest.fixture
def te(): return 1625733222.0548382
@pytest.fixture
def t(): return 3731
@pytest.fixture
def delta(ts, te): return int(te - ts)
@pytest.fixture
def current(): return 4
@pytest.fixture
def total(): return 10


def test_get_delta(ts, te):
    assert get_delta(ts, te) == int(te - ts)


def test_get_hms(t):
    assert get_hms(t) == (1, 2, 11)


def test_get_eta(delta):
    assert get_eta(delta, current=4, total=10) == (8, 59, 54)


def test_get_fancy_time():
    assert get_fancy_time(*get_hms(0)) == "00:00:00"
    assert get_fancy_time(*get_hms(3731)) == "01:02:11"
    assert get_fancy_time(*get_hms(41000)) == "11:23:20"
    assert get_fancy_time(*get_hms(1141000)) == "316:56:40"


def test_get_fancy_eta(ts, te, current, total):
    assert get_fancy_eta(ts, te, current, total) == get_fancy_time(*get_eta(get_delta(ts, te), current, total))
