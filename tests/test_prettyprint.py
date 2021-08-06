from weakvtg.prettyprint import pp


def test_ppf():
    assert pp(56.123456789) == "56.123457"


def test_ppd():
    assert pp({"foo": 1., "bar": "hello", "baz": 2}) == "foo: 1.000000, bar: hello, baz: 2"
