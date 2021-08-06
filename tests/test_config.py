from weakvtg.config import parse_configs


def test_parse_config():
    assert isinstance(parse_configs('{"foo": 0, "bar": "hello"}'), dict)
