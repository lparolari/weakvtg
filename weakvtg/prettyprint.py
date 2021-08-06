import typing as t

from weakvtg.const import float_format


def pp(x: t.Any) -> str:
    if isinstance(x, float):
        return ppf(x)
    if isinstance(x, dict):
        return ppd(x)
    return str(x)


def ppd(d: t.Dict[str, t.Any]) -> str:
    return ", ".join([f"{k}: {pp(v)}" for k, v in d.items()])


def ppf(x: float) -> str:
    return f"{x:{float_format}}"
