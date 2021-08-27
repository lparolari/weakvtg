import pytest

from weakvtg.classes import get_specials, add_specials, load_classes, get_class, get_classes


@pytest.fixture
def classes(): return ["foo", "bar", "baz"]
@pytest.fixture
def specials(): return ["my_foo"]


def test_get_specials():
    assert get_specials() == ["__background__"]


def test_add_specials(classes, specials):
    assert add_specials(classes, specials) == [*specials, *classes]


def test_load_classes(tmp_path):
    cls_path = tmp_path / "classes.txt"
    cls_path.write_text("foo\nbar\nbaz paz")

    classes = load_classes(cls_path)

    assert len(classes) == 3
    assert classes[0] == "foo"
    assert classes[2] == "baz paz"


def test_get_class(classes):
    assert get_class(classes, 0) == "foo"
    assert get_class(classes, 2) == "baz"


def test_get_classes(tmp_path):
    cls_path = tmp_path / "classes.txt"
    cls_path.write_text("foo\nbar\nbaz paz")

    assert get_classes(cls_path) == add_specials(load_classes(cls_path), get_specials())
