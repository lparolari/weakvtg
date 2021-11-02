import pytest

from weakvtg.tokenizer import adj_iter, get_nlp

nlp = get_nlp()


def test_adj_iter():
    doc = nlp("A red apple near a yellow banana")
    it = adj_iter(doc)

    assert next(it).text == "red"
    assert next(it).text == "yellow"
    with pytest.raises(StopIteration):
        next(it)
