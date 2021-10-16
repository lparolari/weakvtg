import pytest

from weakvtg.tokenizer import get_nlp, noun_chunk_iter

nlp = get_nlp()


def test_noun_chunk_iter():
    doc = nlp("A phrase with another phrase occurs..")
    it = noun_chunk_iter(doc)

    assert next(it).text == "A phrase"
    assert next(it).text == "another phrase"
    with pytest.raises(StopIteration):
        next(it)
