import pytest

from weakvtg.tokenizer import get_nlp, root_chunk_iter

nlp = get_nlp()


def test_noun_chunk_iter():
    doc = nlp("A phrase with another phrase occurs..")
    it = root_chunk_iter(doc)

    assert next(it).text == "phrase"
    assert next(it).text == "phrase"
    with pytest.raises(StopIteration):
        next(it)
