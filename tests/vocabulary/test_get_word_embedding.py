import pytest
import torchtext.vocab

from weakvtg.vocabulary import get_word_embedding

Vectors = torchtext.vocab.Vectors


def test_get_word_embedding():
    assert isinstance(get_word_embedding('glove'), Vectors)
    assert isinstance(get_word_embedding('w2v'), Vectors)

    with pytest.raises(ValueError):
        get_word_embedding('foo')

    with pytest.raises(ValueError):
        get_word_embedding('w2v', 400)
