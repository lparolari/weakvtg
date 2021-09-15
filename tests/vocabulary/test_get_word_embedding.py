import os

import pytest
import torchtext.vocab

from weakvtg.vocabulary import get_word_embedding

Vectors = torchtext.vocab.Vectors


def test_vector_cache_exists(vector_cache):
    assert os.path.isdir(vector_cache)


def test_get_word_embedding_given_glove(vector_cache):
    assert isinstance(get_word_embedding('glove', cache=vector_cache), Vectors)


def test_get_word_embedding_given_w2v(vector_cache):
    assert isinstance(get_word_embedding('w2v', cache=vector_cache), Vectors)


def test_get_word_embedding_given_unknown(vector_cache):
    with pytest.raises(ValueError):
        get_word_embedding('foo', cache=vector_cache)


def test_get_word_embedding_given_w2v_wrong_dim(vector_cache):
    with pytest.raises(ValueError):
        get_word_embedding('w2v', 400, cache=vector_cache)


@pytest.fixture
def vector_cache():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".vector_cache")
