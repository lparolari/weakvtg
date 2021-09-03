import pytest
import torchtext

from weakvtg import classes


def test_vocab_loaded(vocab):
    assert isinstance(vocab, list)
    assert "boy" in vocab


def test_word_embedding_given_classes(vocab, glove_words):
    out_of_vocabulary = list(filter(lambda word: word not in glove_words, vocab))

    assert len(out_of_vocabulary) == 295


@pytest.fixture
def vocab(resource_path_root):
    vocab_path = (resource_path_root / "objects_vocab.txt")
    return classes.load_classes(vocab_path)


@pytest.fixture
def glove_embeddings():
    return torchtext.vocab.GloVe("840B", dim=300)


@pytest.fixture
def glove_words(glove_embeddings):
    return glove_embeddings.stoi.keys()
