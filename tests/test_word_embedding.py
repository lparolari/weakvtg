import functools

import pytest
import torch.nn.functional
import torchtext

from weakvtg import classes
from weakvtg.vocabulary import Word2Vec


def test_vocab_loaded(vocab):
    assert isinstance(vocab, list)
    assert "boy" in vocab


def test_attributes_vocab_loaded(attributes_vocab):
    assert isinstance(attributes_vocab, list)
    assert "red" in attributes_vocab


def test_word_embedding_given_classes(vocab, glove_words):
    out_of_vocabulary = list(filter(lambda word: word not in glove_words, vocab))

    assert len(out_of_vocabulary) == 295


def test_attributes_vocab_oov(attributes_vocab, glove_words):
    out_of_vocabulary = list(filter(lambda word: word not in glove_words, attributes_vocab))

    assert len(out_of_vocabulary) == 31


def test_glove_word_embedding_similarity(glove_embeddings):
    _get_similarity = functools.partial(get_similarity, embeddings=glove_embeddings)
    assert round(_get_similarity("person", "woman").item(), 4) == pytest.approx(0.5618)
    assert round(_get_similarity("person", "man").item(), 4) == pytest.approx(0.5557)
    assert round(_get_similarity("man", "woman").item(), 4) == pytest.approx(0.7402)
    assert round(_get_similarity("person", "eggs").item(), 4) == pytest.approx(0.215)


def test_w2v_word_embedding_similarity(w2v_embeddings):
    _get_similarity = functools.partial(get_similarity, embeddings=w2v_embeddings)
    assert round(_get_similarity("person", "woman").item(), 4) == pytest.approx(0.5470)
    assert round(_get_similarity("person", "man").item(), 4) == pytest.approx(0.5342)
    assert round(_get_similarity("man", "woman").item(), 4) == pytest.approx(0.7664)
    assert round(_get_similarity("person", "eggs").item(), 4) == pytest.approx(0.0917)


def test_glove_word_embedding_mean_similarity(glove_embeddings):
    def i(word): return glove_embeddings.stoi[word]
    def w(word): return glove_embeddings.vectors[i(word)].unsqueeze(-2)
    def ph(phrase): return torch.cat([w(word) for word in phrase.split(" ")], dim=-2)

    phrase = torch.mean(ph("grass left side"), dim=-2)
    cls = glove_embeddings.vectors[i("grass")]

    similarity = torch.cosine_similarity(phrase, cls, dim=-1)
    similarity = torch.relu(similarity)

    assert similarity.item() == pytest.approx(0.7312, rel=1e-4)


def get_similarity(word1, word2, embeddings):
    w1_index = embeddings.stoi[word1]
    w2_index = embeddings.stoi[word2]

    w1 = embeddings.vectors[w1_index]
    w2 = embeddings.vectors[w2_index]

    return torch.cosine_similarity(w1, w2, dim=-1)


@pytest.fixture
def vocab(resource_path_root):
    vocab_path = (resource_path_root / "objects_vocab.txt")
    return classes.load_classes(vocab_path)


@pytest.fixture
def attributes_vocab(resource_path_root):
    vocab_path = (resource_path_root / "attributes_vocab.txt")
    return classes.load_classes(vocab_path)


@pytest.fixture
def glove_embeddings():
    return torchtext.vocab.GloVe("840B", dim=300)


@pytest.fixture
def w2v_embeddings():
    return Word2Vec("word2vec-google-news-300")


@pytest.fixture
def glove_words(glove_embeddings):
    return glove_embeddings.stoi.keys()
