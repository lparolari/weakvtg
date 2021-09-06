import pytest
import torch.nn.functional
import torchtext

from weakvtg import classes


def test_vocab_loaded(vocab):
    assert isinstance(vocab, list)
    assert "boy" in vocab


def test_word_embedding_given_classes(vocab, glove_words):
    out_of_vocabulary = list(filter(lambda word: word not in glove_words, vocab))

    assert len(out_of_vocabulary) == 295


def test_word_embedding_similarity(glove_embeddings):
    def _get_similarity(word1, word2):
        w1_index = glove_embeddings.stoi[word1]
        w2_index = glove_embeddings.stoi[word2]

        w1 = glove_embeddings.vectors[w1_index]
        w2 = glove_embeddings.vectors[w2_index]

        return torch.nn.functional.cosine_similarity(w1, w2, dim=-1)

    assert round(_get_similarity("person", "woman").item(), 4) == pytest.approx(0.5618)
    assert round(_get_similarity("person", "man").item(), 4) == pytest.approx(0.5557)
    assert round(_get_similarity("man", "woman").item(), 4) == pytest.approx(0.7402)
    assert round(_get_similarity("person", "eggs").item(), 4) == pytest.approx(0.215)


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
