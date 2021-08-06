from collections import Counter

import pytest
import torch
import torchtext.vocab

from weakvtg.padder import get_number_examples, get_max_length_examples, get_max_length_phrases, \
    get_indexed_phrases_per_example, get_padded_examples


@pytest.fixture
def l0(): return []


@pytest.fixture
def l1(): return [1, 2, 3]


@pytest.fixture
def ll1(): return [[1, 2, 3], [4, 5, 6, 7]]


@pytest.fixture
def lll1(): return [
    [[1, 2, 3], [4, 5, 6, 7, 8, 9]],
    [[11, 12, 13, 14], [14, 15, 16, 17]],
    [[101], [102], [103]]
]


@pytest.fixture
def tokenizer():
    # A very simple tokenizer function
    def f(phrase: str):
        return phrase.split(sep=' ')
    return f


@pytest.fixture
def ph1(): return "the pen is on the table"
@pytest.fixture
def ph2(): return "the dog is running"


@pytest.fixture
def mapping():
    return {
        "the": 3,
        "pen": 1,
        "is": 2,
        "on": 1,
        "table": 1,
        "dog": 1,
        "running": 1,
    }


@pytest.fixture
def counter(mapping):
    return Counter(mapping)


@pytest.fixture
def vocab(counter):
    return torchtext.vocab.vocab(counter)


@pytest.fixture
def indexed(): return [[[0, 1, 2, 3, 0, 4]], [[0, 5, 2, 6]]]  # [2, 1, x]


@pytest.fixture
def indexed_pad(): return [  # [3, 3, 8
    [[0, 1, 2, 3, 0, 4, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 5, 2, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
]


def test_get_number_examples(l0, l1, ll1, lll1):
    assert get_number_examples(l0) == 0
    assert get_number_examples(l1) == 3
    assert get_number_examples(ll1) == 2
    assert get_number_examples(lll1) == 3


def test_get_max_length_examples(l0, l1, ll1, lll1):
    assert get_max_length_examples(l0) == 0
    assert get_max_length_examples(ll1) == 4
    assert get_max_length_examples(lll1) == 3
    with pytest.raises(TypeError):
        get_max_length_examples(l1)


def test_get_max_length_phrases(l0, l1, ll1, lll1):
    assert get_max_length_phrases(l0) == 0
    assert get_max_length_phrases(lll1) == 6

    with pytest.raises(TypeError):
        get_max_length_phrases(l1)

    with pytest.raises(TypeError):
        get_max_length_phrases(ll1)


def test_get_indexed_phrases(ph1, ph2, tokenizer, vocab):
    assert get_indexed_phrases_per_example([[ph1], [ph2]], tokenizer, vocab) == [[[0, 1, 2, 3, 0, 4]], [[0, 5, 2, 6]]]
    assert get_indexed_phrases_per_example([], tokenizer, vocab) == []
    assert get_indexed_phrases_per_example([[]], tokenizer, vocab) == [[[]]]


def test_get_padded_examples(indexed, indexed_pad):
    assert get_padded_examples(indexed, padding_value=0, padding_dim=(3, 2, 8))[0].tolist() == indexed_pad
    assert get_padded_examples([], padding_value=0, padding_dim=(3, 2, 8))[0].tolist() == torch.zeros(3, 2, 8).tolist()
    assert get_padded_examples([[]], padding_value=0, padding_dim=(3, 2, 8))[0].tolist() == torch.zeros(3, 2, 8).tolist()
    assert get_padded_examples([[[]]], padding_value=0, padding_dim=(3, 2, 8))[0].tolist() \
           == torch.zeros(3, 2, 8).tolist()
