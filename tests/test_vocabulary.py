import collections

import pytest

from weakvtg.vocabulary import get_vocab, load_vocab


@pytest.fixture
def vocab():
    return get_vocab(collections.Counter({"the": 3, "pen": 1, "is": 2, "on": 1, "table": 1, "dog": 1, "running": 1}))


def test_load_vocab(tmp_path):
    vocab_json_path = tmp_path / "vocab.json"
    vocab_json_path.write_text('{ "the": 2, "pen": 1, "is": 2, "on": 1, "table": 1, "dog": 1, "running": 1 }')

    vocab = load_vocab(vocab_json_path)

    assert vocab["."] == 0
    assert vocab[""] == 0
    assert vocab["   "] == 0
    assert [vocab[token] for token in "the pen is on the table".split()] == [1, 2, 3, 4, 1, 5]
    assert [vocab[token] for token in "the dog is running".split()] == [1, 6, 3, 7]


def test_vocab(vocab):
    assert vocab["."] == 0
    assert vocab[""] == 0
    assert vocab["   "] == 0
    assert [vocab[token] for token in "the pen is on the table".split()] == [1, 2, 3, 4, 1, 5]
    assert [vocab[token] for token in "the dog is running".split()] == [1, 6, 3, 7]
