import collections
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from weakvtg.mask import get_synthetic_mask
from weakvtg.model import get_image_features, get_phrases_features, create_phrases_embedding_network, \
    get_phrases_representation, create_image_embedding_network
from weakvtg.vocabulary import get_vocab, get_word_embedding

pretrained_embeddings = get_word_embedding("glove")


def test_get_image_features():
    boxes_feat = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    boxes = torch.Tensor([[10, 10, 20, 20], [50, 50, 55, 55]])
    area = torch.Tensor([[100], [25]])

    expected = torch.cat([F.normalize(boxes_feat, p=1, dim=-1), boxes, area], dim=-1)
    actual = get_image_features(boxes, boxes_feat)

    assert torch.equal(expected, actual)


def test_get_phrases_features():
    phrases = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    phrases_mask = torch.Tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]])
    mask = get_synthetic_mask(phrases_mask)

    embedding_network = mock.Mock()
    embedding_network.return_value = phrases

    recurrent_network = mock.Mock()
    recurrent_network.return_value = phrases

    actual = get_phrases_features(phrases, phrases_mask, embedding_network, recurrent_network)
    expected = torch.masked_fill(phrases, mask == 0, value=0)

    assert torch.equal(actual, expected)
    embedding_network.assert_called_with(phrases)
    recurrent_network.assert_called_once()

    # recurrent_network.assert_called_with(phrases, phrases_length, mask) throws an error


def test_create_phrases_embedding_network():
    vocab = get_vocab(collections.Counter({"the": 3, "pen": 1, "is": 2, "on": 1, "table": 1, "dog": 1, "running": 1}))

    text_embedding = create_phrases_embedding_network(vocab, pretrained_embeddings, embedding_size=300, freeze=True)

    phrases = torch.Tensor([[1, 2, 3, 4, 1, 5], [1, 6, 3, 7, 0, 0]]).long()
    phrases_embedded = text_embedding(phrases)

    assert phrases_embedded.size() == torch.Size((2, 6, 300))


def test_get_phrases_representation():
    vocab = get_vocab(collections.Counter({"the": 3, "pen": 1, "is": 2, "on": 1, "table": 1, "dog": 1, "running": 1}))

    phrases = torch.Tensor([[[1, 2, 3, 4, 1, 5], [1, 6, 3, 7, 0, 0]]]).long()
    phrases_mask = torch.Tensor([[[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]]).long()
    phrases_length = torch.sum(phrases_mask.int(), dim=-1)

    text_embedding = create_phrases_embedding_network(vocab, pretrained_embeddings, embedding_size=300, freeze=True)
    phrases_embedded = text_embedding(phrases)

    lstm = nn.LSTM(300, 50, num_layers=2, bidirectional=False, batch_first=False)

    phrases_repr = get_phrases_representation(phrases_embedded, phrases_length, get_synthetic_mask(phrases_mask),
                                              out_features=50, recurrent_network=lstm)

    assert phrases_repr.size() == torch.Size((1, 2, 50))


def test_create_image_embedding_network():
    assert len(create_image_embedding_network(50, 20)) == 5
    assert len(create_image_embedding_network(50, 20, n_hidden_layer=3)) == 7
