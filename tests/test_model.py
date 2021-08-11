from unittest import mock

import torch
import torch.nn.functional as F

from weakvtg.mask import get_synthetic_mask
from weakvtg.model import get_image_features, get_phrases_features


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
    phrases_length = torch.sum(phrases_mask.int(), dim=-1)
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


