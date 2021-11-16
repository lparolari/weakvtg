import torch

from weakvtg.model import get_batch_negative, get_batch_negative_mask


def test_get_batch_negative_mask():
    assert get_batch_negative_mask(torch.rand(2, 3, 4)).shape == (2, 2)
