import torch

from weakvtg.model import get_batched_batch


def test_get_batched_batch():
    assert get_batched_batch(torch.rand(2, 3, 4)).shape == (1, 2, 3, 4)
