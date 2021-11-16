import torch

from weakvtg.matrix import get_ones


def test_get_ones():
    assert torch.equal(get_ones(size=5, n_dims=4, dim=-2), torch.ones(5).view(1, 1, 5, 1))
