import torch

from weakvtg.matrix import get_negative


def test_get_negative():
    x = torch.rand(5, 5, 3, 2)
    ident = torch.eye(5).view(5, 5, 1, 1)

    assert torch.equal(get_negative(x), x * (1 - ident))
