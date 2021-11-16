import torch

from weakvtg.matrix import get_diag


def test_get_diag():
    x = torch.rand(2, 3, 5, 5)
    v = torch.ones(5).view(1, 1, 5, 1)

    assert torch.equal(get_diag(x), x @ v)
