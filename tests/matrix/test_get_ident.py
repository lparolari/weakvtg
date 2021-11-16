import torch

from weakvtg.matrix import get_ident


def test_get_ident():
    assert torch.equal(get_ident(5, 4), torch.eye(5).view(1, 1, 5, 5))
