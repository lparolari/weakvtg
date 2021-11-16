import torch

from weakvtg.matrix import get_diag, get_positive
from weakvtg.utils import invert


def test_get_positive():
    x = torch.rand(5, 5, 3, 2)

    assert torch.equal(get_positive(x), invert(get_diag(invert(x * torch.eye(5).view(5, 5, 1, 1)))))
