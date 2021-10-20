import torch

from weakvtg.loss import get_loss_direction


def test_get_loss_direction():
    a = torch.rand((2, 3))
    b = torch.rand((2, 3))

    assert torch.equal(get_loss_direction(a, b), (a + b - b * (1 - a)) - 1)
