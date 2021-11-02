import torch

from weakvtg.dataset import get_attribute


def test_get_attribute():
    x = torch.rand((2, 3))
    assert torch.equal(get_attribute(x), torch.argmax(x, dim=-1))
