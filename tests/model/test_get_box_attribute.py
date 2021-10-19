import torch

from weakvtg.model import get_box_attribute


def test_get_box_attribute():
    x = torch.rand(2, 3)
    assert torch.equal(get_box_attribute(x), torch.argmax(x, dim=-1))
