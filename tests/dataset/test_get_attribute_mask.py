import torch

from weakvtg.dataset import get_attribute_mask


def test_get_attribute_mask():
    x = torch.tensor([17, 0, 0, 3, 8])

    torch.equal(
        get_attribute_mask(x),
        torch.tensor([True, False, False, True, True])
    )
