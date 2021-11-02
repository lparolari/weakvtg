import torch

from weakvtg.dataset import get_mask_without_background


def test_get_boxes_mask_no_background():
    mask = torch.tensor([True, False, True])
    label = torch.tensor([0, 15, 8])
    background = 0

    assert torch.equal(
        get_mask_without_background(mask, label, background=background),
        torch.tensor([False, False, True])
    )
