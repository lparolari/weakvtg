import pytest
import torch

from weakvtg.dataset import get_boxes_mask_no_background


@pytest.fixture
def boxes_mask(): return torch.tensor([True, False, True])
@pytest.fixture
def boxes_class(): return torch.tensor([0, 15, 8])


def test_get_boxes_mask_no_background(boxes_mask, boxes_class):
    expected = torch.tensor([False, False, True])

    assert torch.equal(get_boxes_mask_no_background(boxes_mask, boxes_class), expected)
