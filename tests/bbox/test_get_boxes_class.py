import pytest
import torch

from weakvtg.bbox import get_boxes_class


@pytest.fixture
def boxes_class_probability(): return torch.tensor([[.15, .1, .75], [.5, .2, .3]])


def test_get_boxes_class(boxes_class_probability):
    assert torch.equal(get_boxes_class(boxes_class_probability), torch.argmax(boxes_class_probability, dim=-1))