import pytest
import torch

from weakvtg.model import get_box_class


def test_get_box_class(probability):
    assert torch.equal(get_box_class(probability), torch.argmax(probability, dim=-1))


@pytest.fixture
def probability():
    return torch.tensor([[[.05, .15, .80], [.4, .2, .4]]])  # [b, n_boxes, n_class]
