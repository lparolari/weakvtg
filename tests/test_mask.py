import pytest
import torch

from weakvtg.mask import get_synthetic_mask


@pytest.fixture
def mask():
    return torch.Tensor([[[1, 1, 0], [1, 1, 1]], [[1, 0, 0], [0, 0, 0]]]).bool()


def test_get_synthetic_mask(mask):
    assert torch.equal(get_synthetic_mask(mask), torch.Tensor([[[True], [True]], [[True], [False]]]).bool())
