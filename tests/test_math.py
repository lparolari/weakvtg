import pytest
import torch

from weakvtg.math import get_max, get_argmax, masked_mean


def test_get_max():
    assert get_max([1, 4, 2, 5, 1]) == 5


def test_get_argmax():
    assert get_argmax([1, 4, 2, 5, 1]) == 3


def test_masked_mean():
    x = torch.tensor([2., 4., 0.])  # last is padded
    mask = torch.tensor([1, 1, 0])

    assert masked_mean(x, mask, dim=-1).item() == pytest.approx(3.)
