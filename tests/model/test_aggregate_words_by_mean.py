import pytest
import torch

from weakvtg.model import aggregate_words_by_mean


def test_aggregate_words_by_mean():
    x = torch.tensor([2., 4., 0.])  # last is padded
    mask = torch.tensor([1, 1, 0])

    assert aggregate_words_by_mean((x, mask), dim=-1).item() == pytest.approx(3.)


def test_aggregate_words_by_mean_given_unused_parameters():
    x = torch.tensor([2., 4., 0.])  # last is padded
    y = x
    mask = torch.tensor([1, 1, 0])

    assert aggregate_words_by_mean((x, mask), (y, mask),
                                   mask=mask, dim=-1, unsued_parameter=42).item() == pytest.approx(3.)
