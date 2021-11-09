import pytest
import torch

from weakvtg.loss import get_multimodal_similarity, loss_maf


def test_get_multimodal_similarity():
    x = torch.rand(2, 3)
    m = torch.ones(2)

    assert list(get_multimodal_similarity(x, m.sum(dim=-1)).size()) == []
    assert torch.mean(torch.max(x, dim=-1)[0], dim=-1)


def test_get_multimodal_similarity_given_mask():
    x = torch.rand(3, 3)
    m = torch.tensor([1, 1, 0], dtype=torch.bool)

    x = torch.masked_fill(x, mask=m.unsqueeze(-1) == 0, value=0)

    assert list(get_multimodal_similarity(x, m.sum(dim=-1)).size()) == []
    assert torch.sum(torch.max(x, dim=-1)[0], dim=-1) / 2


def test_loss_maf():
    from math import log, exp
    a_pos, mask_pos = torch.tensor([[.1, .3], [.2, .1]]), torch.tensor([1, 1])
    a_neg, mask_neg = torch.tensor([[.5], [.7], [.0]]), torch.tensor([1, 1, 0])

    assert loss_maf((a_pos, a_neg), (mask_pos, mask_neg), eps=0).item() == \
           pytest.approx(-log(exp((.3 + .2) / 2) / exp((.5 + .7) / 2)))
