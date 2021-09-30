import torch

from weakvtg.loss import loss_inversely_correlated


def test_loss_inversely_correlated():
    X = torch.tensor([1, -1,  1, -1,  0, 0, .236], dtype=torch.float),
    y = torch.tensor([1, -1, -1,  1, -1, 1, -1], dtype=torch.float)

    assert torch.equal(
        loss_inversely_correlated(X, y),
        torch.tensor([-1, -1, 1, 1, 0, 0, .236])
    )
