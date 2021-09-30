import torch

from weakvtg.loss import loss_orthogonal


def test_loss_orthogonal():
    X = torch.tensor([1, -1,  1, -1,  0, 0, .236, -.751], dtype=torch.float),
    y = torch.tensor([1, -1, -1,  1, -1, 1,   -1,     1], dtype=torch.float)

    assert torch.equal(
        loss_orthogonal(X, y),
        -1 * torch.tensor([1, 0, 0, -1, 0, 0, 0, -.751]) + torch.tensor([0, 1, 1, 0, 0, 0, .236 ** 2, 0])
    )
