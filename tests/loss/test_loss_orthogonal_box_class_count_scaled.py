import torch

from weakvtg.loss import loss_orthogonal_box_class_count_scaled


def test_loss_orthogonal_box_class_count_scaled():
    X = torch.tensor([1, -1,  1, -1,  0, 0, .236, -.751], dtype=torch.float), torch.tensor([3, 1, 1, 1, 0, 1, 1, 0])
    y = torch.tensor([1, -1, -1,  1, -1, 1,   -1,     1], dtype=torch.float)

    eps = 1e-08

    x_pos = X[0] * (y == 1)
    x_neg = X[0] * (y != 1)

    count_pos = X[1] * (y == 1) + eps
    count_neg = X[1] * (y != 1) + eps

    assert torch.equal(
        loss_orthogonal_box_class_count_scaled(X, y),
        -1 * (x_pos / count_pos) + torch.square(x_neg) / count_neg
    )
