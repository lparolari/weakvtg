import torch

from tests.util import is_close
from weakvtg.loss import loss_inversely_correlated_box_class_count_scaled


def test_loss_inversely_correlated():
    X = torch.tensor([1, -1,  1, -1,  0, 0, .236], dtype=torch.float), torch.tensor([3, 1, 1, 1, 0, 1, 1])
    y = torch.tensor([1, -1, -1,  1, -1, 1, -1], dtype=torch.float)

    assert is_close(
        loss_inversely_correlated_box_class_count_scaled(X, y),
        - (X[0] * y) / (X[1] + 1e-08)
    )
