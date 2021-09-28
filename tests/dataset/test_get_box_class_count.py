import torch

from weakvtg.dataset import get_box_class_count


def test_get_box_class_count():
    n_class = 5
    box_class = torch.tensor([4, 2, 4, 0])

    assert torch.equal(
        get_box_class_count(box_class, n_class=n_class),
        torch.tensor([1, 0, 1, 0, 2])
    )
