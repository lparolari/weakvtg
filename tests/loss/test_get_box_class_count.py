import torch

from weakvtg.loss import get_box_class_count


def test_get_box_class_count():
    box_class = torch.tensor([4, 2, 4, 0])
    class_count = torch.tensor([1, 0, 1, 0, 2])

    assert torch.equal(
        get_box_class_count(box_class, class_count),
        torch.tensor([2, 1, 2, 1])
    )
