import torch

from weakvtg.loss import get_predicted_box_by_union_on_max_class


def test_get_predicted_box_by_union_on_max_class():
    box = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.], [30., 30., 40., 40.], [0., 0., 10., 10.]]) / 100
    box_class = torch.tensor([5, 5, 17, 4])
    score = torch.tensor([[.3, .6, .1, .0], [.1, .1, .6, .2]])

    assert torch.equal(
        get_predicted_box_by_union_on_max_class(score, box, box_class),
        torch.tensor([[0., 0., 15., 15.], [30., 30., 40., 40.]]) / 100
    )
