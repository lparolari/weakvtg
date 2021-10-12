import torch

from weakvtg.loss import get_predicted_box_by_max


def test_get_predicted_box_by_max():
    score = torch.tensor([[.1, .1, .5, .3], [.3, .2, .2, .3]])
    box = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.], [30., 30., 40., 40.], [5., 5., 10., 10.]]) / 100

    assert torch.equal(
        get_predicted_box_by_max(score, box),
        torch.tensor([[30., 30., 40., 40.], [0., 0., 10., 10.]]) / 100.
    )
