import torch

from weakvtg.bbox import get_union_box


def test_get_union_box():
    box = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.]]) / 100.

    assert torch.equal(
        get_union_box(box),
        torch.tensor([0., 0., 15., 15.]) / 100.
    )


def test_get_union_box_given_single():
    box = torch.tensor([0., 0., 15., 15.]) / 100.

    assert torch.equal(
        get_union_box(box),
        torch.tensor([0., 0., 15., 15.]) / 100.
    )


def test_get_union_box_given_batch():
    box = torch.tensor([[[0., 0., 10., 10.], [5., 5., 15., 15.]], [[0., 0., 10., 10.], [5., 5., 7., 7.]]]) / 100.

    assert torch.equal(
        get_union_box(box),
        torch.tensor([[0., 0., 15., 15.], [0., 0., 10., 10.]]) / 100.
    )


def test_get_union_box_given_mask_max():
    box = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.], [0., 0., 10., 10.], [5., 5., 7., 7.]]) / 100.
    mask_max = torch.tensor([True, False, True, True]).unsqueeze(-1)

    assert torch.equal(
        get_union_box(box, mask_max=mask_max),
        torch.tensor([0., 0., 10., 10.]) / 100.
    )


def test_get_union_box_given_both_masks():
    box = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.], [0., 0., 10., 10.], [5., 5., 7., 7.]]) / 100.
    mask_min = torch.tensor([False, False, False, True]).unsqueeze(-1)
    mask_max = torch.tensor([True, False, True, True]).unsqueeze(-1)

    assert torch.equal(
        get_union_box(box, mask_min=mask_min, mask_max=mask_max),
        torch.tensor([5., 5., 10., 10.]) / 100.
    )
