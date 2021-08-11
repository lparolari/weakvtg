import pytest
import torch

import weakvtg.loss as losses


@pytest.fixture
def b1(): return [0/100, 0/100, 10/100, 10/100], [1/100, 1/100, 11/100, 11/100]  # overlapping, iou >= 0.5
@pytest.fixture
def b2(): return [0/100, 0/100, 10/100, 10/100], [5/100, 5/100, 15/100, 15/100]  # overlapping, iou < 0.5
@pytest.fixture
def b3(): return [0/100, 0/100, 10/100, 10/100], [20/100, 20/100, 30/100, 30/100]  # non overlapping


@pytest.fixture
def boxes1(b1, b2, b3): return torch.Tensor([b1[0], b1[0], b2[0], b3[0]])  # [b, 4]
@pytest.fixture
def boxes2(b1, b2, b3): return torch.Tensor([b1[1], b1[1], b2[1], b3[1]])  # [b, 4]
@pytest.fixture
def mask(): return torch.Tensor([[[1], [0], [1], [1]]])
@pytest.fixture
def boxes(boxes1, boxes2): return torch.Tensor([boxes2.cpu().numpy()])  # [b, n_boxes, 4]


@pytest.fixture
def ch1(): return [1, 2, 3]
@pytest.fixture
def ch2(): return [5, 1, 0]
@pytest.fixture
def ch3(): return [7, 3, 0]
@pytest.fixture
def qr1(): return [6, 5, 0]
@pytest.fixture
def qr2(): return [2, 3, 4]
@pytest.fixture
def qr3(): return [9, 0, 0]
@pytest.fixture
def chunks(ch1, ch2, ch3): return torch.Tensor([[ch1, ch2, ch3]])   # [b, n_chunks, n_words]
@pytest.fixture
def queries(qr1, qr2, qr3): return torch.Tensor([[qr1, qr2, qr3]])  # [b, n_chunks, n_words]


@pytest.fixture
def boxes_scores(): return torch.Tensor([[[1/10, 5/10, 1/10, 3/10]]])  # [b, n_chunks, n_boxes]


@pytest.fixture
def iou_scores(): return torch.Tensor([[[0.55], [0.86], [0.31], [0.49]]])  # [b, n_chunks, 1]


def test_get_iou_scores(boxes1, boxes2, mask):
    expected = torch.Tensor([9*9 / (100 + 100 - 9*9), 0, 5*5 / (100 + 100 - 5*5), 0])

    scores = losses.get_iou_scores(boxes1, boxes2, mask)  # [b, 1]
    scores = scores.squeeze(-1)                           # [b]

    assert torch.all(torch.isclose(scores, expected)).item() is torch.Tensor([1]).bool().item()


def test_get_accuracy(iou_scores, mask):
    assert losses.get_accuracy(iou_scores, mask).item() == torch.Tensor([1 / 3]).item()


def test_get_pointing_game_accuracy(boxes1, boxes2, mask):
    assert losses.get_pointing_game_accuracy(boxes1, boxes2, mask).item() == torch.Tensor([2 / 3]).item()


def test_get_boxes_predicted(boxes, boxes_scores):
    # If scores are > 0 then corresponding value in mask should be 1 by definition.
    mask = torch.tensor([[[1]]])  # [b, n_chunks, 1]

    expected = torch.tensor([[boxes[0][1].numpy()]])
    predicted = losses.get_boxes_predicted(boxes, boxes_scores, mask)

    assert torch.equal(expected, predicted)
