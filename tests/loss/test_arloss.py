import pytest
import torch

from weakvtg.loss import arloss, loss_inversely_correlated, loss_inversely_correlated_box_class_count_scaled


def test_arloss():
    prediction = torch.tensor([[1., -1., 1., 1.], [.4, .5, .6, 1.]])
    prediction_mask = torch.tensor([[1], [1]])
    box_mask = torch.tensor([1, 1, 1, 0])
    box_class_count = torch.tensor([2, 1, 1, 1])
    concept_direction = torch.tensor([[1., 1., -1., 1.], [1., -1., 1., 1.]])

    p0 = (prediction[0] * concept_direction[0])
    p0 = torch.masked_fill(p0, mask=box_mask == 0, value=0)
    p0 = p0.sum() / box_mask.sum(-1)

    p1 = (prediction[1] * concept_direction[1])
    p1 = torch.masked_fill(p1, mask=box_mask == 0, value=0)
    p1 = p1.sum() / box_mask.sum(-1)

    loss = - (p0 + p1) / len(prediction)

    assert loss.item() == pytest.approx(0.0833, rel=1e-03)
    assert torch.isclose(
        loss,
        arloss(prediction, prediction_mask, box_mask, box_class_count, concept_direction,
               f_loss=loss_inversely_correlated)
    )


def test_arloss_given_scale_by_box_class_count():
    prediction = torch.tensor([[1., -1., 1., 1.], [.4, .5, .6, 1.]])
    prediction_mask = torch.tensor([[1], [1]])
    box_mask = torch.tensor([1, 1, 1, 0])
    box_class_count = torch.tensor([2, 1, 1, 1])
    concept_direction = torch.tensor([[1., 1., -1., 1.], [1., -1., 1., 1.]])

    p0 = ((prediction[0] * concept_direction[0]) / box_class_count)
    p0 = torch.masked_fill(p0, mask=box_mask == 0, value=0)
    p0 = p0.sum() / box_mask.sum(-1)

    p1 = ((prediction[1] * concept_direction[1]) / box_class_count)
    p1 = torch.masked_fill(p1, mask=box_mask == 0, value=0)
    p1 = p1.sum() / box_mask.sum(-1)

    loss = - (p0 + p1) / len(prediction)

    assert loss.item() == pytest.approx(0.2)
    assert torch.isclose(
        loss,
        arloss(prediction, prediction_mask, box_mask, box_class_count, concept_direction,
               f_loss=loss_inversely_correlated_box_class_count_scaled)
    )
