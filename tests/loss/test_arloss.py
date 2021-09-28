import pytest
import torch

from weakvtg.loss import arloss


def test_arloss():
    prediction = torch.tensor([[1., -1., 1., 1.], [.4, .5, .6, 1.]])
    prediction_mask = torch.tensor([[1], [1]])
    box_mask = torch.tensor([1, 1, 1, 0])
    box_class_count = torch.tensor([2, 1, 1, 1])
    concept_direction = torch.tensor([[1., 1., -1., 1.], [1., -1., 1., 1.]])

    # Given Q the set of queries, B the set of boxes, loss can be calculated as
    #   loss = - \sum_{q \in Q} p^q / | Q |
    # where
    #   p^q = \sum_{b \in B} ( ( s_b^q * d_b^q ) / f_b ) / | B |
    #   s_b^q = the similarity of box b wrt query q in [-1, 1]
    #   d_b^q =  the similarity of concept in [-1, 1]
    #   f_b = | { b' \in B | cls(b') = cls(b) } |

    p0 = ((prediction[0] * concept_direction[0]) / box_class_count)
    p0 = torch.masked_fill(p0, mask=box_mask == 0, value=0)
    p0 = p0.sum() / box_mask.sum(-1)

    p1 = ((prediction[1] * concept_direction[1]) / box_class_count)
    p1 = torch.masked_fill(p1, mask=box_mask == 0, value=0)
    p1 = p1.sum() / box_mask.sum(-1)

    loss = - (p0 + p1) / len(prediction)

    assert loss.item() == pytest.approx(0.2)
    assert arloss(prediction, prediction_mask, box_mask, box_class_count, concept_direction, eps=0).item() \
           == pytest.approx(loss)
