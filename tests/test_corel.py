import pytest
import torch

from weakvtg.corel import arloss


def test_arloss():
    prediction = torch.tensor([[1., -1., 1., 0.], [.4, .5, .6, 0.]])
    prediction_mask = torch.tensor([[1], [1]])
    box_mask = torch.tensor([1, 1, 1, 0])
    concept_direction = torch.tensor([[1., 1., -1., 1.], [1., -1., 1., 1.]])

    # Given Q the set of queries, B the set of boxes, loss can be calculated as
    #   loss = - \sum_{q \in Q} p^q / | Q |
    # where
    #   p^q = \sum_{b \in B} ( s_b^q * d_b^q ) / | B |
    #   s_b^q = the similarity of box b wrt query q in [-1, 1]
    #   d_b^q =  the similarity of concept in [-1, 1]
    #
    # So
    #   p^1 = (1 - 1 - 1) / 3
    #   p^2 = (.4 - .5 + .6) / 3 = 1/2 / 3 = 1 / 6
    #   loss = - ( - 1/3 + 1/6 ) / 2 = - ( - 1/6 / 2 ) = - ( - 0.16 / 2 ) = + 0.083

    assert arloss(prediction, prediction_mask, box_mask, concept_direction).item() \
           == pytest.approx(0.0833, abs=1e-3)
