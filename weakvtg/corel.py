import torch.nn as nn


def arloss(prediction, prediction_mask, box_mask, concept_direction, lam):

    def average_over_boxes(x, m):
        return x.sum(dim=-1) / m.sum(dim=-1).unsqueeze(-1)

    def average_over_phrases(x, m):
        return x.sum() / m.sum()

    prediction = prediction * concept_direction
    prediction = average_over_boxes(prediction, box_mask)
    prediction = average_over_phrases(prediction, prediction_mask)

    arloss = -(lam * prediction)

    return arloss


class CosineARLoss(nn.Module):
    """
    Adapted from https://github.com/lparolari/corel2019/blob/master/corel/loss_functions.py.
    See https://arxiv.org/pdf/1812.07627.pdf for more details.
    """
    def __init__(self, lam, device=None):
        super(CosineARLoss, self).__init__()
        self.lam = lam
        self.device = device

    def forward(self, prediction, prediction_mask, box_mask, concept_direction):
        return arloss(prediction, prediction_mask, box_mask, concept_direction, lam=self.lam)
