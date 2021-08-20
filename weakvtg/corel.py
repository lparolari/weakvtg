import torch.nn as nn


def arloss(attraction_tensor, repulsion_tensor, attraction_mask, repulsion_mask, boxes_mask, lam):

    def average_over_boxes(x, m):
        return x.sum(dim=-1) / m.sum(dim=-1).unsqueeze(-1)

    def average_over_phrases(x, m):
        return x.sum() / m.sum()

    loss_attraction = average_over_boxes(attraction_tensor, boxes_mask)
    loss_attraction = average_over_phrases(loss_attraction, attraction_mask)

    loss_repulsion = average_over_boxes(repulsion_tensor, boxes_mask)
    loss_repulsion = average_over_phrases(loss_repulsion, repulsion_mask)

    arloss = -(lam * loss_attraction) + ((1. - lam) * loss_repulsion)

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

    def forward(self, attraction, repulsion, boxes_mask):
        attraction_tensor, attraction_mask = attraction
        repulsion_tensor, repulsion_mask = repulsion

        # The original version of this work used to update the repulsion tensor by taking its max, i.e.,
        #   `repulsion_tensor, _ = repulsion_tensor.max(dim=1)`.
        # However, in our settings, we receive the right `repulsion_tensor` which can be made by a single example
        # (eventually computed with `max`) or by some examples (eventually computed with `topk`). We do not matter:
        # the responsibility is delegated outside this class.

        repulsion_tensor = repulsion_tensor ** 2

        return arloss(attraction_tensor, repulsion_tensor, attraction_mask, repulsion_mask, boxes_mask, lam=self.lam)
