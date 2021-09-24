import torch


def arloss(prediction, prediction_mask, box_mask, concept_direction):

    def average_over_boxes(x, m):
        return x.sum(dim=-1) / m.sum(dim=-1).unsqueeze(-1)

    def average_over_phrases(x, m):
        return x.sum() / m.sum()

    loss = -(prediction * concept_direction)

    # masking padded scores
    n_ph = prediction.size()[-2]

    box_mask_ = box_mask.unsqueeze(-2).repeat(1, n_ph, 1)

    loss = torch.masked_fill(loss, prediction_mask == 0, value=0)
    loss = torch.masked_fill(loss, box_mask_ == 0, value=0)

    # add up contributions
    loss = average_over_boxes(loss, box_mask)
    loss = average_over_phrases(loss, prediction_mask)

    return loss
