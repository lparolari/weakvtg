import torch


def arloss(prediction, prediction_mask, box_mask, box_class_count, concept_direction, eps=1e-08):

    def average_over_boxes(x, m):
        return x.sum(dim=-1) / m.sum(dim=-1).unsqueeze(-1)

    def average_over_phrases(x, m):
        return x.sum() / m.sum()

    # adjust dimensions
    n_ph = prediction.size()[-2]

    box_mask_unsqueeze = box_mask.unsqueeze(-2).repeat(1, n_ph, 1)
    box_class_count = box_class_count.unsqueeze(-2).repeat(1, n_ph, 1)
    box_class_count = box_class_count + eps  # prevent division by zero

    # compute loss
    loss = -1 * (concept_direction * prediction / box_class_count)

    # masking padded scores
    loss = torch.masked_fill(loss, prediction_mask == 0, value=0)
    loss = torch.masked_fill(loss, box_mask_unsqueeze == 0, value=0)

    # add up contributions
    loss = average_over_boxes(loss, box_mask)
    loss = average_over_phrases(loss, prediction_mask)

    return loss
