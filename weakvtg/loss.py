import logging

import torch
import torch.nn as nn

from weakvtg import anchors
from weakvtg.corel import CosineARLoss
from weakvtg.mask import get_synthetic_mask


class WeakVtgLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss = CosineARLoss(lam=0.5, device=device)

    def forward(self, batch, output):
        boxes = batch["pred_boxes"]
        boxes_mask = batch["pred_boxes_mask"]
        phrases_2_crd = batch["phrases_2_crd"]
        phrases_mask = batch["phrases_mask"]
        phrases_mask_negative = batch["phrases_mask_negative"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)
        phrases_synthetic_negative = get_synthetic_mask(phrases_mask_negative)

        (predicted_score_positive, predicted_score_negative) = output[0]  # [b, n_chunks, n_boxes]

        def _get_scores(scores, phrases_mask, boxes_mask, boxes_fill_value):
            """
            Apply scores pipeline which (1) sets padded bounding box scores to 1 and -1 respectively for positive
            and negative phrases and (2) filter out padded phrases.
            """
            # gradient on padded boxes could affect results, for this reason we masked-fill padded boxes with 1 or -1
            # respectively for positive and negative phrases in order to match target value and don't let the gradient
            # to care about this scores.
            n_ph = scores.size()[-2]
            boxes_mask = boxes_mask.unsqueeze(-2).repeat(1, n_ph, 1)

            scores = torch.masked_fill(scores, boxes_mask == 0, value=boxes_fill_value)

            # also in this case, in order to save gradient issues, we completely remove padded phrases from scores.
            scores = filter_scores(scores, phrases_mask)

            return scores

        score_positive_mask = phrases_synthetic
        score_negative_mask = phrases_synthetic_negative
        score_positive = _get_scores(predicted_score_positive, score_positive_mask, boxes_mask, boxes_fill_value=+1)
        score_negative = _get_scores(predicted_score_negative, score_negative_mask, boxes_mask, boxes_fill_value=-1)

        l_disc = self.loss(
            (score_positive, score_positive_mask),
            (score_negative, score_negative_mask)
        )

        def get_validation(boxes, boxes_gt, scores, mask):
            with torch.no_grad():
                boxes_pred = get_boxes_predicted(boxes, scores, mask)

                iou_scores = get_iou_scores(boxes_pred, boxes_gt, mask)

                accuracy = get_accuracy(iou_scores, mask)
                p_accuracy = get_pointing_game_accuracy(boxes_gt, boxes_pred, mask)

            return iou_scores, accuracy, p_accuracy

        # Please note that for validation we should not use scores tensor given to loss module, because they are
        # masked in order to fix gradient problem. For validation we should use instead scores coming directly from
        # model output.
        validation = get_validation(boxes, phrases_2_crd, predicted_score_positive, phrases_synthetic)

        return l_disc, *validation


def get_iou_scores(boxes, gt, mask):
    """
    Compute and return IoU scores between two tensor of bounding boxes, and mask them with `mask`.

    :param boxes: A [..., d, 4] tensor
    :param gt: A [..., d, 4] tensor
    :param mask: A [..., d, 1] tensor
    :return: A [..., d, 1] tensor with IoU scores where mask is 1, else 0
    """
    iou_scores_ref = anchors.bbox_final_iou(boxes, gt)  # [b, n_chunks]
    iou_scores_ref = iou_scores_ref.unsqueeze(-1)       # [b, n_chunks, 1]
    iou_scores = iou_scores_ref * mask
    return iou_scores


def get_boxes_predicted(boxes, scores, mask):
    """
    Extract best bounding boxes from object detector boxes based on given scores.

    :param boxes: A [b, n_boxes, 4] tensor with bounding boxes from object detector
    :param scores: A [b, n_chunks, n_boxes] tensor with bounding box scores for each chunk
    :param mask: A [b, n_chunks, 1] long tensor which represent synthetic chunks
    :return: A [b, n_chunks, 4] tensor with best bounding box for each chunk
    """
    indexes = torch.argmax(scores, dim=-1)                  # [b, n_chunks]
    indexes = indexes.unsqueeze(-1).repeat(1, 1, 4).long()  # [b, n_chunks, 4]

    boxes = torch.gather(boxes, dim=-2, index=indexes)       # [b, n_chunks, 4]
    boxes = torch.masked_fill(boxes, mask == 0, value=0)

    return boxes


def get_accuracy(iou_scores, mask):
    """
    Compute state of the art accuracy by counting positive examples when IoU > 0.5.

    :param iou_scores: A [*, d, 1] tensor with IoU scores
    :param mask: A [*, d, *1] tensor representing whether queries match chunks
    :return: The accuracy as IoU > 0.5 among ground truth and predicted boxes
    """

    accuracy = iou_scores[..., 0] > 0.5  # [b, n_chunks]
    accuracy = accuracy.type(torch.long) * mask.squeeze(dim=-1)
    accuracy = torch.sum(accuracy) / mask.sum()

    return accuracy


def get_pointing_game_accuracy(boxes_gt, boxes_pred, mask):
    """
    Compute the pointing game accuracy by counting positive examples whether their centers fall inside ground truth
    bounding boxes.

    :param boxes_gt: A [b, n_chunks, 4] tensor with ground truth bounding boxes
    :param boxes_pred: A [b, n_chunks, 4] tensor with (best) predicted bounding box for each chunk
    :param mask: A [b, n_chunks] tensor representing whether queries match chunks
    :return: The accuracy counting positive examples with pointing game metric
    """
    # coordinates conversion
    boxes_pred_cthw = anchors.tlbr2cthw(boxes_pred)
    # calculate if the predicted center of the bounding boxes are inside the gt bounding boxes
    x_point = boxes_pred_cthw[..., 0]
    y_point = boxes_pred_cthw[..., 1]
    accuracy_x = torch.logical_and(boxes_gt[..., 0] <= x_point, x_point <= boxes_gt[..., 2])
    accuracy_y = torch.logical_and(boxes_gt[..., 1] <= y_point, y_point <= boxes_gt[..., 3])
    # final accuracy
    accuracy = torch.logical_and(accuracy_x, accuracy_y)
    accuracy = accuracy.int() * mask.squeeze(dim=-1)
    accuracy = torch.sum(accuracy) / mask.sum()
    return accuracy


def filter_scores(scores, mask):
    """
    Return items in `scores` tensor where `mask` is not False.

    Please note the returned tensor may vary in dimension wrt the number of True elements in `mask`.

    :param scores: A [d1, d2, *, dN-1, dN] tensor
    :param mask: A [d1, d2, *, dN-1, 1] bool tensor
    :return: A [d1 * d2 * ... * dN-1, dN] tensor
    """
    index = mask.long().squeeze(-1).nonzero(as_tuple=True)
    return scores[index]
