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
        phrases_2_crd = batch["phrases_2_crd"]
        phrases = batch["phrases"]
        phrases_mask = batch["phrases_mask"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)
        phrases_synthetic_negative = get_synthetic_mask(phrases_mask)  # TODO

        (predicted_score_positive, predicted_score_negative) = output[0]  # [b, n_chunks, n_boxes]

        score_positive = predicted_score_positive
        score_negative = predicted_score_negative
        score_positive_mask = phrases_synthetic
        score_negative_mask = phrases_synthetic_negative

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

        validation = get_validation(boxes, phrases_2_crd, score_positive, score_positive_mask)

        return l_disc, *validation


def get_representation_for_best_bbox(x_repr: torch.Tensor, bbox_scores: torch.Tensor):
    """
    Gather the representation from `x_repr` for the best bounding box from `bbox_scores`.

    :param x_repr: A input representation for each phrase for each bounding box
    :param bbox_scores: A tensor with a list of bounding box logits for each phrase
    :return: A tensor with input representation for best bounding box
    """
    n_pred_box = x_repr.size()[2]

    best_bbox_indices = torch.argmax(bbox_scores, dim=-1)                # [b, n_chunks]
    best_bbox_indices = best_bbox_indices.unsqueeze(-1).unsqueeze(-1)    # [b, n_chunks, 1, 1]
    best_bbox_indices = best_bbox_indices.repeat(1, 1, n_pred_box, 500)  # [b, n_chunks, n_pred_box, 500]
    x_repr_best = torch.gather(x_repr, dim=2, index=best_bbox_indices)   # [b, n_chunks, 500]
    x_repr_best = x_repr_best[:, :, 0, :].contiguous()
    return x_repr_best


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


def get_active_index(active_box_index, shape):
    """
    Returns the bounding boxes active index for each chunk.

    :param active_box_index: A [d1, d3] torch.LongTensor
    :param shape: A [d1, d2, d3] torch.Size object
    :return: A [d1, d2, d3] torch.LongTensor
    """
    return active_box_index.repeat(1, shape[-2]).view(*shape)


def get_score(score, index, mask):
    """
    Returns scores on given index.

    :param score: A [*, d1, d2] tensor
    :param index: A [*, d1, d3] tensor
    :param mask: A [*, d1, 1] torch.BoolTensor
    :return: A [*, d1, d3] tensor
    """
    score = torch.gather(score, dim=-1, index=index)
    score = torch.masked_fill(score, mask == 0, value=0)
    return score
