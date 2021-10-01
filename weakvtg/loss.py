import torch
import torch.nn as nn

from weakvtg import anchors
from weakvtg.bbox import get_boxes_class
from weakvtg.mask import get_synthetic_mask


class WeakVtgLoss(nn.Module):
    def __init__(self, get_concept_similarity_direction, f_loss):
        super().__init__()
        self.get_concept_similarity_direction = get_concept_similarity_direction
        self.f_loss = f_loss

    def forward(self, batch, output):
        get_concept_similarity_direction = self.get_concept_similarity_direction
        f_loss = self.f_loss

        boxes = batch["pred_boxes"]
        boxes_mask = batch["pred_boxes_mask"]
        boxes_class_pred = batch["pred_cls_prob"]
        boxes_class = get_boxes_class(boxes_class_pred)
        class_count = batch["class_count"]
        phrases_2_crd = batch["phrases_2_crd"]
        phrases_mask = batch["phrases_mask"]
        phrases_mask_negative = batch["phrases_mask_negative"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)
        phrases_synthetic_negative = get_synthetic_mask(phrases_mask_negative)
        boxes_class_count = get_box_class_count(boxes_class, class_count)

        (predicted_score_positive, predicted_score_negative) = output[0]  # [b, n_ph, n_box]
        (positive_concept_similarity, negative_concept_similarity) = output[1]  # [b, n_ph, n_box]

        concept_direction = get_concept_similarity_direction(positive_concept_similarity)  # [b, n_ph, n_box]

        score_positive_mask = phrases_synthetic
        score_positive = predicted_score_positive

        l_disc = arloss(
            score_positive,
            score_positive_mask,
            boxes_mask,
            boxes_class_count,
            concept_direction,
            f_loss=f_loss
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


def arloss(prediction, prediction_mask, box_mask, box_class_count, concept_direction, f_loss):
    """
    Given Q the set of queries, B the set of boxes, loss can be calculated as
        loss = - sum_{q in Q} p^q / | Q |
    where
        p^q = sum_{b in B} f_loss(X, y) / | B |
        X = prediction, box_class_count
        y = concept_direction

    :param prediction: A [*, d1, d2] tensor
    :param prediction_mask: A [*, d1, 1] tensor
    :param box_mask: A [*, d2] tensor
    :param box_class_count: A [*, d2] tensor
    :param concept_direction: A [*, d1, d2] tensor
    :param f_loss: A loss function, return a [*, d1, d2] tensor
    :return: A float value
    """

    def average_over_boxes(x, m):
        return x.sum(dim=-1) / m.sum(dim=-1).unsqueeze(-1)

    def average_over_phrases(x, m):
        return x.sum() / m.sum()

    # adjust dimensions
    n_ph = prediction.size()[-2]
    box_mask_unsqueeze = box_mask.unsqueeze(-2).repeat(1, n_ph, 1)
    box_class_count = box_class_count.unsqueeze(-2).repeat(1, n_ph, 1)

    # compute loss
    X = (prediction, box_class_count)
    y = concept_direction

    loss = f_loss(X, y)

    # masking padded scores
    loss = torch.masked_fill(loss, prediction_mask == 0, value=0)
    loss = torch.masked_fill(loss, box_mask_unsqueeze == 0, value=0)

    # add up contributions
    loss = average_over_boxes(loss, box_mask)
    loss = average_over_phrases(loss, prediction_mask)

    return loss


def loss_inversely_correlated(X, y):
    """
    Return
        -1 * (concept_direction * prediction)
    where
        prediction = X[0]
        concept_direction = y
    """
    prediction, *_ = X
    concept_direction = y

    return -1 * (concept_direction * prediction)


def loss_inversely_correlated_box_class_count_scaled(X, y):
    """
    Return
        -1 * (concept_direction * prediction / box_class_count)
    where
        prediction = X[0]
        box_class_count = X[1]
        concept_direction = y
    """
    eps = 1e-08

    prediction, box_class_count, *_ = X
    concept_direction = y

    box_class_count = box_class_count + eps  # prevent division by zero

    return -1 * (concept_direction * prediction / box_class_count)


def loss_orthogonal(X, y):
    """
    Return
        -1 * x_pos + torch.square(x_neg)
    where
        x_pos = X[0] (i.e., predictions) where negative values are zero
        x_neg = X[0] (i.e., predictions) where positive values are zero
    """
    x, *_ = X

    mask_pos = y == 1

    x_pos = torch.masked_fill(x, mask=mask_pos == 0, value=0)
    x_neg = torch.masked_fill(x, mask=mask_pos == 1, value=0)

    return -1 * x_pos + torch.square(x_neg)


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


def get_box_class_count(box_class, class_count):
    """
    Return a tensor with the number of bounding boxes with class c for each bounding box b, where `c = cls(b)`.

    :param box_class: A [*, d1] tensor
    :param class_count: A [*, d2] tensor
    :return: A [*, d1] tensor
    """
    return torch.gather(class_count, dim=-1, index=box_class)
