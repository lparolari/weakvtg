import torch
import torch.nn as nn

from weakvtg import anchors
from weakvtg.bbox import get_boxes_class, get_union_box
from weakvtg.mask import get_synthetic_mask
from weakvtg.utils import expand


class WeakVtgLoss(nn.Module):
    def __init__(self, *, get_concept_similarity_direction, get_attribute_similarity_direction, get_predicted_box,
                 f_loss):
        super().__init__()
        self.get_concept_similarity_direction = get_concept_similarity_direction
        self.get_attribute_similarity_direction = get_attribute_similarity_direction
        self.get_predicted_box = get_predicted_box
        self.f_loss = f_loss

    def forward(self, batch, output):
        get_concept_similarity_direction = self.get_concept_similarity_direction
        get_attribute_similarity_direction = self.get_attribute_similarity_direction
        get_predicted_box = self.get_predicted_box
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
        adjective_mask = batch["adjective_mask"]  # [b, n_np, n_attr]
        box_attribute_mask = batch["attribute_mask"]  # [b, n_box]

        (predicted_score_positive, predicted_score_negative) = output[0]  # [b, n_ph, n_box]
        (positive_concept_similarity, negative_concept_similarity) = output[1]  # [b, n_ph, n_box]
        attribute_similarity = output[2]

        concept_direction = get_concept_similarity_direction(positive_concept_similarity)  # [b, n_ph, n_box]
        attribute_direction = get_attribute_similarity_direction(
            attribute_similarity, box_attribute_mask=box_attribute_mask.unsqueeze(-2),
            adjective_mask=adjective_mask.sum(dim=-1, keepdims=True).to(torch.bool))  # [b, n_ph, n_box]

        loss_direction = get_loss_direction(concept_direction, attribute_direction)  # TODO: temporary disabled

        score_positive_mask, score_negative_mask = phrases_synthetic.squeeze(-1), phrases_synthetic_negative.squeeze(-1)
        score_positive, score_negative = predicted_score_positive, predicted_score_negative

        L = loss_maf((score_positive, score_negative), (score_positive_mask, score_negative_mask))

        def get_validation(boxes, boxes_gt, scores, mask):
            with torch.no_grad():
                boxes_pred = get_predicted_box(scores, boxes, boxes_class)
                boxes_pred = torch.masked_fill(boxes_pred, mask=mask == 0, value=0)

                iou_scores = get_iou_scores(boxes_pred, boxes_gt, mask)

                accuracy = get_accuracy(iou_scores, mask)
                p_accuracy = get_pointing_game_accuracy(boxes_gt, boxes_pred, mask)

            return iou_scores, accuracy, p_accuracy

        # Please note that for validation we should not use scores tensor given to loss module, because they are
        # masked in order to fix gradient problem. For validation we should use instead scores coming directly from
        # model output.
        validation = get_validation(boxes, phrases_2_crd, predicted_score_positive, phrases_synthetic)

        return L, *validation


def arloss(prediction, prediction_mask, box_mask, box_class_count, loss_direction, f_loss):
    """
    Given Q the set of queries, B the set of boxes, loss can be calculated as
        loss = - sum_{q in Q} p^q / | Q |
    where
        p^q = sum_{b in B} f_loss(X, y) / | B |
        X = prediction, box_class_count
        y = loss_direction

    :param prediction: A [*, d1, d2] tensor
    :param prediction_mask: A [*, d1, 1] tensor
    :param box_mask: A [*, d2] tensor
    :param box_class_count: A [*, d2] tensor
    :param loss_direction: A [*, d1, d2] tensor
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
    y = loss_direction

    loss = f_loss(X, y)

    # masking padded scores
    loss = torch.masked_fill(loss, prediction_mask == 0, value=0)
    loss = torch.masked_fill(loss, box_mask_unsqueeze == 0, value=0)

    # add up contributions
    loss = average_over_boxes(loss, box_mask)
    loss = average_over_phrases(loss, prediction_mask)

    return loss


def loss_maf(attention_t, mask_t):
    """
    Returns
                   e^sim(I, S)
        L = - log --------------
                   e^sim(I, S')
    where
        sim(I, S) is the multimodal similarity built on
          attention A_{j,z} with A_{j,z} the similarity
          measure between phrase p_j and box b_z,
        I is the set of bounding box,
        S is the set of queries, and
        S' is the set of negative queries

    :param attention_t: A tuple of [*, n_ph, n_box] tensors
    :param mask_t: A tuple of [*, n_ph] tensors
    :return: A float value
    """
    attention_positive, attention_negative = attention_t
    mask_positive, mask_negative = mask_t

    def _get_multimodal_similarity(attention, mask):
        attention = torch.masked_fill(attention, mask=mask.unsqueeze(-1) == 0, value=0)
        n = mask.sum(dim=-1)

        return get_multimodal_similarity(attention, n)

    multimodal_similarity_positive = torch.exp(_get_multimodal_similarity(attention_positive, mask_positive))
    multimodal_similarity_negative = torch.exp(_get_multimodal_similarity(attention_negative, mask_negative))

    loss = -torch.log(multimodal_similarity_positive / multimodal_similarity_negative)

    return loss.mean()


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

    return -1 * y * x_pos + torch.square(y * x_neg)


def loss_orthogonal_box_class_count_scaled(X, y):
    """
    Return
         -1 * (x_pos / count_pos) + x_neg^2 / count_neg
       = -1 * x_pos * freq_pos + x_neg^2 * freq_neg
    where
        x_pos = X[0] (i.e., predictions) where negative values are zero
        x_neg = X[0] (i.e., predictions) where positive values are zero
        count_pos = X[1] (i.e., counts) where negative values are zero
        count_neg = X[1] (i.e., counts) where positive values are zero
    """
    eps = 1e-08

    x, box_class_count, *_ = X

    mask_pos = y == 1

    x_pos = torch.masked_fill(x, mask=mask_pos == 0, value=0)
    x_neg = torch.masked_fill(x, mask=mask_pos == 1, value=0)

    count_pos = torch.masked_fill(box_class_count, mask=mask_pos == 0, value=0) + eps
    count_neg = torch.masked_fill(box_class_count, mask=mask_pos == 1, value=0) + eps

    return -1 * (y * x_pos / count_pos) + torch.square(y * x_neg) / count_neg


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


def get_predicted_box_by_max(score, box, *_, **__):
    """
    Extract best bounding boxes from object detector boxes based on given scores.

    :param score: A [d*, d1, d2] tensor
    :param box: A [d*, d2, 4] tensor
    :return: A [d*, d1, 4] tensor with best bounding box for each chunk
    """
    index = torch.argmax(score, dim=-1)                 # [*, d1]
    index = expand(index, dim=-1, size=box.size()[-1])  # [*, d1, 4]

    boxes = torch.gather(box, dim=-2, index=index)      # [*, d1, 4]

    return boxes


def get_predicted_box_by_union_on_max_class(score, box, box_class, *_, **__):
    """
    Return the union box among all boxes with the same class as the most confident box per phrase.

    :param score: A [*, d1, d2] tensor
    :param box: A [*, d2, 4] tensor
    :param box_class: A [*, d2] long tensor
    :return: A [*, d1, 4] tensor
    """
    n_ph = score.size()[-2]

    box_full = expand(box, dim=-3, size=n_ph)              # [*, d1, d2, 4]
    box_class_full = expand(box_class, dim=-2, size=n_ph)  # [*, d1, d2]

    best_score_index = torch.argmax(score, dim=-1).unsqueeze(-1)                   # [*, d1, 1]
    best_box_class = torch.gather(box_class_full, dim=-1, index=best_score_index)  # [*, d1, 1]

    box_class_mask = (box_class_full == best_box_class).unsqueeze(-1)  # [*, d1, d2, 1]

    # Please note that the following is not possible because the
    # resulting tensor size depends on the number of positive values
    # in `box_class_mask` which can be different for every per
    # phrase. For this reason the result contains all valid bounding
    # box without distinguishing between phrases, which is not what
    # we want.
    #
    # box_given_max_class = box_full[box_class_mask.nonzero(as_tuple=True)]
    # return get_union_box(box_given_max_class)

    return get_union_box(box_full, mask_min=box_class_mask, mask_max=box_class_mask)  # [*, d1, 4]


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


def get_loss_direction(a, b):
    """
    Return a tensor with values in {-1, 0, 1} implementing the following function
    `f(a, b) = (a + b - b * (1 - a)) - 1`.

    The desired effect is
     * a=1, b=1 -> 1
     * a=1, b=0 -> 0
     * a=0, b=1 -> -1
     * a=0, b=0 -> -1

    :param a: A [d1, ..., dN] tensor with values in {0, 1}
    :param b: A [d1, ..., dN] tensor with values in {0, 1}
    :return: A [d1, ..., dN] tensor with values in {-1, 0, 1}
    """
    return (a + b - b * (1 - a)) - 1


def get_multimodal_similarity(attention, n, eps=1e-08):
    """
    Returns
        1
        - * sum max A_{j,z}
        N    j   z

    Note: attention tensor should be already masked

    :param attention: A [*, d2, d1] tensor
    :param n: A [*] tensor
    :param eps: A float value
    :return: A [*] tensor
    """
    attention, _ = torch.max(attention, dim=-1)  # [*, d2]
    attention = torch.sum(attention, dim=-1)     # [*]

    return attention / (n + eps)   # [*]
