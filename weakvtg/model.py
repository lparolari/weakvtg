import torch
import torch.nn as nn
import torch.nn.functional as F

from weakvtg import anchors
from weakvtg.mask import get_synthetic_mask


class Model(nn.Module):
    def forward(self, batch):
        raise NotImplementedError


class MockModel(Model):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, batch):
        boxes = batch["pred_boxes"]
        phrases = batch["phrases"]
        phrases_mask = batch["phrases_mask"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)

        size = (*phrases_synthetic.size()[:-1], boxes.size()[-2])

        return (torch.rand(size, requires_grad=True), torch.rand(size, requires_grad=True)),


class WeakVtgModel(Model):
    def forward(self, batch):
        boxes = batch["pred_boxes"]  # [b, n_boxes, 4]
        boxes_features = batch["pred_boxes_features"]  # [b, n_boxes, 2048]
        phrases = batch["phrases"]  # [b, n_ph, n_words]
        phrases_mask = batch["phrases_mask"]  # [b, n_ph, n_words]

        img_x = get_image_features(boxes, boxes_features)
        # TODO


def get_image_features(boxes, boxes_feat):
    """
    Normalize bounding box features and concatenate its spacial features (position and area).

    :param boxes: A [*1, 4] tensor
    :param boxes_feat: A [*2, fi] tensor
    :return: A [*3, fi + 5] tensor
    """
    boxes_feat = F.normalize(boxes_feat, p=1, dim=-1)

    boxes_tlhw = anchors.tlbr2tlhw(boxes)  # [*1, 4]
    area = (boxes_tlhw[..., 2] * boxes_tlhw[..., 3]).unsqueeze(-1)  # [*1, 1]

    return torch.cat([boxes_feat, boxes, area], dim=-1)
