import torch
import torch.nn.functional as F

from weakvtg.model import get_image_features


def test_get_image_features():
    boxes_feat = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    boxes = torch.Tensor([[10, 10, 20, 20], [50, 50, 55, 55]])
    area = torch.Tensor([[100], [25]])

    expected = torch.cat([F.normalize(boxes_feat, p=1, dim=-1), boxes, area], dim=-1)
    actual = get_image_features(boxes, boxes_feat)

    assert torch.equal(expected, actual)
