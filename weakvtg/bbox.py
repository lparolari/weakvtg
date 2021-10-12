import torch


def scale_bbox(bbox_list, width, height):
    """
    Normalize a bounding box give max_x and max_y.
    :param bbox_list: list of list of coodinates in format: [xmin, ymin, xmax, ymax]
    :param width: image max width.
    :param height: image max height
    :return: list of list of normalized coordinates.
    """
    results = []
    for i in bbox_list:
        xmin, ymin, xmax, ymax = i
        norm_cr = [xmin / width, ymin / height, xmax / width, ymax / height]
        results.append(norm_cr)
    return results


def get_boxes_class(boxes_class_probability):
    return torch.argmax(boxes_class_probability, dim=-1)


def get_union_box(box, *, mask_min=None, mask_max=None):
    """
    Return the union box along last but one dimension.

    :param box: A [d1, ..., dN-1, dN] tensor, usually dN = 4
    :param mask_min: A [d1, ..., dN-1, 1] tensor
    :param mask_max: A [d1, ..., dN-1, dN] tensor
    :return: A [d1, ..., dN-1] tensor
    """
    if mask_min is None:
        mask_min = torch.ones_like(box)

    if mask_max is None:
        mask_max = torch.ones_like(box)

    box_min = torch.masked_fill(box, mask=mask_min == 0, value=1)
    box_max = torch.masked_fill(box, mask=mask_max == 0, value=0)

    x_min = torch.min(box_min[..., 0], dim=-1)[0].unsqueeze(-1)
    y_min = torch.min(box_min[..., 1], dim=-1)[0].unsqueeze(-1)
    x_max = torch.max(box_max[..., 2], dim=-1)[0].unsqueeze(-1)
    y_max = torch.max(box_max[..., 3], dim=-1)[0].unsqueeze(-1)

    union_box = torch.stack([x_min, y_min, x_max, y_max], dim=-2).squeeze(-1)

    return union_box
