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
