def get_max(xs):
    return max(xs)


def get_argmax(xs):
    return xs.index(get_max(xs))


def masked_mean(x, mask, *args, **kwargs):
    """
    Return the mean among `x` scaling the contribution on number of "1" in `mask`.

    Please note that `x` is required to be a masked tensor for not summing up padded contributes.

    :param x: A [*, d2, d1] tensor
    :param mask: A [*, d2, 1] tensor
    :return: A [*, d2] tensor
    """
    return x.sum(*args, **kwargs) / mask.sum(*args, **kwargs)
