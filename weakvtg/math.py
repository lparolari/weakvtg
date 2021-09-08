def get_max(xs):
    return max(xs)


def get_argmax(xs):
    return xs.index(get_max(xs))


def masked_mean(inp, *_args, dim, **_kwargs):
    """
    Return the mean among `x` scaling the contribution on number of "1" in `mask`.

    .. note::
     Please note that `x` is required to be a masked tensor in order to prevent the sum of padded contributes.

    .. note::
     Please note that `args` and `kwargs` are there only for interface compatibility purposes and they are not
     forwarded. You should provide `mask` and `dim` as kwargs.

    :param inp: A ([*, d1], [*, 1]) tuple of tensors, where (x, mask) = inp
    :param dim: The dimension to reduce
    :return: A [*] tensor
    """
    (x, mask) = inp

    return x.sum(dim=dim) / mask.sum(dim=dim)
