import torch

from weakvtg.utils import invert


def get_ident(size, n_dims, dim=-2):
    """
    Generates a square matrix `size x size` at dim `dim` with `n_dims`.

    :param size: Size of vector
    :param n_dims: Number of dimensions
    :param dim: The dimension to keep of size `size`
    :return: A [1, 1, ..., 1,     size, 1,     ..., 1] tensor
                           dim-1  dim   dim+1
    """
    dims = [1] * n_dims
    dims[dim] = size
    dims[dim + 1] = size

    return torch.eye(size).view(*dims)


def get_ones(size, n_dims, dim=-2):
    """
    Generates a square matrix `size x size` at dim `dim` with `n_dims`.
    :param size: Size of vector
    :param n_dims: Number of dimensions
    :param dim: The dimension to keep of size `size`
    :return: A [1, 1, ..., 1,     size, 1,     ..., 1] tensor
                           dim-1  dim   dim+1
    """
    dims = [1] * n_dims
    dims[dim] = size
    return torch.ones(size).view(dims)


def get_masked(x, mask):
    """
    Returns
        x * mask

    :param x: A [*d] tensor
    :param mask: A [*d] tensor
    :return: A [*d] tensor
    """
    return x * mask


def get_diag(x):
    """
    Returns
        x @ v
    where
        v = [1, 1, ..., 1] e [*, 1, d]

    :param x: A [*, d, d] tensor
    :return: A [*, d, 1] tensor
    """
    v = get_ones(x.size(-1), x.dim())
    return x @ v


def get_positive(x):
    """
    Returns values on diagonal.

    :param x: A [d, d, *] tensor
    :return: A [1, d, *] tensor
    """
    x = invert(x)
    ident = get_ident(x.size(-1), x.dim())

    x = get_masked(x, ident)

    return invert(get_diag(x))


def get_negative(x):
    """
    Returns values outside diagonal

    :param x: A [d, d, *] tensor
    :return: A [d, d, *] tensor
    """
    x = invert(x)
    ident = get_ident(x.size(-1), x.dim())

    x = get_masked(x, 1 - ident)

    return invert(x)
