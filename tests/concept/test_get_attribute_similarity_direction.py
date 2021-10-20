import torch

from tests.util import is_close
from weakvtg.concept import get_attribute_similarity_direction


def test_get_attribute_similarity_direction():
    x = torch.rand((2, 3))
    m1 = torch.tensor([[True, True, False]])
    m2 = torch.tensor([[True], [False]])

    expected = x
    expected = torch.masked_fill(expected, m1 == 0, 1)
    expected = torch.masked_fill(expected, m2 == 0, 1)

    assert is_close(
        get_attribute_similarity_direction(x, m1, m2, f_activation=torch.nn.Identity()),
        expected
    )
