import pytest
import torch
from math import log, exp

from weakvtg.loss import loss_maf, sim_mm
from weakvtg.matrix import get_positive, get_negative


def test_t1():
    x = torch.tensor([[[[0.4345, 0.6505, 0.3944]],
                       [[0.0253, 0.4157, 0.5813]]],
                      [[[0.9978, 0.1090, 0.3867]],
                       [[0.3077, 0.3430, 0.0620]]]])  # [2, 2, 1, 3]

    mask = torch.ones(2, 2)

    b = x.size(0)
    n_box = x.size(-1)

    prediction_p = get_positive(x).squeeze(0)
    prediction_n = get_negative(x).view(b, -1, n_box)
    mask_p = get_positive(mask).squeeze(0).unsqueeze(-1)
    mask_n = get_negative(mask)

    L_actual = loss_maf((prediction_p, prediction_n), (mask_p, mask_n), sim_mm)

    x1 = -log(exp(0.6505) / exp(0.5813))
    x2 = -log(exp(0.3430) / exp(0.9978))

    L_expected = (x1 + x2) / 2

    assert L_expected == pytest.approx(0.2928)
    assert L_actual.item() == pytest.approx(L_expected)
