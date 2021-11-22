import torch

from weakvtg.loss import sim_mm_n


def test_sim_mm_n():
    x = torch.tensor([[[[0.0000, 0.0000],
                        [0.0000, 0.0000]],

                       [[0.3620, 0.6571],
                        [0.4334, 0.8536]]],


                      [[[0.7001, 0.5417],
                        [0.5703, 0.6368]],

                       [[0.0000, 0.0000],
                        [0.0000, 0.0000]]]])  # [b, b, n_ph, n_box]

    index = torch.tensor([[[0, 1], [1, 1]]])  # [1, b, n_ph]
    index = index.squeeze(0)  # [b, n_ph]

    # TODO: test does not work because we exp the matrix befor return!

    assert torch.equal(sim_mm_n(x, index), torch.tensor([[0.4334, 0.8536], [0.6368, 0.6368]]))
