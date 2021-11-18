import torch

from weakvtg.loss import sim_mm_p


def test_sim_mm_p():
    x = torch.tensor([[[[0.7001, 0.5417],
                        [0.5703, 0.6368]],

                       [[0.3620, 0.6571],
                        [0.4334, 0.8536]]]])

    assert torch.equal(sim_mm_p(x)[0], torch.tensor([[[0.7001, 0.6368], [0.6571, 0.8536]]]))
    assert torch.equal(sim_mm_p(x)[1], torch.tensor([[[0, 1], [1, 1]]]))
