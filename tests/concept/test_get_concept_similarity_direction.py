import torch

from weakvtg.concept import get_concept_similarity_direction, binary_threshold


def test_get_concept_similarity_direction():
    similarity = torch.tensor([[-.4, .7], [.8, .0]])
    assert torch.equal(get_concept_similarity_direction(similarity, f_activation=torch.nn.Identity()), similarity)


def test_binary_threshold():
    similarity = torch.tensor([-.4, .0, .1, .2, .8])
    threshold = 0.2

    assert torch.equal(binary_threshold(similarity, threshold), torch.tensor([-1., -1., -1., -1., 1.]))
