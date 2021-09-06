import pytest
import torch

from weakvtg.math import masked_mean
from weakvtg.model import get_concept_similarity


def test_get_concept_similarity(box_class_embedding, phrase_embedding, phrase_mask, f_aggregate):
    assert torch.equal(
        get_concept_similarity(box_class_embedding, phrase_embedding, phrase_mask, f_aggregate=f_aggregate,
                               f_similarity=torch.cosine_similarity),
        torch.tensor([[[1., 0.], [0., 1.]]])
    )


def test_get_concept_similarity_given_padded_phrase(box_class_embedding, phrase_embedding, phrase_mask_padded,
                                                    f_aggregate):
    assert torch.equal(
        get_concept_similarity(box_class_embedding, phrase_embedding, phrase_mask_padded, f_aggregate=f_aggregate,
                               f_similarity=torch.cosine_similarity),
        torch.tensor([[[1., 0.], [.5, .5]]])
    )


@pytest.fixture
def box_class_embedding():
    return torch.tensor([[[1., 1.], [-1., -1.]]])  # [b, n_box, f_emb]


@pytest.fixture
def phrase_embedding():
    return torch.tensor([[[[2., 1.], [-1., 0.]], [[-1., -1.], [5., 7.]]]])  # [b, n_ph, n_word, f_emb]


@pytest.fixture
def phrase_mask():
    return torch.tensor([[[[1], [1]], [[1], [0]]]])  # [b, n_ph, n_word, 1]


@pytest.fixture
def phrase_mask_padded():
    return torch.tensor([[[[1], [1]], [[0], [0]]]])  # [b, n_ph, n_word, 1]


@pytest.fixture
def f_aggregate():
    return masked_mean
