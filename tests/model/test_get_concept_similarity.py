import functools

import torch

from weakvtg.model import get_concept_similarity, aggregate_words_in_phrase


"""
unit = +  +

.  .  .  +  .  .  .
.  G  .  A  .  B  .
.  .  .  +  .  .  .
+  F  +  Z  +  C  +
.  .  .  +  .  .  .
.  E  .  D  .  .  .
.  .  .  +  .  .  .

S1 = {A, B, C} are embeddings for bounding box's class
S2 = {D, E, F, G}  are embeddings for phrase's word

sim(A, B) = 0.7071

sim(G, A) = 0.7071
sim(F, A) = 0
sim(E, A) = -0.7071
sim(D, A) = -1

sim({D, E, F}, {A, B, C}) < sim(G, A)

sim(G, A) is the maximum similarity of {D, E, F, G} wrt {A, B, C}

sim(G, A) = 0.7071
sim(G, B) = 0
sim(G, C) = -0.7071
"""

a, b, c = [0., 1.], [1., 1.], [1., 0.]
d, e, f, g = [0., -1.], [-1., -1.], [-1., 0.], [-1., 1.]

sim_g_a = 0.7071
sim_g_b = 0.
sim_g_c = -0.7071


sim_f_a = -0.
sim_f_b = -0.7071
sim_f_c = -1.

f_aggregate = aggregate_words_in_phrase

_get_concept_similarity = functools.partial(
    get_concept_similarity,
    f_aggregate=f_aggregate,
    f_similarity=torch.cosine_similarity,
    f_activation=torch.nn.Identity()
)

_get_concept_similarity_relu = functools.partial(
    get_concept_similarity,
    f_aggregate=f_aggregate,
    f_similarity=torch.cosine_similarity,
    f_activation=torch.relu
)


def test_get_concept_similarity():
    concept_similarity = torch.tensor([[sim_g_a, sim_g_b, sim_g_c], [sim_f_a, sim_f_b, sim_f_c]])

    box_class_embedding = torch.tensor([[a, b, c]])  # [b, n_box, f_emb]
    phrase_embedding = torch.tensor([[[d, e, f, g], [f, f, f, f]]])  # [b, n_ph, n_word, f_emb]

    box_class_mask = torch.ones(1, 3, 1)
    phrase_mask = torch.ones(1, 2, 4, 1)

    assert is_close(
        _get_concept_similarity((box_class_embedding, box_class_mask), (phrase_embedding, phrase_mask)),
        concept_similarity
    )


def test_get_concept_similarity_given_padded_words():
    concept_similarity = torch.tensor([[sim_f_a, sim_f_b, sim_f_c], [-1., -1., -1.]])

    box_class_embedding = torch.tensor([[a, b, c]])  # [b, n_box, f_emb]
    phrase_embedding = torch.tensor([[[d, e, f, g], [g, g, g, g]]])  # [b, n_ph, n_word, f_emb]

    box_class_mask = torch.ones(1, 3, 1)
    phrase_mask = torch.tensor([[[0, 1, 1, 0], [0, 0, 0, 0]]]).unsqueeze(-1)

    assert is_close(
        _get_concept_similarity((box_class_embedding, box_class_mask), (phrase_embedding, phrase_mask)),
        concept_similarity
    )


def test_get_concept_similarity_given_padded_box():
    concept_similarity = torch.tensor([[-1., sim_g_b, -1.], [-1., sim_g_b, -1.]])

    box_class_embedding = torch.tensor([[a, b, c]])  # [b, n_box, f_emb]
    phrase_embedding = torch.tensor([[[d, e, f, g], [g, g, g, g]]])  # [b, n_ph, n_word, f_emb]

    box_class_mask = torch.tensor([[0, 1, 0]]).unsqueeze(-1)
    phrase_mask = torch.ones(1, 2, 4, 1)

    assert is_close(
        _get_concept_similarity((box_class_embedding, box_class_mask), (phrase_embedding, phrase_mask)),
        concept_similarity
    )


def test_get_concept_similarity_given_relu():
    concept_similarity = torch.tensor([[sim_g_a, sim_g_b, 0.], [sim_f_a, 0., 0.]])

    box_class_embedding = torch.tensor([[a, b, c]])  # [b, n_box, f_emb]
    phrase_embedding = torch.tensor([[[d, e, f, g], [f, f, f, f]]])  # [b, n_ph, n_word, f_emb]

    box_class_mask = torch.ones(1, 3, 1)
    phrase_mask = torch.ones(1, 2, 4, 1)

    assert is_close(
        _get_concept_similarity_relu((box_class_embedding, box_class_mask), (phrase_embedding, phrase_mask)),
        concept_similarity
    )


def is_close(*args, **kwargs):
    return all(torch.isclose(*args, **kwargs, rtol=1e-04).detach().numpy().ravel())


def test_is_close():
    assert is_close(torch.tensor([1., 0.11111]), torch.tensor([1., 0.11112]))
    assert not is_close(torch.tensor([1., 0.1111]), torch.tensor([1., 0.1112]))
    assert not is_close(torch.tensor([[0.1111]]), torch.tensor([[0.1112]]))
    assert not is_close(torch.tensor([[[0.1111]]]), torch.tensor([[[0.1112]]]))

