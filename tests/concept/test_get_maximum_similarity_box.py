import torch

from tests.util import is_close
from weakvtg.concept import get_maximum_similarity_box

"""
.  .  .  +  .  .  .
.  .  .  A  .  B  .
.  .  .  +  .  .  .
+  +  +  +  +  C  +
.  .  .  +  .  .  .
.  .  .  E  .  D  .
.  .  .  +  .  .  .

S1 = {a, b} is the set of class embeddings
S2 = {c, d, e} is the set of word embeddings
"""


f_similarity = torch.cosine_similarity

a, b = [0., 1.], [1., 1.]  # class
c, d, e = [1., 0.], [1., -1.], [0., -1.]  # word

sim_c_a = 0.
sim_c_b = 0.7071
sim_d_a = -0.7071
sim_d_b = 0.
sim_e_a = -1.
sim_e_b = -0.7071


def test_get_maximum_similarity_box():
    phrase_embedding_t = torch.tensor([[c, d], [d, e]]), torch.tensor([[1, 1], [0, 1]]).unsqueeze(-1)
    box_class_embedding_t = torch.tensor([a, b]), torch.tensor([1, 1]).unsqueeze(-1)

    maximum_similarity_box_t = get_maximum_similarity_box(phrase_embedding_t, box_class_embedding_t, f_similarity)

    assert is_close(maximum_similarity_box_t[0], torch.tensor([[sim_c_b, sim_d_b], [-1., sim_e_b]]))
    assert is_close(maximum_similarity_box_t[1], torch.tensor([[1, 1], [0, 1]]))
