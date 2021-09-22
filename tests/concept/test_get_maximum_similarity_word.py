import torch

from tests.util import is_close
from weakvtg.concept import get_maximum_similarity_word

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


def test_get_maximum_similarity_word():
    phrase_embedding_t = torch.tensor([[c, d], [d, e]]), torch.tensor([[1, 1], [0, 1]]).unsqueeze(-1)
    maximum_similarity_box_t = torch.tensor([[sim_c_b, sim_d_b], [-1., sim_e_b]]), torch.tensor([[1, 1], [0, 1]])

    assert is_close(get_maximum_similarity_word(phrase_embedding_t, maximum_similarity_box_t), torch.tensor([c, e]))
