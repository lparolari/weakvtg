import torch

from weakvtg.concept import aggregate_words_by_max

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


def test_aggregate_words_by_max():
    phrase_embedding_t = torch.tensor([[c, d], [d, e]]), torch.tensor([[1, 1], [0, 1]]).unsqueeze(-1)
    box_class_embedding_t = torch.tensor([a, b]), torch.tensor([1, 1]).unsqueeze(-1)

    assert torch.equal(
        aggregate_words_by_max(phrase_embedding_t, box_class_embedding_t, f_similarity=f_similarity),
        torch.tensor([c, e])
    )
