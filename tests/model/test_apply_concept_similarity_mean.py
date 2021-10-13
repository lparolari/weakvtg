import torch

from weakvtg.model import apply_concept_similarity_mean


def test_apply_concept_similarity_mean():
    logits = torch.tensor([1, 2, 3])
    concept_similarity = torch.tensor([4, 5, 6])

    assert torch.equal(
        apply_concept_similarity_mean(logits, concept_similarity),
        (logits + concept_similarity) / 2
    )


def test_apply_concept_similarity_mean_given_weight():
    logits = torch.tensor([1, 2, 3])
    concept_similarity = torch.tensor([4, 5, 6])

    assert torch.equal(
        apply_concept_similarity_mean(logits, concept_similarity, lam=0.2),
        0.2 * logits + (1 - 0.2) * concept_similarity
    )
