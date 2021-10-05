import torch

from weakvtg.model import apply_concept_similarity_product


def test_apply_concept_similarity_product():
    logits = torch.tensor([1, 2, 3])
    concept_similarity = torch.tensor([4, 5, 6])

    assert torch.equal(
        apply_concept_similarity_product(logits, concept_similarity),
        logits * torch.abs(concept_similarity)
    )
