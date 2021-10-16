from tests.tokenizer.util import text
from weakvtg.tokenizer import get_nlp, get_torchtext_tokenizer_adapter

nlp = get_nlp()


def test_torchtext_adapter():
    adapted_nlp = get_torchtext_tokenizer_adapter(nlp)
    assert adapted_nlp("hills/cliffs") == text(nlp("hills/cliffs"))
