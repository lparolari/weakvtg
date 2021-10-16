import spacy

from tests.tokenizer.util import text
from weakvtg.tokenizer import get_nlp

nlp = get_nlp()


def test_spacy_load_english_model():
    spacy.load('en_core_web_sm')


def test_spacy_tokenizer():
    assert isinstance(nlp("lower stone wall"), object)


def test_spacy_special_cases():
    assert text(nlp("hills/cliffs")) == ["hills", "/", "cliffs"]
    assert text(nlp("sky w/no clouds")) == ["sky", "w", "/", "no", "clouds"]
    assert text(nlp("bottom-mid air")) == ["bottom", "-", "mid", "air"]

    assert text(nlp("/cliffs")) == ["/", "cliffs"]
    assert text(nlp("/no clouds")) == ["/", "no", "clouds"]
    assert text(nlp("-mid")) == ["-", "mid"]

    assert text(nlp("........anything")) == ["........", "anything"]
