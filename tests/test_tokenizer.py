import pytest
import spacy

from weakvtg.tokenizer import get_nlp


def text(doc):
    return [t.text for t in doc]


@pytest.fixture
def nlp():
    return get_nlp()


def test_spacy_load_english_model():
    spacy.load('en_core_web_sm')


def test_spacy_tokenizer(nlp):
    assert isinstance(nlp("lower stone wall"), object)


def test_spacy_special_cases(nlp):
    assert text(nlp("hills/cliffs")) == ["hills", "/", "cliffs"]
    assert text(nlp("sky w/no clouds")) == ["sky", "w", "/", "no", "clouds"]
    assert text(nlp("bottom-mid air")) == ["bottom", "-", "mid", "air"]

    assert text(nlp("/cliffs")) == ["/", "cliffs"]
    assert text(nlp("/no clouds")) == ["/", "no", "clouds"]
    assert text(nlp("-mid")) == ["-", "mid"]

    assert text(nlp("........anything")) == ["........", "anything"]
