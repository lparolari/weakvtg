import pytest
import spacy

from weakvtg.tokenizer import get_nlp, get_torchtext_tokenizer_adapter, get_noun_phrases, get_text, noun_chunk_iter, \
    root_chunk_iter


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


def test_torchtext_adapter(nlp):
    adapted_nlp = get_torchtext_tokenizer_adapter(nlp)
    assert adapted_nlp("hills/cliffs") == text(nlp("hills/cliffs"))


def test_noun_chunk_iter(nlp):
    doc = nlp("A phrase with another phrase occurs..")
    it = noun_chunk_iter(doc)

    assert next(it).text == "A phrase"
    assert next(it).text == "another phrase"
    with pytest.raises(StopIteration):
        next(it)


def test_get_noun_phrases_given_noun_chunking(nlp):
    doc = nlp("The cat and the dog sleep in the basket near the door with a person.")
    np = get_noun_phrases(doc, f_chunking=noun_chunk_iter)

    assert len(np) == 5
    assert np[0] == "The cat"
    assert np[1] == "the dog sleep"
    assert np[2] == "the basket"
    assert np[3] == "the door"
    assert np[4] == "a person"


def test_get_noun_phrases_given_root_chunking(nlp):
    doc = nlp("The cat and the dog sleep in the basket near the door with a person.")
    np = get_noun_phrases(doc, f_chunking=root_chunk_iter)

    assert len(np) == 5
    assert np[0] == "cat"
    assert np[1] == "sleep"
    assert np[2] == "basket"
    assert np[3] == "door"
    assert np[4] == "person"


def test_get_text(nlp):
    doc = nlp("The cat")
    assert get_text(doc) == doc.text
