from weakvtg.tokenizer import get_text, get_nlp

nlp = get_nlp()


def test_get_text():
    doc = nlp("The cat")
    assert get_text(doc) == doc.text
