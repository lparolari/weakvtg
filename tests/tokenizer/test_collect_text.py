from weakvtg.tokenizer import collect_text, get_nlp

nlp = get_nlp()


def test_collect_text():
    assert collect_text(nlp("The pen is on the table"), phrase_iter) == ['The', 'pen', 'is', 'on', 'the', 'table']


def phrase_iter(doc):
    for tok in doc:
        yield tok
