from weakvtg.tokenizer import get_nlp, get_adjectives, adj_iter

nlp = get_nlp()


def test_get_noun_phrases_given_noun_chunking():
    doc = nlp("A red apple")
    xs = get_adjectives(doc, f_adjective=adj_iter)

    assert len(xs) == 1
    assert xs[0] == "red"
