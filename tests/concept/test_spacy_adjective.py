import spacy

nlp = spacy.load("en_core_web_sm")


def get_adjective(doc):
    return list(filter(lambda x: x.pos_ == "ADJ", doc))


def test_spacy_adjective():
    assert len(get_adjective(nlp("A red apple"))) == 1
    assert len(get_adjective(nlp("A red person on right"))) == 2
    assert len(get_adjective(nlp("The person on right"))) == 1
    assert len(get_adjective(nlp("A little boy is jumping in front of a fountain"))) == 1
    assert len(get_adjective(nlp("Grass on left of people"))) == 0
    assert len(get_adjective(nlp("Big rock top right"))) == 1
