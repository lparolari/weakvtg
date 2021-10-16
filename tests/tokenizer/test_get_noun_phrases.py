from weakvtg.tokenizer import get_nlp, get_noun_phrases, noun_chunk_iter, root_chunk_iter

nlp = get_nlp()


def test_get_noun_phrases_given_noun_chunking():
    doc = nlp("The cat and the dog sleep in the basket near the door with a person.")
    np = get_noun_phrases(doc, f_chunking=noun_chunk_iter)

    assert len(np) == 5
    assert np[0] == "The cat"
    assert np[1] == "the dog sleep"
    assert np[2] == "the basket"
    assert np[3] == "the door"
    assert np[4] == "a person"


def test_get_noun_phrases_given_root_chunking():
    doc = nlp("The cat and the dog sleep in the basket near the door with a person.")
    np = get_noun_phrases(doc, f_chunking=root_chunk_iter)

    assert len(np) == 5
    assert np[0] == "cat"
    assert np[1] == "sleep"
    assert np[2] == "basket"
    assert np[3] == "door"
    assert np[4] == "person"
