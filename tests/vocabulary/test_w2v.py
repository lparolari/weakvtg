from weakvtg.vocabulary import Word2Vec


def test_w2v_exists():
    w2v = Word2Vec("word2vec-google-news-300")
    assert w2v.stoi["dog"] == 2043
    assert len(w2v.itos) == 3000000
