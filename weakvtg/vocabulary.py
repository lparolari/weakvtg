import collections
import typing as t

import torch
from torchtext import vocab

from weakvtg import iox

T = t.TypeVar('T')


def load_vocab_from_json(filename: str):
    vocab_dict = iox.load_json(filename)
    vocab_counter = collections.Counter(vocab_dict)
    return get_vocab(vocab_counter)


def load_vocab_from_list(xs: t.List[T]):
    vocab_counter = collections.Counter(xs)
    return get_vocab(vocab_counter)


def get_vocab(counter: t.Counter):
    vocabulary = vocab.vocab(counter)

    specials = ["<unk>"]

    for i, special in enumerate(specials):
        if special not in vocabulary:
            vocabulary.insert_token(special, i)

    # make default index same as index of unknown token
    vocabulary.set_default_index(vocabulary["<unk>"])

    return vocabulary


def get_word_embedding(kind: str, dim: int = 300, **kwargs):
    """
    Return a `torchtext.vocab.Vectors` object, instanced with word embeddings wrt `kind`
    (i.e., fixed, pretrained models).

    :param kind: One of 'glove', 'w2v'
    :param dim: The embedding size
    :param kwargs: Other `torchtext.vocab.Vectors.__init__` parameters
    :return: A `torchtext.vocab.Vectors` instance
    """
    if kind == "glove":
        return vocab.GloVe(name='840B', dim=dim, **kwargs)
    if kind == "w2v":
        if dim != 300:
            raise ValueError(f"The specified embedding size ({dim}) is not valid for Word2Vec. Please use `dim=300`.")
        return Word2Vec(name="word2vec-google-news-300", **kwargs)
    raise ValueError(f"Invalid embedding kind ({kind}), should be in 'glove', 'w2v'.")


class Word2Vec(vocab.Vectors):
    """
    Word2Vec adapter for `torchtext.vocab.Vectors`.
    """

    def cache(self, name, cache, url=None, max_vectors=None):
        import gensim.downloader as api

        model = api.load(name)

        self.itos = model.index_to_key
        self.stoi = model.key_to_index
        self.vectors = torch.tensor(model.vectors)
        self.dim = model.vector_size
