import collections
import typing as t

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
