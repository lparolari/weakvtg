from functools import partial
from typing import Callable, Iterator, Any, List, Optional

import spacy

Doc = spacy.language.Doc
Span = Any  # spacy does not export Span type
ChunkingF = Optional[Callable[[Doc], Iterator[Span]]]


def _spacy_nlp():
    """
    Returns a `nlp` object with custom rules for "/" and "-" prefixes.

    Resources:
    - [Customizing spaCy’s Tokenizer class](https://spacy.io/usage/linguistic-features#native-tokenizers)
    - [Modifying existing rule sets](https://spacy.io/usage/linguistic-features#native-tokenizer-additions)
    """
    nlp = spacy.load('en_core_web_sm')

    prefixes = nlp.Defaults.prefixes + [r"""/""", r"""-"""]
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search

    return nlp


def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]


def get_nlp():
    return _spacy_nlp()


def get_torchtext_tokenizer_adapter(nlp):
    return partial(_spacy_tokenize, spacy=nlp)


def get_noun_phrases(doc: Doc, f_chunking: ChunkingF = None) -> List[str]:
    if f_chunking is None:
        f_chunking = noun_chunk_iter

    np = list(f_chunking(doc))
    np = list(map(get_text, np))

    return np


def noun_chunk_iter(doc: Doc) -> Iterator[Span]:
    return doc.noun_chunks


def root_chunk_iter(doc: Doc) -> Iterator[Span]:
    for chunk in noun_chunk_iter(doc):
        yield chunk.root


def get_text(doc: Doc) -> str:
    return doc.text
