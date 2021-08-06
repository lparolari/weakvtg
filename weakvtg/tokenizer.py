from functools import partial


def _spacy_nlp():
    """
    Returns a `nlp` object with custom rules for "/" and "-" prefixes.

    Resources:
    - [Customizing spaCy’s Tokenizer class](https://spacy.io/usage/linguistic-features#native-tokenizers)
    - [Modifying existing rule sets](https://spacy.io/usage/linguistic-features#native-tokenizer-additions)
    """
    import spacy

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
