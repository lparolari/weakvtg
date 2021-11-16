import functools
import tempfile
from typing import Any, Dict, Optional

__defaults = {
    "batch_size": 64,
    "num_workers": 0,
    "prefetch_factor": 2,
    "image_filepath": "data/refer/data/images",
    "data_filepath": "data/referit_raw/preprocessed",
    "train_idx_filepath": "data/referit_raw/train.txt",
    "valid_idx_filepath": "data/referit_raw/val.txt",
    "test_idx_filepath": "data/referit_raw/test.txt",
    "vocab_filepath": "data/referit_raw/vocab.json",
    "classes_vocab_filepath": "data/objects_vocab.txt",
    "attributes_vocab_filepath": "data/attributes_vocab.txt",
    "learning_rate": 0.001,
    "word_embedding": "glove",
    "text_embedding_size": 300,
    "text_semantic_size": 500,
    "text_semantic_num_layers": 1,
    "text_recurrent_network_type": "lstm",
    "image_embedding_size": 2053,
    "image_projection_size": 500,
    "image_projection_hidden_layers": 2,
    "image_projection_net": "mlp",
    "concept_similarity_aggregation_strategy": "mean",
    "concept_similarity_activation_threshold": 0.,
    "apply_concept_similarity_strategy": "one",
    "apply_concept_similarity_weight": 0.5,
    "loss": "inversely_correlated",
    "n_box": 100,
    "n_epochs": 15,
    "device_name": "cpu",
    "save_folder": tempfile.gettempdir(),
    "suffix": "default",
    "restore": None,
    "use_spell_correction": False,
    "use_replace_phrase_with_noun_phrase": False,
    "localization_strategy": "max",
    "attribute_similarity_aggregation_strategy": "max",
    "attribute_similarity_apply_strategy": "one",
    "attribute_similarity_apply_weight": 0.5,
    "attribute_similarity_direction_function": "binary_threshold",
    "attribute_similarity_direction_threshold": 0.,
}

__device = None


def get_config(config, defaults=None):
    """
    Returns current values or defaults
    """
    if defaults is None:
        defaults = __defaults

    # remove entries with None value to let defaults replace them
    config = {k: v for k, v in config.items() if v is not None}

    new_config = defaults.copy()
    new_config.update(config)

    return new_config


def make_options(name: str, options: Dict[str, Any]):
    def get_option(option: str, *, params: Optional[Dict[str, Any]] = None):
        if option not in options:
            raise ValueError(f"Provided option for {name} ({option}) is not supported. Please use one "
                             f"of {list(options.keys())}")

        if params is None:
            return options[option]

        return functools.partial(options[option], **params.get(option, {}))

    return get_option


def set_global_device(device):
    global __device
    __device = device


def get_global_device():
    return __device
