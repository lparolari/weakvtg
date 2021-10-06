import tempfile

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
    "learning_rate": 0.001,
    "word_embedding": "glove",
    "text_embedding_size": 300,
    "text_semantic_size": 500,
    "text_semantic_num_layers": 1,
    "text_recurrent_network_type": "lstm",
    "image_embedding_size": 2053,
    "image_semantic_size": 500,
    "image_semantic_hidden_layers": 2,
    "concept_similarity_aggregation_strategy": "mean",
    "concept_similarity_activation_threshold": 0.,
    "apply_concept_similarity_strategy": "one",
    "concept_similarity_application_weight": 1.,
    "loss": "inversely_correlated",
    "n_box": 100,
    "n_epochs": 15,
    "device_name": "cpu",
    "save_folder": tempfile.gettempdir(),
    "suffix": "default",
    "restore": None,
    "use_spell_correction": False,
    "use_replace_phrase_with_noun_phrase": False,
}


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
