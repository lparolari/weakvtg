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
    "learning_rate": 0.001,
    "text_embedding_size": 300,
    "text_semantic_size": 500,
    "text_semantic_num_layers": 1,
    "text_recurrent_network_type": "lstm",
    "image_embedding_size": 2053,
    "image_semantic_size": 500,
    "image_semantic_hidden_layers": 2,
    "n_box": 100,
    "n_epochs": 15,
    "device_name": "cpu",
    "save_folder": tempfile.gettempdir(),
    "suffix": "default",
    "restore": None,
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
