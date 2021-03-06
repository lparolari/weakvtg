import argparse
import functools
import logging

import numpy as np
import spellchecker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchtext
import wandb

from weakvtg.config import get_config, make_options
from weakvtg.classes import get_classes, load_classes
from weakvtg.dataset import VtgDataset, collate_fn, process_example
from weakvtg.loss import WeakVtgLoss, loss_inversely_correlated, loss_inversely_correlated_box_class_count_scaled, \
    loss_orthogonal, get_predicted_box_by_max, get_predicted_box_by_union_on_max_class, \
    loss_orthogonal_box_class_count_scaled
from weakvtg.math import get_argmax, get_max
from weakvtg.model import WeakVtgModel, create_phrases_embedding_network, create_image_embedding_network, init_rnn, \
    get_phrases_representation, get_phrases_embedding
from weakvtg.model import apply_concept_similarity_one
from weakvtg.model import apply_concept_similarity_product
from weakvtg.model import apply_concept_similarity_mean
from weakvtg.concept import get_concept_similarity, aggregate_words_by_max, aggregate_words_by_mean, binary_threshold, \
    get_concept_similarity_direction, get_attribute_similarity_direction
from weakvtg.tokenizer import get_torchtext_tokenizer_adapter, get_nlp, get_noun_phrases, root_chunk_iter, adj_iter, \
    get_adjectives
from weakvtg.train import train, load_model, test_example, test, classes_frequency, concepts_frequency
from weakvtg.vocabulary import load_vocab_from_json, load_vocab_from_list, get_word_embedding


make_phrases_recurrent = make_options("RNN type", options={"lstm": nn.LSTM, "rnn": nn.RNN})
make_concept_similarity_f_aggregate = make_options("concept similarity aggregation strategy",
                                                   options={"max": aggregate_words_by_max,
                                                            "mean": aggregate_words_by_mean})
make_attribute_similarity_f_aggregate = make_options("attribute similarity aggregation strategy",
                                                     options={"max": aggregate_words_by_max,
                                                              "mean": aggregate_words_by_mean})
make_f_loss = make_options("loss", options={
        "inversely_correlated": loss_inversely_correlated,
        "inversely_correlated_box_class_count_scaled": loss_inversely_correlated_box_class_count_scaled,
        "orthogonal": loss_orthogonal,
        "orthogonal_box_class_count_scaled": loss_orthogonal_box_class_count_scaled,
    })
make_apply_concept_similarity = make_options("apply concept similarity strategy", options={
        "one": apply_concept_similarity_one,
        "product": apply_concept_similarity_product,
        "mean": apply_concept_similarity_mean
    })
make_apply_attribute_similarity = make_options("apply attribute similarity strategy", options={
        "one": apply_concept_similarity_one,
        "product": apply_concept_similarity_product,
        "mean": apply_concept_similarity_mean
    })
make_localization_strategy = make_options("localization strategy", options={
        "max": get_predicted_box_by_max,
        "union_max_class": get_predicted_box_by_union_on_max_class,
    })
make_image_projection_net = make_options("image projection net", options={
        "none": torch.nn.Identity,
        "mlp": create_image_embedding_network,
    })
make_attribute_similarity_direction_function = make_options("attribute similarity direction function", options={
    "binary_threshold": binary_threshold,
    "one": lambda x, *_, **__: torch.ones_like(x),
})


def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, test or plot some example with `weakvtg` model.")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--image-filepath", type=str, default=None)
    parser.add_argument("--data-filepath", type=str, default=None)
    parser.add_argument("--train-idx-filepath", type=str, default=None)
    parser.add_argument("--valid-idx-filepath", type=str, default=None)
    parser.add_argument("--test-idx-filepath", type=str, default=None)
    parser.add_argument("--vocab-filepath", type=str, default=None)
    parser.add_argument("--classes-vocab-filepath", type=str, default=None)
    parser.add_argument("--attributes-vocab-filepath", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--word-embedding", type=str, default=None)
    parser.add_argument("--text-embedding-size", type=int, default=None)
    parser.add_argument("--text-semantic-size", type=int, default=None)
    parser.add_argument("--text-semantic-num-layers", type=int, default=None)
    parser.add_argument("--text-recurrent-network-type", type=str, default=None)
    parser.add_argument("--image-embedding-size", type=int, default=None)
    parser.add_argument("--image-projection-net", type=str, default=None)
    parser.add_argument("--image-projection-size", type=int, default=None)
    parser.add_argument("--image-projection-hidden-layers", type=int, default=None)
    parser.add_argument("--concept-similarity-aggregation-strategy", type=str, default=None)
    parser.add_argument("--concept-similarity-activation-threshold", type=float, default=None)
    parser.add_argument("--apply-concept-similarity-strategy", type=str, default=None)
    parser.add_argument("--apply-concept-similarity-weight", type=float, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--n-box", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--device-name", type=str, default=None)
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--use-spell-correction", action="store_true", default=None)
    parser.add_argument("--use-replace-phrase-with-noun-phrase", action="store_true", default=None)
    parser.add_argument("--localization-strategy", type=str, default=None)
    parser.add_argument("--attribute-similarity-aggregation-strategy", type=str, default=None)
    parser.add_argument("--attribute-similarity-direction-function", type=str, default=None)
    parser.add_argument("--attribute-similarity-direction-threshold", type=float, default=None)
    parser.add_argument("--attribute-similarity-apply-strategy", type=str, default=None)
    parser.add_argument("--attribute-similarity-apply-weight", type=float, default=None)

    parser.add_argument("--workflow", type=str, choices=["train", "valid", "test", "test-example", "classes-frequency",
                                                         "concepts-frequency"],
                        default="train")

    parser.add_argument("--log-level", dest="log_level", type=int, default=logging.DEBUG, help="Log verbosity")
    parser.add_argument("--log-file", dest="log_file", type=str, default=None, help="Log filename")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true", default=False, help="Wandb log")

    return parser.parse_args()


def main():
    print("Hello, World!")

    np.random.seed(42)
    torch.manual_seed(42)

    args = parse_args()

    config = get_config({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "image_filepath": args.image_filepath,
        "data_filepath": args.data_filepath,
        "train_idx_filepath": args.train_idx_filepath,
        "valid_idx_filepath": args.valid_idx_filepath,
        "test_idx_filepath": args.test_idx_filepath,
        "vocab_filepath": args.vocab_filepath,
        "classes_vocab_filepath": args.classes_vocab_filepath,
        "attributes_vocab_filepath": args.attributes_vocab_filepath,
        "learning_rate": args.learning_rate,
        "word_embedding": args.word_embedding,
        "text_embedding_size": args.text_embedding_size,
        "text_semantic_size": args.text_semantic_size,
        "text_semantic_num_layers": args.text_semantic_num_layers,
        "text_recurrent_network_type": args.text_recurrent_network_type,
        "image_embedding_size": args.image_embedding_size,
        "image_projection_net": args.image_projection_net,
        "image_projection_size": args.image_projection_size,
        "image_projection_hidden_layers": args.image_projection_hidden_layers,
        "concept_similarity_aggregation_strategy": args.concept_similarity_aggregation_strategy,
        "concept_similarity_activation_threshold": args.concept_similarity_activation_threshold,
        "apply_concept_similarity_strategy": args.apply_concept_similarity_strategy,
        "apply_concept_similarity_weight": args.apply_concept_similarity_weight,
        "loss": args.loss,
        "n_box": args.n_box,
        "n_epochs": args.n_epochs,
        "device_name": args.device_name,
        "save_folder": args.save_folder,
        "suffix": args.suffix,
        "restore": args.restore,
        "use_spell_correction": args.use_spell_correction,
        "use_replace_phrase_with_noun_phrase": args.use_replace_phrase_with_noun_phrase,
        "localization_strategy": args.localization_strategy,
        "attribute_similarity_aggregation_strategy": args.attribute_similarity_aggregation_strategy,
        "attribute_similarity_direction_function": args.attribute_similarity_direction_function,
        "attribute_similarity_direction_threshold": args.attribute_similarity_direction_threshold,
        "attribute_similarity_apply_strategy": args.attribute_similarity_apply_strategy,
        "attribute_similarity_apply_weight": args.attribute_similarity_apply_weight,
    })

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    prefetch_factor = config["prefetch_factor"]
    image_filepath = config["image_filepath"]
    data_filepath = config["data_filepath"]
    train_idx_filepath = config["train_idx_filepath"]
    valid_idx_filepath = config["valid_idx_filepath"]
    test_idx_filepath = config["test_idx_filepath"]
    vocab_filepath = config["vocab_filepath"]
    classes_vocab_filepath = config["classes_vocab_filepath"]
    attributes_vocab_filepath = config["attributes_vocab_filepath"]
    learning_rate = config["learning_rate"]
    word_embedding = config["word_embedding"]
    text_embedding_size = config["text_embedding_size"]
    text_semantic_size = config["text_semantic_size"]
    text_semantic_num_layers = config["text_semantic_num_layers"]
    text_recurrent_network_type = config["text_recurrent_network_type"]
    image_embedding_size = config["image_embedding_size"]
    image_projection_net = config["image_projection_net"]
    image_projection_size = config["image_projection_size"]
    image_projection_hidden_layers = config["image_projection_hidden_layers"]
    concept_similarity_aggregation_strategy = config["concept_similarity_aggregation_strategy"]
    concept_similarity_activation_threshold = config["concept_similarity_activation_threshold"]
    apply_concept_similarity_strategy = config["apply_concept_similarity_strategy"]
    apply_concept_similarity_weight = config["apply_concept_similarity_weight"]
    loss = config["loss"]
    n_box = config["n_box"]
    n_epochs = config["n_epochs"]
    device_name = config["device_name"]
    save_folder = config["save_folder"]
    suffix = config["suffix"]
    restore = config["restore"]
    use_spell_correction = config["use_spell_correction"]
    use_replace_phrase_with_noun_phrase = config["use_replace_phrase_with_noun_phrase"]
    localization_strategy = config["localization_strategy"]
    attribute_similarity_aggregation_strategy = config["attribute_similarity_aggregation_strategy"]
    attribute_similarity_direction_function = config["attribute_similarity_direction_function"]
    attribute_similarity_direction_threshold = config["attribute_similarity_direction_threshold"]
    attribute_similarity_apply_strategy = config["attribute_similarity_apply_strategy"]
    attribute_similarity_apply_weight = config["attribute_similarity_apply_weight"]

    device = torch.device(device_name)

    wandb.init(project='weakvtg', entity='vtkel-solver', mode="online" if args.use_wandb else "disabled")
    wandb.config.update(config)

    logging.basicConfig(filename=args.log_file, level=args.log_level)

    logging.info(f"Model started with following parameters: {config}")

    # create core tools
    nlp = get_nlp()
    tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=get_torchtext_tokenizer_adapter(nlp))
    f_spell_correction = spellchecker.SpellChecker().correction if use_spell_correction else None

    concept_f_aggregate = make_concept_similarity_f_aggregate(concept_similarity_aggregation_strategy)
    attribute_f_aggregate = make_attribute_similarity_f_aggregate(attribute_similarity_aggregation_strategy)

    vocab = load_vocab_from_json(vocab_filepath)
    classes_vocab = load_vocab_from_list(load_classes(classes_vocab_filepath))
    attributes_vocab = load_vocab_from_list(load_classes(attributes_vocab_filepath))

    word_embedding = get_word_embedding(word_embedding, text_embedding_size)

    phrases_embedding_net = create_phrases_embedding_network(vocab, word_embedding, embedding_size=text_embedding_size,
                                                             f_spell_correction=f_spell_correction, freeze=True)
    classes_embedding_net = create_phrases_embedding_network(classes_vocab, word_embedding,
                                                             embedding_size=text_embedding_size, freeze=True)
    attributes_embedding_net = create_phrases_embedding_network(attributes_vocab, word_embedding,
                                                                embedding_size=text_embedding_size, freeze=True)

    phrases_recurrent_layer = make_phrases_recurrent(text_recurrent_network_type)
    phrases_recurrent_net = phrases_recurrent_layer(text_embedding_size, text_semantic_size,
                                                    num_layers=text_semantic_num_layers, bidirectional=False,
                                                    batch_first=False)
    phrases_recurrent_net = init_rnn(phrases_recurrent_net)

    f_image_projection_net = make_image_projection_net(image_projection_net)
    image_embedding_net = f_image_projection_net(image_embedding_size, image_projection_size,
                                                 n_hidden_layer=image_projection_hidden_layers)

    _get_classes_embedding = functools.partial(get_phrases_embedding, embedding_network=classes_embedding_net)
    _get_attributes_embedding = functools.partial(get_phrases_embedding, embedding_network=attributes_embedding_net)
    _get_phrases_embedding = functools.partial(get_phrases_embedding, embedding_network=phrases_embedding_net)
    _get_phrases_representation = functools.partial(get_phrases_representation,
                                                    recurrent_network=phrases_recurrent_net,
                                                    out_features=text_semantic_size,
                                                    device=device)
    _get_concept_similarity = functools.partial(get_concept_similarity, f_aggregate=concept_f_aggregate,
                                                f_similarity=torch.cosine_similarity, f_activation=torch.nn.Identity())
    _get_attribute_similarity = functools.partial(get_concept_similarity, f_aggregate=attribute_f_aggregate,
                                                  f_similarity=torch.cosine_similarity,
                                                  f_activation=torch.nn.Identity())
    _concept_similarity_direction_f_activation = functools.partial(binary_threshold,
                                                                   threshold=concept_similarity_activation_threshold)
    _get_concept_similarity_direction = functools.partial(get_concept_similarity_direction,
                                                          f_activation=_concept_similarity_direction_f_activation)
    _attribute_similarity_direction_f_activation = make_attribute_similarity_direction_function(
        attribute_similarity_direction_function,
        params={"binary_threshold": {"threshold": attribute_similarity_direction_threshold}})
    _get_attribute_similarity_direction = functools.partial(get_attribute_similarity_direction,
                                                            f_activation=_attribute_similarity_direction_f_activation)
    _get_predicted_box = make_localization_strategy(localization_strategy)
    _apply_concept_similarity_params = {"mean": {"lam": apply_concept_similarity_weight}}
    _apply_concept_similarity = make_apply_concept_similarity(apply_concept_similarity_strategy,
                                                              params=_apply_concept_similarity_params)
    _apply_attribute_similarity_params = {"mean": {"lam": attribute_similarity_apply_weight}}
    _apply_attribute_similarity = make_apply_attribute_similarity(attribute_similarity_apply_strategy,
                                                                  params=_apply_attribute_similarity_params)

    # create dataset adapter
    f_get_noun_phrase = functools.partial(get_noun_phrases, f_chunking=root_chunk_iter)
    f_get_adjective = functools.partial(get_adjectives, f_adjective=adj_iter)
    process_fn = functools.partial(process_example, n_boxes_to_keep=n_box, f_extract_noun_phrase=f_get_noun_phrase,
                                   f_extract_adjective=f_get_adjective, f_nlp=nlp,
                                   use_replace_phrase_with_noun_phrase=use_replace_phrase_with_noun_phrase)

    train_dataset = VtgDataset(image_filepath, data_filepath, idx_filepath=train_idx_filepath, process_fn=process_fn)
    valid_dataset = VtgDataset(image_filepath, data_filepath, idx_filepath=valid_idx_filepath, process_fn=process_fn)
    test_dataset = VtgDataset(image_filepath, data_filepath, idx_filepath=test_idx_filepath, process_fn=process_fn)

    # setup dataloader
    collate_function = functools.partial(collate_fn, tokenizer=tokenizer, vocab=vocab)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_function,
                                               num_workers=num_workers, prefetch_factor=prefetch_factor)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_function,
                                               num_workers=num_workers, prefetch_factor=prefetch_factor)

    # create model, optimizer and criterion
    model = WeakVtgModel(
        phrases_embedding_net=phrases_embedding_net,
        phrases_recurrent_net=phrases_recurrent_net,
        image_embedding_net=image_embedding_net,
        get_classes_embedding=_get_classes_embedding,
        get_attributes_embedding=_get_attributes_embedding,
        get_phrases_embedding=_get_phrases_embedding,
        get_phrases_representation=_get_phrases_representation,
        get_concept_similarity=_get_concept_similarity,
        get_attribute_similarity=_get_attribute_similarity,
        f_similarity=F.cosine_similarity,
        apply_concept_similarity=_apply_concept_similarity,
        apply_attribute_similarity=_apply_attribute_similarity
    )
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = WeakVtgLoss(
        get_concept_similarity_direction=_get_concept_similarity_direction,
        get_attribute_similarity_direction=_get_attribute_similarity_direction,
        get_predicted_box=_get_predicted_box,
        f_loss=make_f_loss(loss)
    )

    # restore model, if needed
    start_epoch = 0
    if restore is not None:
        start_epoch = load_model(restore, model, optimizer, device=device)

    # start the game

    def do_train():
        _, valid_history = train(train_loader, valid_loader, model, optimizer, criterion,
                                 start_epoch=start_epoch, n_epochs=n_epochs, save_folder=save_folder, suffix=suffix)

        # log data
        valid_loss = valid_history["loss"]
        valid_accuracy = valid_history["accuracy"]

        logging.info(f"Best hist validation loss at epoch {get_argmax(valid_loss)}: {get_max(valid_loss)}")
        logging.info(f"Best hist validation accuracy at epoch {get_argmax(valid_accuracy)}: {get_max(valid_accuracy)}")

    def do_test():
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_function,
                                             num_workers=num_workers, prefetch_factor=prefetch_factor)

        test(loader, model, optimizer, criterion)

    def do_test_example():
        dataset = valid_dataset
        loader = torchdata.DataLoader(dataset, batch_size=1, collate_fn=collate_function, num_workers=num_workers,
                                      prefetch_factor=prefetch_factor)
        classes = get_classes("data/objects_vocab.txt")
        test_example(dataset, loader, model, optimizer, criterion, vocab=vocab, classes=classes)

    def do_classes_frequency():
        dataset = test_dataset
        loader = torchdata.DataLoader(dataset, batch_size=1, collate_fn=collate_function, num_workers=num_workers,
                                      prefetch_factor=prefetch_factor)
        classes = get_classes("data/objects_vocab.txt")
        classes_frequency(loader, model, optimizer, classes)

    def do_concepts_frequency():
        dataset = test_dataset
        loader = torchdata.DataLoader(dataset, batch_size=1, collate_fn=collate_function, num_workers=num_workers,
                                      prefetch_factor=prefetch_factor)
        concepts_frequency(loader, vocab, classes_vocab, _get_classes_embedding, _get_phrases_embedding,
                           f_similarity=torch.cosine_similarity)

    if args.workflow == "train":
        do_train()
    if args.workflow == "test":
        do_test()
    if args.workflow == "test-example":
        do_test_example()
    if args.workflow == "classes-frequency":
        do_classes_frequency()
    if args.workflow == "concepts-frequency":
        do_concepts_frequency()

    print("Goodbye, World!")


if __name__ == "__main__":
    main()
