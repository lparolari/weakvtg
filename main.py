import argparse
import functools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchtext
import wandb

from weakvtg.config import get_config
from weakvtg.dataset import VtgDataset, collate_fn
from weakvtg.loss import WeakVtgLoss
from weakvtg.math import get_argmax, get_max
from weakvtg.model import WeakVtgModel, create_phrases_embedding_network, create_image_embedding_network, init_rnn, \
    get_phrases_representation, get_phrases_embedding
from weakvtg.tokenizer import get_torchtext_tokenizer_adapter, get_nlp
from weakvtg.train import train, load_model, test_example
from weakvtg.vocabulary import load_vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, test or plot some example with `weakvtg` model.")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--image-filepath", type=str, default=None)
    parser.add_argument("--data-filepath", type=str, default=None)
    parser.add_argument("--train-idx-filepath", type=str, default=None)
    parser.add_argument("--valid-idx-filepath", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device-name", type=str, default=None)
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--restore", type=str, default=None)

    parser.add_argument("--workflow", type=str, choices=["train", "valid", "test", "test-example"], default="train")

    parser.add_argument("--log-level", dest="log_level", type=int, default=logging.DEBUG, help="Log verbosity")
    parser.add_argument("--log-file", dest="log_file", type=str, default=None, help="Log filename")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true", default=False, help="Wandb log")

    return parser.parse_args()


if __name__ == "__main__":
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
        "learning_rate": args.learning_rate,
        "device_name": args.device_name,
        "save_folder": args.save_folder,
        "suffix": args.suffix,
        "restore": args.restore,
    })

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    prefetch_factor = config["prefetch_factor"]
    image_filepath = config["image_filepath"]
    data_filepath = config["data_filepath"]
    train_idx_filepath = config["train_idx_filepath"]
    valid_idx_filepath = config["valid_idx_filepath"]
    learning_rate = config["learning_rate"]
    device_name = config["device_name"]
    save_folder = config["save_folder"]
    suffix = config["suffix"]
    restore = config["restore"]

    device = torch.device(device_name)

    wandb.init(project='weakvtg', entity='vtkel-solver', mode="online" if args.use_wandb else "disabled")
    wandb.config.update(config)

    logging.basicConfig(filename=args.log_file, level=args.log_level)

    logging.info(f"Model started with following parameters: {config}")

    # create dataset adapter
    train_dataset = VtgDataset(image_filepath, data_filepath, idx_filepath=train_idx_filepath)
    valid_dataset = VtgDataset(image_filepath, data_filepath, idx_filepath=valid_idx_filepath)

    # create core tools
    # * tokenizer
    # * vocab
    # * phrases embedding net
    # * phrases recurrent net

    tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=get_torchtext_tokenizer_adapter(get_nlp()))

    vocab = load_vocab("data/referit_raw/vocab.json")

    phrases_embedding_net = create_phrases_embedding_network(vocab, embedding_size=300, freeze=True)

    phrases_recurrent_net = init_rnn(nn.LSTM(300, 500, num_layers=1, bidirectional=False, batch_first=False))

    image_embedding_net = create_image_embedding_network(2053, 500)

    _get_phrases_embedding = functools.partial(get_phrases_embedding, embedding_network=phrases_embedding_net)
    _get_phrases_representation = functools.partial(get_phrases_representation,
                                                    recurrent_network=phrases_recurrent_net,
                                                    out_features=500,
                                                    device=device)

    # setup dataloader
    collate_function = functools.partial(collate_fn, tokenizer=tokenizer, vocab=vocab)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_function,
                                               num_workers=num_workers, prefetch_factor=prefetch_factor)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_function,
                                               num_workers=num_workers, prefetch_factor=prefetch_factor)

    # create core tools for training
    model = WeakVtgModel(
        phrases_embedding_net=phrases_embedding_net,
        phrases_recurrent_net=phrases_recurrent_net,
        image_embedding_net=image_embedding_net,
        get_phrases_embedding=_get_phrases_embedding,
        get_phrases_representation=_get_phrases_representation,
        f_similarity=F.cosine_similarity
    )
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = WeakVtgLoss(device=device)

    start_epoch = 0
    if restore is not None:
        start_epoch = load_model(restore, model, optimizer, device=device)

    def do_train():
        _, valid_history = train(train_loader, valid_loader, model, optimizer, criterion,
                                 start_epoch=start_epoch, save_folder=save_folder, suffix=suffix)

        # log data
        valid_loss = valid_history["loss"]
        valid_accuracy = valid_history["accuracy"]

        logging.info(f"Best hist validation loss at epoch {get_argmax(valid_loss)}: {get_max(valid_loss)}")
        logging.info(f"Best hist validation accuracy at epoch {get_argmax(valid_accuracy)}: {get_max(valid_accuracy)}")

    def do_test_example():
        dataset = valid_dataset
        loader = torchdata.DataLoader(dataset, batch_size=1, collate_fn=collate_function, num_workers=num_workers,
                                      prefetch_factor=prefetch_factor)
        test_example(dataset, loader, model, optimizer, criterion, vocab=vocab)


    if args.workflow == "train":
        do_train()

    if args.workflow == "test-example":
        do_test_example()

    print("Goodbye, World!")
