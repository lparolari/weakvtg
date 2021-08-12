import argparse
import functools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import wandb

from weakvtg.config import get_config
from weakvtg.dataset import VtgDataset, collate_fn
from weakvtg.loss import WeakVtgLoss
from weakvtg.math import get_argmax, get_max
from weakvtg.model import WeakVtgModel, create_phrases_embedding_network, create_phrases_recurrent_network, \
    create_image_embedding_network, init_rnn
from weakvtg.tokenizer import get_torchtext_tokenizer_adapter, get_nlp
from weakvtg.train import train
from weakvtg.vocabulary import load_vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, test or plot some example with `weakvtg` model.")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--data-filepath", type=str, default=None)
    parser.add_argument("--train-idx-filepath", type=str, default=None)
    parser.add_argument("--valid-idx-filepath", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

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
        "data_filepath": args.data_filepath,
        "train_idx_filepath": args.train_idx_filepath,
        "valid_idx_filepath": args.valid_idx_filepath,
        "learning_rate": args.learning_rate,
    })

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    prefetch_factor = config["prefetch_factor"]
    data_filepath = config["data_filepath"]
    train_idx_filepath = config["train_idx_filepath"]
    valid_idx_filepath = config["valid_idx_filepath"]
    learning_rate = config["learning_rate"]

    device = None

    wandb.init(project='weakvtg', entity='vtkel-solver', mode="online" if args.use_wandb else "disabled")
    wandb.config.update(config)

    logging.basicConfig(filename=args.log_file, level=args.log_level)

    logging.info(f"Model started with following parameters: {config}")

    # create dataset adapter
    train_dataset = VtgDataset(data_filepath=data_filepath, idx_filepath=train_idx_filepath)
    valid_dataset = VtgDataset(data_filepath=data_filepath, idx_filepath=valid_idx_filepath)

    # create core tools
    # * tokenizer
    # * vocab
    # * phrases embedding net
    # * phrases recurrent net

    tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=get_torchtext_tokenizer_adapter(get_nlp()))

    vocab = load_vocab("data/referit_raw/vocab.json")

    phrases_embedding_net = create_phrases_embedding_network(vocab, embedding_size=300, freeze=True)

    lstm = init_rnn(nn.LSTM(300, 500, num_layers=1, bidirectional=False, batch_first=False))
    phrases_recurrent_net = functools.partial(create_phrases_recurrent_network,
                                              features_size=500, recurrent_network=lstm, device=device)

    image_embedding_network = create_image_embedding_network(2053, 500)

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
        image_embedding_network=image_embedding_network,
        f_similarity=F.cosine_similarity,
    )
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = WeakVtgLoss(torch.device("cpu"))

    # start the training
    _, valid_history = train(train_loader, valid_loader, model, optimizer, criterion)

    # log data
    valid_loss = valid_history["loss"]
    valid_accuracy = valid_history["accuracy"]

    logging.info(f"Best hist validation loss at epoch {get_argmax(valid_loss)}: {get_max(valid_loss)}")
    logging.info(f"Best hist validation accuracy at epoch {get_argmax(valid_accuracy)}: {get_max(valid_accuracy)}")

    print("Goodbye, World!")
