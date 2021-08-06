import argparse
import functools
import logging

import torch
import torchtext
import wandb

from weakvtg.config import parse_configs
from weakvtg.dataset import VtgDataset, collate_fn
from weakvtg.math import get_argmax, get_max
from weakvtg.tokenizer import get_torchtext_tokenizer_adapter, get_nlp
from weakvtg.train import train
from weakvtg.vocabulary import load_vocab


class MockTensor:
    def __init__(self, value):
        self.value = value

    def __format__(self, format_spec):
        return f"{self.value:{format_spec}}"

    def __str__(self):
        return str(self.value)

    def item(self):
        return self.value

    def backward(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, test or plot some example with `weakvtg` model.")

    parser.add_argument("--configs", dest="configs", type=str, default=None,
                        help="Model parameters as a JSON dictionary.")
    parser.add_argument("--log-level", dest="log_level", type=int, default=logging.DEBUG, help="Log verbosity")
    parser.add_argument("--log-file", dest="log_file", type=str, default=None, help="Log filename")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true", default=False, help="Wandb log")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs = parse_configs(args.configs)

    wandb.init(project='weakvtg', entity='vtkel-solver', mode="online" if args.use_wandb else "disabled")
    wandb.config.update(configs)

    logging.basicConfig(filename=args.log_file, level=args.log_level)

    logging.info(f"Model started with following parameters: {configs}")

    dataset = VtgDataset("data/referit_raw/preprocessed", "data/referit_raw/train.txt")

    tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=get_torchtext_tokenizer_adapter(get_nlp()))
    vocab = load_vocab("data/referit_raw/vocab.json")

    collate_function = functools.partial(collate_fn, tokenizer=tokenizer, vocab=vocab)

    # TODO: replace mock
    def c(_1, _2):
        return MockTensor(1.), MockTensor(2.), MockTensor(3.), MockTensor(4.)
    from unittest import mock
    train_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_function)
    valid_loader = [{"id": range(10)}, {"id": range(10)}, {"id": range(10)}, {"id": range(10)}]
    model = mock.Mock()
    optimizer = mock.Mock()
    criterion = c

    _, valid_history = train(train_loader, valid_loader, model, optimizer, criterion)

    valid_loss = valid_history["loss"]
    valid_accuracy = valid_history["accuracy"]

    logging.info(f"Best hist validation loss at epoch {get_argmax(valid_loss)}: {get_max(valid_loss)}")
    logging.info(f"Best hist validation accuracy at epoch {get_argmax(valid_accuracy)}: {get_max(valid_accuracy)}")
