import argparse
import logging

from tests.mock_tensor import MockTensor
from weakvtg.config import parse_configs
from weakvtg.math import get_argmax, get_max
from weakvtg.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, test or plot some example with `weakvtg` model.")

    parser.add_argument("--configs", dest="configs", type=str, default=None,
                        help="Model parameters as a JSON dictionary.")
    parser.add_argument("--log-level", dest="log_level", type=int, default=logging.DEBUG, help="Log verbosity")
    parser.add_argument("--log-file", dest="log_file", type=str, default=None, help="Log filename")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs = parse_configs(args.configs)

    logging.basicConfig(filename=args.log_file, level=args.log_level)

    logging.info(f"Model started with following parameters: {configs}")

    # TODO: replace mock
    def c(_1, _2):
        return MockTensor(1.), MockTensor(2.), MockTensor(3.), MockTensor(4.)
    from unittest import mock
    train_loader = [{"id": range(10)}, {"id": range(10)}, {"id": range(10)}, {"id": range(10)}]
    valid_loader = [{"id": range(10)}, {"id": range(10)}, {"id": range(10)}, {"id": range(10)}]
    model = mock.Mock()
    optimizer = mock.Mock()
    criterion = c

    _, valid_history = train(train_loader, valid_loader, model, optimizer, criterion)

    valid_loss = valid_history["loss"]
    valid_accuracy = valid_history["accuracy"]

    logging.info(f"Best hist validation loss at epoch {get_argmax(valid_loss)}: {get_max(valid_loss)}")
    logging.info(f"Best hist validation accuracy at epoch {get_argmax(valid_accuracy)}: {get_max(valid_accuracy)}")

