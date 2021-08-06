import logging
import os
import sys
import time

from tqdm import tqdm

from weakvtg.prettyprint import pp
from weakvtg.timeit import get_fancy_eta, get_fancy_time, get_hms, get_delta
from weakvtg.utils import get_batch_size, percent, pivot


def epoch(loader, model, optimizer, criterion, train=True):
    fancy_mode = "Train" if train else "Valid"
    n_batch = len(loader)

    total_examples = 0
    total_loss = 0.
    total_accuracy = 0.
    total_p_accuracy = 0.

    start_time = time.time()

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        output = model(batch)

        loss, iou, accuracy, p_accuracy = criterion(batch, output)

        if train:
            loss.backward()
            optimizer.step()

        total_examples += get_batch_size(batch)

        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_p_accuracy += p_accuracy.item()

        # computing some statistics
        fancy_batch_no = i + 1
        end_time = time.time()
        eta = get_fancy_eta(start_time, end_time, current=fancy_batch_no, total=n_batch)
        start_time = end_time
        logging.debug(f"{fancy_mode} {fancy_batch_no}/{n_batch}, loss: {pp(loss)}, accuracy: {pp(accuracy)}, "
                      f"p_accuracy: {pp(p_accuracy)} | ETA: {eta}")

    loss = total_loss / total_examples
    accuracy = percent(total_accuracy / total_examples)
    p_accuracy = percent(total_p_accuracy / total_examples)

    return {"loss": loss, "accuracy": accuracy, "p_accuracy": p_accuracy}


def train(train_loader, valid_loader, model, optimizer, criterion, n_epochs=15):
    train_results = []
    valid_results = []

    start_time = time.time()

    for i in range(n_epochs):
        logging.info(f"Start epoch {i + 1}")

        train_out = epoch(train_loader, model, optimizer, criterion, train=True)
        logging.info(f"Training completed. {pp(train_out)}")

        valid_out = epoch(valid_loader, model, optimizer, criterion, train=False)
        logging.info(f"Validation completed. {pp(valid_out)}")

        train_results += [train_out]
        valid_results += [valid_out]

        # computing some statistics
        fancy_epoch_no = i + 1
        end_time = time.time()
        running_time = get_fancy_time(*get_hms(get_delta(start_time, end_time)))
        start_time = end_time
        logging.info(f"Complete epoch {fancy_epoch_no} in {running_time}")

    train_results = pivot(train_results)
    valid_results = pivot(valid_results)

    logging.info(f"Training completed.")

    return train_results, valid_results
