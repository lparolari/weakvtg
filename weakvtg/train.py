import logging

from tqdm import tqdm

from weakvtg.utils import get_batch_size, percent


def epoch(loader, model, optimizer, criterion, train=True):
    total_examples = 0
    total_loss = 0.
    total_accuracy = 0.
    total_p_accuracy = 0.

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

        logging.debug(f"loss: {loss:.6}, accuracy: {accuracy:.4}, p_accuracy: {p_accuracy:.4}")

    loss = total_loss / total_examples
    accuracy = percent(total_accuracy / total_examples)
    p_accuracy = percent(total_p_accuracy / total_examples)

    return {"loss": loss, "accuracy": accuracy, "p_accuracy": p_accuracy}


def train(train_loader, valid_loader, model, optimizer, criterion, n_epochs=15):
    for epoch in tqdm(range(n_epochs)):
        pass

    return {"loss": 0., "accuracy": 0., "p_accuracy": 0}
