from unittest import mock

import pytest

from tests.mock_tensor import MockTensor
from tests.utils import contains
from weakvtg.train import train, epoch


@pytest.fixture
def loader():
    return [{"id": [1, 2]}]


@pytest.fixture
def model():
    return mock.Mock()


@pytest.fixture
def optimizer(): return mock.Mock()


@pytest.fixture
def criterion():
    def c(_1, _2):
        return MockTensor(1.), MockTensor(2.), MockTensor(3.), MockTensor(4.)
    return c


def test_train(loader, model, optimizer, criterion):
    train_history, valid_history = train(train_loader=loader, valid_loader=loader, model=model, optimizer=optimizer,
                                         criterion=criterion, n_epochs=15)

    assert isinstance(train_history, dict)
    assert isinstance(valid_history, dict)


def test_epoch(loader, model, optimizer, criterion):
    out = epoch(loader=loader, model=model, optimizer=optimizer, criterion=criterion, train=True)

    assert isinstance(out, dict)
    assert contains(["loss", "accuracy", "p_accuracy"], out.keys())
