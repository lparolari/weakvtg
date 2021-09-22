import torch


def is_close(*args, **kwargs):
    return all(torch.isclose(*args, **kwargs, rtol=1e-04).detach().numpy().ravel())


def test_is_close():
    assert is_close(torch.tensor([1., 0.11111]), torch.tensor([1., 0.11112]))
    assert not is_close(torch.tensor([1., 0.1111]), torch.tensor([1., 0.1112]))
    assert not is_close(torch.tensor([[0.1111]]), torch.tensor([[0.1112]]))
    assert not is_close(torch.tensor([[[0.1111]]]), torch.tensor([[[0.1112]]]))
