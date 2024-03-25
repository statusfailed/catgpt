""" test reference implementations match pytorch """
import torch
from catgpt.reference import mean, r_mean, var, r_var

def test_reference_mean():
    shape = (100, 100, 100, 100)
    dims = (-2, -1)
    x = torch.normal(10, 2, shape)

    expected = x.mean(dims)
    actual   = mean(x, dims=dims)
    assert torch.allclose(expected, actual)

def test_reference_r_mean():
    shape = (100, 100, 100, 100)
    dims = (-2, -1)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.ones((100,100))
    expected = x.mean(dims).grad_fn(dy)
    actual   = r_mean(x, dy, dims=dims)
    assert torch.allclose(expected, actual)


# see https://pytorch.org/docs/stable/generated/torch.var.html
def test_reference_var():
    shape = (100, 100, 100, 100)
    dims = (-2, -1)
    x = torch.normal(10, 2, shape)

    expected = x.var(dims)
    actual   = var(x, dims)
    assert torch.allclose(expected, actual)


def test_reference_r_var():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.tensor(1)
    expected = x.var().grad_fn(dy)
    actual   = r_var(x, dy)
    assert torch.allclose(expected, actual)
