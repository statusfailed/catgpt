""" test reference implementations match pytorch """
import torch
from catgpt.reference import mean, r_mean, var, r_var

def test_op_mean():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape)

    expected = x.mean()
    actual   = mean(x)
    assert torch.allclose(expected, actual)

def test_op_r_mean():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.tensor(1)
    expected = x.mean().grad_fn(dy)
    actual   = r_mean(x, dy)
    assert torch.allclose(expected, actual)


# see https://pytorch.org/docs/stable/generated/torch.var.html
def test_op_var():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape)

    expected = x.var()
    actual   = var(x)
    assert torch.allclose(expected, actual)


def test_op_r_var():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.tensor(1)
    expected = x.var().grad_fn(dy)
    actual   = r_var(x, dy)
    assert torch.allclose(expected, actual)
