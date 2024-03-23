# Test some basic ops are implemented the same
#
#   - mean
#   - variance
import torch

def prod(xs):
    a = 1
    for x in xs:
        a *= x
    return a

# not sure what internal impl. is. Hope this works!
def mean(x):
    n_dim = prod(x.shape)
    return x.sum() / n_dim

def r_mean(x, dy):
    N = prod(x.shape)
    return (1/N)*dy

# reference implementation according to docs
def var(x, correction=1):
    N = prod(x.shape)
    m = mean(x)
    d = max(0, N - correction)
    return (1 / d) * ((x - m)**2).sum()

# https://math.stackexchange.com/questions/2836083/derivative-of-the-variance-wrt-x-i#2887894
def r_var(x, dy, correction=1):
    N = prod(x.shape)
    d = max(0, N - correction)
    return (x - x.mean()) * (2 / d)

def test_op_mean():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape)

    expected = x.mean()
    actual   = mean(x)
    assert torch.allclose(expected, actual)

# see https://pytorch.org/docs/stable/generated/torch.var.html
def test_op_var():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape)

    expected = x.var()
    actual   = var(x)
    assert torch.allclose(expected, actual)

def test_op_r_mean():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.tensor(1)
    expected = x.mean().grad_fn(dy)
    actual   = r_mean(x, dy)
    assert torch.allclose(expected, actual)

def test_op_r_var():
    shape = (100, 100, 100)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.tensor(1)
    expected = x.var().grad_fn(dy)
    actual   = r_var(x, dy)
    assert torch.allclose(expected, actual)
