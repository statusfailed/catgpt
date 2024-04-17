""" test reference implementations match pytorch """
import unittest

import torch
from catgpt.reference import mean, r_mean, var, r_var, softmax, r_softmax, layer_norm

import torch.nn.functional as functional

def test_reference_mean():
    shape = (50,40,30,20,10)
    dims = (-2, -1)
    x = torch.normal(10, 2, shape)

    expected = x.mean(dims)
    actual   = mean(x, dims=dims)
    assert torch.allclose(expected, actual)

def test_reference_r_mean():
    shape = (50,40,30,20,10)
    dims = (-2, -1)
    x = torch.normal(10, 2, shape, requires_grad=True)

    dy = torch.arange(50*40*30).view((50,40,30))
    expected = x.mean(dims).grad_fn(dy)
    actual   = r_mean(x, dy, dims=dims)
    assert torch.allclose(expected, actual)


# see https://pytorch.org/docs/stable/generated/torch.var.html
def test_reference_var():
    shape = (50,40,30,20,10)
    dims = (-2, -1)

    x = torch.normal(10, 2, shape)

    expected = x.var(dims)
    actual   = var(x, dims)

    assert torch.allclose(expected, actual)

def test_reference_r_var():
    shape = (50,40,30,20,10)
    dims = (-2, -1)

    x = torch.normal(10, 2, shape, requires_grad=True)
    dy = torch.arange(50*40*30).view((50,40,30))

    expected = x.var(dims).grad_fn(dy)
    actual   = r_var(x, dy, dims)
    assert torch.allclose(expected, actual)

def test_reference_softmax():
    shape = (50,40,30)
    dim = -1 # torch softmax only supports one dim

    x = torch.normal(10, 2, shape)

    expected = functional.softmax(x, dim=-1)
    actual   = softmax(x)
    assert torch.allclose(expected, actual)

def randomly_zero_out(tensor):
    size = tensor.shape
    mask = torch.zeros_like(tensor)
    indices = torch.randint(0, size[-1], (size[:-1]))
    mask.scatter_(-1, indices.unsqueeze(-1), 1)
    return tensor * mask

def test_reference_r_softmax():
    shape = (50,40,30)

    x = torch.rand(shape) - 1
    x.requires_grad = True
    dy = torch.rand(shape, dtype=x.dtype) - 1

    # randomly zero out some entries of dy (as if it came from NLL loss)
    dy = randomly_zero_out(dy)

    expected = functional.softmax(x, dim=-1).grad_fn(dy)
    actual   = r_softmax(x, dy)

    # TODO: tolerance here needs to be quite high; why?
    assert torch.allclose(expected, actual)


def test_reference_layer_norm():
    # shape = (50,40,30)
    # normalized_shape = (40,30)
    shape = (50,2,10)
    normalized_shape = (shape[-1],)

    x = torch.normal(10, 2, shape)

    expected = functional.layer_norm(x, normalized_shape=normalized_shape, weight=None, bias=None, eps=1e-05)
    actual   = layer_norm(x)
    # we use a high tolerance because we don't actually use torch's layernorm-
    # ours will do.
    assert torch.allclose(expected, actual, atol=1e-5)
