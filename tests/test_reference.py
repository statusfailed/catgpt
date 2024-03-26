""" test reference implementations match pytorch """
import unittest

import torch
from catgpt.reference import mean, r_mean, var, r_var, softmax, r_softmax

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
    actual   = softmax(x, dim=-1)
    assert torch.allclose(expected, actual)

@unittest.skip("TODO")
def test_reference_r_softmax():
    # shape = (50,40,30)
    shape = (2,10)
    dim = -1 # torch softmax only supports one dim

    x = torch.normal(10, 2, shape, requires_grad=True)
    dy = torch.ones(shape, dtype=x.dtype)

    expected = functional.softmax(x, dim=-1).grad_fn(dy)
    actual   = r_softmax(x.detach().numpy(), dy.numpy(), dim=-1)
    breakpoint()
    assert torch.allclose(expected, actual)
