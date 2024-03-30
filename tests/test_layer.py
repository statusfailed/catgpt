import unittest
import torch

from catgrad import NdArrayType, Dtype
from catgrad.compile import rdiff

from catgrad.bidirectional.functor import Forget
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend.torch import Torch

from catgpt.layer import *
from catgpt import reference

import torch.nn.functional as functional

F = Forget()

def test_mean_fwd():
    N = NdArrayType((50,40,30), Dtype.float32)
    T = NdArrayType((20,10), Dtype.float32)

    c = mean_fwd(N, T)
    Fc = F(c)
    f = to_python_function(Fc, array_backend=Torch)
    x = torch.normal(10, 2, (N+T).shape)

    expected = reference.mean(x, dims=(-2, -1))
    [actual] = f(x)
    assert torch.all(expected == actual)

def test_mean_rev():
    N = NdArrayType((50,40,30), Dtype.float32)
    T = NdArrayType((20,10), Dtype.float32)

    c = mean_rev(N, T)
    Fc = F(c)
    f = to_python_function(Fc, array_backend=Torch)

    x = torch.normal(10, 2, (N+T).shape)
    dy = torch.normal(10, 2, N.shape)

    expected = reference.r_mean(x, dy, dims=(-2, -1))
    [actual] = f(x, dy)
    # TODO: why is this not exact?
    assert torch.allclose(expected, actual)

def test_var_fwd():
    N = NdArrayType((50,40,30), Dtype.float32)
    T = NdArrayType((20,10), Dtype.float32)

    c = var_fwd(N, T)
    Fc = F(c)
    f = to_python_function(Fc, array_backend=Torch)
    x = torch.normal(10, 2, (N+T).shape)

    expected = reference.var(x, dims=(-2, -1))
    [actual] = f(x)
    assert torch.all(expected == actual)

def test_var_rev():
    N = NdArrayType((50,40,30), Dtype.float32)
    T = NdArrayType((20,10), Dtype.float32)

    c = var_rev(N, T)
    Fc = F(c)
    f = to_python_function(Fc, array_backend=Torch)
    x = torch.normal(10, 2, (N+T).shape)
    dy = torch.normal(10, 2, N.shape)

    expected = reference.r_var(x, dy, dims=(-2, -1))
    [actual] = f(x, dy)
    assert torch.all(expected == actual)

def test_softmax_fwd():
    N = NdArrayType((40,30,20), Dtype.float32)
    T = NdArrayType((100,), Dtype.float32)

    dim = -1 # torch softmax only supports one dim

    Fc = Softmax(N, T).arrow()
    f = to_python_function(Fc, array_backend=Torch)

    x = torch.normal(10, 2, (N+T).shape)

    expected = reference.softmax(x)
    [actual] = f(x)
    assert torch.allclose(expected, actual)

def test_softmax_rev():
    N = NdArrayType((2,), Dtype.float32)
    T = NdArrayType((10,), Dtype.float32)

    dim = -1 # torch softmax only supports one dim

    # differentiate softmax and map to core
    e = Softmax(N, T)
    c = F(e.rev())
    f = to_python_function(c, array_backend=Torch)

    x = torch.normal(10, 2, (N+T).shape, requires_grad=True)
    dy = torch.normal(10, 2, (N+T).shape)

    # expected = r_softmax(x).grad_fn(dy)[0]
    expected = reference.r_softmax(x, dy)
    # NOTE: we're testing rev, and rev expects x input to be softmax(x)!
    [actual] = f(reference.softmax(x), dy)
    assert torch.allclose(expected, actual)

def test_layer_norm():
    N = NdArrayType((50,40,30), Dtype.float32)
    T = NdArrayType((20,), Dtype.float32)

    c = layer_norm(N, T)
    Fc = F(c)
    f = to_python_function(Fc, array_backend=Torch)
    x = torch.normal(10, 2, (N+T).shape)

    expected = reference.layer_norm(x)
    [actual] = f(x)
    assert torch.allclose(expected, actual)
