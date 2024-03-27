import unittest
import torch

from catgrad import NdArrayType, Dtype
from catgrad.compile import rdiff

from catgrad.bidirectional.functor import Forget
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend.torch import Torch

from catgpt.layer import *
from catgpt.reference import mean, r_mean, var, r_var, softmax

import torch.nn.functional as functional

F = Forget()

def test_mean_fwd():
    N = NdArrayType((50,40,30), Dtype.float32)
    T = NdArrayType((20,10), Dtype.float32)

    c = mean_fwd(N, T)
    Fc = F(c)
    f = to_python_function(Fc, array_backend=Torch)
    x = torch.normal(10, 2, (N+T).shape)

    expected = mean(x, dims=(-2, -1))
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

    expected = r_mean(x, dy, dims=(-2, -1))
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

    expected = var(x, dims=(-2, -1))
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

    expected = r_var(x, dy, dims=(-2, -1))
    [actual] = f(x, dy)
    assert torch.all(expected == actual)

# def fn(x0):
    # x1 = Torch.nmax((-1,), x0)
    # x4 = Torch.constant(2.718281828459045, (2, 10), Dtype.float32)
    # x2 = -x1
    # x3 = Torch.ncopy((10,), x2) # x.max(dim=-1, keepdim=True)
    # x5 = x4 ** x3 # exp(x3)
    # x6, x7 = (x5, x5)
    # x8 = Torch.nadd((-1,), x7)
    # x9 = Torch.ncopy((10,), x8)
    # x10 = x6 / x9

    # x = x0
    # z = x - x.max(dim=-1, keepdim=True).values
    # num = z.exp()
    # den = num.sum(dim=-1, keepdim=True)
    # result = num / den

    # breakpoint()
    # return [x10]

def test_softmax_fwd():
    N = NdArrayType((40,30,20), Dtype.float32)
    T = NdArrayType((100,), Dtype.float32)

    dim = -1 # torch softmax only supports one dim

    Fc = Softmax(N, T).arrow()
    f = to_python_function(Fc, array_backend=Torch)

    x = torch.normal(10, 2, (N+T).shape)

    expected = softmax(x)
    [actual] = f(x)
    assert torch.allclose(expected, actual)

@unittest.skip("TODO")
def test_softmax_rev():
    N = NdArrayType((2,), Dtype.float32)
    T = NdArrayType((10,), Dtype.float32)

    dim = -1 # torch softmax only supports one dim

    # differentiate softmax and map to core
    # c = op(Softmax(N, T))
    Fc = F(rdiff(op(Softmax(N, T))))
    # f = to_python_function(Fc, array_backend=Torch)
    f = fn

    x = torch.normal(10, 2, (N+T).shape, requires_grad=True)
    # dy = torch.normal(10, 2, (N+T).shape)
    dy = torch.ones((N+T).shape, dtype=x.dtype)

    expected = softmax(x, dim=dim).grad_fn(dy)[0]
    [actual] = f(x, dy)
    breakpoint()
    assert torch.allclose(expected, actual)
