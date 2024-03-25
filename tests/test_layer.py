import torch

from catgrad import NdArrayType, Dtype
from catgrad.bidirectional.functor import Forget
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend.torch import Torch

from catgpt.layer import *
from catgpt.reference import mean, r_mean, var, r_var

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
    assert torch.allclose(expected, actual)

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
    assert torch.allclose(expected, actual)
