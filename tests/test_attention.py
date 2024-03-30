import torch

from catgrad import compile_model
from catgrad.combinators import identity
from catgrad.signature import NdArrayType, Dtype, op
from catgrad.bidirectional.operation import *
from catgrad.bidirectional.functor import Forget
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend.torch import Torch

from catgpt.layer.attention import heads, splitter, self_attention, value, attention, block

F = Forget()

num_heads = 8
head_size = 48
d_model = num_heads * head_size
sequence_length = 32
batch_size = 64

B, T, C = [ NdArrayType((i,), Dtype.float32) for i in [batch_size, sequence_length, d_model] ]
N = NdArrayType((num_heads,), B.dtype)
K = NdArrayType((C.shape[0]//num_heads,), B.dtype)
C3 = NdArrayType((C.shape[0]*3,), Dtype.float32)

def _compile_id(c):
    CompiledModel, ParamType, model_ast = compile_model(c, identity, identity)
    import ast
    print(ast.unparse(model_ast))
    return CompiledModel

def test_heads():
    # TODO: test with parameter input
    c = heads(B, T, C)
    CompiledModel = _compile_id(c)
    p0 = torch.ones((C+C3).shape)
    x0 = torch.ones((B+T+C).shape)
    CompiledModel(Torch).predict(p0, x0)

def test_splitter():
    c = splitter(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x = torch.ones((B+T+C+3).shape)
    CompiledModel(Torch).predict(x)


# TODO: move me!
def reference_self_attention(q_x, k_x):
    import torch
    import torch.nn.functional as functional

    mask = ~torch.tril(torch.ones(T.shape[0], T.shape[0], dtype=bool))
    w0 = q_x @ k_x.transpose(-2, -1)
    w1 = w0 / torch.sqrt(torch.tensor(d_model)) # d_k = d_model
    w2 = w1.masked_fill(mask, float('-inf'))
    w3 = functional.softmax(w2, dim=-1) # (B, T, T)
    return w3

def test_self_attention():
    c = self_attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x = torch.normal(0, 1, (B+N+T+K).shape)
    [actual] = CompiledModel(Torch).predict(x, x)
    expected = reference_self_attention(x, x)
    assert torch.allclose(expected, actual)

def test_value():
    c = value(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x0 = torch.ones((B+N+T+T).shape)
    x1 = torch.ones((B+N+T+K).shape)
    CompiledModel(Torch).predict(x0, x1)

def test_attention():
    c = attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    p0 = torch.ones((C+C3).shape)
    x0 = torch.ones((B+T+C).shape)
    CompiledModel(Torch).predict(p0, x0)

def test_block():
    c = block(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    p = torch.ones((C+C3).shape)
    x = torch.ones((B+T+C).shape)
    CompiledModel(Torch).predict(p, x)
