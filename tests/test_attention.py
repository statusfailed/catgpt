import torch

from catgrad import compile_model
from catgrad.combinators import identity
from catgrad.signature import NdArrayType, Dtype, op
from catgrad.bidirectional.operation import *
from catgrad.bidirectional.functor import Forget
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend.torch import Torch

from catgpt import reference
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

def reference_heads_splitter(p, x):
    B, T, C = x.size()
    q_x, k_x, v_x = (x @ p).split(d_model, dim=2) # split along 3*C dim
    q_x = q_x.view(B, T, num_heads, C // num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
    k_x = k_x.view(B, T, num_heads, C // num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
    v_x = v_x.view(B, T, num_heads, C // num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
    return q_x, k_x, v_x

def test_heads_splitter():
    c = heads(B, T, C) >> splitter(B, T, C, num_heads)
    CompiledModel = _compile_id(c)

    p = torch.normal(0, 1, (C+C3).shape)
    x = torch.normal(0, 1, (B+T+C).shape)

    actual = CompiledModel(Torch).predict(p, x)
    expected = reference_heads_splitter(p, x)
    for e, a in zip(expected, actual):
        assert torch.all(e == a)


MASK = ~torch.tril(torch.ones(sequence_length, sequence_length, requires_grad=False, dtype=bool))
# TODO: move to reference module
def reference_self_attention(q_x, k_x):
    import torch
    import torch.nn.functional as functional

    w0 = q_x @ k_x.transpose(-2, -1)
    w1 = w0 / torch.sqrt(torch.tensor(d_model)) # d_k = d_model
    w2 = w1.masked_fill(MASK, float('-inf'))
    # w3 = functional.softmax(w2, dim=-1) # (B, T, T)
    w3 = reference.softmax(w2)
    return w3

def test_self_attention():
    c = self_attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    
    x0 = torch.normal(0, 9, (B+N+T+K).shape)
    x1 = torch.normal(0, 9, (B+N+T+K).shape)

    [actual] = CompiledModel(Torch).predict(x0, x1)
    expected = reference_self_attention(x0, x1)
    
    assert torch.allclose(expected, actual)
    assert torch.all(expected == actual)

def reference_value(a, v):
    r = a @ v
    return r.transpose(1,2).reshape(B.shape[0], T.shape[0], C.shape[0])

def test_value():
    c = value(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x0 = torch.normal(0, 1, (B+N+T+T).shape)
    x1 = torch.normal(0, 1, (B+N+T+K).shape)
    [actual] = CompiledModel(Torch).predict(x0, x1)
    expected = reference_value(x0, x1)
    assert torch.all(expected == actual)

def reference_attention(p, x):
    import torch.nn.functional as functional
    B, T, C = x.size()

    # attention heads + splitting
    q_x, k_x, v_x = reference_heads_splitter(p, x)

    # self_attention
    w3 = reference_self_attention(q_x, k_x)

    # w4 = w3 @ v_x
    # w5 = w4.transpose(1,2).contiguous().view(B, T, C) # reassemble heads
    w5 = reference_value(w3, v_x)
    return w5

def test_attention_layer():
    c = attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    p = torch.normal(1, 9, (C+C3).shape)
    x = torch.normal(2, 3, (B+T+C).shape)
    [actual] = CompiledModel(Torch).predict(p, x)
    expected = reference_attention(p, x)
    assert torch.all(expected == actual)

def reference_block(p, x0):
    x1 = reference.layer_norm(x0)
    x2 = reference_attention(p, x1)
    x3 = x0 + x2
    return x3

def test_block():
    c = block(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    p = torch.normal(1, 9, (C+C3).shape)
    x = torch.normal(1, 9, (B+T+C).shape)
    [actual] = CompiledModel(Torch).predict(p, x)
    expected = reference_block(p, x)
    assert torch.all(expected == actual)
