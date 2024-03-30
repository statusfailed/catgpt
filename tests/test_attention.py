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
from catgpt.settings import *

F = Forget()

def _compile_id(c):
    CompiledModel, ParamType, model_ast = compile_model(c, identity, identity)
    import ast
    print(ast.unparse(model_ast))
    return CompiledModel

################################################################################
# Tests

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

def test_heads_splitter():
    c = heads(B, T, C) >> splitter(B, T, C, num_heads)
    CompiledModel = _compile_id(c)

    p = torch.normal(0, 1, (C+C3).shape)
    x = torch.normal(0, 1, (B+T+C).shape)

    actual = CompiledModel(Torch).predict(p, x)
    expected = reference.heads_splitter(p, x)
    for e, a in zip(expected, actual):
        assert torch.all(e == a)

def test_self_attention():
    c = self_attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    
    x0 = torch.normal(0, 9, (B+N+T+K).shape)
    x1 = torch.normal(0, 9, (B+N+T+K).shape)

    [actual] = CompiledModel(Torch).predict(x0, x1)
    expected = reference.self_attention(x0, x1)
    
    assert torch.allclose(expected, actual)
    assert torch.all(expected == actual)

def test_value():
    c = value(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x0 = torch.normal(0, 1, (B+N+T+T).shape)
    x1 = torch.normal(0, 1, (B+N+T+K).shape)
    [actual] = CompiledModel(Torch).predict(x0, x1)
    expected = reference.value(x0, x1)
    assert torch.all(expected == actual)

def test_attention_layer():
    c = attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    p = torch.normal(1, 9, (C+C3).shape)
    x = torch.normal(2, 3, (B+T+C).shape)
    [actual] = CompiledModel(Torch).predict(p, x)
    expected = reference.attention(p, x)
    assert torch.all(expected == actual)

def test_block():
    c = block(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    p = torch.normal(1, 9, (C+C3).shape)
    x = torch.normal(1, 9, (B+T+C).shape)
    [actual] = CompiledModel(Torch).predict(p, x)
    expected = reference.block(p, x)
    assert torch.all(expected == actual)
