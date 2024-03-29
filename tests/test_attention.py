import torch

from catgrad import compile_model
from catgrad.combinators import identity
from catgrad.signature import NdArrayType, Dtype, op
from catgrad.bidirectional.operation import *
from catgrad.bidirectional.functor import Forget
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend.torch import Torch

from catgpt.attention import heads, splitter, self_attention, value, attention, block

F = Forget()

num_heads = 8
B, T, C = [ NdArrayType((i,), Dtype.float32) for i in [64, 32, 384] ]
N = NdArrayType((num_heads,), B.dtype)
K = NdArrayType((C.shape[0]//num_heads,), B.dtype)
C3 = NdArrayType((C.shape[0]*3,), Dtype.float32)

def _compile_id(c):
    CompiledModel, ParamType, model_ast = compile_model(c, identity, identity)
    import ast
    print(ast.unparse(model_ast))
    return CompiledModel

def test_heads():
    c = heads(B, T, C)
    CompiledModel = _compile_id(c)
    p = torch.ones((C+C3).shape)
    x = torch.ones((B+T+C).shape)
    CompiledModel(Torch).predict(p, x)

from dataclasses import dataclass
from catgrad.signature import Dtype
from catgrad.target.python.array_backend import ArrayBackend

def test_heads():
    # TODO: test with parameter input
    heads(B, T, C)

def test_splitter():
    c = splitter(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x = torch.ones((B+T+C+3).shape)
    CompiledModel(Torch).predict(x)

def test_self_attention():
    c = self_attention(B, T, C, num_heads)
    # CompiledModel = _compile_id(c)
    # x = torch.ones((B+T+C3).shape)
    # CompiledModel(Torch).predict(x)

def test_value():
    value(B, T, C, num_heads)

def test_attention():
    attention(B, T, C, num_heads)

def test_block():
    c = block(B, T, C, num_heads)
    
    # CompiledModel, ParamType, model_ast = compile_model(c, identity, identity)
    CompiledModel = _compile_id(c)

    p = torch.ones((C+C3).shape)
    x = torch.ones((B+T+C).shape)
    CompiledModel(Torch).predict(p, x)
