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

def test_self_attention():
    c = self_attention(B, T, C, num_heads)
    CompiledModel = _compile_id(c)
    x = torch.ones((B+N+T+K).shape)
    CompiledModel(Torch).predict(x, x)

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
    
    # CompiledModel, ParamType, model_ast = compile_model(c, identity, identity)
    CompiledModel = _compile_id(c)
    p = torch.ones((C+C3).shape)
    x = torch.ones((B+T+C).shape)
    CompiledModel(Torch).predict(p, x)
