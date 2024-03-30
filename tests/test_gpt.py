import torch

from catgrad import compile_model
from catgrad.combinators import identity
from catgrad.signature import NdArrayType, Dtype, op
from catgrad.target.python.array_backend.torch import Torch
from catgpt.gpt import gpt
from catgpt.settings import *

B, T, C, V = [ NdArrayType((i,), Dtype.float32) for i in [batch_size, sequence_length, d_model, vocab_size] ]
N = NdArrayType((num_heads,), B.dtype)
K = NdArrayType((C.shape[0]//num_heads,), B.dtype)
C3 = NdArrayType((C.shape[0]*3,), Dtype.float32)

def test_gpt():
    model = gpt(B, T, C, vocab_size=vocab_size, num_heads=num_heads, num_blocks=6)
    CompiledModel, ParamType, model_ast = compile_model(model, identity, identity)
    x = torch.ones((B+T+V).shape, dtype=torch.float32)
    p = [ torch.normal(0, 1, t.shape) for t in ParamType ]
    CompiledModel(Torch).predict(*p, x)
