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

from catgpt.reference.model.picogpt import GPT

def test_picogpt():
    """ verify we get identical results from the picoGPT torch reference implementation and catGPT """
    B1 = NdArrayType((1,), Dtype.float32)
    model = gpt(B1, T, C, vocab_size=vocab_size, num_heads=num_heads, num_blocks=1)
    CompiledModel, ParamType, model_ast = compile_model(model, identity, identity)

    p = [ torch.normal(0, 1, t.shape) for t in ParamType ]
    q = [ q.T for q in p ] # torch params
    x = torch.normal(0, 9, (B1+T+V).shape, dtype=torch.float32)
    
    torch_model = GPT('device', vocab_size, d_model, num_heads, num_layers=1)
    state_dict = {k: v for k, v in zip(torch_model.state_dict().keys(), q)}
    torch_model.load_state_dict(state_dict)

    [actual] = CompiledModel(Torch).predict(*p, x)
    expected, _ = torch_model(x)

    assert torch.allclose(actual, expected)
    assert torch.all(actual == expected)
