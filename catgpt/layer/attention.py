import math
import torch # needed for sqrt; math.sqrt gives different results(!)
from typing import List

from catgrad.signature import Dtype, op
from catgrad.bidirectional.operation import *
from catgrad.bidirectional.operation import *
from catgrad.layers import linear

from catgpt.layer.statistics import layer_norm
from catgpt.layer.softmax import Softmax
from catgpt.layer.masked_fill_constant import MaskedFillConstant

def _check_type(T: NdArrayType):
    assert len(T.shape) == 1, "NdArrayType must have dimension 1"

def _check_types(*Ts: List[NdArrayType]):
    if len(Ts) == 0:
        return

    dtype = Ts[0].dtype
    for T in Ts:
        assert T.dtype == dtype
        _check_type(T)

def _get_N_K_types(C: NdArrayType, num_heads: int):
    (head_size, rem) = divmod(C.shape[0], num_heads)
    assert rem == 0, f"{num_heads=} did not evenly divide {C.shape=}"

    N = NdArrayType((num_heads,), C.dtype)
    K = NdArrayType((head_size,), C.dtype)
    return N, K

def heads(B: NdArrayType, T: NdArrayType, C: NdArrayType):
    _check_types(B, T, C)
    # Target has shape of C*3, but we explicitly reshape to be splittable
    C3 = NdArrayType((C.shape[0]*3,), C.dtype)
    # NOTE: we do 3+C rather than C+3 here then permute to keep consistency with reference impl
    return linear(B+T, C, C3) >> op(Reshape(B+T+C3, B+T+3+C)) >> op(Permute(B+T+3+C, [0, 1, 3, 2]))

def splitter(B: NdArrayType, T: NdArrayType, C: NdArrayType, num_heads: int):
    _check_types(B, T, C)
    N, K = _get_N_K_types(C, num_heads)
    C3 = NdArrayType((C.shape[0]*3,), C.dtype)
    X = B+T+C
    Y = B+T+N+K

    s = op(NSplit(B+T+C, 3))
    r = op(Reshape(X, Y))
    p = op(Permute(Y, [0, 2, 1, 3]))

    return s >> (r @ r @ r) >> (p @ p @ p)

def query_key(B: NdArrayType, T: NdArrayType, C: NdArrayType, num_heads: int):
    _check_types(B, T, C)
    d_model = C.shape[0]
    N, K = _get_N_K_types(C, num_heads)
    BNTK = B+N+T+K
    BNKT = B+N+K+T

    a = identity(obj(BNTK)) @ op(Permute(BNTK, [0, 1, 3, 2]))
    b = op(MatrixMultiply(B+N, T, K, T))
    scale_factor = torch.sqrt(torch.tensor(d_model)).item() # NOTE: torch.sqrt != math.sqrt.

    c = scale_inverse(scale_factor)(obj(B+N+T+T))
    mask = ~torch.tril(torch.ones(T.shape[0], T.shape[0], dtype=bool))
    d = op(MaskedFillConstant(B+N, T+T, mask, float('-inf')))
    e = op(Softmax(B+N+T, T))

    return a >> b >> c >> d >> e

def value(B: NdArrayType, T: NdArrayType, C: NdArrayType, num_heads: int):
    _check_types(B, T, C)
    d_model = C.shape[0]
    N, K = _get_N_K_types(C, num_heads)

    m = op(MatrixMultiply(B+N, T, T, K))
    p = op(Permute(B+N+T+K, [0, 2, 1, 3]))
    r = op(Reshape(B+T+N+K, B+T+C))

    return m >> p >> r

def attention(B, T, C, num_heads: int):
    _check_types(B, T, C)
    N, K = _get_N_K_types(C, num_heads)
    return heads(B, T, C) >> splitter(B, T, C, num_heads) >> (query_key(B, T, C, num_heads) @ identity(obj(B+N+T+K))) >> value(B, T, C, num_heads)

def block(B, T, C, num_heads):
    X = obj(B+T+C)
    ln_attn = layer_norm(B+T, C) >> attention(B, T, C, num_heads)
    return copy(X) >> (ln_attn @ identity(X)) >> add(X)
