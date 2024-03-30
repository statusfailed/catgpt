import math
from typing import List

from catgrad.signature import Dtype, op
from catgrad.bidirectional.operation import *
from catgrad.bidirectional.operation import *
from catgrad.layers import linear

from catgpt.layer.statistics import layer_norm
from catgpt.layer.softmax import Softmax

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

# hax
import torch
from catgrad.target.python.special import PythonOp
@dataclass(frozen=True)
class MaskedFillConstant(PythonOp, Dagger):
    N: NdArrayType # batch size
    T: NdArrayType # mask type
    mask: torch.tensor # NOTE: unchecked, but must have type T
    c: constant

    # core
    def source(self): return obj(self.N+self.T)
    def target(self): return obj(self.N+self.T)
    def __call__(self, x):
        # in fwd pass, set values @ mask positions to c.
        result = x.masked_fill(self.mask, self.c)
        return [result]

    # bidirectional
    def to_core(self): return op(self)
    def rev(self):
        # in rev pass, since values were set to a constant, they are zeroed.
        return op(MaskedFillConstant(self.N, self.T, self.mask, 0))

def heads(B: NdArrayType, T: NdArrayType, C: NdArrayType):
    _check_types(B, T, C)
    # Target has shape of C*3, but we explicitly reshape to be splittable
    C3 = NdArrayType((C.shape[0]*3,), C.dtype)
    return linear(B+T, C, C3) >> op(Reshape(B+T+C3, B+T+C+3))

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

def self_attention(B: NdArrayType, T: NdArrayType, C: NdArrayType, num_heads: int):
    _check_types(B, T, C)
    d_model = C.shape[0]
    N, K = _get_N_K_types(C, num_heads)
    BNTK = B+N+T+K
    BNKT = B+N+K+T

    a = identity(obj(BNTK)) @ op(Permute(BNTK, [0, 1, 3, 2]))
    b = op(MatrixMultiply(B+N, T, K, T))
    c = scale_inverse(math.sqrt(d_model))(obj(B+N+T+T))

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
    return heads(B, T, C) >> splitter(B, T, C, num_heads) >> (self_attention(B, T, C, num_heads) @ identity(obj(B+N+T+K))) >> value(B, T, C, num_heads)

def block(B, T, C, num_heads):
    X = obj(B+T+C)
    ln_attn = layer_norm(B+T, C) >> attention(B, T, C, num_heads)
    return copy(X) >> (ln_attn @ identity(X)) >> add(X)
