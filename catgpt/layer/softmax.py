from dataclasses import dataclass

from catgrad.signature import NdArrayType, obj, op
from catgrad.combinators import identity, permutation
from catgrad.special.definition import Definition
from catgrad.bidirectional.operation import Lens, constant, subtract, copy, multiply, discard, add, negate, NAdd, NCopy, Reshape, MatrixMultiply
import catgrad.core.operation as ops

from catgpt.layer.masked_fill_constant import MaskedFillConstant

@dataclass(frozen=True)
class Softmax(Definition, Lens):
    N: NdArrayType
    T: NdArrayType
    def source(self): return obj(self.N + self.T)
    def target(self): return obj(self.N + self.T)

    def __post_init__(self):
        if not self.T.dtype.is_floating():
            raise ValueError("Sigmoid is not defined for non-floating-point dtypes")

    ########################################
    # Softmax as a Core definition

    # The definition of the Softmax function in terms of Core ops
    def arrow(self):
        N = self.N
        T = self.T
        X = obj(N + T)

        # maximum over T, negate, and broadcast
        # --[max]---[negate]---[copy]---
        max_neg = op(ops.NMax(N, T)) >> op(ops.Negate(N)) >> op(ops.NCopy(N, T))

        #     /----------------\
        # ---●                  ●----
        #     \----[max_neg]---/
        x_minus_max = op(ops.Copy(N+T)) >> (identity(X) @ max_neg) >> op(ops.Add(N+T))


        # ---[Σ]--[broadcast]---
        sum_bcast = op(ops.NAdd(N, T)) >> op(ops.NCopy(N, T))

        #            /----------------------\
        # ---[exp]--●                        [÷]----
        #            \---[Σ]--[broadcast]---/
        z = ops.exp1(N + T) >> op(ops.Copy(N+T)) >> (identity(X) @ sum_bcast) >> op(ops.Divide(N + T))

        return x_minus_max >> z

    ########################################
    # Softmax as an Optic

    # we want this to appear as a Definition in core, so we just return the op
    # as a singleton diagram.
    def to_core(self): return op(self)

    # The forward map is like Lens, but we copy the *output*, not the input.
    def fwd(self):
        return op(self) >> copy(self.target())

    def rev(self):
        N, T = self.N, self.T
        X = obj(N + T)

        _1 = NdArrayType((1,), T.dtype)
        outer_prod = copy(obj(N+T)) >> ( op(Reshape(N+T, N+T+1)) @ op(Reshape(N+T, N+1+T)) ) >> op(MatrixMultiply(N, T, _1, T))
        diag = diagonal(N, T)

        jacobian = copy(obj(N+T)) >> (diag @ outer_prod) >> subtract(obj(N+T+T))
        unsqueeze = op(Reshape(N+T, N+T+_1))
        matmul = op(MatrixMultiply(N, T, T, _1)) >> op(Reshape(N+T+_1, N+T))
        return (jacobian @ unsqueeze) >> matmul

# TODO FIXME: this is torch-only; we need an op in catgrad.
def diagonal(N, T):
    assert len(T.shape) == 1
    import torch
    mask = ~torch.eye(T.shape[0], dtype=bool) # off-diagonals
    return op(NCopy(N+T, T)) >> op(MaskedFillConstant(N, T+T, mask, 0))
