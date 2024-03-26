from dataclasses import dataclass

from catgrad.signature import NdArrayType, obj, op
from catgrad.combinators import identity
from catgrad.special.definition import Definition
from catgrad.bidirectional.operation import Lens, constant, subtract, copy, multiply
import catgrad.core.operation as ops

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

    # The reverse map is the same as sigmoid: σ(x) · (1 - σ(x)) · dy
    def rev(self):
        raise NotImplementedError("This is not correct")
        # σ * (1 - σ) * dy
        #
        #         /----------\
        # σ(x) --●            *---\
        #         \-- (1-) --/     *---
        #                         /
        # dy   ------------------/
        X = obj(self.N + self.T)
        id_X = identity(X)
        dec_1 = (constant(1)(X) @ id_X) >> subtract(X) # 1 - x
        grad = copy(X) >> (id_X @ dec_1) >> multiply(X) # σ * (1 - σ)
        return (grad @ identity(X)) >> multiply(X) # σ * (1 - σ) * dy
