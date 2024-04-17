""" a hack: implement attention using the PythonOp escape hatch. """
from dataclasses import dataclass
import torch

from catgrad.signature import Dtype, op
from catgrad.bidirectional.operation import *
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

