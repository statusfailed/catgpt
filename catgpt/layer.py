from catgrad.bidirectional.operation import *

from catgpt.util import product

# Mean over the T dimensions
def mean_fwd(N: NdArrayType, T: NdArrayType):
    s = 1 / product(T.shape) # scale factor

    a = op(NAdd(N, T))
    b = scale(s)(obj(N))

    return a >> b

# TODO: if we put this into an op (class Mean), we should have it be a Dagger
# and drop the discard.
def mean_rev(N: NdArrayType, T: NdArrayType):
    X = obj(N+T)
    Y = obj(N)

    s = 1 / product(T.shape) # scale factor
    a = scale(s)(Y)
    b = op(NCopy(N, T))

    return discard(X) @ (a >> b)
