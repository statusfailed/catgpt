from catgrad.bidirectional.operation import *
from catgpt.util import product

# Mean over the T dimensions
def mean_fwd(N: NdArrayType, T: NdArrayType):
    s = product(T.shape)
    a = op(NAdd(N, T))
    b = scale_inverse(s)(obj(N))
    return a >> b

# TODO: if we put this into an op (class Mean), we should have it be a Dagger
# and drop the discard.
def mean_rev(N: NdArrayType, T: NdArrayType):
    X = obj(N+T)
    Y = obj(N)

    s = product(T.shape) # scale factor
    a = scale_inverse(s)(Y) # * (1 / s)
    b = op(NCopy(N, T))

    return discard(X) @ (a >> b)

def x_minus_mu(N: NdArrayType, T: NdArrayType):
    X = obj(N + T)
    neg_mean = mean_fwd(N, T) >> op(NCopy(N, T)) >> negate(X) # -μ(x)
    return copy(X) >> (neg_mean @ identity(X)) >> add(X) # (x - μ)

def var_fwd(N: NdArrayType, T: NdArrayType):
    # constants
    X = obj(N + T)
    correction = 1
    n_elem = product(T.shape)
    d = max(0, n_elem - correction)

    # circuits
    square_sum_scale = copy(X) >> multiply(X) >> op(NAdd(N,T)) >> scale(1/d)(obj(N)) # ^2 ; Σ ; (1/d)

    return x_minus_mu(N, T) >> square_sum_scale

def var_rev(N: NdArrayType, T: NdArrayType):
    correction = 1
    n_elem = product(T.shape)
    d = max(0, n_elem - correction)
    X = obj(N + T)

    top = x_minus_mu(N, T) >> scale(2/d)(X)
    bot = op(NCopy(N, T))

    return (top @ bot) >> multiply(X)
