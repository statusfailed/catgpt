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

def var_fwd(N: NdArrayType, T: NdArrayType, correction=1):
    # constants
    X = obj(N + T)
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

def layer_norm_fwd(N: NdArrayType, T: NdArrayType, epsilon=1e-05):
    # NOTE: layer_norm is always over the final dimension only
    X = obj(N+T)
    assert len(T.shape) == 1, "layer_norm is only over the final dimension"

    v = var_fwd(N, T, correction=0) >> op(NCopy(N, T))
    stddev = v >> increment(epsilon)(X) >> exponentiate(-0.5)(X)

    return copy(X) >> (x_minus_mu(N, T) @ stddev) >> multiply(X)
