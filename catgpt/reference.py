""" reference implementations of forward and reverse maps """
import math
from catgpt.util import product

# not sure what internal impl. is. Hope this works!
def mean(x, dims):
    n_dim = product(x.shape[d] for d in dims)
    return x.sum(dims) / n_dim

def r_mean(x, dy, dims):
    n_dim = product(x.shape[d] for d in dims)

    values = (1/n_dim)*dy # scale
    result = values.view(dy.shape + (1,)*len(dims)).broadcast_to(x.shape) # "copy"

    return result

# reference implementation according to docs
# https://pytorch.org/docs/stable/generated/torch.var.html
def var(x, dims, correction=1):
    N = product(x.shape[d] for d in dims)

    # This is a torch one-liner, but catgrad core will require explicit broadcasting
    # m = x.mean(dims, keepdim=True)
    m = mean(x, dims) # (N₀ N₁ ..., T₀, T₁, ...)
    m = m.view(tuple(m.shape + (1,)*len(dims))) # broadcast to (N₀, N₁, ..., 1, 1, ..., 1)

    var_sum = ((x - m)**2).sum(dims)
    d = max(0, N - correction)
    return (1 / d) * var_sum

# https://math.stackexchange.com/questions/2836083/derivative-of-the-variance-wrt-x-i#2887894
def r_var(x, dy, dims, correction=1):
    n_elem = product(x.shape[d] for d in dims)
    d = max(0, n_elem - correction)

    m = mean(x, dims) # (N₀ N₁ ..., T₀, T₁, ...)
    m = m.view(tuple(m.shape + (1,)*len(dims))) # broadcast to (N₀, N₁, ..., 1, 1, ..., 1)
    grad = (x - m) * (2 / d)
    result = grad * dy.view(dy.shape + (1,)*len(dims))
    return result

def softmax(x):
    """ softmax over final dimension """
    z = x - x.max(dim=-1, keepdim=True).values
    # num = z.exp()
    # NOTE: e power for bitwise accuracy with catgrad
    num = torch.tensor(math.e) ** z
    den = num.sum(dim=-1, keepdim=True)
    return num / den

def r_softmax(x, dy):
    softmax_x = softmax(x)
    dot_product = (softmax_x * dy).sum(dim=-1, keepdim=True)
    grad_x = softmax_x * (dy - dot_product)
    return grad_x

# NOTE: this implementation doesn't use the layer norm weights;
import torch
def layer_norm(x, epsilon=1e-05):
    # m = x.mean(dim=-1, keepdim=True)
    # v = x.var(dim=-1, keepdim=True, unbiased=False)
    m = mean(x, dims=(-1,)).unsqueeze(-1).broadcast_to(x.shape)
    v = var(x, dims=(-1,), correction=0).unsqueeze(-1).broadcast_to(x.shape)

    e = torch.tensor(-0.5) # NOTE: convert to torch.tensor to get bit-level reproducibility
    return (x - m) * (v + epsilon)**e
