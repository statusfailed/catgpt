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

def r_softmax(z, dy):
    # Compute softmax along the last dimension
    s = softmax(z)

    # Compute outer product of s with itself.
    s_expanded = s.unsqueeze(-1)  # shape [..., n, 1]
    outer_prod = s_expanded * s.unsqueeze(-2)  # shape [..., n, n]

    # Jacobian of softmax
    jacobian = torch.diag_embed(s) - outer_prod  # shape [..., n, n]

    # Multiply the Jacobian by dy
    dx = torch.matmul(jacobian, dy.unsqueeze(-1)).squeeze(-1)

    return dx

# NOTE: this implementation doesn't use the layer norm weights;
import torch
def layer_norm(x, epsilon=1e-05):
    # m = x.mean(dim=-1, keepdim=True)
    # v = x.var(dim=-1, keepdim=True, unbiased=False)
    m = mean(x, dims=(-1,)).unsqueeze(-1).broadcast_to(x.shape)
    v = var(x, dims=(-1,), correction=0).unsqueeze(-1).broadcast_to(x.shape)

    e = torch.tensor(-0.5) # NOTE: convert to torch.tensor to get bit-level reproducibility
    return (x - m) * (v + epsilon)**e

################################################################################
# Attention functions

from catgpt.settings import *

def heads_splitter(p, x):
    B, T, C = x.size()
    q_x, k_x, v_x = (x @ p).split(d_model, dim=2) # split along 3*C dim
    q_x = q_x.view(B, T, num_heads, C // num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
    k_x = k_x.view(B, T, num_heads, C // num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
    v_x = v_x.view(B, T, num_heads, C // num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
    return q_x, k_x, v_x

MASK = ~torch.tril(torch.ones(sequence_length, sequence_length, requires_grad=False, dtype=bool))
# TODO: move to reference module
def self_attention(q_x, k_x):
    import torch
    import torch.nn.functional as functional

    w0 = q_x @ k_x.transpose(-2, -1)
    w1 = w0 / torch.sqrt(torch.tensor(d_model)) # d_k = d_model
    w2 = w1.masked_fill(MASK, float('-inf'))
    # w3 = functional.softmax(w2, dim=-1) # (B, T, T)
    w3 = softmax(w2)
    return w3

def value(a, v):
    r = a @ v
    return r.transpose(1,2).reshape(B.shape[0], T.shape[0], C.shape[0])

def attention(p, x):
    import torch.nn.functional as functional
    B, T, C = x.size()

    # attention heads + splitting
    q_x, k_x, v_x = heads_splitter(p, x)

    # self_attention
    w3 = self_attention(q_x, k_x)

    # w4 = w3 @ v_x
    # w5 = w4.transpose(1,2).contiguous().view(B, T, C) # reassemble heads
    w5 = value(w3, v_x)
    return w5

def block(p, x0):
    x1 = layer_norm(x0)
    x2 = attention(p, x1)
    x3 = x0 + x2
    return x3
