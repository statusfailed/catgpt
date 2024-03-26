""" reference implementations of forward and reverse maps """
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
    z = x - x.max()
    num = z.exp()
    den = num.sum(dim=-1, keepdim=True)
    return num / den
