""" reference implementations of forward and reverse maps """
from catgpt.util import product

# not sure what internal impl. is. Hope this works!
def mean(x, dims):
    n_dim = product(x[dims].shape)
    return x.sum(dims) / n_dim

def r_mean(x, dy, dims):
    N = product(x[dims].shape)
    return (1/N)*dy

# reference implementation according to docs
# https://pytorch.org/docs/stable/generated/torch.var.html
def var(x, dims, correction=1):
    N = product(x[dims].shape)

    # This is a torch one-liner, but catgrad core will require explicit broadcasting
    # m = x.mean(dims, keepdim=True)
    m = mean(x, dims) # (N₀ N₁ ..., T₀, T₁, ...)
    m = m.view(tuple(m.shape + (1,)*len(dims))) # broadcast to (N₀, N₁, ..., 1, 1, ..., 1)

    var_sum = ((x - m)**2).sum(dims)
    d = max(0, N - correction)
    return var_sum / d

# https://math.stackexchange.com/questions/2836083/derivative-of-the-variance-wrt-x-i#2887894
def r_var(x, dy, correction=1):
    N = product(x.shape)
    d = max(0, N - correction)
    return (x - x.mean()) * (2 / d)

