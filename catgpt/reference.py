""" reference implementations of forward and reverse maps """
from catgpt.util import product

# not sure what internal impl. is. Hope this works!
def mean(x):
    n_dim = product(x.shape)
    return x.sum() / n_dim

def r_mean(x, dy):
    N = product(x.shape)
    return (1/N)*dy

# reference implementation according to docs
# https://pytorch.org/docs/stable/generated/torch.var.html
def var(x, correction=1):
    N = product(x.shape)
    m = mean(x)
    d = max(0, N - correction)
    return (1 / d) * ((x - m)**2).sum()

# https://math.stackexchange.com/questions/2836083/derivative-of-the-variance-wrt-x-i#2887894
def r_var(x, dy, correction=1):
    N = product(x.shape)
    d = max(0, N - correction)
    return (x - x.mean()) * (2 / d)

