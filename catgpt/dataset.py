""" 'Shakespeare' dataset code
    adapted from nanoGPT https://github.com/karpathy/nanoGPT
    under MIT license
"""
import torch
from typing import List

class Dataset:
    def __init__(self, input_file):
        self.input_file = input_file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # list of chars in the vocabulary
        chars = list(sorted(set(text)))
        self.vocab_size = len(chars)

        self._atoi = { c: i for i, c in enumerate(chars) }
        self._itoa = { i: c for c, i in self._atoi.items() }

        self.encoded_text = torch.tensor(self.encode(text), dtype=int)

    def encode(self, s: str):
        return [ self._atoi[c] for c in s ]

    def decode(self, x: List[int]):
        return "".join(self._itoa[i] for i in x)

def random_sequences(N, T, x, generator=None):
    """ Select N sequences of length T from the 1D tensor x. """
    K = len(x) # training data as one big sequence; total length
    assert x.dim() == 1
    # 0 1 2 ... T
    r = torch.arange(T) # 0 1 2  ... T

    # 10, 42, 992, ...
    starts = torch.randint(0, K-T, (N,), generator=generator)

    # 10, 11, 12, ... 10+T
    # 42, 43, 44, ..., 42+T
    # ⋮
    indices = starts.unsqueeze(1) + r.unsqueeze(0)
    sequences = x[indices]
    return sequences

def random_batch(N, T, v, generator=None):
    xy = random_sequences(N, T+1, v, generator)
    x = xy[..., :-1]
    y = xy[..., 1:]
    return x, y
