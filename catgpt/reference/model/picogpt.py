""" picoGPT model: a stripped-down version of nanoGPT https://github.com/karpathy/nanoGPT
    (adapted under MIT license)
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from catgpt.util import product
import catgpt.reference.function as reference

sequence_length = 32

MASK = ~torch.tril(torch.ones(sequence_length, sequence_length, requires_grad=False, dtype=bool))

class SelfAttention(nn.Module):
    """ vectorised self-attention, adapted & simplified from nanoGPT """
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.attention = nn.Linear(d_model, 3*d_model, bias=False)

        self.d_model = d_model
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        q_x, k_x, v_x = self.attention(x).split(self.d_model, dim=2) # split along 3*C dim
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)

        # w0 = q_x @ k_x.transpose(-2, -1)
        # w1 = w0 / torch.sqrt(torch.tensor(self.d_model)) # d_k = d_model
        # w2 = w1.masked_fill(MASK, float('-inf'))
        # # w3 = functional.softmax(w2, dim=-1) # (B, T, T)
        # w3 = reference.softmax(w2)
        w3 = reference.query_key(q_x, k_x)

        # w4 = w3 @ v_x
        # w5 = w4.transpose(1,2).reshape(B, T, C)
        w5 = reference.value(w3, v_x)
        return w5

class Block(nn.Module):
    """ Encoder block with causal self-attention """

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        head_size, rem = divmod(d_model, num_heads)
        assert rem == 0 # check it divides evenly
        self.attention = SelfAttention(d_model, num_heads)

    def forward(self, x0):
        x1 = reference.layer_norm(x0)
        x2 = self.attention(x1)
        x3 = x0 + x2
        return x3

class GPT(nn.Module):
    def __init__(self, device, vocab_size, d_model, num_heads=8, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        # self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.token_embedding = nn.Linear(vocab_size, d_model, bias=False)
        # self.position_embedding_table = nn.Embedding(sequence_length, d_model)
        self.blocks = nn.Sequential(*[Block(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx: a B-sized batch of sequences length T.
        # Each element is a single token stored as an integer.
        B, T, V = idx.shape

        token_embedding = self.token_embedding(idx) # (B, T, C)
        # position_embedding = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)

        x0 = token_embedding # + position_embedding
        x1 = self.blocks(x0)
        x2 = reference.layer_norm(x1)
        logits = self.linear(x2)

        loss = None
        if targets is not None:
            probs = reference.softmax(logits)
            log_probs = torch.log(probs)
            log_probs_flat = log_probs.view(B*T, self.vocab_size)
            targets_flat = targets.reshape(B*T)
            loss = F.nll_loss(log_probs_flat, targets_flat)

        return logits, loss
