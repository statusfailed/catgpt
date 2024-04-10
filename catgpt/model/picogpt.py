from catgrad.signature import NdArrayType, Dtype, obj
from catgrad.combinators import identity
from catgrad.layers import linear

from catgpt.layer.attention import block
from catgpt.layer.softmax import Softmax
from catgpt.layer.statistics import layer_norm

from catgrad.bidirectional.operation import discard, copy

def picogpt(B, T, C, vocab_size: int, num_heads: int, num_blocks: int):
    assert num_blocks >= 0
    V = NdArrayType((vocab_size,), Dtype.float32)

    token_embedding = linear(B+T, V, C)

    blocks = identity(obj(B+T+C))
    for i in range(num_blocks):
        blocks = blocks >> block(B, T, C, num_heads)
    norm = layer_norm(B+T, C)
    output = linear(B+T, C, V) # project back into vocab size

    return token_embedding >> blocks >> norm >> output
