import torch
from typing import List

from catgrad import compile_model
from catgrad.combinators import identity
from catgrad.signature import NdArrayType, Dtype, op
from catgrad.target.python.array_backend.torch import Torch

from catgpt import reference
from catgpt.model.picogpt import picogpt
from catgpt.settings import *

# useful types
B, T, C, V = [ NdArrayType((i,), Dtype.float32) for i in [batch_size, sequence_length, d_model, vocab_size] ]
B1 = NdArrayType((1,), Dtype.float32) # batch size 1 for predictions


# decode model outputs to chars
chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
atoi = { c: i for i, c in enumerate(chars) }
itoa = { i: c for c, i in atoi.items() }
def decode(x: List[int]):
    return "".join(itoa[i] for i in x)

def generate(fwd, generator=None):
    context = torch.zeros(sequence_length, dtype=int)

    while True:
        # cap context at sequence_length
        context = context[-sequence_length:]

        # one-hot the inputs (model uses Linear not Embedding layer)
        x = context.reshape(1, 32)
        x_hot = torch.eye(vocab_size)[x]

        logits = fwd(x_hot)
        final_logit = logits[:, -1, :] # all batches, last elem of seq, all probabilities
        probs = reference.softmax(final_logit)

        context_next = torch.multinomial(probs.squeeze(0), num_samples=1, generator=generator)
        context = torch.cat((context, context_next))
        print(decode(context_next.tolist()), end='', flush=True)

def main():
    G = torch.Generator()
    G.manual_seed(42)

    num_blocks=6

    model = picogpt(B1, T, C, vocab_size=vocab_size, num_heads=num_heads, num_blocks=num_blocks)
    CompiledModel, ParamType, model_ast = compile_model(model, identity, identity)
    
    # load params
    # NOTE: dodgy hack here: we're relying on serialized params being in the same order as the model!
    torch_p = torch.load('model.pt')
    ps = [ v for v in torch_p.values() ]
    for i in range(0,len(ps)):
        ps[i] = ps[i].T # torch has opposite convention for matrices, except for Embedding!?

    c = CompiledModel(Torch)
    x = torch.ones((B1+T+V).shape, dtype=torch.float32)

    catgpt_fwd = lambda x_hot: c.predict(*ps, x_hot)[0]
    generate(catgpt_fwd, generator=G)

if __name__ == "__main__":
    main()
