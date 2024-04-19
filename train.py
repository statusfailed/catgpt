import sys
import torch

from catgrad import layers, compile_model
from catgrad.signature import op
from catgrad.combinators import identity
from catgrad.target.python.array_backend.torch import Torch

# import catgpt.dataset as dataset
from catgpt.dataset import Dataset, random_batch
from catgpt.layer import Softmax
from catgpt.model.picogpt import picogpt
from catgpt.settings import *

def main():
    learning_rate = 1e-1 # for SGD
    max_iters = 10_000
    num_blocks = 6
    eval_interval = 100
    device = 'cpu'

    dataset = Dataset("input.txt")

    model = picogpt(B, T, C, vocab_size=vocab_size, num_heads=num_heads, num_blocks=num_blocks)
    encoded_text = dataset.encoded_text.to(device)

    # NOTE: we put a softmax on the output of the model here.
    CompiledModel, ParamType, model_ast = compile_model(model >> op(Softmax(B+T, V)), identity, identity)
    run = CompiledModel(Torch)

    # load torch-initialized weights for reproducibility
    torch_p = torch.load('model.pt')
    ps = [ v for v in torch_p.values() ]
    for i in range(0,len(ps)):
        # torch has opposite convention for matrices (except for Embedding)
        ps[i] = ps[i].T

    # set manual seed to get same behaviour as train_reference.py
    torch.manual_seed(0)
    with torch.no_grad():
        for i in range(max_iters):
            # get batch & one-hot it.
            xb, yb = random_batch(batch_size, sequence_length, encoded_text)
            xb_hot = torch.eye(dataset.vocab_size)[xb]

            [probs] = run.predict(*ps, xb_hot)

            # We compute displacement manually here by manually backpropping loss through NLL and log.
            log_probs = torch.log(probs) # grad ~= -0.0005
            log_probs_flat = log_probs.view(batch_size*sequence_length, vocab_size)
            targets_flat = yb.reshape(batch_size*sequence_length)

            # we don't actually need to compute this, we just do it to compare
            # to the reference implementation.
            loss = torch.nn.functional.nll_loss(log_probs_flat, targets_flat)

            # compute log_probs.grad
            dlog_probs = torch.zeros_like(log_probs_flat)
            dlog_probs[torch.arange(len(targets_flat)), targets_flat] = - 1 / (batch_size * sequence_length)
            dlog_probs = dlog_probs.reshape(batch_size, sequence_length, vocab_size)

            # compute probs.grad
            dprobs = (1 / probs) * dlog_probs

            # update params
            dps = run.rev_p(*ps, xb_hot, dprobs)
            for p, dp in zip(ps, dps):
                p.add_(-learning_rate * dp)

            # print loss at each iteration; compare to reference implementation
            print(i, 'loss', loss.item())

            if i % 100 == 0:
                print(i, 'saving weights...')
                torch.save(ps, 'catgpt.pt')


if __name__ == "__main__":
    main()
