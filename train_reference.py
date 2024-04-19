import sys
import torch

from catgpt.dataset import Dataset, random_batch
from catgpt.reference.model.picogpt import GPT

from catgpt.reference import softmax
from catgpt.settings import *

def main():
    torch.manual_seed(0)

    # learning_rate = 3e-4 # for AdamW
    learning_rate = 1e-1 # for SGD

    # max_iters = 10_000
    max_iters = 10
    eval_interval = 100

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    dataset = Dataset("input.txt")

    model = GPT(device, dataset.vocab_size, d_model=384, num_heads=6, num_layers=6)
    m = model.to(device)
    encoded_text = dataset.encoded_text.to(device)

    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # re-seed generator for reproducibility with train.py
    torch.manual_seed(0)

    # load existing weights if they exist
    try:
        model.load_state_dict(torch.load('model.pt'))
    except FileNotFoundError:
        print("no weights found!")

    for i in range(max_iters):
        if i % 100 == 0:
            print("saving weights...")
            torch.save(model.state_dict(), 'model.pt')

        xb, yb = random_batch(batch_size, sequence_length, encoded_text)
        xb_hot = torch.eye(dataset.vocab_size)[xb]

        logits, loss = model(xb_hot, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(i, loss.item())


if __name__ == "__main__":
    main()
