# catgpt

A GPT model implemented with [catgrad](https://github.com/statusfailed/catgrad).

NOTE: requires `catgrad 0.2.0`

# Run

Install dependencies

    python -m venv venv
    source venv/bin/activate
    pip install torch catgpt==0.2.0

Get some pretrained model weights:

    ./get_weights.sh

Generate some text

    python generate.py

# Architecture

The architecture is a **very** stripped-down [nanoGPT](https://github.com/karpathy/nanoGPT).
Several layers have been removed which impact the quality of generated text.
In order of importance, the removed layers are:

- [ ] Positional encodings (!)
- [ ] self-attention output layer
- [ ] `FeedForward` after attention in each `block`
- [ ] Learnable layer norm weights
