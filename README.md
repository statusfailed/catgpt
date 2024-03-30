# catgpt

A GPT model implemented with [catgrad](https://github.com/statusfailed/catgrad).

The architecture is a very-stripped-down [nanoGPT](https://github.com/karpathy/nanoGPT).
Several layers have been removed which reduce the quality of results.
In order of importance, the removed layers are:

- [ ] Positional encodings (!)
- [ ] self-attention output layer
- [ ] `FeedForward` after attention in each `block`
- [ ] Learnable layer norm weights
