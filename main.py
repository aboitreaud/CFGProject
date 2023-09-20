import time

import torch
import torch.nn as nn
import numpy as np
from model import GPT, GPTConfig
from context_free_grammar import CFG
from grammar import Grammar

if __name__ == "__main__":
    cfg = CFG(L=3, ns=[1, 3, 3, 10], nr=[2, 2, 2], T=[8, 8, 8])

    nspl = 1000
    # Generate ns[0] * nspl sentences in total
    # Vocab size is ns[L]
    # Each sentence is a tensor of shape (T_0,T_1,...,T_{L-1})
    # Flattened sentences are of length np.prod(T) product of length of the rules at each level

    sentence_length = np.prod(cfg.T)
    start = time.time()
    config = GPTConfig(vocab_size=cfg.ns[-1], n_embd=384, n_head=6, n_layer=6)
    m = GPT(config)
    m = nn.DataParallel(m)
    m.to(config.device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    max_iters = 1
    eval_interval = 1
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    # data loading
    def get_batch(config: GPTConfig = GPTConfig()):
        sentence = cfg.sample_flattened(1)[0][0].view(sentence_length)  # reshape in a 1d tensor
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(0, sentence_length - config.block_size, size=(config.batch_size,))
        x = torch.stack([sentence[i: i + config.block_size] for i in ix])
        y = torch.stack([sentence[i+1: i + config.block_size + 1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        m.eval()
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch()
            logits = m(X)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out["val"] = losses.mean()
        m.train()
        return out

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: val loss {losses['val']:.4f}"
            )
        if iter % 1000 == 0 and iter > 1:
            learning_rate /= 2
            optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

        # sample a batch of data
        xb, yb = get_batch()

        # evaluate the loss
        logits = m(xb)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
        loss.backward()
        optimizer.step()
    end1 = time.time()
    model = m.module
    # generate n_gen sentences from the model and check their correctness
    n_gen = 10
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    mistakes = []
    for i in range(n_gen):
        gen_sentence = model.generate(context, max_new_tokens=sentence_length)[0].tolist()
        # remove root symbol at the beginning
        _, err = cfg.collapse_and_get_err(torch.tensor(gen_sentence[1:]).view(*cfg.T))
        mistakes.append(err)
    print(mistakes)
    end2 = time.time()
    print(end1 - start)
    print(end2 - start)


