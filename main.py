import torch
import numpy as np
from model import GPT, GPTConfig
from context_free_grammar import CFG
from grammar import Grammar

if __name__ == "__main__":
    cfg = CFG(L=3, ns=[1, 3, 3, 3], nr=[2, 2, 2], T=[8, 8, 8])

    nspl = 1000
    # Generate ns[0] * nspl sentences in total
    # Vocab size is ns[L]
    # Each sentence is a tensor of shape (T_0,T_1,...,T_{L-1})
    # Flattened sentences are of length np.prod(T) product of length of the rules at each level

    #s, labels = cfg.sample_flattened(nspl)
    # for i, sentence in enumerate(s):
    #     print('Sentence {i} is: {sent}'.format(i=i, sent=sentence.detach().numpy()))
    # print(s.shape, labels)

    # gram = Grammar(n_levels=4, n_symbols=[1, 3, 3, 3], n_children=[5, 5, 5], n_rules=[2, 2, 2])
    # print(gram.generate_n_sentences(nspl=1))

    # get rid of labels and reshape the sentences.
    train_data = cfg.sample_flattened(int(.9*nspl))[0][0]
    val_data = cfg.sample_flattened(int(.1*nspl))[0][0]
    sentence_length = np.prod(cfg.T)

    config = GPTConfig(vocab_size=cfg.ns[-1], n_embd=384, n_head=6, n_layer=6)
    model = GPT(config)
    m = model.to(config.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    max_iters = 50
    eval_interval = 10
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # data loading
    print(train_data.shape)
    def get_batch(split: str, config: GPTConfig = GPTConfig()):
        data = train_data if split == "train" else val_data
        print('LENDATA: ', len(data))
        print('blk size: ', config.block_size)
        # generate a small batch of data of inputs x and targets y
        chosen_sentence = np.random.randint(len(data))
        ix = torch.randint(0, sentence_length - config.block_size, size=(config.batch_size,))
        x = torch.stack([data[chosen_sentence, i: i + config.block_size] for i in ix])
        y = torch.stack([data[chosen_sentence, i+1: i + config.block_size + 1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        print('x.shape', x.shape)
        print('y.shape', y.shape)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(m.generate(context, max_new_tokens=500)[0].tolist())
