import wandb
import numpy as np

from ngram_model import NGramModel
from context_free_grammar import CFG

wandb.login()

max_epochs = 500
batch_size = 5000
eval_iters = 100

# cfg = CFG(L=5, ns=[1, 3, 3, 3, 9, 10], nr=[2, 2, 2, 2, 2], T=[2, 2, 4, 4, 8])
cfg = CFG(L=3, ns=[1, 10, 100, 1000], nr=[1, 1, 1], T=[8, 8, 8])

n = 3
k_smoothing = 0.1
m = NGramModel(n, k_smoothing, cfg)

test_set = cfg.sample_flattened(eval_iters)[0]


def train(model: NGramModel):
    running_acc_mean = []
    for epoch in range(max_epochs):
        sentences = cfg.sample_flattened(batch_size)[0]
        for s in range(sentences.size(1)):
            model.simple_ngrams(sentences[0, s, :])
        perplexity = model.compute_perplexity(test_set)

        gen_sentences = model.gen_sentence(test_set[0, :, :model.n], np.prod(cfg.T))
        gen_sentences = gen_sentences.view([eval_iters] + cfg.T)
        acc = cfg.frac_of_gramatically_correct_sentences(gen_sentences)
        running_acc_mean.append(acc*100)

        # compute per-level errors
        # a sentence can only be good at level i if it was good at all levels beteewn L and i+1
        correct_sentences = np.zeros(cfg.L)
        for sentence in gen_sentences:
            _, err = cfg.collapse_and_get_err(sentence)
            for i in range(len(err) - 1, -1, -1):
                if err[i].sum() != 0:
                    break
                else:
                    correct_sentences[i] += 1
        errors = np.array(correct_sentences) / eval_iters * 100

        log_dict = {"nb sentences seen": (epoch + 1) * batch_size,
                    "perplexity": perplexity,
                    "accuracy": acc * 100,
                    }
        for i, err in enumerate(errors):
            log_dict[f'% of correct sentences at level {i}'] = err
        wandb.log(log_dict)
        print(f"Epoch [{epoch+1}/{max_epochs}] perplexity: {perplexity}, accuracry: {acc*100}%")
    # log the mean accuracy over the last 50 epochs
    wandb.log({"Accuracy mean over 50 last epochs": np.mean(running_acc_mean[-50:])})

conf = {'model.n': m.n,
        'model.k_smoothing': k_smoothing,
        'cfg': {'L': cfg.L,
                'ns': cfg.ns,
                'nr': cfg.nr,
                'T': cfg.T},
        'eval_iters': eval_iters,
        'batch_size': batch_size}
wandb.init(project='CFG-ngram_first_try', name=f'{m.n}-gram', config=conf)

# wandb.watch(m, log='all')
train(m)
wandb.finish()
