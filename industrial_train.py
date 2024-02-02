import math
import torch
import torch.nn as nn
import numpy as np
from model import GPT, GPTConfig
from context_free_grammar import CFG
import wandb

wandb.login()

# data loading = sample new sentences to fill-in the mini-batch
def get_batch(config: GPTConfig = GPTConfig()):
    data, _ = cfg.sample(config.batch_size)  # dropping labels (useless for the task)
    N = data.shape[0]  # should be equal to config.batch_size
    data = data.view(N, sentence_length)  # flatten them to be (N,sentence_length)
    x = data[:, 0:sentence_length - 1]  # (bsz,sentence_length-1)
    y = data[:, 1:sentence_length].contiguous()  # (bsz,sentence_length-1)
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss(m, eval_iters=100):
    # This validation function samples a new batch of sentences and evaluates the loss of the model
    # Takes 20s for 100 sentences
    m.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        logits = m(X)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def eval_errors(m, context, n_gen=100):
    # for generating sentences from the model, we first sample real sentences from the grammar
    # then, the model is given the first 'context_length' symbols and asked to complete the sentence
    # Takes 40s for 100 sentences
    if isinstance(m, nn.DataParallel):
        m = m.module

    m.eval()
    context_length = context.size()[1]
    gen_sentences = m.generate(context, max_new_tokens=sentence_length - context_length, temperature=0.1)

    # compute accuracy
    gen_sentences = gen_sentences.view([n_gen] + cfg.T).cpu()
    acc = cfg.frac_of_gramatically_correct_sentences(gen_sentences)

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

    return acc, np.array(correct_sentences) / n_gen * 100


def get_lr(i, i_final):
    coeff = 0.5 * (1.0 + math.cos(math.pi * i / i_final))  # decays from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(m):
    print(f'One epoch is {training_parameters["batches_per_epoch"]} steps, ' +
          f'validation loss is computed at the end of every epoch and quality metric is ' +
          f'averaged over {training_parameters["quality_metric_iters"]} sentences')
    print(f'Will run for {training_parameters["num_epoch"]} epochs')
    total_num_iter = training_parameters['num_epoch'] * training_parameters['batches_per_epoch']
    # Build one context, to be reused each time we generate sentences in eval_errors
    context_length = 3
    context = cfg.sample(training_parameters['quality_metric_iters'])[0].view(
        training_parameters['quality_metric_iters'], sentence_length)[:, :context_length].to(config.device)
    running_acc_mean = []
    for epoch in range(training_parameters['num_epoch']):
        train_loss_sum = .0
        m.train()
        # determine and set the learning rate for this epoch
        lr = get_lr(epoch, training_parameters['num_epoch']) if decay_lr else max_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for iter in range(training_parameters['batches_per_epoch']):
            # sample a batch of data
            xb, yb = get_batch(config)
            # evaluate the loss
            optimizer.zero_grad()
            logits = m(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        # evaluate the loss on newly generated sentences at the end of every epoch
        train_loss = train_loss_sum / config.batch_size
        val_loss = estimate_loss(m, training_parameters["eval_iters"])
        acc, errors = eval_errors(m, context, training_parameters['quality_metric_iters'])
        running_acc_mean.append(acc*100)
        log_dict = {"nb sentences seen": (epoch + 1) * training_parameters['batches_per_epoch'] * config.batch_size,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "accuracy": acc * 100,
                    "learning_rate": optimizer.param_groups[0]["lr"]}
        for i, err in enumerate(errors):
            log_dict[f'% of correct sentences at level {i}'] = err

        if epoch % 10 == 0:
            formatted_stats = ', '.join(f'{key}: {value}' for key, value in log_dict.items())
            print(formatted_stats)
        wandb.log(log_dict)
    # log the mean accuracy over the last 50 epochs
    wandb.log({"Accuracy mean over 50 last epochs": np.mean(running_acc_mean[-50:])})


# training content
cfg = CFG(L=5, ns=[1, 3, 3, 3, 9, 10], nr=[2, 2, 2, 2, 2], T=[2, 2, 4, 4, 8])
sentence_length = np.prod(cfg.T)
for layer in [2]:
    for head in [1, 2, 4, 8, 12]:
        for dim in [8, 16, 32, 64, 128]:
            config = GPTConfig(vocab_size=cfg.ns[-1],
                               block_size=sentence_length - 1,
                               n_embd=dim,
                               n_head=head,
                               n_layer=layer,
                               batch_size=100)
            m = GPT(config)
            m = nn.DataParallel(m)
            m = m.to(config.device)

            # print the number of parameters in the model
            million_params = sum(p.numel() for p in m.parameters()) / 1e6
            print(million_params, "M parameters")
            max_lr = 6e-4  # max learning rate
            min_lr = max_lr / 10
            decay_lr = True

            weight_decay = 1e-1
            beta1 = 0.9
            beta2 = 0.95
            grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

            optimizer = m.module.configure_optimizers(weight_decay, max_lr, (beta1, beta2), device_type='cuda')

            training_parameters = {'num_epoch': 200,
                                   'batches_per_epoch': 50,
                                   'eval_iters': 100,
                                   'quality_metric_iters': 100,
                                   'learning_rate': 6e-4,
                                   'architecture': f'GPT {million_params:.3f}M',
                                   'model': config,
                                   'grammar': cfg.__dict__}
            wandb.init(project='CFG-HarderGrammar5', config=training_parameters,
                       name=f'{config.n_head}h {config.n_layer}l embd={config.n_embd} {million_params:.1f}M')

            wandb.watch(m, log='all')
            train(m)
            wandb.finish()
