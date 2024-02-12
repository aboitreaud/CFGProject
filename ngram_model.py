import numpy as np
import torch
from collections import defaultdict
from context_free_grammar import CFG


class NGramModel:
    def __init__(self, n, cfg: CFG):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.cfg = cfg

    def block_separated_ngrams(self, sentence):
        assert sentence.size(1) == np.prod(self.cfg.T)
        block_size = self.cfg.T[-1]

        nb_blocks = np.prod(self.cfg.T) // block_size
        for i in range(nb_blocks):
            for j in range(self.n, block_size):
                context = tuple(sentence[i*block_size+j-self.n-1: i*block_size+j].tolist())
                next_token = sentence[i*block_size+j]
                self.ngrams[context][next_token] += 1

    def simple_ngrams(self, sentence):
        assert sentence.size() == np.prod(self.cfg.T)

        # Generate n-grams and count occurrences
        for i in range(len(sentence) - self.n + 1):
            ngram = sentence[i:i+self.n]
            context = tuple(ngram[:-1].tolist())
            next_token = ngram[-1].item()
            self.ngrams[context][next_token] += 1

    def predict_next_token(self, context):
        # Given a context, predict the next token
        context = tuple(context.tolist())
        next_tokens = list(self.ngrams[context].keys())
        probabilities = [self.ngrams[context][token] for token in next_tokens]
        if np.sum(probabilities) == 0:
            next_token = np.random.randint(self.cfg.ns[-1])
        else:
            probabilities = np.array(probabilities) / np.sum(probabilities)
            next_token = np.random.choice(next_tokens, p=probabilities)
        return next_token

    def gen_sentence(self, context, sentence_length):
        # Given a tensor of contexts, complete them to form full sentences of size sentence_length
        n_gen = context.size(0)
        sentences = torch.zeros(n_gen, sentence_length)
        sentences[:, :context.size(1)] = context
        for s in range(n_gen):
            for i in range(context.size(0), sentence_length):
                context = sentences[s, i-self.n:i]
                sentences[s, i] = self.predict_next_token(context)
        return sentences

    def print_dict(self):
        # Printing the defaultdict
        for key_outer, inner_dict in self.ngrams.items():
            print(f"Outer Key: {key_outer}")
            for key_inner, value in inner_dict.items():
                print(f"Inner Key: {key_inner}, Value: {value}")

    def compute_perplexity(self, test_set, smoothing_factor, vocab_size):
        total_log_prob = 0
        num_tokens = 0

        for s in range(test_set.size(1)):
            sentence = test_set[0, s, :]

            for i in range(self.n - 1, sentence.size(0)):
                context = tuple(sentence[i - self.n + 1:i].tolist())
                next_token = sentence[i].item()
                # Compute the probability of the next token given the context
                prob = (self.ngrams[context][next_token] + smoothing_factor) / \
                    (sum(self.ngrams[context].values()) + (vocab_size * smoothing_factor))

                # Update total log probability and number of tokens
                total_log_prob += np.log(prob)
                num_tokens += 1

        perplexity = np.exp(-total_log_prob / num_tokens)
        return perplexity
