import numpy as np
import torch
from collections import defaultdict
from context_free_grammar import CFG


class NGramModel:
    def __init__(self, n, smoothing_factor, cfg: CFG):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.cfg = cfg

    def simple_ngrams(self, sentence):
        assert sentence.size() == np.prod(self.cfg.T)

        # Generate n-grams and count occurrences
        for i in range(len(sentence) - self.n + 1):
            ngram = sentence[i:i+self.n]
            context = tuple(ngram[:-1].tolist())
            next_token = ngram[-1].item()
            self.ngrams[context][next_token] += 1

    def predict_next_token(self, context):
        assert len(context) == self.n - 1
        # Given a context, predict the next token
        context = tuple(context.tolist())
        next_tokens = list(self.ngrams[context].keys())
        probabilities = [self._calculate_smoothed_prob(context, token) for token in next_tokens]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        next_token = np.random.choice(next_tokens, p=probabilities)
        return next_token

    def gen_sentence(self, context, sentence_length):
        # Given a tensor of contexts, complete them to form full sentences of size sentence_length
        n_gen = context.size(0)
        context_len = context.size(1)
        sentences = torch.zeros(n_gen, sentence_length)
        sentences[:, :context_len] = context
        for s in range(n_gen):
            for i in range(context_len, sentence_length):
                context = sentences[s, i-self.n + 1:i]
                sentences[s, i] = self.predict_next_token(context)
        return sentences

    def _calculate_smoothed_prob(self, context, next_token):
        # Calculate smoothed probability of next_token given context
        prob = (self.ngrams[context][next_token] + self.smoothing_factor) / \
            (sum(self.ngrams[context].values()) + (self.cfg.ns[-1] * self.smoothing_factor))
        return prob

    def compute_perplexity(self, test_set):
        total_log_prob = 0
        num_tokens = 0

        for s in range(test_set.size(1)):
            sentence = test_set[0, s, :]

            for i in range(self.n - 1, sentence.size(0)):
                context = tuple(sentence[i - self.n + 1:i].tolist())
                assert len(context) == self.n - 1
                next_token = sentence[i].item()
                # Compute the probability of the next token given the context
                prob = self._calculate_smoothed_prob(context, next_token)
                # Update total log probability and number of tokens
                total_log_prob += np.log(prob)
                num_tokens += 1

        perplexity = np.exp(-total_log_prob / num_tokens)
        return perplexity

    class HierarchicalNGram:

        def __init__(self, cfg: CFG) -> None:
            self.cfg = cfg
            self.ngram = {lev: defaultdict(lambda: defaultdict(int)) for lev in range(self.cfg.L)}
            # counter for attributing new symbols at each level
            self.symbol_counters = {lev: 0 for lev in range(self.cfg.L)}

        def simple_ngrams(self, sentence):
            for lev in range(self.cfg.L - 1, -1, -1):
                n = self.cfg.T[lev]
                assert sentence.size() == np.prod(self.cfg.T[:lev])
                upper_level_sentence = torch.zeros(sentence.size()//n)
                # Generate n-grams and count occurrences
                j = 0
                for i in range(0, len(sentence) - n + 1, n):
                    context = tuple(sentence[i:i+n].tolist())
                    next_group = tuple(sentence[i+n, i+2*n].tolist())
                    self.ngrams[context][next_group] += 1

                    # Get the upper level symbol corresponsing to context
                    upper_level_symbol = self.ngram[lev].get(context, -1)

                    # In case this group of n symbols at level lev has never been seen,
                    # map it to a new upper level symbol and increment the new symbol counter
                    if upper_level_symbol == -1:
                        upper_level_symbol = self.symbol_counters[lev]
                        self.symbol_counters[lev] += 1
                    # Append it to the upper-level sentence
                    upper_level_sentence[j] = upper_level_symbol
                    j += 1
                sentence = upper_level_sentence
