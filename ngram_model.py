import numpy as np
import torch
from collections import defaultdict
from context_free_grammar import CFG
from sklearn.cluster import KMeans


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
        # Leaves are considered level 0 of the grammar, root is level L
        # Mapping between groups of symbols at level i and (arbitrarily) chosen symbols at level i+1
        self.ngrams = {lev: {} for lev in range(self.cfg.L-1)}
        # Mapping between symbols at level i and groups of symbols of level i-1, used for generation
        self.reverse_dict = {lev: {} for lev in range(1, self.cfg.L)}
        # Counter for attributing new symbols at each level
        self.symbol_counters = {lev: 0 for lev in range(self.cfg.L)}
        self.root_expansion_freq = defaultdict(int)

    def simple_ngrams(self, sentence):
        for lev in range(self.cfg.L-1):
            n = self.cfg.T[lev]
            # TODO: check in the CFG code how T is ordered [4,2] != [2,4] for e.g.
            assert sentence.size() == np.prod(self.cfg.T[lev:])
            upper_level_sentence = torch.zeros(sentence.size(0)//n)
            # Generate n-grams and count occurrences
            j = 0
            for i in range(0, len(sentence) - n + 1, n):
                context = tuple(sentence[i:i+n].tolist())
                next_group = tuple(sentence[i+n: i+2*n].tolist())

                curr = self.ngrams[lev].get(context, NgramEntry(context))

                # Increment the next-group counter for the current context
                curr.next_tokens[next_group] += 1

                # In case this group of n symbols at level lev has never been seen,
                # map it to a new upper level symbol and increment the new symbol counter
                if curr.upper_level_symbol == -1:
                    curr.upper_level_symbol = self.symbol_counters[lev]
                    self.symbol_counters[lev] += 1
                # Append it to the upper-level sentence
                upper_level_sentence[j] = curr.upper_level_symbol

                # Put the element in both dicts
                self.ngrams[lev][context] = curr
                self.reverse_dict[lev+1][curr.upper_level_symbol] = context
                j += 1
            sentence = upper_level_sentence
        # Add the second to last level sentence to the root expansion dict
        self.root_expansion_freq[tuple(sentence.tolist())] += 1

    def generate_sentence(self, nspl, verbose=False):
        sentences = torch.zeros((nspl, np.prod(self.cfg.T)))
        for iter in range(nspl):
            # Randomly choose below-root level sequence
            prob = [self.root_expansion_freq[c] / sum(self.root_expansion_freq.values())
                    for c in self.root_expansion_freq.keys()]
            idx = np.random.choice(len(self.root_expansion_freq), p=prob)
            seq = list(self.root_expansion_freq.keys())[idx]

            # Expand that sequence until the leaf level
            for lev in range(self.cfg.L - 1, 0, -1):
                if verbose:
                    print("Expanding level", lev)
                next_level_seq = []
                for symbol in seq:
                    random_idx = np.random.randint(0, len(self.reverse_dict[lev][symbol]))
                    if verbose:
                        print(random_idx, self.reverse_dict[lev][symbol])
                    next_level_seq += [self.reverse_dict[lev][symbol][random_idx]]
                seq = next_level_seq
            sentences[iter, :] = torch.tensor(seq)
        return sentences

    def get_upper_level_symbol(self, lev, group):
        if group == ():
            return -1
        else:
            return self.ngrams[lev][group].upper_level_symbol

    def compress_upper_level(self, lev):
        # Going up the cfg tree from level lev+1 to level lev, we have created ns[l] * nr[l] symbols
        # There should only be ns[l] of them
        # This functions reduces the number of upper-level symbols from ns[l] * nr[l] to ns[l]
        # by applying K Means clustering on the dictionary of T[l+1]-grams of level l+1,
        # in which the groups of symbols are replaced by their upper-level symbol,
        # and looking for ns[l] clusters, we can match the identical symbols from level l.

        # Start by transforming the n-grams from level lev+1 into a dict of level lev symbols
        d = {}
        for group in self.ngrams[lev-1].keys():
            up = self.get_upper_level_symbol(lev-1, group)
            temp = {self.get_upper_level_symbol(lev-1, g): self.ngrams[lev-1][group].next_tokens[g]
                    for g in self.ngrams[lev-1][group].next_tokens.keys()}
            d[up] = dict(sorted(temp.items(), key=lambda x: x[0]))

        # Transform that dict into an array
        # There are self.cfg.ns[lev] * self.cfg.nr[lev] + 1 possible symbols including the termination empty tuple ()
        vectors = np.zeros((len(d), self.cfg.ns[lev] * self.cfg.nr[lev] + 1))
        for row in d.items():
            for i in row[1].keys():
                vectors[row[0], i] = row[1][i]
        kmeans = KMeans(n_clusters=self.cfg.ns[lev], n_init='auto', random_state=0)
        kmeans.fit(vectors)
        cluster_labels = kmeans.labels_

        # Update reverse_dict
        temp = {i: [] for i in range(self.cfg.ns[lev])}
        for old_symbol in self.reverse_dict[lev].keys():
            new_symbol = cluster_labels[old_symbol]
            temp[new_symbol].append(self.reverse_dict[lev][old_symbol])
        self.reverse_dict[lev] = temp

        # Update ngrams
        for ngram in self.ngrams[lev-1].keys():
            old_symbol = self.get_upper_level_symbol(lev-1, ngram)
            self.ngrams[lev-1][ngram].upper_level_symbol = cluster_labels[old_symbol]

        # If we are compressing below-root level, also update the stored root expansion sequences
        if lev == self.cfg.L - 1:
            new_dict = defaultdict(int)
            for old_seq in self.root_expansion_freq.keys():
                new_seq = []
                for s in old_seq:
                    new_seq.append(cluster_labels[int(s)])
                new_dict[tuple(new_seq)] += self.root_expansion_freq[old_seq]
            self.root_expansion_freq = new_dict


class NgramEntry:
    def __init__(self, context) -> None:
        self.next_tokens = defaultdict(int)
        self.upper_level_symbol = -1

    def __str__(self) -> str:
        s = f"Next tokens: {self.next_tokens}, Upper-level-symbol: {self.upper_level_symbol}"
        return s

    def __repr__(self) -> str:
        s = f"Next tokens: {self.next_tokens}, Upper-level-symbol: {self.upper_level_symbol}"
        return s
