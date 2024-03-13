# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch
from context_free_grammar import CFG


# %%
class CFGBacktracker:
    def __init__(self, cfg, nb_allowed_differing_words=1) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_allowed_differing_words = nb_allowed_differing_words
        # for each level, we store the rules at each level
        self.backtracked_rules = {lev: {} for lev in range(self.cfg.L)}

    def find_closest_pair(self, level, sentences):
        # sentences = sentences.to(self.device)

        distances = (sentences.unsqueeze(1) ^ sentences.unsqueeze(0)).sum(dim=2)

        # Find the min distance, among sentences not exactly equal, so with > 0 distance
        zero_mask = (distances == 0)
        # Mask the 0 with the maximal hamming distance of two sentences of the grammar
        distances[zero_mask] = np.prod(self.cfg.T) * (self.cfg.ns[-1] + 1)
        min_index = distances.argmin(dim=None)
        i = min_index // distances.shape[0]
        j = min_index % distances.shape[0]
        i, j = i.item(), j.item()

        # Check how many words are different in the two sentences
        diff_mask = torch.ne(sentences[i], sentences[j])
        num_differing_items = diff_mask.sum().item()
        # Return pair of sentences if they differ by nb_allowed_differing_words words max
        if num_differing_items <= self.nb_allowed_differing_words * self.cfg.T[level]:
            return i, j
        else:
            return None

    def find_synonyms(self, sentence1, sentence2, word_size):
        # Initialize list to store the tuples of synonyms 
        synonyms = []

        for c1, c2 in zip(sentence1.split(word_size), sentence2.split(word_size)):
            if torch.any(c1 != c2):
                synonyms.append((c1, c2))
        return synonyms

    def apply_synonyms_change(self, synonyms, sentences):
        """
        Replaces the synonyms[1] subsections in all sentences with synonyms[0].

        Returns:
            torch.Tensor: The modified sentences with the synonym2 replaced with synonym1 in every sentences.
        """
        for (synonym1, synonym2) in synonyms:
            new_sentences = torch.zeros_like(sentences)
            # Check if the synonyms are valid
            assert len(synonym1) == len(synonym2), "Synonyms must have the same length"
            assert synonym1.dim() == 1 and synonym2.dim() == 1, "Synonyms must be 1-dimensional"
            word_size = synonym1.size(0)
            new_sentences = torch.zeros_like(sentences)
            for i in range(sentences.size(0)):
                words = list(sentences[i].split(word_size))
                for j, word in enumerate(words):
                    if word.equal(synonym2):
                        words[j] = synonym1
                new_sentences[i] = torch.cat(list(words))
            return new_sentences
        return sentences

    def find_all_synonym_pairs(self, level, sentences):
        # Find synonyms at current level
        pairs_of_synonyms = []
        iter = 0
        min_dist_pair = self.find_closest_pair(level=level, sentences=sentences)
        while min_dist_pair is not None:
            synonyms = self.find_synonyms(sentences[min_dist_pair[0]], sentences[min_dist_pair[1]], self.cfg.T[level])
            for t in synonyms:
                pairs_of_synonyms.append(t)
            # Modifiy sentences by merging synonyms
            sentences = self.apply_synonyms_change(synonyms, sentences)
            iter += 1
            # Compute the pair for the next iter, that will occur only if it's not None
            min_dist_pair = self.find_closest_pair(level=level, sentences=sentences)
            if min_dist_pair is not None:
                if min_dist_pair[0] == min_dist_pair[1]:
                    return pairs_of_synonyms, sentences
        return pairs_of_synonyms, sentences

    def store_rules(self, level, pairs_of_synonyms):
        # Store rules by arbitrarily attributing groups of synonyms a word of the level above
        generation_rules = {}
        word_to_upper_level_symbol = {}
        for i in range(len(pairs_of_synonyms)):
            generation_rules[i] = pairs_of_synonyms[i]
            for w in pairs_of_synonyms[i]:
                word_to_upper_level_symbol[tuple(w.tolist())] = i
        self.backtracked_rules[level] = generation_rules
        return word_to_upper_level_symbol

    def build_upper_level_seq(self, level, curr_level_sentences, word_to_upper_level_symbol):
        # Create the sentences at the level above
        old_sentences = torch.unique(curr_level_sentences, dim=0)
        upper_level_sentences = torch.zeros((old_sentences.size(0), old_sentences.size(1) // self.cfg.T[level]), dtype=torch.long)
        for k in range(old_sentences.size(0)):
            words = torch.split(old_sentences[k], self.cfg.T[level])
            for i, w in enumerate(words):
                upper_level_sentences[k, i] = word_to_upper_level_symbol[tuple(w.tolist())]
        return upper_level_sentences

    def backtrack_cfg(self, sentences):
        for lev in range(self.cfg.L - 1, 0, -1):
            print(f'Working on level {lev}')
            pairs_of_synonyms, sentences = self.find_all_synonym_pairs(level=lev, sentences=sentences)
            if len(pairs_of_synonyms) == 0:
                print(f"Failed finding synonyms, no sentence in the corpus differs by only one word, stopping at level {lev}")
                return
            print(pairs_of_synonyms)
            word_to_upper_level_symbol = self.store_rules(lev, pairs_of_synonyms)
            sentences = self.build_upper_level_seq(lev, sentences, word_to_upper_level_symbol)
            print(self.backtracked_rules)
# %%
cfg = CFG(L=3, ns=[1, 3, 9, 10], nr=[2, 2, 2], T=[4, 4, 2])
sentences = cfg.sample_flattened(15000)[0].squeeze(0)

# %%
backtracker = CFGBacktracker(cfg)
backtracker.backtrack_cfg(sentences)
