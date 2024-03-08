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
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # for each level, we store the 
        self.backtracked_rules = [{} for _ in range(self.cfg.L)]


    def find_closest_sentences(self, sentences, nb_allowed_differing_words, word_size):
        min_dist = self.hamming_distance(sentences[0], sentences[1])
        min_dist_pair = (-1, -1)
        nb_sentences = sentences.size(0)
        for i in range(nb_sentences):
            for j in range(i+1, nb_sentences):
                dist = self.hamming_distance(sentences[i], sentences[j])
                if   0 < dist < min_dist: # We are not interested in exact same sentences
                    # Check how many words are different in the two sentences
                    diff_mask = torch.ne(sentences[i], sentences[j])
                    num_differing_items = diff_mask.sum().item()
                    if num_differing_items <= nb_allowed_differing_words * word_size:
                        min_dist = dist
                        min_dist_pair = (i,j)
        return min_dist_pair

    def hamming_distance(self, vec1, vec2):
        """
        Computes the Hamming distance between two vectors.
        """
        assert len(vec1) == len(vec2), "Vectors must have the same length"

        # Convert each vector to a binary string
        bin_vec1 = ''.join(np.binary_repr(num, width=4) for num in vec1)
        bin_vec2 = ''.join(np.binary_repr(num, width=4) for num in vec2)

        # Count the number of differing bits
        distance = sum(c1 != c2 for c1, c2 in zip(bin_vec1, bin_vec2))

        return distance
    
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
    
    def find_all_synonym_pairs(self, sentences):
        # Find synonyms at current level
        min_dist_pair = (-2, -2)
        iter = 0
        pairs_of_synonyms = []
        while min_dist_pair != (-1, -1):
            min_dist_pair = self.find_closest_sentences(sentences, 1, 8)
            synonyms = self.find_synonyms(sentences[min_dist_pair[0]], sentences[min_dist_pair[1]], 8)
            for t in synonyms:
                pairs_of_synonyms.append(t)
            # Modifiy sentences by merging synonyms
            sentences = self.apply_synonyms_change(synonyms, sentences)
            iter += 1
        return pairs_of_synonyms
    
    def store_rules(self, pairs_of_synonyms, level):
        # Store rules by arbitrarily attributing groups of synonyms a word of the level above
        generation_rules = {}
        word_to_upper_level_symbol = {}
        for i in range(len(pairs_of_synonyms)):
            generation_rules[i] = pairs_of_synonyms[i]
            for w in pairs_of_synonyms[i]:
                word_to_upper_level_symbol[tuple(w.tolist())] = i
        self.backtracked_rules[level] = generation_rules
        print(generation_rules, word_to_upper_level_symbol)
        return word_to_upper_level_symbol
    
    def build_upper_level_seq(self, level, curr_level_sentences, word_to_upper_level_symbol):
        # Create the sentences at the level above
        old_sentences = torch.unique(curr_level_sentences, dim=0)
        upper_level_sentences = torch.zeros((old_sentences.size(0), old_sentences.size(1) // self.cfg.T[level]))
        for k in range(old_sentences.size(0)):
            words = torch.split(old_sentences[k], self.cfg.T[level])
            for i, w in enumerate(words):
                upper_level_sentences[k, i] = word_to_upper_level_symbol[tuple(w.tolist())]
        return upper_level_sentences
    
    def backtrack_cfg(self, sentences):
        for lev in range(self.cfg.L -1, 1, -1):
            pairs_of_synonyms = self.find_all_synonym_pairs(sentences)
            if len(pairs_of_synonyms) == 0:
                print(f"Failed finding synonyms, no sentence in the corpus differs by only one word, stopping at level {lev}")
                return
            word_to_upper_level_symbol = self.store_rules(sentences, lev)
            sentences = self.build_upper_level_seq(lev, word_to_upper_level_symbol)
            print(sentences)
# %%
cfg = CFG(L=3, ns=[1, 9, 9, 10], nr=[2, 2, 2], T=[2, 2, 2])
backtracker = CFGBacktracker(cfg)
sentences = cfg.sample_flattened(1000)[0].squeeze(0)
backtracker.backtrack_cfg(sentences)

# %%
