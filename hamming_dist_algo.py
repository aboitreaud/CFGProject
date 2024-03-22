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
    def __init__(self, cfg, nb_allowed_differing_words=1, on_gpu=True) -> None:
        self.cfg = cfg
        # Max nb of differing between two sentences we allow to match synonyms
        self.nb_allowed_differing_words = nb_allowed_differing_words
        # for each level, we store the generation rule (mapping level i symbol -> word of level i+1)
        self.backtracked_rules = {lev: {} for lev in range(self.cfg.L)}
        # Store the words of level L-1 to start generation process
        self.below_root_seq = None
        self.device = torch.device("cuda") if on_gpu else "cpu"

    def find_closest_pair(self, level, sentences):
        """
        Given a corpus of sentences, find the two closest ones, with respect to the hamming distance
        It only returns the pair of closest sentences if they are less than 'self.nb_allowed_differing_words' apart
        Otherwise, None is returned, meaning that no synonyms will further be declared at that level on these sentences
        """
        sentences = sentences.to(self.device)

        distances = (sentences.unsqueeze(1) ^ sentences.unsqueeze(0)).sum(dim=2)

        # Find the min distance, among sentences not exactly equal, so with > 0 distance
        zero_mask = (distances == 0)
        # Mask the 0 with the theoretical maximal hamming distance of two sentences of the grammar
        distances[zero_mask] = np.prod(self.cfg.T) * (self.cfg.ns[-1] + 1)
        min_index = distances.argmin(dim=None)
        i = min_index // distances.shape[0]
        j = min_index % distances.shape[0]
        i, j = i.item(), j.item()

        # Check how many words are different in the two sentences
        diff_mask = torch.ne(sentences[i], sentences[j])
        num_differing_items = diff_mask.sum().item()
        # Return pair of sentences if they differ by less than nb_allowed_differing_words words
        if num_differing_items <= self.nb_allowed_differing_words * self.cfg.T[level]:
            return i, j
        else:
            return None

    def find_synonyms(self, sentence1, sentence2, word_size):
        """
        Given two sentences, find the words at same position that are not equal and declare them synonyms
        """
        synonyms = [] # List to store pairs (tuples) of synonyms

        for c1, c2 in zip(sentence1.split(word_size), sentence2.split(word_size)):
            if torch.any(c1 != c2):
                # Synonyms are the differing words (at same position) in the pair of sentences
                synonyms.append((c1, c2))
        return synonyms

    def apply_synonyms_change(self, synonyms, sentences):
        """
        Replaces the synonyms[1] substrings in all sentences with synonyms[0].

        Returns:
            torch.Tensor: The modified sentences with the synonym[1] replaced with synonym[0] in all sentences.
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
        """
        Given a corpus and a cfg level, find all pairs of synonyms declarable in these sentences
        A pair of synonyms can only be found in sentences that are less than self.nb_allowed_differing_words apart
        This function also updates the sentences after having found a pair of synonyms

        Returns:
            pairs_of_synonyms (list[tuples]): all synonyms found in the sentences at the given level
            sentences (torch.Tensor): updated corpus of sentences after merging synonyms
        """
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
        """
        Given a level and the list of pairs of synonyms, store the generation rules
        Rules are the mapping: char at level i:[words at level i-1]
        The pair of synonyms at position j in the list will be attributed j for symbol at level above
        For creating the sequence at level i from the sentence at level i-1, we also return the dictionary of
        the reverse generation rules. This is the mapping (word at level i-1: symbol at level i)
        For the pair of synonyms at position j in the list, this dictionary get two new entries:
        pairs_of_synonyms[j][0]:j and pairs_of_synonyms[j][1]:j
        """
        generation_rules = {}
        word_to_upper_level_symbol = {}
        for i in range(len(pairs_of_synonyms)):
            generation_rules[i] = pairs_of_synonyms[i]
            for w in pairs_of_synonyms[i]:
                word_to_upper_level_symbol[tuple(w.tolist())] = i
        self.backtracked_rules[level] = generation_rules
        return word_to_upper_level_symbol

    def build_upper_level_seq(self, level, curr_level_sentences, word_to_upper_level_symbol):
        """
        Given the level i, the sentences at level i, and the mapping (word at level i: symbol at level i+1),
        Return the corpus of reconstructed sentences at level i+1
        """
        old_sentences = torch.unique(curr_level_sentences, dim=0)
        upper_level_sentences = torch.zeros((old_sentences.size(0), old_sentences.size(1) // self.cfg.T[level]), dtype=torch.long)
        for k in range(old_sentences.size(0)):
            words = torch.split(old_sentences[k], self.cfg.T[level])
            for i, w in enumerate(words):
                upper_level_sentences[k, i] = word_to_upper_level_symbol[tuple(w.tolist())]
        return upper_level_sentences

    def backtrack_cfg(self, sentences):
        """
        Wrapper function calling all previous backtracking utilities
        Given a corpus of cfg sentences, try to recover all cfg generation rules
        At each level, declare as many synonyms as possible under the nb_allowed_differing_word constraint
        With these synonyms and the mapping (word at level i: symbol at level i+1), build sequences at level i+1.

        If not enough synonyms were found, some words won't appear in the mapping and the algo
        will fail to build sentences at the level above.

        Otherwise, the algo goes up one level and applies the same steps to the reconstructed sequences of level i+1
        """
        for lev in range(self.cfg.L - 1, 0, -1):
            print(f'Working on level {lev}')
            pairs_of_synonyms, sentences = self.find_all_synonym_pairs(level=lev, sentences=sentences)
            if len(pairs_of_synonyms) == 0:
                print(f"Failed finding synonyms, no sentence in the corpus differs by only {self.nb_allowed_differing_words} words,\
                       stopping at level {lev}")
                return None
            word_to_upper_level_symbol = self.store_rules(lev, pairs_of_synonyms)
            sentences = self.build_upper_level_seq(lev, sentences, word_to_upper_level_symbol)
            if lev == 1:
                self.below_root_seq = torch.unique(sentences, dim=0)
        return self.backtracked_rules

    def generate_sentences(self, nspl):
        sentences = torch.zeros((nspl, int(np.prod(self.cfg.T))), dtype=torch.int)
        for i in range(nspl):
            idx = np.random.randint(len(self.below_root_seq))
            seq = self.below_root_seq[idx]
            for lev in range(1, self.cfg.L):
                new_seq = []
                for s in seq.tolist():
                    choices = list(self.backtracked_rules[lev][s])
                    idx = np.random.randint(len(choices))
                    new_seq.extend(choices[idx].tolist())
                seq = torch.tensor(new_seq)
            sentences[i] = seq
        return sentences

# %%
cfg = CFG(L=3, ns=[1, 3, 9, 10], nr=[2, 2, 2], T=[8, 8, 8])
for _ in range(20):
    sentences = cfg.sample_flattened(8000)[0].squeeze(0)
    backtracker = CFGBacktracker(cfg, nb_allowed_differing_words=3, on_gpu=False)
    rules = backtracker.backtrack_cfg(sentences)
    if rules is not None:
        nspl = 100
        gen_sentences = backtracker.generate_sentences(nspl)
        cfg.frac_of_gramatically_correct_sentences(gen_sentences.view([nspl] + cfg.T))
    print()

# %%
