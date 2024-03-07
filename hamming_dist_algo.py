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
class SynonymFinder:

    def find_closest_sentences(self, sentences, nb_allowed_differing_words, word_size):
        min_dist = sentences.size(0)*10
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
                return (c1, c2)
        
    
    def apply_synonyms_change(self, synonyms, sentences):
        """
        Replaces the synonyms[1] subsections in all sentences with synonyms[0].
            
        Returns:
            torch.Tensor: The modified sentences with the synonym2 replaced with synonym1 in every sentences.
        """
        if synonyms is not None:
            (synonym1, synonym2) = synonyms
            # Check if the synonyms are valid
            assert len(synonym1) == len(synonym2), "Synonyms must have the same length"
            assert synonym1.dim() == 1 and synonym2.dim() == 1, "Synonyms must be 1-dimensional"
            word_size = synonym1.size(0)
            for i, sentence in enumerate(sentences):
                # Find the starting index of the subsection, if it exists
                synonym_positions = (sentence.unfold(0, word_size, word_size) == synonym2.unsqueeze(0)).all(dim=1).squeeze().nonzero().squeeze(1)
                
                # Replace the subsection with zeros if found
                for start_idx in synonym_positions:
                    start_idx *= word_size
                    end_idx = start_idx + word_size
                    sentence[start_idx:end_idx] = synonym1
                    sentences[i] = sentence
        return sentences

# %%
s = SynonymFinder()
cfg = CFG(L=2, ns=[1, 9, 10], nr=[2, 2], T=[8, 8])
sentences = cfg.sample_flattened(50)[0].squeeze(0)

min_dist_pair = (-2, -2)
while min_dist_pair != (-1, -1):
    min_dist_pair = s.find_closest_sentences(sentences, 1, 8)
    print(sentences[min_dist_pair[0]])
    print(sentences[min_dist_pair[1]])
    synonyms = s.find_synonyms(sentences[min_dist_pair[0]], sentences[min_dist_pair[1]], 8)
    print(synonyms)
    sentences = s.apply_synonyms_change(synonyms, sentences)
    print(min_dist_pair)
