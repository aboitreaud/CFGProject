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

    def find_closest_sentences(self, corpus):
        min_dist = 512*10
        min_dist_pair = (-1, -1)
        nb_sentences = corpus.size(0)
        for i in range(nb_sentences):
            for j in range(i+1, nb_sentences):
                dist = self.hamming_distance(corpus[i], corpus[j])
                if dist < min_dist:
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
        
    
    def apply_synonym_change(self, synonym1, synonym2, corpus):
        """
        Replaces the synonym2 subsections in the sentences with synonym1.
        
        Args:
            tensor (torch.Tensor): The input tensor of length 512.
            subsection (torch.Tensor): The subsection tensor of length 8 to search for.
            
        Returns:
            torch.Tensor: The modified corpus with the synonym2 replaced with synonym1 in every sentences.
        """
        for sentence in corpus:
            # Check if the subsection tensor is valid
            assert len(synonym1) == len(synonym2), "Synonyms must have the same length"
            assert synonym1.dim() == 1 and synonym2.dim() == 1, "Synonyms must be 1-dimensional"
            
            # Find the starting index of the subsection, if it exists
            start_idx = (sentence.unfold(0, 8) == synonym2.unsqueeze(0)).all(dim=2).squeeze().nonzero().squeeze(1)
            
            # Replace the subsection with zeros if found
            if start_idx.numel() > 0:
                end_idx = start_idx + 8
                sentence[start_idx:end_idx] = synonym1
            
        return corpus

# %%
s = SynonymFinder()

# %%
# Example usage
vectors = np.random.randint(0, 10, size=(5, 512))

# %%

# %%
print(min_dist_pair)

# %%
min_dist

# %%
