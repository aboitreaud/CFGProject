import torch
import numpy as np


class Grammar:

    # Define a one-class contex-free-grammar with:
    #
    # - n_levels: from the root symbol at level 0 up to sentence at last (nb_levels) level
    # - n_symbols: list of the number of symbols per level from level one to level nb_levels (length = nb_levels)
    #   Note that nb_symbols[0] is the number of "classes"
    # - n_children: list of the length of the rules at each level (length = nb_levels - 1) each
    #   symbol from level i will be expanded into nb_children symbols at level i+1
    # - n_rules: list of the number of rules per level. At level i, there are nb_rules[i] possibilities to expand a
    # symbol from level i into nb_children[i] symbols of level i+1. Rules are drawn in the constructor,
    # so they are fixed for all sentence generations
    def __init__(self, n_levels: int, n_symbols: list, n_children: list, n_rules: list):
        self.n_levels = n_levels
        self.n_symbols = n_symbols
        self.n_children = n_children
        self.n_rules = n_rules
        self.rules = []

        assert len(n_symbols) == n_levels, "Please check the list of number of symbols"
        assert len(n_children) == n_levels - 1, "Please check the list of number of children"
        assert len(n_rules) == n_levels - 1, "Please check the list of number of rules"

        for l in range(n_levels - 1):
            # for each of the n_symbols[l] symbols at level l, we draw n_rules[l] of length n_children[l]
            self.rules.append(torch.randint(0, n_symbols[l+1], size=(n_symbols[l], n_rules[l], n_children[l])))

    def generate_sentence(self, root: torch.tensor, debug: bool):
        # init sequence with root symbol
        sentence = root
        # -1 here to take into account the fact that level 0 is the root symbol
        for level in range(self.n_levels - 1):
            sentence = self.expand_one_level(sentence, level)
            if debug:
                print(f"sentence after level {level} has shape {sentence.shape}: {sentence}")
        return sentence

    def expand_one_level(self, input_seq: torch.tensor, level: int):
        rules_chosen = torch.randint(0, self.n_rules[level], size=input_seq.shape)
        output_seq = self.rules[level][input_seq, rules_chosen]
        return output_seq

    def generate_n_sentences(self, nspl:int, debug: bool = False):
        sentences = torch.empty(nspl)
        for i in range(nspl):
            sentences[i] = self.generate_sentence(torch.tensor(0), debug)

