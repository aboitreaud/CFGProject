import torch
import numpy as np


class CFG:
    """
    ns = [ ns_0, ns_1, ..., ns_{L-1}, ns_L ]   list containing L+1 integers
    nr = [ nr_0, ns_1, ..., nr_{L-1}       ]   list containing L integers
    T  = [  T_0,  T_1, ...,  T_{L-1}       ]   list containing L integers
    ns_l is the number of symbol at level l
    nr_l is the number of rules per symbol at level l
    T_l is the length of the rule at level l
    L is the number of levels

    We have a L+1 levels:
    Level 0: class labels. There are ns_0 labels. Labels are level-0 symbols.
    Level 1: Tensors of shape (T_0,) containing level-1 symbols (there are ns_1 level-1 symbols)
    Level 2: Tensors of shape (T_0,T_1) containing level-2 symbols (there are ns_2 level-2 symbols)
    .
    .
    Level L: Tensors of shape (T_0,T_1,...,T_{L-1}) containing level-L symbols (there are ns_L level-L symbols)

    We have L sets of rules:
    Rules of level l are used to expand symbols of level l into symbols of level l+1
    rules[l]: LongTensor of shape (ns_l, nr_l, T_l) with entries in { 0,1,...,ns_{l+1}-1 }
    """

    def __init__(self, L, ns, nr, T):
        assert L == len(nr) and L == len(T) and L == len(ns) - 1
        self.ns = ns
        self.nr = nr
        self.T = T
        self.L = L
        self.rules = (
            []
        )  # rules[l] has shape (ns_l, nr_l, T_l) with entries in { 0,1,...,ns_{l+1}-1 }
        for l in range(L):
            self.rules.append(torch.randint(0, ns[l + 1], size=(ns[l], nr[l], T[l])))
            print("Level {level} rule: {rule} with shape {shape}".format(level=l, rule=self.rules[-1], shape=self.rules[-1].shape))

    ######################################################################
    # FUNCTIONS TO GENERATE A SEQUENCE OF SYMBOLS ACCORDING TO THE GRAMMAR
    ######################################################################

    def expand_symbols_one_level(self, S, l):
        """
        This function uses rules of level l to expand symbols of level l into symbols of level l+1
        INPUT:    S: LongTensor containing symbols from level l
                     More specifically, S has shape (*) with entries in {0,1,...,ns_l-1}
        OUTPUT:   S_expanded: LongTensor containing symbols of level l+1
                              More specifically, S_expanded has shape (*,T_l) with entries in {0,1,...,ns_{l+1}-1}
        """
        RND = torch.randint(0, self.nr[l], size=S.shape)
        return self.rules[l][S, RND]

    def expand_symbols(self, S, l=0):
        """
        Expand from level l to level L
        """
        for lev in range(l, self.L):
            S = self.expand_symbols_one_level(S, lev)
        return S

    def expand_symbols_and_keep_latent_variables(self, S, l=0):
        """
        This function is useful for debugging purpose
        """
        latent = [S]
        for lev in range(l, self.L):
            S = self.expand_symbols_one_level(S, lev)
            latent.append(S)
        return latent

    def sample(self, nspl):
        """
        This function generates nspl data points per class, as well as the corresponding label vector.
        """
        labels = torch.arange(self.ns[0]).repeat_interleave(nspl, dim=0)
        S = self.expand_symbols(labels)
        return S, labels

    def sample_flattened(self, nspl):
        """
        This function generates nspl data points per class, as well as the corresponding label vector.
        """
        labels = torch.arange(self.ns[0]).repeat_interleave(nspl, dim=0)
        S = self.expand_symbols(labels)
        return torch.reshape(S, (self.ns[0], nspl, -1)), labels


    ###################################################################################
    #  FUNCTIONS TO DECIDE WETHER OR NOT A SEQUENCE OF SYMBOLS IS GRAMMATICALLY CORRECT
    ###################################################################################

    def closest_rule(self, seq, l):
        """
        seq is a LongTensor of length T_l with entries in {0,1, ..., ns_{l+1}-1}
        This function find the rule of level l which is the closest to seq (with respect to the Hamming distance)

        Recall that rules of level l are contained in a Tensor rules[l] of shape  (ns_l, nr_l, T_l)
        with entries in {0,...,ns_{l+1}

        This function find i and j that solves

            Minimize  dist( seq , rules[l][i,j,:] )  over all  0 <= i <= ns_l and 0<= j <= nr_l

        So i is the index of the symbol that (approximately) generated seq, and j is the index of the specific rule.

        The function also returns the distance between seq and the closest rule.
        """
        assert seq.shape[0] == self.T[l], "seq doesn't have the right length"
        assert torch.all(
            seq < self.ns[l + 1]
        ).item(), "some symbols in seq are too large"

        # hamming distance between seq and all rules of level i
        H = torch.sum(self.rules[l] != seq.view(1, 1, -1), dim=-1)  # (ns_l,nr_l)

        # find smallest distance
        min_dst = torch.min(H)

        # Indicator matrix of the entries that are the smallest
        BoolMat = H == min_dst

        # indices of the first smallest entries (there might be multiple entries)
        i_and_j = BoolMat.nonzero()[0]

        # unpack
        i = i_and_j[0].item()
        j = i_and_j[1].item()
        hamming_dst = min_dst.item()

        return i, j, hamming_dst

    def collapse_symbols_one_level(self, S, l):
        """
        This function uses rules of level l to coarsen symbols of level l+1 into symbols of level l
        This is the inverse of the expand function.
        INPUT:    S: LongTensor containing symbols from level i+1
                     More specifically, S has shape (*,L_i) with entries in {0,1,...,ns_{i+1}-1}
        OUTPUT:   S_coarsened: LongTensor containing symbols of level i
                              More specifically, S_coarsened has shape (*) with entries in {0,1,...,ns_{i}-1}
        """

        # suppose S has shape (T_0,T_1,T_2).
        # We start by flattening it to have shape (T_0*T_1 , T_2)
        S_flat = S.view(-1, S.shape[-1])

        # For each of the T_0*T_1 sequences in S_flat we find the index of the symbol that generated it
        # (and also the hamming distance error if there is no exact match)
        S_coarsened = []
        err = []
        for seq in S_flat:
            i, j, hamming_dst = self.closest_rule(seq, l)
            S_coarsened.append(i)
            err.append(hamming_dst)

        S_coarsened = torch.tensor(S_coarsened)
        err = torch.tensor(err)

        # We now reshape S_coarsened and error to have shape (T_0,T_1)
        target_shape = S.shape[:-1]
        S_coarsened = S_coarsened.view(target_shape)
        err = err.view(target_shape)

        return S_coarsened, err

    def collapse_and_get_err(self, S):
        latent = []
        err = []
        for l in range(self.L - 1, -1, -1):
            S, e = self.collapse_symbols_one_level(S, l)
            latent.append(S)
            err.append(e)

        # reverse order of the lists so that latent[l] gives the latent variables of level l
        latent.reverse()
        err.reverse()

        return latent, err

    def get_vocab_size(self):
        return np.prod(self.T)

    def get_sentence_length(self):
        return np.prod(self.T)