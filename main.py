import torch

from context_free_grammar import CFG
from grammar import Grammar

if __name__ == '__main__':
    
    cfg = CFG(L=3, ns=[1, 3, 3, 3], nr=[2, 2, 2] , T=[5, 5, 5])

    nspl = 1
    # Generate ns[0] * nspl sentences in total
    # Vocab size is ns[L]
    # Each sentence is a tensor of shape (T_0,T_1,...,T_{L-1})
    # Flattened sentences are of length np.prod(T) product of length of the rules at each level

    # Task:
    #  - classification (we have the labels of the sentences) -> how many classes ?
    #  - generation -> do we need classes ? hamming distance appears to be the cost for the moment

    s, labels = cfg.sample(nspl)
    # for i, sentence in enumerate(s):
    #     print('Sentence {i} is: {sent}'.format(i=i, sent=sentence.detach().numpy()))
    # print(s.shape, labels)
    print('\n')
    print("\n")
    gram = Grammar(n_levels=4, n_symbols=[1, 3, 3, 3], n_children=[5, 5, 5], n_rules=[2, 2, 2])
    print(gram.generate_n_sentences(nspl=1))
