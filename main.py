from context_free_grammar import CFG

if __name__ == '__main__':
    
    cfg = CFG(L=3, ns=[2, 3, 3, 3], nr=[5, 5, 5] , T=[2, 2, 2])

    nspl = 3
    # Generate ns[0] * nspl sentences in total
    # Vocab size is ns[L]
    # Each sentence is a tensor of shape (T_0,T_1,...,T_{L-1})
    # Flattened sentences are of length np.prod(T) product of length of the rules at each level

    # Task:
    #  - classification (we have the labels of the sentences) -> how many classes ?
    #  - generation -> how many classes ?

    s, labels = cfg.sample_flattened(nspl)
    # for i, sentence in enumerate(s):
    #     print('Sentence {i} is: {sent}'.format(i=i, sent=sentence.detach().numpy()))
    print(s, labels)

    print(s.shape, labels)

