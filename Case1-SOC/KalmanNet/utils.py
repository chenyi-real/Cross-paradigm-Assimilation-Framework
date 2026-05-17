import numpy as np

def pad_sequence(seqs, pad_value=0.0):
    N = len(seqs)
    d = seqs[0].shape[1]
    max_T = max([s.shape[0] for s in seqs])
    padded = np.full((N, max_T, d), pad_value, dtype=float)
    for i, s in enumerate(seqs):
        T = s.shape[0]
        padded[i, :T, :] = s
    return padded

def pad_labels(seqs, pad_value=0.0):
    N = len(seqs)
    max_T = max([len(s) for s in seqs])
    padded = np.full((N, max_T), pad_value, dtype=float)
    for i, s in enumerate(seqs):
        T = len(s)
        padded[i, :T] = s
    return padded
