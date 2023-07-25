# A DIY Jaccard Score calculator that could admit inputs with different sizes

import numpy as np

def JaccardScoreDIY(C1, C2):
    # Inputs:
    # C1 ........................................................... A subset of [n]
    # C2 ........................................................... Another subset of [n]

    # Outputs:
    # score ........................................................ The Jaccard score

    top = len(np.intersect1d(C1, C2))
    bottom = len(np.union1d(C1, C2))

    score = top/bottom

    return score

