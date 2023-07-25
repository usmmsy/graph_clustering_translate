# ======================================== Acknowledgement ================================================
# This is part of series of work to translate Dr.Daniel Mckenzie's LsqClusterPursuit code from MatLab into 
# Python; The purpose of this project is first to understand how the algorithm works and second to practice 
# coding ability in python language;


import numpy as np
import math
from scipy.sparse.linalg import lsqr


def LsqClusterPursuit(L: np.ndarray, Gamma_a: np.ndarray, Omega_a: np.ndarray, n_a: int, reject: float):

    # Inputs:
    # L ........................................... Laplacian matrix
    # Gamma_a ..................................... Labeled data for C_a
    # Omega_a ..................................... Superset containing C_a
    # n_a ......................................... (Estimated) size of C_a

    # Outputs:
    # C ........................................... Estimate of C_a (including Gamma_a)
    # v ........................................... vector of probabilities of not being in C
    
    Phi = L[:, Omega_a]                             # Keep columns of L with indices in Omega_a
    n = len(L)                                      # Number of vertices in graph
    vdeg = Phi.sum(axis=1)                          # Get the vector of vertex degrees
    k = math.floor(n_a/5)
    indices = np.sort(np.matmul(np.transpose(np.abs(Phi)), np.abs(vdeg)))[:k]
    Phi[:, indices.astype('int')] = 0
    g = len(Gamma_a)
    h = len(Omega_a)
    sparsity = math.ceil(1.1*h - (n_a - g))
    if sparsity <= 0:
        C = np.union1d(Omega_a, Gamma_a)
        v = np.zeros((n,1))
    else:
        v = lsqr(Phi, vdeg)[0]
        Lambda_a = []
        for i in range(len(v)):
            if v[i] > reject:
                Lambda_a += [i] 
        B = np.setdiff1d(Omega_a, Omega_a[Lambda_a])        
        C = np.union1d(B, Gamma_a)
    return C, v

