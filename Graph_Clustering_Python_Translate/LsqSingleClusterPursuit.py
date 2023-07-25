# ======================================== Acknowledgement ================================================
# This is part of series of work to translate Dr.Daniel Mckenzie's LsqClusterPursuit code from MatLab into 
# Python; The purpose of this project is first to understand how the algorithm works and second to practice 
# coding ability in python language;

# The original MATLAB code is based on the code of SingleClusterPursuit algorithm by Dr. Daniel Mckenzie, 
# with the parameter 'reject' in Mckenzie's SingleClusterPursuit is removed.


import numpy as np
from scipy.sparse import diags
from scipy.sparse import eye
from LsqRandomWalkThresh import LsqRandomWalkThresh
from LsqClusterPursuit import LsqClusterPursuit


def LsqSingleClusterPursuit(A: np.ndarray, Gamma: np.ndarray, n0: int, epsilon: float, t: int, reject: float):

    # Inputs:
    # A ......................................................... Adjacency matrix of data converted to graph form
    # Gamma ..................................................... Vector. Labelled data within cluster of interest
    # n0 ........................................................ (estimated) size of C_a
    # epsilon ................................................... Omega_a will be of size (1+epsilon)n0(a)
    # t ......................................................... Depth of random walk

    # Outputs:
    # Cluster ................................................... Vector. The elements in the cluster of interest.

    # Initialization:
    n = len(A)
    degvec = A.sum(axis=1)
    Dinv = diags(1/degvec, 0, shape=(n,n)).toarray() 
    DinvA = np.matmul(Dinv, A)
    L = eye(n).toarray() - DinvA

    # Call the subroutines:
    Omega = LsqRandomWalkThresh(A, Gamma, n0, epsilon, t)
    Cluster = LsqClusterPursuit(L, Gamma, Omega, n0, reject)[0]
    
    return Cluster


# Test:

# A = np.array([
#     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#     [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
#     ])

# Gamma = np.array([0,2,3,5]) 

# Result = LsqSingleClusterPursuit(A, Gamma, 4, .4, 20, .1)

# print(Result)


