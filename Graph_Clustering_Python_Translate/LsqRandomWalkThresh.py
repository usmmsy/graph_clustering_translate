# ======================================== Acknowledgement ================================================
# This is part of series of work to translate Dr.Daniel Mckenzie's LsqClusterPursuit code from MatLab into 
# Python; The purpose of this project is first to understand how the algorithm works and second to practice 
# coding ability in python language;

# The function coded here determines the superset Omega needed for SSCP by running a random walk for k steps
# starting from Gamma and returning the (1+epsilon)n0_hat vertices with highest probabilities of being 
# visited. See [Lai and Mckenzie, 2020]

# The original MATLAB code was written by Daniel Mckenzie under Dr.Ming-Jun Lai's supervision on 2 March 2019
# Modified by Zhaiming Shen under Dr.Ming-Jun Lai's supervision in Dec.2019


import numpy as np
from scipy.sparse import diags
from scipy.sparse import coo_matrix


def LsqRandomWalkThresh(A: np.ndarray, Gamma: np.ndarray, n0_hat: float, epsilon: float, t: int):

    # Inputs:
    # A ................................................ Adjacency matrix
    # Gamma ............................................ Seed vertices/labelled data
    # n0_hat ........................................... Estimate on the size of the cluster
    # epsilon .......................................... Oversampling parameter
    # t ................................................ Depth of random walk

    # Outputs:
    # Omega ............................................ Superset containing a large fraction of the vertices in cluster

    # Initialization: 
    n = len(A) 
    Dtemp = A.sum(axis=1) 
    Dinv = diags(1/Dtemp, 0, shape=(n,n)).toarray() 
    m = len(Gamma)
    v0 = coo_matrix((Dtemp[Gamma], (Gamma, np.zeros(m))), shape = (n,1)).toarray() 
    P = np.matmul(A, Dinv)

    # Random Walk and Threshold:
    v = v0
    for i in range(t):
        v = np.matmul(P, v)
    v = np.concatenate(v)
    w = np.sort(v)[::-1]
    IndsThresh = np.argsort(v)[::-1]
    FirstZero = np.where(w == 0)[0]
    if FirstZero.size>0 and FirstZero[0]<np.ceil(epsilon*n0_hat)-1:
        print('Warning: the size of Omega is smaller than (1+delta) times the user specified cluster size. Try a larger value of k')
        T = FirstZero[0]
    else:
        T = np.ceil(epsilon*n0_hat)
    
    T = int(T)
    
    Omega = np.union1d(IndsThresh[:T], Gamma)
    
    return Omega

# Test:

if __name__ == '__main__': 
    A = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ])
    
    Gamma = np.array([2,3,5]) 
    Omega = LsqRandomWalkThresh(A, Gamma, 5, .5, 10) 
    print(Omega)
