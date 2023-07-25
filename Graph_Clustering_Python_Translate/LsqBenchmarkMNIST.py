# ======================================== Acknowledgement ================================================
# This is part of series of work to translate Dr.Daniel Mckenzie's LsqClusterPursuit code from MatLab into 
# Python; The purpose of this project is first to understand how the algorithm works and second to practice 
# coding ability in python language;

# LsqSingleClusterPursuit Algorithm on MNIST data set;

# This demo is based on the demo Benchmark_MNIST by Dr. Daniel Mckenzie, while his SingleClusterPursuit is 
# replaced by LsqSingleClusterPursuit.


import numpy as np
import math
import time
from scipy.io import loadmat 
from scipy.sparse import csc_matrix
# from sklearn.metrics import jaccard_score 
from JaccardScoreDIY import JaccardScoreDIY
import matplotlib.pyplot as plt
from LsqSingleClusterPursuit import LsqSingleClusterPursuit

# Parameters and load the data:
MNIST_data = loadmat('MNIST_KNN_Mult_K=15.mat') 
A_csc = MNIST_data['A'] 
y = MNIST_data['y']

A = A_csc.toarray()
A = A[:700, :700]
y = y[:700]

# print(np.shape(A), np.shape(y))

epsilon = 0.8

num_trials = 5
num_sizes = 5
k = 10 # number of clusters

# Find the ground truth clusters:
TrueClusters = []
n0vec = []
for a in range(k):
    Ctemp = np.where(y==a)
    TrueClusters.append(Ctemp)
    n0vec.append(len(Ctemp))
# TrueClusters = np.array(TrueClusters)
# n0vec = np.array(n0vec)

# Define all vectors of interest:
time_SCP_mat = np.zeros((k,num_sizes))
Jaccard_SCP_mat = np.zeros((k,num_sizes))
accuracy_SCP_mat = np.zeros((k,num_sizes))

for j in range(num_sizes):
    sample_frac = 0.001*(j+1)
    for i in range(k):
        TrueCluster = TrueClusters[i][0]
        # print(TrueCluster, np.size(TrueCluster))
        n0 = len(TrueCluster)
        # n0_equal = 7000
        n0_equal = 700

        # Draw Seed set:
        Gamma = np.random.choice(TrueCluster, math.ceil(sample_frac*n0_equal), replace=False)

        # SingleClusterPursuit:
        tic = time.perf_counter()
        Cluster_SCP = LsqSingleClusterPursuit(A, Gamma, n0_equal, epsilon, 3, 0)
        toc = time.perf_counter() - tic
        time_SCP_mat[i][j] = time_SCP_mat[i][j] + toc 
        Jaccard_SCP_mat[i][j] = JaccardScoreDIY(TrueCluster, Cluster_SCP)
        accuracy_SCP_mat[i][j] = len(np.intersect1d(Cluster_SCP, TrueCluster))/n0
 
# plt.plot(np.mean(accuracy_SCP_mat))

# plt.boxplot(Jaccard_SCP_mat)
# plt.show()

# plt.boxplot(accuracy_SCP_mat)
# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('LSQ single cluster pursuit method')

# ax1.plot(np.mean(accuracy_SCP_mat))

# ax2.boxplot(Jaccard_SCP_mat)
# ax2.xscale('Percentage of vertices used as seeds')
# ax2.yscale('Jaccard Score')

# ax3.boxplot(accuracy_SCP_mat)
# ax3.xscale('Percentage of vertices used as seeds')
# ax3.yscale('Accuracy')

# plt.show()

print(Jaccard_SCP_mat, accuracy_SCP_mat)

plt.figure()

plt.subplot(221)
plt.plot(np.mean(accuracy_SCP_mat, axis=0))
plt.xlabel('Percentage of vertices used as seeds')
plt.ylabel('Average Jaccard Score')
# plt.grid(True)

plt.subplot(222)
plt.boxplot(Jaccard_SCP_mat)
plt.xlabel('Percentage of vertices used as seeds')
plt.ylabel('Jaccard Score')
# plt.title('LSQ single cluster pursuit method')
# plt.grid(True)

plt.subplot(223)
plt.boxplot(accuracy_SCP_mat)
plt.xlabel('Percentage of vertices used as seeds')
plt.ylabel('Accuracy')
# plt.title('LSQ single cluster pursuit method')
# plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.suptitle('LSQ Single Cluster Pursuit Method')

plt.show()


