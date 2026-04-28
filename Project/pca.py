import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import scripts.util as util
import scripts.plot as myplt

def compute_mu_C(D):
    mu = util.vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):
    mu, C = compute_mu_C(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D

def plot_hist_pca(D, L):
    P = compute_pca(D, D.shape[0]) # Get all directions
    DP = apply_pca(P, D)
    
    D0 = DP[:, L==0]
    D1 = DP[:, L==1]

    if not os.path.exists('out/plots'):
        os.makedirs('out/plots')

    for dIdx in range(DP.shape[0]):
        plt.figure()
        plt.xlabel(f'PCA Direction {dIdx+1}')
        plt.ylabel('Density')        
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'False')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'True')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'out/plots/hist_pca_{dIdx+1}.pdf')
        plt.close()

if __name__ == '__main__':
    myplt.plot_init()
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = 'data/trainData.txt'
    
    D, L = util.load(fname)
    plot_hist_pca(D, L)
