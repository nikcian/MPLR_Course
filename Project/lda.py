import os
import sys
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import scripts.util as util
import scripts.plot as myplt

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = util.vcol(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = util.vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):
    Sb, Sw = compute_Sb_Sw(D, L)
    U, s, _ = np.linalg.svd(Sw)
    P = np.dot(U * util.vrow(1.0/(s**0.5)), U.T)
    Sb2 = np.dot(P, np.dot(Sb, P.T))
    U2, s2, _ = np.linalg.svd(Sb2)
    P2 = U2[:, 0:m]
    return np.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

def plot_hist_lda(D, L):
    # For binary classification, LDA can only extract 1 meaningful direction (C-1 = 1)
    U = compute_lda_geig(D, L, m=1)
    DP = apply_lda(U, D)
    
    D0 = DP[:, L==0]
    D1 = DP[:, L==1]

    if not os.path.exists('out/plots'):
        os.makedirs('out/plots')

    plt.figure()
    plt.xlabel('LDA Direction 1')
    plt.ylabel('Density')        
    plt.hist(D0[0, :], bins = 10, density = True, alpha = 0.4, label = 'False')
    plt.hist(D1[0, :], bins = 10, density = True, alpha = 0.4, label = 'True')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('out/plots/hist_lda_1.pdf')
    plt.close()

if __name__ == '__main__':
    myplt.plot_init()
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = 'data/trainData.txt'
    
    D, L = util.load(fname)
    plot_hist_lda(D, L)
