import sys
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks
import scripts.util as util

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

def analyze(D, L):
    P = compute_pca(D, D.shape[0])
    DP = apply_pca(P, D)
    
    D0 = DP[:, L==0]
    D1 = DP[:, L==1]

    for dIdx in range(DP.shape[0]):
        print(f"--- PCA Direction {dIdx+1} ---")
        
        mean0, std0 = np.mean(D0[dIdx, :]), np.std(D0[dIdx, :])
        mean1, std1 = np.mean(D1[dIdx, :]), np.std(D1[dIdx, :])
        
        print(f"Class False (0) - Mean: {mean0:.4f}, Std: {std0:.4f}")
        print(f"Class True  (1) - Mean: {mean1:.4f}, Std: {std1:.4f}")
        print(f"Distance between means: {abs(mean0 - mean1):.4f}")
        
        # Estimate number of peaks using KDE
        for cls, D_cls in [("False", D0), ("True", D1)]:
            kde = stats.gaussian_kde(D_cls[dIdx, :])
            x = np.linspace(np.min(D_cls[dIdx, :]), np.max(D_cls[dIdx, :]), 1000)
            y = kde(x)
            peaks, _ = find_peaks(y)
            print(f"Class {cls} number of peaks (clusters): {len(peaks)}")
        print()

if __name__ == '__main__':
    fname = 'data/trainData.txt'
    D, L = util.load(fname)
    analyze(D, L)
