import numpy as np
import scripts.util as util
from lda import compute_lda_geig, apply_lda

def analyze(D, L):
    U = compute_lda_geig(D, L, m=1)
    DP = apply_lda(U, D)
    
    D0 = DP[:, L==0][0]
    D1 = DP[:, L==1][0]

    mean0, std0 = np.mean(D0), np.std(D0)
    mean1, std1 = np.mean(D1), np.std(D1)
    
    print(f"LDA Direction 1")
    print(f"Class False (0) - Mean: {mean0:.4f}, Std: {std0:.4f}")
    print(f"Class True  (1) - Mean: {mean1:.4f}, Std: {std1:.4f}")
    print(f"Distance between means: {abs(mean0 - mean1):.4f}")

if __name__ == '__main__':
    fname = 'data/trainData.txt'
    D, L = util.load(fname)
    analyze(D, L)
