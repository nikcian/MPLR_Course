import numpy as np
from utils.math_utils import compute_mu_C

def compute_pca(D, m):
    mu, C = compute_mu_C(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D
