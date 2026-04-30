import numpy as np
from utils.math_utils import compute_mu_C, logpdf_GAU_ND, vcol, vrow

def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams

def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams

def compute_llr(DVAL, mu1, C1, mu0, C0):
    ll1 = logpdf_GAU_ND(DVAL, mu1, C1)
    ll0 = logpdf_GAU_ND(DVAL, mu0, C0)
    return ll1 - ll0

def predict_from_llr(LLR, threshold=0):
    PVAL = np.zeros(LLR.shape, dtype=np.int32)
    PVAL[LLR >= threshold] = 1
    return PVAL

def compute_error_rate(PVAL, LVAL):
    return (PVAL != LVAL).sum() / float(LVAL.size) * 100
