import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import scripts.util as util
import lda
import pca

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)

def classify():
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = 'data/trainData.txt'
    
    D, L = util.load(fname)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # 1. Compute LDA direction on training data
    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)

    # 2. Project training data
    DTR_lda = lda.apply_lda(ULDA, DTR)

    # 3. Check orientation: mean of class True (1) should be larger than mean of class False (0)
    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    # 4. Compute threshold on training data
    mean_cls0 = DTR_lda[0, LTR==0].mean()
    mean_cls1 = DTR_lda[0, LTR==1].mean()
    threshold = (mean_cls0 + mean_cls1) / 2.0

    # 5. Project validation data
    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    # 6. Predict on validation data
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0

    # 7. Compute errors
    errors = (PVAL != LVAL).sum()
    total = LVAL.size
    error_rate = errors / float(total) * 100

    print("--- LDA Classification ---")
    print(f"Threshold computed on DTR: {threshold:.4f}")
    print(f"Number of errors on DVAL: {errors} (out of {total} samples)")
    print(f"Error rate: {error_rate:.2f}%")

    print("\n--- PCA + LDA Classification Sweep ---")
    for m in range(1, 7):
        # 1. Compute PCA on training data
        UPCA = pca.compute_pca(DTR, m=m)
        
        # 2. Project training and validation data
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)

        # 3. Compute LDA on PCA-projected training data
        ULDA = lda.compute_lda_geig(DTR_pca, LTR, m=1)

        # 4. Project PCA-training data to LDA
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)

        # 5. Check orientation
        if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
            ULDA = -ULDA
            DTR_lda = lda.apply_lda(ULDA, DTR_pca)

        # 6. Compute threshold
        mean_cls0 = DTR_lda[0, LTR==0].mean()
        mean_cls1 = DTR_lda[0, LTR==1].mean()
        threshold = (mean_cls0 + mean_cls1) / 2.0

        # 7. Project PCA-validation data to LDA
        DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)

        # 8. Predict and calculate errors
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        
        errors_pca = (PVAL != LVAL).sum()
        error_rate_pca = errors_pca / float(total) * 100
        print(f"PCA m={m} -> Errors: {errors_pca}, Error rate: {error_rate_pca:.2f}%")

if __name__ == '__main__':
    classify()
