import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger
from models import pca, lda


def threshold_search_and_classify(D, L):
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    # Base LDA
    print("--- LDA Classification ---")
    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)
    DTR_lda = lda.apply_lda(ULDA, DTR)

    if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    mean_cls0 = DTR_lda[0, LTR == 0].mean()
    mean_cls1 = DTR_lda[0, LTR == 1].mean()
    base_threshold = (mean_cls0 + mean_cls1) / 2.0

    thresholds = np.linspace(DVAL_lda.min(), DVAL_lda.max(), 1000)
    best_err = len(LVAL)
    best_t = base_threshold

    for t in thresholds:
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= t] = 1
        PVAL[DVAL_lda[0] < t] = 0
        err = (PVAL != LVAL).sum()
        if err < best_err:
            best_err = err
            best_t = t

    PVAL_base = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL_base[DVAL_lda[0] >= base_threshold] = 1
    PVAL_base[DVAL_lda[0] < base_threshold] = 0
    err_base = (PVAL_base != LVAL).sum()

    print(f"Threshold computed on DTR: {base_threshold:.4f}")
    print(f"Number of errors on DVAL: {err_base} (out of {len(LVAL)} samples)")
    print(f"Error rate: {err_base/len(LVAL)*100:.2f}%\n")

    print("--- LDA Threshold Search (on DVAL) ---")
    print(f"Best Threshold found: {best_t:.4f}")
    print(f"Best Error: {best_err} ({best_err/len(LVAL)*100:.2f}%)")
    print(f"Improvement: {(err_base - best_err)/len(LVAL)*100:.2f}%\n")

    print("--- PCA + LDA Classification Sweep ---")
    for m in range(1, 7):
        UPCA = pca.compute_pca(DTR, m=m)
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)

        ULDA_pca = lda.compute_lda_geig(DTR_pca, LTR, m=1)
        DTR_lda_pca = lda.apply_lda(ULDA_pca, DTR_pca)

        if DTR_lda_pca[0, LTR == 0].mean() > DTR_lda_pca[0, LTR == 1].mean():
            ULDA_pca = -ULDA_pca
            DTR_lda_pca = lda.apply_lda(ULDA_pca, DTR_pca)

        mean_cls0_pca = DTR_lda_pca[0, LTR == 0].mean()
        mean_cls1_pca = DTR_lda_pca[0, LTR == 1].mean()
        threshold_pca = (mean_cls0_pca + mean_cls1_pca) / 2.0

        DVAL_lda_pca = lda.apply_lda(ULDA_pca, DVAL_pca)

        PVAL_pca = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL_pca[DVAL_lda_pca[0] >= threshold_pca] = 1
        PVAL_pca[DVAL_lda_pca[0] < threshold_pca] = 0

        errors_pca = (PVAL_pca != LVAL).sum()
        error_rate_pca = errors_pca / float(len(LVAL)) * 100
        print(f"PCA m={m} -> Errors: {errors_pca}, Error rate: {error_rate_pca:.2f}%")


if __name__ == "__main__":
    logger.setup_logger("out/05_threshold_classification.txt")

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    threshold_search_and_classify(D, L)
