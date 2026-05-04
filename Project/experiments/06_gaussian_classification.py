import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger
from models import gaussian_models, pca
from utils.math_utils import vcol, vrow


def evaluate_models(DTR, LTR, DVAL, LVAL):
    # MVG
    hParams_MVG = gaussian_models.Gau_MVG_ML_estimates(DTR, LTR)
    LLR_MVG = gaussian_models.compute_llr(
        DVAL, hParams_MVG[1][0], hParams_MVG[1][1], hParams_MVG[0][0], hParams_MVG[0][1]
    )
    PVAL_MVG = gaussian_models.predict_from_llr(LLR_MVG)
    err_MVG = gaussian_models.compute_error_rate(PVAL_MVG, LVAL)

    # Naive Bayes
    hParams_Naive = gaussian_models.Gau_Naive_ML_estimates(DTR, LTR)
    LLR_Naive = gaussian_models.compute_llr(
        DVAL,
        hParams_Naive[1][0],
        hParams_Naive[1][1],
        hParams_Naive[0][0],
        hParams_Naive[0][1],
    )
    PVAL_Naive = gaussian_models.predict_from_llr(LLR_Naive)
    err_Naive = gaussian_models.compute_error_rate(PVAL_Naive, LVAL)

    # Tied
    hParams_Tied = gaussian_models.Gau_Tied_ML_estimates(DTR, LTR)
    LLR_Tied = gaussian_models.compute_llr(
        DVAL,
        hParams_Tied[1][0],
        hParams_Tied[1][1],
        hParams_Tied[0][0],
        hParams_Tied[0][1],
    )
    PVAL_Tied = gaussian_models.predict_from_llr(LLR_Tied)
    err_Tied = gaussian_models.compute_error_rate(PVAL_Tied, LVAL)

    return err_MVG, err_Naive, err_Tied


def print_correlation_matrices(DTR, LTR):
    print("--- Covariance and Correlation Analysis ---")
    hParams_MVG = gaussian_models.Gau_MVG_ML_estimates(DTR, LTR)

    for cls in [0, 1]:
        print(f"\nClass {cls}")
        C = hParams_MVG[cls][1]
        print("Covariance Matrix:")
        print(np.array2string(C, precision=4, suppress_small=True))

        # Pearson correlation matrix
        Corr = C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))
        print("Correlation Matrix:")
        print(np.array2string(Corr, precision=4, suppress_small=True))


if __name__ == "__main__":
    logger.setup_logger("out/06_gaussian_classification.txt")

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    # Correlation Analysis
    print_correlation_matrices(DTR, LTR)
    print("\n" + "=" * 50 + "\n")

    # Analysis 1: All features (1 to 6)
    print("--- Classification with ALL Features (1 to 6) ---")
    err_MVG, err_Naive, err_Tied = evaluate_models(DTR, LTR, DVAL, LVAL)
    print(f"MVG Error rate:         {err_MVG:.2f}%")
    print(f"Naive Bayes Error rate: {err_Naive:.2f}%")
    print(f"Tied Error rate:        {err_Tied:.2f}%")
    print()

    # Analysis 2: Features 1 to 4
    print("--- Classification with Features 1 to 4 ---")
    DTR_1_4 = DTR[0:4, :]
    DVAL_1_4 = DVAL[0:4, :]
    err_MVG, err_Naive, err_Tied = evaluate_models(DTR_1_4, LTR, DVAL_1_4, LVAL)
    print(f"MVG Error rate:         {err_MVG:.2f}%")
    print(f"Naive Bayes Error rate: {err_Naive:.2f}%")
    print(f"Tied Error rate:        {err_Tied:.2f}%")
    print()

    # Analysis 3: Features 1 and 2
    print("--- Classification with Features 1 and 2 ---")
    DTR_1_2 = DTR[0:2, :]
    DVAL_1_2 = DVAL[0:2, :]
    err_MVG, err_Naive, err_Tied = evaluate_models(DTR_1_2, LTR, DVAL_1_2, LVAL)
    print(f"MVG Error rate:         {err_MVG:.2f}%")
    print(f"Naive Bayes Error rate: {err_Naive:.2f}%")
    print(f"Tied Error rate:        {err_Tied:.2f}%")
    print()

    # Analysis 4: Features 3 and 4
    print("--- Classification with Features 3 and 4 ---")
    DTR_3_4 = DTR[2:4, :]
    DVAL_3_4 = DVAL[2:4, :]
    err_MVG, err_Naive, err_Tied = evaluate_models(DTR_3_4, LTR, DVAL_3_4, LVAL)
    print(f"MVG Error rate:         {err_MVG:.2f}%")
    print(f"Naive Bayes Error rate: {err_Naive:.2f}%")
    print(f"Tied Error rate:        {err_Tied:.2f}%")
    print("\n" + "=" * 50 + "\n")

    # Analysis 5: PCA Pre-processing
    print("--- PCA + Generative Gaussian Models ---")
    for m in range(1, 7):
        UPCA = pca.compute_pca(DTR, m=m)
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)

        err_MVG, err_Naive, err_Tied = evaluate_models(DTR_pca, LTR, DVAL_pca, LVAL)

        print(f"PCA m={m}")
        print(f"  MVG Error rate:         {err_MVG:.2f}%")
        print(f"  Naive Bayes Error rate: {err_Naive:.2f}%")
        print(f"  Tied Error rate:        {err_Tied:.2f}%")
