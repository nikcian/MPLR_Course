"""
Laboratory 5 - Project task: Generative Gaussian classifiers on the fingerprint-spoofing dataset.

We implement three models and evaluate them on the same 2/3-1/3 split used in lab 3:
  - MVG         : Multivariate Gaussian Classifier (class-conditional Σ_c).
  - Naive Bayes : Diagonal class-conditional covariances Σ_c = diag(Σ_c) — features assumed
                  independent given the class.
  - Tied        : All classes share a single Σ = (1/N) Σ_c N_c Σ_c.  This is exactly the LDA model.

Lab questions addressed (with the answers we expect from this dataset):
  - LLR vs. uniform-prior decision: threshold t = 0  ⇒  predict class 1 if llr > 0.
  - Compare the covariance matrices and Pearson correlation matrices of the two classes — how
    correlated are the features?  (Generally very weakly: this is why Naive Bayes is competitive
    despite its strong independence assumption.)
  - Discard features 5-6 (which are not Gaussian — see Lab 4) and re-run: error rate drops for all
    three models, confirming that the bad fit hurt the classifiers.
  - Restrict to features 1-2: similar means, different variances ⇒ MVG should beat the Tied model.
  - Restrict to features 3-4: different means, similar variances ⇒ MVG and Tied behave similarly.
  - PCA pre-processing: m = 1 .. 6.  PCA usually does not help here (the dataset is already
    low-dimensional), but it can sometimes recover from numerical issues.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger
from utils.math_utils import vcol, vrow
from models import gaussian_models, pca


def evaluate_models(DTR, LTR, DVAL, LVAL):
    # Trains MVG / Naive Bayes / Tied on DTR and reports error rate on DVAL.
    # The LLR convention is log p(x|1) - log p(x|0); class 1 = genuine = "True" hypothesis.
    results = {}
    for name, estimator in [
        ("MVG", gaussian_models.Gau_MVG_ML_estimates),
        ("Naive Bayes", gaussian_models.Gau_Naive_ML_estimates),
        ("Tied", gaussian_models.Gau_Tied_ML_estimates),
    ]:
        hParams = estimator(DTR, LTR)
        LLR = gaussian_models.compute_llr(
            DVAL, hParams[1][0], hParams[1][1], hParams[0][0], hParams[0][1]
        )
        # Uniform priors  ⇒  threshold = -log(π1/π0) = 0.
        PVAL = gaussian_models.predict_from_llr(LLR, threshold=0.0)
        err = gaussian_models.compute_error_rate(PVAL, LVAL)
        results[name] = err
    return results


def print_results(title, results):
    print(f"\n--- {title} ---")
    for name in ["MVG", "Naive Bayes", "Tied"]:
        print(f"  {name:<12} error rate: {results[name]:.2f}%")


def correlation_analysis(DTR, LTR):
    # The lab asks for the per-class covariance and Pearson correlation matrices.
    # Pearson rho_ij = Cov_ij / (sqrt(Var_i) * sqrt(Var_j)) — diagonals are 1, off-diagonals in
    # [-1, +1]; values near 0 indicate weak linear dependence.
    print("\n--- Per-class covariance / correlation (full 6-feature MVG) ---")
    hParams = gaussian_models.Gau_MVG_ML_estimates(DTR, LTR)
    for cls in [0, 1]:
        C = hParams[cls][1]
        print(f"\nClass {cls}")
        print("Covariance:")
        print(np.array2string(C, precision=4, suppress_small=True))
        Corr = C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))
        print("Correlation (Pearson):")
        print(np.array2string(Corr, precision=4, suppress_small=True))
        # The off-diagonals are typically small in absolute value (well below the diagonal scale),
        # i.e. features are weakly correlated — the Naive Bayes assumption is not catastrophic.


if __name__ == "__main__":
    logger.setup_logger("out/lab05_gaussian_classification.txt")

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    # 1) Correlation / covariance inspection
    correlation_analysis(DTR, LTR)

    # 2) Full feature set
    res_all = evaluate_models(DTR, LTR, DVAL, LVAL)
    print_results("All features (1-6)", res_all)

    # 3) Drop the non-Gaussian features 5-6
    res_1_4 = evaluate_models(DTR[0:4, :], LTR, DVAL[0:4, :], LVAL)
    print_results("Features 1-4 (drop the non-Gaussian 5-6)", res_1_4)
    # We expect all three error rates to drop here: features 5-6 violate the Gaussian assumption
    # (cf. Lab 4 plots), so removing them yields a model whose assumptions match the data better.

    # 4) Features 1-2: similar means, different variances ⇒ MVG wins
    res_1_2 = evaluate_models(DTR[0:2, :], LTR, DVAL[0:2, :], LVAL)
    print_results("Features 1-2 only (similar means, different variances)", res_1_2)
    # The Tied model is forced to use one shared Σ; when variances differ between classes it cannot
    # capture this and underperforms.  MVG can model the per-class spread → lower error.

    # 5) Features 3-4: different means, similar variances ⇒ MVG and Tied behave similarly
    res_3_4 = evaluate_models(DTR[2:4, :], LTR, DVAL[2:4, :], LVAL)
    print_results("Features 3-4 only (different means, similar variances)", res_3_4)
    # Here the tied-Σ assumption matches the data; MVG's extra flexibility brings little benefit
    # (and can even hurt slightly due to more parameters to estimate on the same data).

    # 6) PCA pre-processing
    print("\n--- PCA + Gaussian classifiers (m = 1 .. 6) ---")
    for m in range(1, 7):
        UPCA = pca.compute_pca(DTR, m=m)
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)
        res = evaluate_models(DTR_pca, LTR, DVAL_pca, LVAL)
        print(
            f"  m={m}  MVG: {res['MVG']:.2f}%  "
            f"Naive: {res['Naive Bayes']:.2f}%  Tied: {res['Tied']:.2f}%"
        )
    # PCA is generally not helpful for the Gaussian classifiers on this dataset because the original
    # space is already low-dimensional (6 features). For Naive Bayes it can even help a little, since
    # PCA decorrelates the features and that matches Naive Bayes's diagonal-covariance assumption.
