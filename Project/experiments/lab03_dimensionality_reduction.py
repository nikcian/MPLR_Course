"""
Laboratory 3 - Project task: Dimensionality reduction (PCA + LDA) and LDA-based classification.

What the lab asks for, mapped to this script:
  - Apply PCA to the 6 features and plot histograms of all 6 principal directions.
    Question: does PCA help separate the classes?  (Answer: no, with 6 dims it's just a rotation.)
  - Apply LDA (1-D, since binary task) and plot the histogram of the projected samples.
    Question: do the classes overlap less along the LDA direction than along the raw features?
  - Use LDA as a classifier on a 2/3-1/3 model-training / validation split. Report the error rate
    obtained with the "mean of projected class means" threshold, then try other thresholds.
  - Apply PCA *before* LDA (m = 1 .. 6) and check whether PCA pre-processing helps.

Important: PCA and LDA are estimated only on the model-training subset (DTR), never on validation
data, otherwise we leak information from the test set into the trained model.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, plot_utils, logger
from models import pca, lda


PLOT_DIR = "out/plots/lab03"


def pca_analysis(D, L):
    print("\n--- PCA: projection of all 6 features ---")
    # PCA with m = 6 is just a rotation of the feature space (information is preserved).
    # We compute the projection of the full dataset so we can visualise the per-direction histograms.
    P = pca.compute_pca(D, D.shape[0])
    DP = pca.apply_pca(P, D)
    plot_utils.plot_hist(
        DP, L, filename_prefix="hist_pca", out_dir=PLOT_DIR, xlabel_prefix="PCA dir"
    )
    # Scatter for every pair of PCA directions — same convention as lab 2: both classes on the
    # same figure so the overlap (or absence of it) is immediately visible.
    plot_utils.plot_scatter(
        DP, L, filename_prefix="scatter_pca", out_dir=PLOT_DIR, all_pairs=True
    )

    # Per-direction summary statistics — useful to confirm that the leading directions concentrate
    # most of the variance (and that classes are not magically separated).
    for dIdx in range(DP.shape[0]):
        m0, s0 = DP[dIdx, L == 0].mean(), DP[dIdx, L == 0].std()
        m1, s1 = DP[dIdx, L == 1].mean(), DP[dIdx, L == 1].std()
        print(
            f"PCA dir {dIdx + 1}: "
            f"class0 (μ={m0:+.4f}, σ={s0:.4f})  class1 (μ={m1:+.4f}, σ={s1:.4f})  |Δμ|={abs(m0 - m1):.4f}"
        )

    # Note: the histograms look different from the original features, but recall that PCA with
    # m = dim(D) is just an orthogonal rotation. The clusters have not actually moved, just the basis
    # used to describe them — this is a classic visual trap noted by the lab text.


def lda_analysis(D, L):
    print("\n--- LDA: 1 discriminant direction (binary problem) ---")
    # Binary task ⇒ at most 1 LDA direction (in general K-1 with K classes).
    U = lda.compute_lda_geig(D, L, m=1)
    DP = lda.apply_lda(U, D)

    m0, s0 = DP[0, L == 0].mean(), DP[0, L == 0].std()
    m1, s1 = DP[0, L == 1].mean(), DP[0, L == 1].std()
    print(f"class0 (μ={m0:+.4f}, σ={s0:.4f})  class1 (μ={m1:+.4f}, σ={s1:.4f})  |Δμ|={abs(m0 - m1):.4f}")

    plot_utils.plot_lda_hist(DP, L, filename="hist_lda_1.png", out_dir=PLOT_DIR)
    # The LDA direction maximises the Fisher ratio Sb/Sw on the whole training set: classes overlap
    # noticeably less than along any single original feature, but the projection cannot "create"
    # separation that wasn't already present in the original feature space.


def _orient_lda_so_class1_is_higher(ULDA, DTR, LTR):
    # The sign of the LDA direction is arbitrary (eigenvectors are defined up to ±1). We fix the
    # orientation so that the projected mean of class 1 (genuine) is larger than that of class 0
    # — required for the "samples above threshold = class 1" decision rule to make sense.
    DTR_lda = lda.apply_lda(ULDA, DTR)
    if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)
    return ULDA, DTR_lda


def lda_classification(D, L):
    print("\n--- LDA classifier (basic threshold + best threshold) ---")
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)
    ULDA, DTR_lda = _orient_lda_so_class1_is_higher(ULDA, DTR, LTR)
    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    # Default threshold: half-way between the projected class means computed on the TRAINING split.
    base_threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
    PVAL_base = (DVAL_lda[0] >= base_threshold).astype(np.int32)
    err_base = (PVAL_base != LVAL).sum()
    print(f"Mean-of-means threshold = {base_threshold:.4f}")
    print(f"  Errors on DVAL: {err_base}/{len(LVAL)}  ({err_base / len(LVAL) * 100:.2f}%)")

    # Threshold sweep on the validation set itself. This is a small "oracle" search just to gauge
    # how much room for improvement is left — we are NOT doing model selection here, only studying
    # the calibration loss of the default threshold.
    thresholds = np.linspace(DVAL_lda.min(), DVAL_lda.max(), 1000)
    best_err = len(LVAL)
    best_t = base_threshold
    for t in thresholds:
        err = ((DVAL_lda[0] >= t).astype(np.int32) != LVAL).sum()
        if err < best_err:
            best_err = err
            best_t = t
    print(f"Best threshold (sweep on DVAL) = {best_t:.4f}")
    print(f"  Errors: {best_err}/{len(LVAL)}  ({best_err / len(LVAL) * 100:.2f}%)")
    print(f"  Improvement over mean-of-means: {(err_base - best_err) / len(LVAL) * 100:.2f}%")


def pca_then_lda(D, L):
    print("\n--- PCA pre-processing + LDA classification (m = 1 .. 6) ---")
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    # Reducing to m = 1 makes LDA trivial; reducing to m = 6 is just a rotation and does not change
    # the LDA subspace. We sweep all values to see whether intermediate m's actually help here.
    for m in range(1, 7):
        UPCA = pca.compute_pca(DTR, m=m)
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)

        ULDA = lda.compute_lda_geig(DTR_pca, LTR, m=1)
        ULDA, DTR_lda = _orient_lda_so_class1_is_higher(ULDA, DTR_pca, LTR)
        DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)

        threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
        PVAL = (DVAL_lda[0] >= threshold).astype(np.int32)
        err = (PVAL != LVAL).sum()
        print(f"PCA m={m} -> errors {err}/{len(LVAL)}  ({err / len(LVAL) * 100:.2f}%)")

    # Empirically PCA pre-processing offers a marginal change here — the dataset only has 6 features
    # and the LDA direction is already close to the variance-maximising direction.


if __name__ == "__main__":
    logger.setup_logger("out/lab03_dimensionality_reduction.txt")
    plot_utils.plot_init()
    plot_utils.ensure_dir(PLOT_DIR)

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)

    pca_analysis(D, L)
    lda_analysis(D, L)
    lda_classification(D, L)
    pca_then_lda(D, L)
