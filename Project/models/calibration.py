"""Score calibration and score-level fusion via prior-weighted logistic regression.

A 1-D affine calibrator on raw scores s is obtained by training a prior-weighted (non
regularised) binary logistic regression with the target prior pT, then converting the
posterior-like LR output back to a LLR-like score:

    f_cal(s) = alpha * s + gamma ,   gamma = b - log(pT / (1 - pT))

with (alpha, b) the LR weight/bias.  The same recipe extends to score-level fusion of
multiple systems by stacking their scores as a (M, N) matrix and training an M-D LR
calibrator (the resulting fuser is an affine function of all system scores).

For both calibration and fusion we also provide a K-fold pooling routine: a separate
calibrator is fit on K-1 folds, applied to the held-out fold, and the held-out scores
(plus their labels, kept aligned) are pooled.  Pooled scores power the model-selection
metrics in lab 10.
"""

import numpy

from utils.math_utils import vrow
from models.logistic_regression import trainWeightedLogRegBinary


def train_calibration(scores_1d, labels, pT, l=0.0):
    """Fit an affine calibrator on a 1-D array of raw scores.

    Returns (alpha, gamma) so that calibrated = alpha * s + gamma is LLR-like for
    the application prior pT.
    """
    w, b, _ = trainWeightedLogRegBinary(vrow(scores_1d), labels, l, pT)
    alpha = float(w[0])
    gamma = float(b) - numpy.log(pT / (1.0 - pT))
    return alpha, gamma


def apply_calibration(scores_1d, alpha, gamma):
    return alpha * scores_1d + gamma


def train_fusion(scores_2d, labels, pT, l=0.0):
    """Fit a linear fuser over multiple systems.

    scores_2d has shape (M, N) with one system per row.  Returns (alphas, gamma) so
    that fused = alphas @ scores + gamma is LLR-like for the application prior pT.
    """
    w, b, _ = trainWeightedLogRegBinary(scores_2d, labels, l, pT)
    gamma = float(b) - numpy.log(pT / (1.0 - pT))
    return w, gamma


def apply_fusion(scores_2d, alphas, gamma):
    return (vrow(alphas) @ scores_2d).ravel() + gamma


def extract_train_val_folds_from_ary(X, idx, K):
    """Pull fold idx out of X as validation; pool the remaining K-1 folds as training.

    Works on both 1-D score arrays and (M, N) score matrices (split along columns).
    """
    if X.ndim == 1:
        train = numpy.hstack([X[jdx::K] for jdx in range(K) if jdx != idx])
        val = X[idx::K]
    else:
        train = numpy.hstack([X[:, jdx::K] for jdx in range(K) if jdx != idx])
        val = X[:, idx::K]
    return train, val


def kfold_calibrate(scores_1d, labels, pT, K=5, l=0.0):
    """K-fold pooled calibration of a single system.

    Trains K calibrators (each on K-1 folds), applies each to its held-out fold, and
    returns the pooled calibrated scores together with the corresponding pooled labels
    (re-ordered to match the score pooling).
    """
    pooled_s, pooled_l = [], []
    for foldIdx in range(K):
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_1d, foldIdx, K)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, K)
        alpha, gamma = train_calibration(SCAL, LCAL, pT, l=l)
        pooled_s.append(apply_calibration(SVAL, alpha, gamma))
        pooled_l.append(LVAL)
    return numpy.hstack(pooled_s), numpy.hstack(pooled_l)


def kfold_fuse(scores_2d, labels, pT, K=5, l=0.0):
    """K-fold pooled fusion of multiple systems.

    scores_2d has shape (M, N).  Returns pooled fused scores and pooled labels (the
    label pooling order is the same as the score pooling order).
    """
    pooled_s, pooled_l = [], []
    for foldIdx in range(K):
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_2d, foldIdx, K)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, K)
        alphas, gamma = train_fusion(SCAL, LCAL, pT, l=l)
        pooled_s.append(apply_fusion(SVAL, alphas, gamma))
        pooled_l.append(LVAL)
    return numpy.hstack(pooled_s), numpy.hstack(pooled_l)
