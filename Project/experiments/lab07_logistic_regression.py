"""
Laboratory 7 - Project task: Binary logistic regression on the fingerprint-spoofing dataset.

Target application: pi_T = 0.1 (most users are impostors). For every model we sweep
lambda in numpy.logspace(-4, 2, 13) and report actDCF / minDCF.  Conversion to LLR-like scores
is required by our DCF tooling:
  - non-weighted LR : subtract log(pi_emp / (1 - pi_emp))
  - prior-weighted LR (target prior pT): subtract log(pT / (1 - pT))

Experiments (in order):
  1. Standard LR, full DTR. Regularization should be mostly irrelevant - lots of training samples.
  2. Standard LR, reduced DTR (DTR[:, ::50]).  Few samples -> regularization matters a lot
     (under-regularised models overfit, over-regularised models underfit and lose probabilistic
     interpretation).
  3. Prior-weighted LR (pT = 0.1), full DTR.  Should look very similar to (1) on this task.
  4. Quadratic LR (feature expansion phi(x) = [vec(x x^T), x]), full DTR.  Captures non-linear
     boundaries.
  5. Comparison against Gaussian models (MVG / Naive / Tied) at pi_T = 0.1.

The best (w, b) and the validation LLR-like scores for each non-reduced variant are saved
under out/lab07/ for use in the calibration laboratory.
"""

import os
import sys

import numpy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, evaluation, logger
from models import gaussian_models, logistic_regression as lr


PLOT_DIR = "out/plots/lab07"
DATA_DIR = "out/lab07"

TARGET_PRIOR = 0.1
LAMBDAS = numpy.logspace(-4, 2, 13)


def sweep_lambda_standard(DTR, LTR, DVAL, LVAL, pi_emp):
    rows = []
    best = None
    for l in LAMBDAS:
        w, b, Jstar = lr.trainLogRegBinary(DTR, LTR, l)
        s = lr.compute_scores(w, b, DVAL)
        sLLR = lr.scores_to_llr(s, pi_emp)
        actDCF = evaluation.compute_actDCF_binary_fast(sLLR, LVAL, TARGET_PRIOR, 1.0, 1.0)
        minDCF = evaluation.compute_minDCF_binary_fast(sLLR, LVAL, TARGET_PRIOR, 1.0, 1.0)
        rows.append((l, Jstar, actDCF, minDCF))
        if best is None or minDCF < best["minDCF"]:
            best = {"l": l, "w": w, "b": b, "Jstar": Jstar,
                    "actDCF": actDCF, "minDCF": minDCF, "sLLR": sLLR}
    return rows, best


def sweep_lambda_weighted(DTR, LTR, DVAL, LVAL, pT):
    rows = []
    best = None
    for l in LAMBDAS:
        w, b, Jstar = lr.trainWeightedLogRegBinary(DTR, LTR, l, pT)
        s = lr.compute_scores(w, b, DVAL)
        sLLR = lr.scores_to_llr(s, pT)
        actDCF = evaluation.compute_actDCF_binary_fast(sLLR, LVAL, TARGET_PRIOR, 1.0, 1.0)
        minDCF = evaluation.compute_minDCF_binary_fast(sLLR, LVAL, TARGET_PRIOR, 1.0, 1.0)
        rows.append((l, Jstar, actDCF, minDCF))
        if best is None or minDCF < best["minDCF"]:
            best = {"l": l, "w": w, "b": b, "Jstar": Jstar,
                    "actDCF": actDCF, "minDCF": minDCF, "sLLR": sLLR}
    return rows, best


def print_table(title, rows):
    print(f"\n--- {title} ---")
    print(f"{'lambda':>10} | {'J*(w,b)':>12} | {'actDCF':>7} | {'minDCF':>7}")
    print("-" * 46)
    for l, Jstar, actDCF, minDCF in rows:
        print(f"{l:>10.1e} | {Jstar:>12.6e} | {actDCF:>7.4f} | {minDCF:>7.4f}")


def plot_dcf_vs_lambda(rows, title, fname):
    ls = [r[0] for r in rows]
    act = [r[2] for r in rows]
    mn = [r[3] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ls, act, marker="o", label="actDCF", color="r")
    plt.plot(ls, mn, marker="s", label="minDCF", color="b", linestyle="--")
    plt.xscale("log", base=10)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(f"DCF (pi_T = {TARGET_PRIOR})")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()


def gaussian_baseline(DTR, LTR, DVAL, LVAL):
    out = {}
    for name, estimator in [
        ("MVG", gaussian_models.Gau_MVG_ML_estimates),
        ("Naive Bayes", gaussian_models.Gau_Naive_ML_estimates),
        ("Tied", gaussian_models.Gau_Tied_ML_estimates),
    ]:
        hParams = estimator(DTR, LTR)
        LLR = gaussian_models.compute_llr(
            DVAL, hParams[1][0], hParams[1][1], hParams[0][0], hParams[0][1]
        )
        actDCF = evaluation.compute_actDCF_binary_fast(LLR, LVAL, TARGET_PRIOR, 1.0, 1.0)
        minDCF = evaluation.compute_minDCF_binary_fast(LLR, LVAL, TARGET_PRIOR, 1.0, 1.0)
        out[name] = {"actDCF": actDCF, "minDCF": minDCF}
    return out


def save_model(tag, best, DVAL_shape):
    numpy.save(f"{DATA_DIR}/{tag}_w.npy", best["w"])
    numpy.save(f"{DATA_DIR}/{tag}_b.npy", numpy.array(best["b"]))
    numpy.save(f"{DATA_DIR}/{tag}_sLLR.npy", best["sLLR"])
    print(f"  saved {tag}: lambda={best['l']:.1e}, minDCF={best['minDCF']:.4f}, "
          f"actDCF={best['actDCF']:.4f}, DVAL_size={DVAL_shape[1]}")


if __name__ == "__main__":
    logger.setup_logger("out/lab07_logistic_regression.txt")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)
    pi_emp = (LTR == 1).sum() / LTR.size
    print(f"DTR shape: {DTR.shape}, DVAL shape: {DVAL.shape}, empirical pi_1 = {pi_emp:.4f}")

    # 1) Standard logistic regression on the full training set.
    rows_full, best_full = sweep_lambda_standard(DTR, LTR, DVAL, LVAL, pi_emp)
    print_table("Standard LR (full DTR)", rows_full)
    plot_dcf_vs_lambda(rows_full, "Standard LR - full training set",
                       "01_std_full.png")
    # Observation: with ~4000 training samples regularization has very little effect on minDCF;
    # actDCF grows with lambda since the score scale is shrunk and the empirical-prior correction
    # is no longer enough to keep them calibrated.

    # 2) Standard logistic regression on a heavily sub-sampled training set (1 every 50 samples).
    DTR_small = DTR[:, ::50]
    LTR_small = LTR[::50]
    pi_emp_small = (LTR_small == 1).sum() / LTR_small.size
    print(f"\nReduced DTR shape: {DTR_small.shape}, empirical pi_1 = {pi_emp_small:.4f}")
    rows_small, _ = sweep_lambda_standard(DTR_small, LTR_small, DVAL, LVAL, pi_emp_small)
    print_table("Standard LR (DTR[:, ::50])", rows_small)
    plot_dcf_vs_lambda(rows_small, "Standard LR - reduced training set (1 out of 50)",
                       "02_std_reduced.png")
    # Observation: with very few samples a small lambda overfits (minDCF and actDCF degrade for
    # small lambda), an intermediate lambda gives the best minDCF, and very large lambda underfits
    # (minDCF rises again).

    # 3) Prior-weighted logistic regression on the full training set, target prior pi_T = 0.1.
    rows_w, best_w = sweep_lambda_weighted(DTR, LTR, DVAL, LVAL, TARGET_PRIOR)
    print_table(f"Prior-weighted LR (pT = {TARGET_PRIOR}, full DTR)", rows_w)
    plot_dcf_vs_lambda(rows_w, f"Prior-weighted LR (pT = {TARGET_PRIOR}) - full training set",
                       "03_weighted_full.png")
    # Observation: minDCF is essentially the same as the non-weighted model (the loss only reweights
    # the contribution of each sample, the decision boundary moves slightly).  The prior-weighted
    # model needs to know the target prior at training time, which is a practical disadvantage.

    # 4) Quadratic logistic regression on the full training set.
    DTR_q = lr.expand_features_quadratic(DTR)
    DVAL_q = lr.expand_features_quadratic(DVAL)
    print(f"\nQuadratic expansion: DTR_q shape = {DTR_q.shape}, DVAL_q shape = {DVAL_q.shape}")
    rows_q, best_q = sweep_lambda_standard(DTR_q, LTR, DVAL_q, LVAL, pi_emp)
    print_table("Quadratic LR (full DTR)", rows_q)
    plot_dcf_vs_lambda(rows_q, "Quadratic LR - full training set",
                       "04_quadratic_full.png")
    # Observation: the quadratic expansion strictly enlarges the hypothesis space, so it can fit
    # non-linear separation boundaries that the linear LR misses.  On this dataset we expect a
    # marked drop in minDCF compared to (1) - similar in spirit to the gap between Tied and MVG
    # Gaussian classifiers, which differ only in whether the boundary is linear or quadratic.

    # 5) Comparison with Gaussian models at pi_T = 0.1.
    gauss = gaussian_baseline(DTR, LTR, DVAL, LVAL)
    print("\n--- Comparison at pi_T = 0.1 ---")
    print(f"{'model':>32} | {'actDCF':>7} | {'minDCF':>7}")
    print("-" * 54)
    for name in ["MVG", "Naive Bayes", "Tied"]:
        print(f"{('Gaussian ' + name):>32} | "
              f"{gauss[name]['actDCF']:>7.4f} | {gauss[name]['minDCF']:>7.4f}")
    print(f"{('Standard LR (best lambda)'):>32} | "
          f"{best_full['actDCF']:>7.4f} | {best_full['minDCF']:>7.4f}  "
          f"[lambda = {best_full['l']:.1e}]")
    print(f"{('Prior-weighted LR (best lambda)'):>32} | "
          f"{best_w['actDCF']:>7.4f} | {best_w['minDCF']:>7.4f}  "
          f"[lambda = {best_w['l']:.1e}]")
    print(f"{('Quadratic LR (best lambda)'):>32} | "
          f"{best_q['actDCF']:>7.4f} | {best_q['minDCF']:>7.4f}  "
          f"[lambda = {best_q['l']:.1e}]")
    # Reading guide:
    #   - The linear LR matches the Tied Gaussian (both impose a linear boundary, just with
    #     different training criteria).
    #   - The quadratic LR matches the MVG model in expressiveness (quadratic boundary). On a
    #     dataset where the per-class covariances differ, we expect both to outperform the linear
    #     ones in minDCF.
    #   - Naive Bayes is competitive only because the features are nearly uncorrelated; otherwise it
    #     would underperform MVG.

    # 6) Persist best models / validation scores for the calibration laboratory.
    print("\n--- Saving best models for upcoming calibration lab ---")
    save_model("std_full", best_full, DVAL.shape)
    save_model("weighted_full", best_w, DVAL.shape)
    save_model("quadratic_full", best_q, DVAL_q.shape)
