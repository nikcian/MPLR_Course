"""
Laboratory 8 - Project task: Support Vector Machines on the fingerprint-spoofing dataset.

Target application: pi_T = 0.1.  SVM scores are NOT probabilistic (no LLR interpretation), so
actDCF is computed assuming the score behaves like an LLR - it is expected to be miscalibrated.
minDCF is still meaningful as a measure of pure class separation.

Experiments (in order):
  1. Linear SVM, K = 1, C in numpy.logspace(-5, 0, 11).  Non-centered data.
  2. Linear SVM, same C grid, on CENTERED data (DTR/DVAL mean-shifted using DTR's mean).
  3. Polynomial kernel SVM, d = 2, c = 1, xi = 0, non-centered.
  4. RBF kernel SVM with xi = 1, grid over gamma in {e^-4, e^-3, e^-2, e^-1} and
     C in numpy.logspace(-3, 2, 11).  One line per gamma in the plots.
  5. [OPTIONAL]  Polynomial kernel d = 4, c = 1, xi = 0, C in numpy.logspace(-5, 0, 11).

Best (by minDCF) validation scores from non-optional models are persisted under out/lab08/.
"""

import os
import sys

import numpy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, evaluation, logger
from utils.math_utils import vcol, vrow
from models import svm


PLOT_DIR = "out/plots/lab08"
DATA_DIR = "out/lab08"

TARGET_PRIOR = 0.1
C_LINEAR = numpy.logspace(-5, 0, 11)
C_RBF = numpy.logspace(-3, 2, 11)
RBF_GAMMAS = [numpy.exp(-4), numpy.exp(-3), numpy.exp(-2), numpy.exp(-1)]
RBF_GAMMA_LABELS = [r"$\gamma = e^{-4}$", r"$\gamma = e^{-3}$",
                    r"$\gamma = e^{-2}$", r"$\gamma = e^{-1}$"]
C_OPTIONAL = numpy.logspace(-5, 0, 11)


def evaluate_scores(scores, LVAL):
    actDCF = evaluation.compute_actDCF_binary_fast(scores, LVAL, TARGET_PRIOR, 1.0, 1.0)
    minDCF = evaluation.compute_minDCF_binary_fast(scores, LVAL, TARGET_PRIOR, 1.0, 1.0)
    return actDCF, minDCF


def sweep_linear(DTR, LTR, DVAL, LVAL, K=1.0):
    rows = []
    best = None
    for C in C_LINEAR:
        w, b, p, d = svm.train_dual_SVM_linear(DTR, LTR, C, K=K)
        sVal = (vrow(w) @ DVAL + b).ravel()
        act, mn = evaluate_scores(sVal, LVAL)
        rows.append((C, p, d, p - d, act, mn))
        if best is None or mn < best["minDCF"]:
            best = {"C": C, "w": w, "b": b, "actDCF": act, "minDCF": mn, "scores": sVal}
    return rows, best


def sweep_kernel(DTR, LTR, DVAL, LVAL, kernel_factory, C_list, eps):
    rows = []
    best = None
    for C in C_list:
        fScore, p, d = svm.train_dual_SVM_kernel(DTR, LTR, C, kernel_factory, eps=eps)
        sVal = fScore(DVAL)
        act, mn = evaluate_scores(sVal, LVAL)
        rows.append((C, p, d, p - d, act, mn))
        if best is None or mn < best["minDCF"]:
            best = {"C": C, "actDCF": act, "minDCF": mn, "scores": sVal}
    return rows, best


def print_table(title, rows):
    print(f"\n--- {title} ---")
    print(f"{'C':>10} | {'primal':>12} | {'dual':>12} | {'dual.gap':>10} | "
          f"{'actDCF':>7} | {'minDCF':>7}")
    print("-" * 76)
    for C, p, d, gap, act, mn in rows:
        print(f"{C:>10.1e} | {p:>12.6e} | {d:>12.6e} | {gap:>10.2e} | "
              f"{act:>7.4f} | {mn:>7.4f}")


def plot_dcf_vs_C(rows, title, fname, optional=False):
    Cs = [r[0] for r in rows]
    act = [r[4] for r in rows]
    mn = [r[5] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(Cs, act, marker="o", label="actDCF", color="r")
    plt.plot(Cs, mn, marker="s", label="minDCF", color="b", linestyle="--")
    plt.xscale("log", base=10)
    plt.xlabel("C")
    plt.ylabel(f"DCF (pi_T = {TARGET_PRIOR})")
    suffix = "  [OPTIONAL]" if optional else ""
    plt.title(title + suffix)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()


def plot_rbf_grid(grid_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for label, color, rows in zip(RBF_GAMMA_LABELS, colors, grid_results):
        Cs = [r[0] for r in rows]
        axes[0].plot(Cs, [r[5] for r in rows], marker="s", linestyle="--",
                     label=label, color=color)
        axes[1].plot(Cs, [r[4] for r in rows], marker="o",
                     label=label, color=color)
    for ax, name in zip(axes, ["minDCF", "actDCF"]):
        ax.set_xscale("log", base=10)
        ax.set_xlabel("C")
        ax.set_ylabel(f"{name} (pi_T = {TARGET_PRIOR})")
        ax.set_title(f"RBF SVM ({name}) - grid over gamma")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/04_rbf_grid.png")
    plt.close()


def save_scores(tag, best):
    numpy.save(f"{DATA_DIR}/{tag}_scores.npy", best["scores"])
    print(f"  saved {tag}: C={best['C']:.1e}, minDCF={best['minDCF']:.4f}, "
          f"actDCF={best['actDCF']:.4f}")


if __name__ == "__main__":
    logger.setup_logger("out/lab08_svm.txt")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)
    print(f"DTR shape: {DTR.shape}, DVAL shape: {DVAL.shape}")
    pi_emp = (LTR == 1).sum() / LTR.size
    print(f"Empirical pi_1 on DTR: {pi_emp:.4f}")

    # 1) Linear SVM, K = 1, non-centered.
    print("\n############### 1) LINEAR SVM (K=1, non-centered) ###############")
    rows_lin, best_lin = sweep_linear(DTR, LTR, DVAL, LVAL, K=1.0)
    print_table("Linear SVM, K=1, non-centered", rows_lin)
    plot_dcf_vs_C(rows_lin, "Linear SVM (K=1) - non-centered", "01_linear.png")
    # Observation: for SVM, *low* C means strong regularization (large margin). minDCF is
    # essentially flat across C - the hyperplane is not the bottleneck here. actDCF is
    # consistently larger than minDCF, reflecting score miscalibration.

    # 2) Linear SVM on centered data (subtract DTR mean from both DTR and DVAL).
    print("\n############### 2) LINEAR SVM (K=1, centered) ###############")
    muTR = vcol(DTR.mean(1))
    DTRc = DTR - muTR
    DVALc = DVAL - muTR
    rows_lin_c, best_lin_c = sweep_linear(DTRc, LTR, DVALc, LVAL, K=1.0)
    print_table("Linear SVM, K=1, centered", rows_lin_c)
    plot_dcf_vs_C(rows_lin_c, "Linear SVM (K=1) - centered", "02_linear_centered.png")
    # Observation: SVMs (without the extended-feature trick) are affine-invariant for the loss,
    # but here we regularize the bias too: centering shifts the optimal b, so we expect small but
    # non-zero differences. On this dataset the impact is mild.

    # 3) Polynomial kernel d=2, c=1, xi=0, non-centered.
    print("\n############### 3) POLY KERNEL SVM (d=2, c=1, xi=0) ###############")
    rows_poly, best_poly = sweep_kernel(
        DTR, LTR, DVAL, LVAL,
        kernel_factory=svm.polyKernel(degree=2, c=1.0),
        C_list=C_LINEAR,
        eps=0.0,
    )
    print_table("Poly SVM (d=2, c=1, xi=0)", rows_poly)
    plot_dcf_vs_C(rows_poly, "Polynomial SVM (d=2, c=1, xi=0)", "03_poly_d2.png")
    # Observation: the degree-2 expansion gives a quadratic separation surface. We expect minDCF
    # to drop substantially compared to the linear SVM, consistent with the gap between linear and
    # quadratic logistic regression and between Tied and MVG Gaussian classifiers.

    # 4) RBF grid (gamma x C).
    print("\n############### 4) RBF KERNEL SVM (grid gamma x C, xi=1) ###############")
    rbf_results = []
    best_rbf = None
    best_rbf_gamma = None
    for gamma, glabel in zip(RBF_GAMMAS, RBF_GAMMA_LABELS):
        rows_rbf, best_rbf_g = sweep_kernel(
            DTR, LTR, DVAL, LVAL,
            kernel_factory=svm.rbfKernel(gamma=gamma),
            C_list=C_RBF,
            eps=1.0,
        )
        print_table(f"RBF SVM ({glabel}, xi=1)", rows_rbf)
        rbf_results.append(rows_rbf)
        if best_rbf is None or best_rbf_g["minDCF"] < best_rbf["minDCF"]:
            best_rbf = best_rbf_g
            best_rbf_gamma = gamma
            best_rbf["gamma"] = gamma
    plot_rbf_grid(rbf_results)
    print(f"\nBest RBF model: gamma = {best_rbf_gamma:.4f}, C = {best_rbf['C']:.1e}, "
          f"minDCF = {best_rbf['minDCF']:.4f}, actDCF = {best_rbf['actDCF']:.4f}")
    # Observation: gamma controls the locality of the kernel (small gamma => wide bumps => smoother
    # boundary). On this dataset, an intermediate value should give the best minDCF. actDCF is
    # often poor without calibration.

    # 5) [OPTIONAL] Polynomial kernel d=4, c=1, xi=0.
    print("\n############### 5) [OPTIONAL] POLY KERNEL SVM (d=4, c=1, xi=0) ###############")
    rows_poly4, best_poly4 = sweep_kernel(
        DTR, LTR, DVAL, LVAL,
        kernel_factory=svm.polyKernel(degree=4, c=1.0),
        C_list=C_OPTIONAL,
        eps=0.0,
    )
    print_table("[OPTIONAL] Poly SVM (d=4, c=1, xi=0)", rows_poly4)
    plot_dcf_vs_C(rows_poly4, "Polynomial SVM (d=4, c=1, xi=0)",
                  "05_poly_d4_optional.png", optional=True)
    # Optional discussion (lab hint): focus on features 4-6. A degree-2 map y -> z = y0 * y1 sends
    # the four clusters (++ -- +- -+) to a 1-D space where the genuine class concentrates near
    # large |z| and the impostors near 0 (or vice versa). A degree-4 monomial gives even more
    # finely tuned interval-like boundaries on these products, which is why d=4 can outperform
    # d=2 on this particular dataset.

    # --- Comparison summary ---
    print("\n############### Comparison at pi_T = 0.1 ###############")
    print(f"{'model':>40} | {'actDCF':>7} | {'minDCF':>7}")
    print("-" * 62)
    print(f"{'Linear SVM (best C, non-centered)':>40} | "
          f"{best_lin['actDCF']:>7.4f} | {best_lin['minDCF']:>7.4f}  [C={best_lin['C']:.1e}]")
    print(f"{'Linear SVM (best C, centered)':>40} | "
          f"{best_lin_c['actDCF']:>7.4f} | {best_lin_c['minDCF']:>7.4f}  [C={best_lin_c['C']:.1e}]")
    print(f"{'Poly SVM d=2 (best C)':>40} | "
          f"{best_poly['actDCF']:>7.4f} | {best_poly['minDCF']:>7.4f}  [C={best_poly['C']:.1e}]")
    print(f"{'RBF SVM (best gamma, C)':>40} | "
          f"{best_rbf['actDCF']:>7.4f} | {best_rbf['minDCF']:>7.4f}  "
          f"[gamma={best_rbf['gamma']:.4f}, C={best_rbf['C']:.1e}]")
    print(f"{'[OPTIONAL] Poly SVM d=4 (best C)':>40} | "
          f"{best_poly4['actDCF']:>7.4f} | {best_poly4['minDCF']:>7.4f}  [C={best_poly4['C']:.1e}]")

    # --- Persist best scores for calibration lab ---
    print("\n--- Saving best (by minDCF) validation scores ---")
    save_scores("linear", best_lin)
    save_scores("linear_centered", best_lin_c)
    save_scores("poly_d2", best_poly)
    save_scores("rbf", best_rbf)
    save_scores("poly_d4_OPTIONAL", best_poly4)
