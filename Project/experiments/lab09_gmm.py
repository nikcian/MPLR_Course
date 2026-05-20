"""
Laboratory 9 - Project task: Gaussian Mixture Models on the fingerprint-spoofing dataset.

Target application: pi_T = 0.1.  We follow the lab project section:

  1. Train full-covariance GMMs with K_c components per class, K_c in {1, 2, 4, 8, 16}.
     Sweep the full Cartesian product (K_fake, K_genuine) - 25 combos.  Score the validation set
     with the standard LLR llr(x) = log GMM_1(x) - log GMM_0(x) (no offset: GMM ML scores are
     already proper LLRs).  Report minDCF/actDCF at pi_T = 0.1.
  2. Compare the best GMM with the best LR (lab 7) and the best SVM (lab 8) at pi_T = 0.1.
  3. Bayes error plot over prior log-odds in (-4, +4) for the three best models.

Conventions follow CLAUDE.md: ML eigenvalue threshold psi = 1e-2 (same as the lab IRIS example),
LBG alpha = 0.1, EM stop on average ll delta < 1e-6.

The best GMM scores are saved under out/lab09/ for the calibration laboratory.
"""

import os
import sys

import numpy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, evaluation, logger
from models import gmm as gmm_module


PLOT_DIR = "out/plots/lab09"
DATA_DIR = "out/lab09"

TARGET_PRIOR = 0.1
COMPONENT_GRID = [1, 2, 4, 8, 16]
PSI_EIG = 1e-2
LBG_ALPHA = 0.1

# Locations where lab 7 and lab 8 saved their best validation scores (LLR-like).
LAB07_DIR = "out/lab07"
LAB08_DIR = "out/lab08"


def train_per_class(DTR, LTR, K0, K1, covType="Full"):
    """Train one GMM per class with K0 / K1 components."""
    gmm0 = gmm_module.train_GMM_LBG_EM(
        DTR[:, LTR == 0], K0, covType=covType, psiEig=PSI_EIG, lbgAlpha=LBG_ALPHA,
    )
    gmm1 = gmm_module.train_GMM_LBG_EM(
        DTR[:, LTR == 1], K1, covType=covType, psiEig=PSI_EIG, lbgAlpha=LBG_ALPHA,
    )
    return gmm0, gmm1


def sweep_components(DTR, LTR, DVAL, LVAL, covType="Full"):
    rows = []
    best = None
    for K0 in COMPONENT_GRID:
        for K1 in COMPONENT_GRID:
            gmm0, gmm1 = train_per_class(DTR, LTR, K0, K1, covType=covType)
            llr = gmm_module.compute_llr_binary(DVAL, gmm1, gmm0)
            actDCF = evaluation.compute_actDCF_binary_fast(llr, LVAL, TARGET_PRIOR, 1.0, 1.0)
            minDCF = evaluation.compute_minDCF_binary_fast(llr, LVAL, TARGET_PRIOR, 1.0, 1.0)
            rows.append((K0, K1, actDCF, minDCF))
            print(f"  K0={K0:>2}  K1={K1:>2}   actDCF={actDCF:.4f}   minDCF={minDCF:.4f}")
            if best is None or minDCF < best["minDCF"]:
                best = {"K0": K0, "K1": K1, "actDCF": actDCF, "minDCF": minDCF, "llr": llr,
                        "gmm0": gmm0, "gmm1": gmm1}
    return rows, best


def print_matrix_table(title, rows, key_idx):
    print(f"\n--- {title} ---")
    header = "        " + " ".join(f"K1={K1:>2}" for K1 in COMPONENT_GRID)
    print(header)
    by_K0 = {K0: {K1: None for K1 in COMPONENT_GRID} for K0 in COMPONENT_GRID}
    for K0, K1, actDCF, minDCF in rows:
        by_K0[K0][K1] = (actDCF, minDCF)[key_idx - 2]
    for K0 in COMPONENT_GRID:
        line = f"K0={K0:>2}   " + " ".join(f"{by_K0[K0][K1]:>5.4f}" for K1 in COMPONENT_GRID)
        print(line)


def bayes_error_plot(llrs_by_name, LVAL, fname):
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))

    n = len(llrs_by_name)
    plt.figure(figsize=(6 * n, 4.5))
    for idx, (name, llr) in enumerate(llrs_by_name.items()):
        actL, minL = [], []
        for pi_t in effPriors:
            actL.append(evaluation.compute_actDCF_binary_fast(llr, LVAL, pi_t, 1.0, 1.0))
            minL.append(evaluation.compute_minDCF_binary_fast(llr, LVAL, pi_t, 1.0, 1.0))
        plt.subplot(1, n, idx + 1)
        plt.plot(effPriorLogOdds, actL, color="r", label="actDCF")
        plt.plot(effPriorLogOdds, minL, color="b", linestyle="--", label="minDCF")
        plt.title(f"Bayes error - {name}")
        plt.xlabel("prior log-odds")
        plt.ylabel("DCF")
        plt.ylim([0, 1.1])
        plt.xlim([-4, 4])
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()


def load_or_none(path):
    return numpy.load(path) if os.path.exists(path) else None


if __name__ == "__main__":
    logger.setup_logger("out/lab09_gmm.txt")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)
    print(f"DTR shape: {DTR.shape}, DVAL shape: {DVAL.shape}")
    print(f"Class 0 (fake): {(LTR == 0).sum()}  |  Class 1 (genuine): {(LTR == 1).sum()}")

    # 1) Full-covariance GMM, full Cartesian product over (K0, K1).
    print("\n############### Full-covariance GMM (K0 x K1 sweep) ###############")
    rows_full, best_full = sweep_components(DTR, LTR, DVAL, LVAL, covType="Full")
    print_matrix_table("Full-covariance minDCF (pi_T = 0.1)", rows_full, key_idx=3)
    print_matrix_table("Full-covariance actDCF (pi_T = 0.1)", rows_full, key_idx=2)
    print(f"\nBest Full GMM: K0={best_full['K0']}, K1={best_full['K1']}  "
          f"minDCF={best_full['minDCF']:.4f}, actDCF={best_full['actDCF']:.4f}")
    # Observations to look for:
    # - On this dataset the data is bimodal-ish (cf. lab 4); going to >1 components per class
    #   should significantly improve minDCF compared to the 1-component MVG baseline.
    # - actDCF tends to grow with the number of components (more flexible model -> mild
    #   miscalibration), but a GMM is still a generative model so the gap stays modest.
    # - Symmetric (K0 == K1) configurations are not always optimal; one class may need more
    #   components than the other if its density is more complex.

    # 1b) Quick sanity check on the Diagonal variant (optional in the lab, but cheap).
    print("\n############### [Quick check] Diagonal-covariance GMM (K0 x K1 sweep) ###############")
    rows_diag, best_diag = sweep_components(DTR, LTR, DVAL, LVAL, covType="Diagonal")
    print_matrix_table("Diagonal minDCF (pi_T = 0.1)", rows_diag, key_idx=3)
    print(f"\nBest Diagonal GMM: K0={best_diag['K0']}, K1={best_diag['K1']}  "
          f"minDCF={best_diag['minDCF']:.4f}, actDCF={best_diag['actDCF']:.4f}")
    # Diagonal can do well here because the features are nearly uncorrelated (cf. lab 5
    # Pearson analysis); more components compensate for the lost off-diagonal modelling.

    # 2) Compare best GMM with best LR (lab 7) and best SVM (lab 8) at pi_T = 0.1.
    print("\n############### Best per-method comparison at pi_T = 0.1 ###############")

    # Load saved best LLR-like scores from previous labs.
    lr_quadratic_sLLR = load_or_none(f"{LAB07_DIR}/quadratic_full_sLLR.npy")
    lr_std_sLLR = load_or_none(f"{LAB07_DIR}/std_full_sLLR.npy")
    svm_rbf_scores = load_or_none(f"{LAB08_DIR}/rbf_scores.npy")
    svm_polyd2_scores = load_or_none(f"{LAB08_DIR}/poly_d2_scores.npy")

    candidates = {"GMM (Full, best K0xK1)": best_full["llr"]}
    if lr_quadratic_sLLR is not None:
        candidates["LR quadratic (best lambda)"] = lr_quadratic_sLLR
    if lr_std_sLLR is not None:
        candidates["LR linear (best lambda)"] = lr_std_sLLR
    if svm_rbf_scores is not None:
        candidates["SVM RBF (best gamma, C)"] = svm_rbf_scores
    if svm_polyd2_scores is not None:
        candidates["SVM poly d=2 (best C)"] = svm_polyd2_scores

    print(f"{'model':>32} | {'actDCF':>7} | {'minDCF':>7}")
    print("-" * 54)
    summary_by_minDCF = []
    for name, scores in candidates.items():
        if scores.shape[0] != LVAL.shape[0]:
            print(f"WARNING: skipping {name} - score size {scores.shape[0]} != {LVAL.shape[0]}")
            continue
        act = evaluation.compute_actDCF_binary_fast(scores, LVAL, TARGET_PRIOR, 1.0, 1.0)
        mn = evaluation.compute_minDCF_binary_fast(scores, LVAL, TARGET_PRIOR, 1.0, 1.0)
        print(f"{name:>32} | {act:>7.4f} | {mn:>7.4f}")
        summary_by_minDCF.append((mn, act, name, scores))
    summary_by_minDCF.sort(key=lambda x: x[0])

    # 3) Bayes error plot for the top three by minDCF (one for each method family if available).
    top_three = {}
    seen_families = set()
    for mn, act, name, scores in summary_by_minDCF:
        family = "GMM" if "GMM" in name else ("LR" if "LR" in name else "SVM")
        if family not in seen_families:
            top_three[name] = scores
            seen_families.add(family)
        if len(top_three) == 3:
            break

    print(f"\n--- Bayes error plot for best-of-family ---")
    for name in top_three:
        print(f"  included: {name}")
    bayes_error_plot(top_three, LVAL, "01_bayes_error_top3.png")
    # What to look for in the figure:
    # - Are minDCF curves preserved in ranking across log-odds?  If yes, model choice is robust.
    # - Is GMM's actDCF close to its minDCF over most of the range?  GMMs are generative,
    #   so we expect mild miscalibration.
    # - SVM scores are not LLRs -> expect actDCF >> minDCF away from the empirical prior.
    # - LR (linear / quadratic) is in between - it learns log-posterior-like scores and we
    #   subtract log-odds of the empirical prior to make them LLR-like, so calibration should
    #   be reasonable in the middle of the range.

    # --- Save best GMM scores for the calibration laboratory ---
    print("\n--- Saving best GMM (Full) scores for upcoming calibration lab ---")
    numpy.save(f"{DATA_DIR}/gmm_full_best_llr.npy", best_full["llr"])
    print(f"  saved gmm_full_best_llr.npy  (K0={best_full['K0']}, K1={best_full['K1']}, "
          f"minDCF={best_full['minDCF']:.4f}, actDCF={best_full['actDCF']:.4f})")
