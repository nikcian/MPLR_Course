"""
Laboratory 6 - Project task: Bayes decisions and model evaluation.

Two evaluation tools are used here:
  - DCF (Detection Cost Function): empirical Bayes risk normalised by the cost of a "dummy" system
    that uses only the prior.
  - minDCF: lower bound on DCF achievable by sweeping the threshold on the evaluation set itself —
    measures pure class separation, ignoring calibration losses.

Lab applications (π1, Cfn, Cfp):
  - (0.5, 1, 1)  uniform prior + cost
  - (0.9, 1, 1)  most users are legit
  - (0.1, 1, 1)  most users are impostors
  - (0.5, 1, 9)  strong security — false positives are very costly
  - (0.5, 9, 1)  legit-user ease of use — false negatives are very costly

A binary problem with arbitrary (π1, Cfn, Cfp) is equivalent to one with effective prior
    π̃ = π1·Cfn / (π1·Cfn + (1−π1)·Cfp)
and costs = 1.  So we can study the three target effective priors π̃ ∈ {0.1, 0.5, 0.9} and the Bayes
error plot over the prior log-odds gives a complete picture across applications.

For each model we compute both actDCF (the risk of the actual decisions taken at the theoretical
threshold) and minDCF (the best achievable risk on this evaluation set). The gap actDCF − minDCF
is the calibration loss.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger, evaluation
from models import gaussian_models


PLOT_DIR = "out/plots/lab06"


def effective_prior(pi, Cfn, Cfp):
    return (pi * Cfn) / (pi * Cfn + (1 - pi) * Cfp)


def compute_model_llrs(DTR, LTR, DVAL):
    # Same three models as in lab 5; we collect their LLRs on the validation set so the Bayes
    # analysis can be done once.
    out = {}
    for name, estimator in [
        ("MVG", gaussian_models.Gau_MVG_ML_estimates),
        ("Naive Bayes", gaussian_models.Gau_Naive_ML_estimates),
        ("Tied", gaussian_models.Gau_Tied_ML_estimates),
    ]:
        hParams = estimator(DTR, LTR)
        out[name] = gaussian_models.compute_llr(
            DVAL, hParams[1][0], hParams[1][1], hParams[0][0], hParams[0][1]
        )
    return out


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


if __name__ == "__main__":
    logger.setup_logger("out/lab06_bayes_evaluation.txt")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    llrs = compute_model_llrs(DTR, LTR, DVAL)
    model_order = ["MVG", "Naive Bayes", "Tied"]

    # 1) Map each lab application to its effective prior.
    section("Lab applications -> effective prior")
    apps = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0),
    ]
    for pi, Cfn, Cfp in apps:
        print(
            f"(π1={pi}, Cfn={Cfn}, Cfp={Cfp})  ->  effective prior π̃ = {effective_prior(pi, Cfn, Cfp):.4f}"
        )
    # Observation: a high false-positive cost (strong security) is equivalent to assuming a *lower*
    # prior probability of the genuine class — the recogniser becomes more conservative about
    # accepting a sample as genuine. Symmetric reasoning for ease-of-use.

    # 2) actDCF and minDCF at the three reduced applications π̃ ∈ {0.1, 0.5, 0.9}.
    section("DCF table at π̃ ∈ {0.1, 0.5, 0.9} (Cfn = Cfp = 1)")
    print(f"{'model':>13} | {'π̃':>4} | {'actDCF':>7} | {'minDCF':>7} | {'cal-loss':>8}")
    print("-" * 55)
    for pi_t in [0.1, 0.5, 0.9]:
        for name in model_order:
            LLR = llrs[name]
            actDCF = evaluation.compute_actDCF_binary_fast(LLR, LVAL, pi_t, 1.0, 1.0)
            minDCF = evaluation.compute_minDCF_binary_fast(LLR, LVAL, pi_t, 1.0, 1.0)
            print(
                f"{name:>13} | {pi_t:>4.1f} | {actDCF:>7.4f} | {minDCF:>7.4f} | {actDCF - minDCF:>8.4f}"
            )
        print("-" * 55)
    # Interpretation tips:
    #   - Lower minDCF ⇒ better intrinsic class separation.  Compare models on minDCF to rank
    #     them by discriminative power.
    #   - actDCF - minDCF is the calibration loss.  Generative Gaussian models often look well
    #     calibrated near π̃ = 0.5 but drift when the application is asymmetric.

    # 3) Bayes-error plots for the three models over prior log-odds in (-4, +4).
    section("Bayes error plots (prior log-odds -4 .. +4)")
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    plt.figure(figsize=(18, 5))
    for idx, name in enumerate(model_order):
        LLR = llrs[name]
        actDCF_list, minDCF_list = [], []
        for pi_t in effPriors:
            actDCF_list.append(evaluation.compute_actDCF_binary_fast(LLR, LVAL, pi_t, 1.0, 1.0))
            minDCF_list.append(evaluation.compute_minDCF_binary_fast(LLR, LVAL, pi_t, 1.0, 1.0))

        plt.subplot(1, 3, idx + 1)
        plt.plot(effPriorLogOdds, actDCF_list, color="r", label="actDCF")
        plt.plot(effPriorLogOdds, minDCF_list, color="b", linestyle="--", label="minDCF")
        plt.title(f"Bayes error plot: {name}")
        plt.xlabel("Prior log-odds")
        plt.ylabel("DCF")
        plt.ylim([0, 1.1])
        plt.xlim([-4, 4])
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    out_path = f"{PLOT_DIR}/bayes_error_plots.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Bayes error plot to {out_path}")
    # What to look for in the saved figure:
    #   - Curves where actDCF stays close to minDCF over the whole range ⇒ well-calibrated model.
    #   - Where actDCF is much higher than minDCF (often at the extremes) ⇒ calibration loss; a
    #     score-calibration step (e.g. logistic regression on the LLRs) would close the gap.
    #   - Model ranking by minDCF may or may not be consistent across π̃; check whether one model
    #     dominates everywhere or only in part of the range.
