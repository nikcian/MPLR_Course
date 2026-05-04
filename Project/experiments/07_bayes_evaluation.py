import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger, evaluation
from models import gaussian_models


def evaluate_effective_prior(pi, Cfn, Cfp):
    return (pi * Cfn) / (pi * Cfn + (1 - pi) * Cfp)


def print_separator(title):
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}\n")


def compute_model_llrs(DTR, LTR, DVAL):
    # MVG
    hParams_MVG = gaussian_models.Gau_MVG_ML_estimates(DTR, LTR)
    LLR_MVG = gaussian_models.compute_llr(
        DVAL, hParams_MVG[1][0], hParams_MVG[1][1], hParams_MVG[0][0], hParams_MVG[0][1]
    )

    # Naive Bayes
    hParams_Naive = gaussian_models.Gau_Naive_ML_estimates(DTR, LTR)
    LLR_Naive = gaussian_models.compute_llr(
        DVAL,
        hParams_Naive[1][0],
        hParams_Naive[1][1],
        hParams_Naive[0][0],
        hParams_Naive[0][1],
    )

    # Tied
    hParams_Tied = gaussian_models.Gau_Tied_ML_estimates(DTR, LTR)
    LLR_Tied = gaussian_models.compute_llr(
        DVAL,
        hParams_Tied[1][0],
        hParams_Tied[1][1],
        hParams_Tied[0][0],
        hParams_Tied[0][1],
    )

    return LLR_MVG, LLR_Naive, LLR_Tied


if __name__ == "__main__":
    logger.setup_logger("out/07_bayes_evaluation.txt")

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)

    LLR_MVG, LLR_Naive, LLR_Tied = compute_model_llrs(DTR, LTR, DVAL)
    models_list = [
        ("MVG", LLR_MVG),
        ("Naive Bayes", LLR_Naive),
        ("Tied Gaussian", LLR_Tied),
    ]

    print_separator("Applications and Effective Priors")
    apps = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0),
    ]

    for pi, Cfn, Cfp in apps:
        eff_pi = evaluate_effective_prior(pi, Cfn, Cfp)
        print(
            f"Application (pi={pi}, Cfn={Cfn}, Cfp={Cfp}) -> Effective Prior: {eff_pi:.4f}"
        )

    print("\nObservation:")
    print(
        "Higher cost for false positives (Cfp=9.0, aiming for strong security) shifts the effective prior to a lower value (0.1)."
    )
    print(
        "Higher cost for false negatives (Cfn=9.0, aiming for ease of use) shifts the effective prior to a higher value (0.9).\n"
    )

    print_separator("Optimal Bayes Decisions & DCF for Effective Priors")
    target_priors = [0.1, 0.5, 0.9]

    for pi_tilde in target_priors:
        print(f"--- Effective Prior: {pi_tilde} ---")
        for name, LLR in models_list:
            actDCF = evaluation.compute_actDCF_binary_fast(
                LLR, LVAL, pi_tilde, 1.0, 1.0
            )
            minDCF = evaluation.compute_minDCF_binary_fast(
                LLR, LVAL, pi_tilde, 1.0, 1.0
            )
            cal_loss = actDCF - minDCF
            print(
                f"{name:>15} -> actDCF: {actDCF:.4f}, minDCF: {minDCF:.4f}, cal_loss: {cal_loss:.4f}"
            )
        print()

    print_separator("Generating Bayes Error Plots")
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    plt.figure(figsize=(18, 5))

    for idx, (name, LLR) in enumerate(models_list):
        actDCF_list = []
        minDCF_list = []
        for effPrior in effPriors:
            actDCF = evaluation.compute_actDCF_binary_fast(
                LLR, LVAL, effPrior, 1.0, 1.0
            )
            minDCF = evaluation.compute_minDCF_binary_fast(
                LLR, LVAL, effPrior, 1.0, 1.0
            )
            actDCF_list.append(actDCF)
            minDCF_list.append(minDCF)

        plt.subplot(1, 3, idx + 1)
        plt.plot(effPriorLogOdds, actDCF_list, label="actDCF", color="r", linestyle="-")
        plt.plot(
            effPriorLogOdds, minDCF_list, label="minDCF", color="b", linestyle="--"
        )
        plt.title(f"Bayes Error Plot: {name}")
        plt.xlabel("Prior Log-Odds")
        plt.ylabel("DCF")
        plt.ylim([0, 1.1])
        plt.xlim([-4, 4])
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if not os.path.exists("out/plots"):
        os.makedirs("out/plots")
    plot_path = "out/plots/bayes_error_plots.png"
    plt.savefig(plot_path)
    print(f"Saved Bayes Error Plots to {plot_path}\n")
