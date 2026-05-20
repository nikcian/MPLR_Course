"""
Laboratory 4 - Project task: fit a 1-D Gaussian (Maximum Likelihood) to every feature of every class.

For each (class, feature) we:
  1. Compute the ML estimates µ_ML and Σ_ML (= empirical mean and variance).
  2. Overlay the resulting Gaussian density on top of the normalized histogram of the samples.

Lab question: for which features does the Gaussian assumption look adequate, and for which does it
fail?  This is preparatory work for Lab 5: a poor 1-D fit warns us that the Naive Bayes assumption
(diagonal Σ, independent Gaussian features) will struggle on that feature.

Notes:
  - We exponentiate the log-density returned by `logpdf_GAU_ND` because the histogram is on the
    linear (probability density) scale.
  - The lab says we may use either the whole dataset or just the model-training split. We use the
    whole dataset here because this is exploratory analysis only.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger
from utils.math_utils import compute_mu_C, logpdf_GAU_ND, vrow


PLOT_DIR = "out/plots/lab04"


if __name__ == "__main__":
    logger.setup_logger("out/lab04_density_estimation.txt")

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    print("--- 1-D Gaussian ML fit per class per feature ---")
    print(f"{'feature':>8} | {'class':>5} | {'mu_ML':>8} | {'sigma2_ML':>10}")
    print("-" * 50)

    # We collect the ML estimates first (one Gaussian per (class, feature)), then plot both classes
    # together on a single figure per feature so the goodness of fit and the class overlap can be
    # judged at a glance.
    class_styles = {
        0: {"hist_color": "steelblue", "line_color": "navy", "name": "False (0)"},
        1: {"hist_color": "darkorange", "line_color": "saddlebrown", "name": "True (1)"},
    }

    estimates = {}
    for cls in [0, 1]:
        DCls = D[:, L == cls]
        for i in range(DCls.shape[0]):
            feature_data = DCls[i, :]
            # Reshape to (1, N) — logpdf_GAU_ND expects samples as columns of a matrix.
            mu, C = compute_mu_C(vrow(feature_data))
            estimates[(cls, i)] = (feature_data, mu, C)
            print(f"{i + 1:8d} | {cls:5d} | {mu[0, 0]:+8.4f} | {C[0, 0]:10.4f}")

    for i in range(D.shape[0]):
        plt.figure()
        # Determine a common plotting range that covers both class histograms and 4σ of either fit.
        x_min = min(estimates[(0, i)][0].min(), estimates[(1, i)][0].min())
        x_max = max(estimates[(0, i)][0].max(), estimates[(1, i)][0].max())
        for cls in [0, 1]:
            _, mu, C = estimates[(cls, i)]
            x_min = min(x_min, mu[0, 0] - 4 * np.sqrt(C[0, 0]))
            x_max = max(x_max, mu[0, 0] + 4 * np.sqrt(C[0, 0]))
        XPlot = np.linspace(x_min, x_max, 1000)

        for cls in [0, 1]:
            feature_data, mu, C = estimates[(cls, i)]
            style = class_styles[cls]
            plt.hist(
                feature_data.ravel(),
                bins=50,
                density=True,
                alpha=0.45,
                color=style["hist_color"],
                edgecolor="black",
                label=f"Hist {style['name']}",
            )
            pdfGau = np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C))
            plt.plot(
                XPlot,
                pdfGau,
                color=style["line_color"],
                linewidth=2,
                label=f"Gaussian fit {style['name']}",
            )

        plt.xlabel(f"Feature {i + 1}")
        plt.ylabel("Density")
        plt.title(f"Feature {i + 1}: per-class histogram and Gaussian ML fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/feature_{i + 1}.png")
        plt.close()

    # Observations (from the saved plots):
    #   - Features 1-4 are reasonably well approximated by a uni-modal Gaussian per class — the
    #     density curve tracks the histogram, with limited skew/kurtosis.
    #   - Features 5 and 6 are clearly multi-modal (two well separated peaks per class). A single
    #     Gaussian is a poor fit here and the Naive-Bayes assumption will pay a price for it in Lab 5.
    print(
        "\nQualitative observation: features 5-6 show bimodal distributions, so a single 1-D "
        "Gaussian is a poor fit.  Features 1-4 are well approximated by a Gaussian.  This is "
        "consistent with the worse performance of Naive Bayes on the full 6-feature set vs. the "
        "1-4 subset that we will see in lab 5."
    )
