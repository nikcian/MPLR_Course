"""
Laboratory 2 - Project task: exploratory analysis of the fingerprint-spoofing dataset.

Lab questions addressed in this script:
  1. First two features:  do classes overlap?  similar means?  similar variances?  how many modes?
  2. Third / fourth features: same questions.
  3. Last two features: overlap, modes, number of clusters visible in the scatter plots.

The dataset is 6-dimensional (binary fingerprint spoofing detection); class 0 = fake, class 1 = genuine.
We load it as a (6, N) matrix, plot per-feature normalized histograms and per-pair scatter plots, then
report empirical statistics (mean, covariance, variance, std) for each class so the questions above can
be answered quantitatively.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, plot_utils, logger


def explore(D, L):
    print("--- Per-class statistics ---")
    # We compute mean / covariance / variance / std separately for the two classes so that the lab
    # questions about "similar mean", "similar variance" can be answered with concrete numbers.
    for cls in [0, 1]:
        print(f"\nClass {cls} ({'fake' if cls == 0 else 'genuine'})")
        DCls = D[:, L == cls]
        # Column vector mean as in Lab 2: D.mean(1).reshape(D.shape[0], 1)
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print("Mean:")
        print(mu)
        # Covariance via the matrix form C = (1/N) Dc Dc^T (broadcasting handles the centering).
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print("Covariance:")
        print(C)
        # Variance is the diagonal of C; std is its square root.
        var = DCls.var(1)
        std = DCls.std(1)
        print("Variance:", var)
        print("Std. dev.:", std)

    print("\n--- Feature-pair comparison (means and variances) ---")
    mu0 = D[:, L == 0].mean(1)
    mu1 = D[:, L == 1].mean(1)
    var0 = D[:, L == 0].var(1)
    var1 = D[:, L == 1].var(1)

    # We highlight the three pairs the project explicitly asks about:
    #   features 1-2, features 3-4, features 5-6.
    groups = {
        "Features 1-2 (Q1)": range(0, 2),
        "Features 3-4 (Q2)": range(2, 4),
        "Features 5-6 (Q3)": range(4, 6),
    }
    for label, indices in groups.items():
        print(f"\n{label}")
        for i in indices:
            print(
                f"  Feature {i + 1}: "
                f"mean0={mu0[i]:+.4f}  mean1={mu1[i]:+.4f}  |Δμ|={abs(mu0[i] - mu1[i]):.4f}  "
                f"var0={var0[i]:.4f}  var1={var1[i]:.4f}  |Δσ²|={abs(var0[i] - var1[i]):.4f}"
            )

    # Notes (answers to the lab questions, based on a typical run of this dataset):
    #  - Features 1-2: means are close between the two classes; variances differ; histograms look
    #    roughly uni-modal but with a marked overlap.
    #  - Features 3-4: means are clearly separated, variances are comparable; histograms uni-modal,
    #    less overlap than 1-2.
    #  - Features 5-6: histograms are clearly multi-modal (peaks around -1 and +1) and the scatter
    #    plot shows ~4 clusters per class — strong non-Gaussian structure that will impact later models.


if __name__ == "__main__":
    logger.setup_logger("out/lab02_iris_dataset.txt")
    plot_utils.plot_init()

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)
    print(f"Dataset shape: {D.shape}  Labels shape: {L.shape}")
    print(f"Samples in class 0 (fake):    {(L == 0).sum()}")
    print(f"Samples in class 1 (genuine): {(L == 1).sum()}")

    print("\nGenerating histograms and scatter plots (PNG, in out/plots/lab02)...")
    plot_utils.plot_hist(D, L, filename_prefix="hist", out_dir="out/plots/lab02")
    # all_pairs=True gives every (i, j) with i < j — this is what the lab asks for.
    plot_utils.plot_scatter(D, L, filename_prefix="scatter", out_dir="out/plots/lab02", all_pairs=True)

    explore(D, L)
