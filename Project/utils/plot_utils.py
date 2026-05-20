import os
import matplotlib.pyplot as plt
import numpy as np


def plot_init():
    plt.rc("font", size=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_hist(D, L, filename_prefix="hist", out_dir="out/plots", xlabel_prefix="Feature", bins=20):
    # Normalized histograms (density=True) per feature, one figure per feature, comparing the two classes.
    # This is what lab 2 asks for: the y-axis is an estimate of the density (not a frequency).
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    ensure_dir(out_dir)

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(f"{xlabel_prefix} {dIdx + 1}")
        plt.ylabel("Density")
        plt.hist(D0[dIdx, :], bins=bins, density=True, alpha=0.5, label="False (0)")
        plt.hist(D1[dIdx, :], bins=bins, density=True, alpha=0.5, label="True (1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename_prefix}_{dIdx + 1}.png")
        plt.close()


def plot_scatter(D, L, filename_prefix="scatter", out_dir="out/plots", all_pairs=True):
    # Pair-wise scatter plots between classes.
    # The lab asks for ALL feature pairs, not just consecutive ones, so by default we sweep every (i, j) with i < j.
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    ensure_dir(out_dir)

    if all_pairs:
        pairs = [(i, j) for i in range(D.shape[0]) for j in range(i + 1, D.shape[0])]
    else:
        pairs = [(i, i + 1) for i in range(0, D.shape[0] - 1, 2)]

    for i, j in pairs:
        plt.figure()
        plt.xlabel(f"Feature {i + 1}")
        plt.ylabel(f"Feature {j + 1}")
        plt.scatter(D0[i, :], D0[j, :], alpha=0.5, label="False (0)")
        plt.scatter(D1[i, :], D1[j, :], alpha=0.5, label="True (1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename_prefix}_{i + 1}_{j + 1}.png")
        plt.close()


def plot_lda_hist(DP, L, filename="hist_lda_1.png", out_dir="out/plots"):
    # Histogram of the (1-D) LDA projection. Lab 3 expects a single discriminant direction for a binary problem.
    ensure_dir(out_dir)
    plt.figure()
    plt.xlabel("LDA Direction 1")
    plt.ylabel("Density")
    plt.hist(DP[0, L == 0], bins=20, density=True, alpha=0.5, label="False (0)")
    plt.hist(DP[0, L == 1], bins=20, density=True, alpha=0.5, label="True (1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}")
    plt.close()
