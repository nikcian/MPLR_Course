import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, plot_utils, logger
from models import lda


def analyze_lda(D, L):
    print("--- LDA Analysis ---")
    U = lda.compute_lda_geig(D, L, m=1)
    DP = lda.apply_lda(U, D)

    D0 = DP[:, L == 0][0]
    D1 = DP[:, L == 1][0]

    mean0, std0 = np.mean(D0), np.std(D0)
    mean1, std1 = np.mean(D1), np.std(D1)

    print("LDA Direction 1")
    print(f"Class False (0) - Mean: {mean0:.4f}, Std: {std0:.4f}")
    print(f"Class True  (1) - Mean: {mean1:.4f}, Std: {std1:.4f}")
    print(f"Distance between means: {abs(mean0 - mean1):.4f}")


if __name__ == "__main__":
    logger.setup_logger("out/03_lda_analysis.txt")
    plot_utils.plot_init()

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)

    print("Generating LDA plots...")
    U = lda.compute_lda_geig(D, L, m=1)
    DP = lda.apply_lda(U, D)
    # LDA projection is 1D
    import matplotlib.pyplot as plt

    if not os.path.exists("out/plots"):
        os.makedirs("out/plots")

    plt.figure()
    plt.xlabel("LDA Direction 1")
    plt.ylabel("Density")
    plt.hist(DP[0, L == 0], bins=10, density=True, alpha=0.4, label="False")
    plt.hist(DP[0, L == 1], bins=10, density=True, alpha=0.4, label="True")
    plt.legend()
    plt.tight_layout()
    plt.savefig("out/plots/hist_lda_1.pdf")
    plt.close()

    analyze_lda(D, L)
