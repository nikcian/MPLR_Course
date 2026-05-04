import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, plot_utils, logger


def explore(D, L):
    print("--- Dataset Statistics ---")
    for cls in [0, 1]:
        print(f"\nClass {cls}")
        DCls = D[:, L == cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print("Mean:")
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print("Covariance:")
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print("Variance:", var)
        print("Std. dev.:", std)

    print("\n--- Feature Analysis ---")
    mu0 = D[:, L == 0].mean(1)
    mu1 = D[:, L == 1].mean(1)
    var0 = D[:, L == 0].var(1)
    var1 = D[:, L == 1].var(1)

    for indices in [range(2), range(2, 4)]:
        print(f"\nAnalysis for features: {list(indices)}")
        for i in indices:
            print(f"Feature {i}:")
            print(
                f"  Mean0: {mu0[i]:.4f}, Mean1: {mu1[i]:.4f}, Diff: {abs(mu0[i]-mu1[i]):.4f}"
            )
            print(
                f"  Var0: {var0[i]:.4f}, Var1: {var1[i]:.4f}, Diff: {abs(var0[i]-var1[i]):.4f}"
            )


if __name__ == "__main__":
    logger.setup_logger("out/01_exploration.txt")
    plot_utils.plot_init()

    fname = "data/trainData.txt"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    D, L = data_utils.load(fname)

    print("Generating plots...")
    plot_utils.plot_hist(D, L, filename_prefix="hist")
    plot_utils.plot_scatter(D, L)

    explore(D, L)
