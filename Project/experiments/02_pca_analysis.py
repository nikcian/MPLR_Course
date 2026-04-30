import sys
import os
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, plot_utils, logger
from models import pca

def analyze_pca(D, L):
    print("--- PCA Analysis ---")
    P = pca.compute_pca(D, D.shape[0])
    DP = pca.apply_pca(P, D)
    
    D0 = DP[:, L==0]
    D1 = DP[:, L==1]

    for dIdx in range(DP.shape[0]):
        print(f"\n--- PCA Direction {dIdx+1} ---")
        
        mean0, std0 = np.mean(D0[dIdx, :]), np.std(D0[dIdx, :])
        mean1, std1 = np.mean(D1[dIdx, :]), np.std(D1[dIdx, :])
        
        print(f"Class False (0) - Mean: {mean0:.4f}, Std: {std0:.4f}")
        print(f"Class True  (1) - Mean: {mean1:.4f}, Std: {std1:.4f}")
        print(f"Distance between means: {abs(mean0 - mean1):.4f}")
        
        for cls, D_cls in [("False", D0), ("True", D1)]:
            kde = stats.gaussian_kde(D_cls[dIdx, :])
            x = np.linspace(np.min(D_cls[dIdx, :]), np.max(D_cls[dIdx, :]), 1000)
            y = kde(x)
            peaks, _ = find_peaks(y)
            print(f"Class {cls} number of peaks (clusters): {len(peaks)}")

if __name__ == '__main__':
    logger.setup_logger('out/02_pca_analysis.txt')
    plot_utils.plot_init()
    
    fname = 'data/trainData.txt'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        
    D, L = data_utils.load(fname)
    
    print("Generating PCA plots...")
    P = pca.compute_pca(D, D.shape[0])
    DP = pca.apply_pca(P, D)
    plot_utils.plot_hist(DP, L, filename_prefix='hist_pca')
    
    analyze_pca(D, L)
