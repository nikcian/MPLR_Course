import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, logger
from utils.math_utils import compute_mu_C, logpdf_GAU_ND, vrow

if __name__ == '__main__':
    logger.setup_logger('out/04_density_estimation.txt')
    
    fname = 'data/trainData.txt'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        
    D, L = data_utils.load(fname)
    
    if not os.path.exists('out/density_plots'):
        os.makedirs('out/density_plots')
        
    print("--- 1D Density Estimation Analysis ---")
    print("Feature | Class | Mean | Variance | Normal Test p-value")
    print("-" * 60)

    try:
        from scipy.stats import normaltest
        has_scipy = True
    except ImportError:
        has_scipy = False

    for cls in [0, 1]:
        DCls = D[:, L==cls]
        
        for i in range(DCls.shape[0]):
            feature_data = DCls[i, :]
            
            # ML Estimates for 1D Gaussian
            mu, C = compute_mu_C(vrow(feature_data))
            
            # Goodness of fit (Normality test)
            if has_scipy:
                stat, p = normaltest(feature_data.ravel())
                p_val_str = f"{p:.4e}"
            else:
                p_val_str = "N/A"
            
            feature_num = i + 1
            print(f"{feature_num:7d} | {cls:5d} | {mu[0,0]:7.4f} | {C[0,0]:8.4f} | {p_val_str}")

            plt.figure()
            # Histogram
            plt.hist(feature_data.ravel(), bins=50, density=True, alpha=0.6, color='b', edgecolor='black')
            
            # Density
            # Extending the range slightly to cover the tails of the Gaussian
            x_min = min(feature_data.min(), mu[0,0] - 4*np.sqrt(C[0,0]))
            x_max = max(feature_data.max(), mu[0,0] + 4*np.sqrt(C[0,0]))
            XPlot = np.linspace(x_min, x_max, 1000)
            
            pdfGau = np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C))
            
            plt.plot(XPlot.ravel(), pdfGau, color='r', linewidth=2)
            plt.xlabel(f"Feature {feature_num} Value")
            plt.ylabel("Density")
            plt.title(f"Class {cls} - Feature {feature_num}")
            
            plt.savefig(f'out/density_plots/class_{cls}_feature_{feature_num}.png')
            plt.close()
