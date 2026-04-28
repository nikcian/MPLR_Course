import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import scripts.util as util

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def logpdf_GAU_ND_fast(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

if __name__ == '__main__':
    D, L = util.load('data/trainData.txt')
    
    if not os.path.exists('out/density_plots'):
        os.makedirs('out/density_plots')
        
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
            
            pdfGau = np.exp(logpdf_GAU_ND_fast(vrow(XPlot), mu, C))
            
            plt.plot(XPlot.ravel(), pdfGau, color='r', linewidth=2)
            plt.xlabel(f"Feature {feature_num} Value")
            plt.ylabel("Density")
            plt.title(f"Class {cls} - Feature {feature_num}")
            
            plt.savefig(f'out/density_plots/class_{cls}_feature_{feature_num}.png')
            plt.close()
