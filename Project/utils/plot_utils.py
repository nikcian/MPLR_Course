import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def plot_init():
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

def plot_hist(D, L, filename_prefix='hist'):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    if not os.path.exists('out/plots'):
        os.makedirs('out/plots')

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(f'Feature {dIdx+1}')
        plt.ylabel('Density')        
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'False')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'True')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'out/plots/{filename_prefix}_{dIdx+1}.pdf')
        plt.close()
        
def plot_scatter(D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    if not os.path.exists('out/plots'):
        os.makedirs('out/plots')
        
    for dIdx in range(0, D.shape[0]-1, 2):
        if dIdx+1 < D.shape[0]:
            plt.figure()
            plt.xlabel(f'Feature {dIdx+1}')
            plt.ylabel(f'Feature {dIdx+2}')
            plt.scatter(D0[dIdx, :], D0[dIdx+1, :], label = 'False')
            plt.scatter(D1[dIdx, :], D1[dIdx+1, :], label = 'True')
        
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'out/plots/scatter_{dIdx+1}_{dIdx+2}.pdf')
            plt.close()
