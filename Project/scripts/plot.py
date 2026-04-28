import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from . import util

def plot_init():
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    if not os.path.exists('out/plots'):
        os.makedirs('out/plots')

    for dIdx in range(6):
        plt.figure()
        plt.xlabel(f'Feature {dIdx+1}')
        plt.ylabel('Density')        
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'False')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'True')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('out/plots/hist_%d.pdf' % (dIdx+1))
        plt.close()
        
def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    if not os.path.exists('out/plots'):
        os.makedirs('out/plots')
        
    for dIdx in range(0,6,2):
        plt.figure()
        plt.xlabel(f'Feature {dIdx+1}')
        plt.ylabel(f'Feature {dIdx+2}')
        plt.scatter(D0[dIdx, :], D0[dIdx+1, :], label = 'False')
        plt.scatter(D1[dIdx, :], D1[dIdx+1, :], label = 'True')
    
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('out/plots/scatter_%d_%d.pdf' % (dIdx+1, dIdx+2))
        plt.close()
    