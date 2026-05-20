import os

import numpy
import sklearn.datasets 

import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D

def plot_scatter(D, L):
    # 1. Calcoliamo la matrice di proiezione P per le prime 2 direzioni
    P = compute_pca(D, m=2)
    
    # 2. Applichiamo la PCA proiettando il dataset D sulle prime due direzioni
    DP = apply_pca(P, D)
    
    DP[1, :] = -DP[1, :]        #OPZIONALE: fa combaciare la soluzione
    
    # DP avrà dimensione (2, 150).
    # DP[0, :] conterrà le coordinate sulla prima direzione
    # DP[1, :] conterrà le coordinate sulla seconda direzione
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 3. Creiamo il grafico scatter separando le tre classi (0, 1, 2)
    plt.figure()
    plt.xlabel('Prima Direzione (PC1)')
    plt.ylabel('Seconda Direzione (PC2)')
    plt.scatter(DP[0, L==0], DP[1, L==0], label='Setosa')
    plt.scatter(DP[0, L==1], DP[1, L==1], label='Versicolor')
    plt.scatter(DP[0, L==2], DP[1, L==2], label='Virginica')
    
    # Aggiungiamo etichette e legenda per renderlo più chiaro
    plt.legend()
    plt.tight_layout()
    # Mostriamo a schermo il grafico
    plt.savefig('plots/scatter_pca_%d_%d.pdf' % (1, 2))
    plt.close()
    

if __name__ == '__main__':

    D, L = load_iris()
    mu, C = compute_mu_C(D)
    print("Mean:")
    print(mu)
    print("Centered Data:")
    print(C)
    P = compute_pca(D, m = 4)
    print("PCA:")
    print(P)
    PSol = numpy.load('IRIS_PCA_matrix_m4.npy') # May have different signs for the different directions
    print("Solution:")
    print(PSol)
    
    plot_scatter(D, L)