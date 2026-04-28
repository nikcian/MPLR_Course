import os

import numpy
import sklearn.datasets 

import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

def load_iris(): # Same as in pca script
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca script
    return x.reshape((1, x.size))

def compute_mu_C(D): # Same as in pca script
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

def plot_scatter(D, L):

    D, L = load_iris()
    
    # 1. Calcoliamo la matrice di proiezione U per le prime 2 direzioni (usiamo geig)
    U = compute_lda_geig(D, L, m = 2)
    
    # 2. Applichiamo la LDA proiettando il dataset D sulle nuove direzioni
    DP = apply_lda(U, D)
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 3. Creiamo il grafico scatter separando le tre classi (0, 1, 2)
    plt.figure()
    plt.xlabel('Prima Direzione (LDA1)')
    plt.ylabel('Seconda Direzione (LDA2)')
    
    # L==0: Setosa, L==1: Versicolor, L==2: Virginica
    plt.scatter(DP[0, L==0], DP[1, L==0], label='Setosa')
    plt.scatter(DP[0, L==1], DP[1, L==1], label='Versicolor')
    plt.scatter(DP[0, L==2], DP[1, L==2], label='Virginica')
    
    # Aggiungiamo etichette e legenda per chiarezza
    plt.legend()
    plt.tight_layout()
    
    # Mostriamo a schermo il grafico
    plt.savefig('plots/scatter_lda_%d_%d.pdf' % (1, 2))
    plt.close()

if __name__ == '__main__':

    D, L = load_iris()
    U = compute_lda_geig(D, L, m = 2)
    print(U)
    print(compute_lda_JointDiag(D, L, m=2)) # May have different signs for the different directions
    USol = numpy.load('IRIS_LDA_matrix_m2.npy') # May have different signs for different directions
    print(USol)
    print(numpy.linalg.svd(numpy.hstack([U, USol]))[1])
    
    plot_scatter(D, L)
