import numpy as np
import os

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load(fname):
    
    DList = []
    labelsList = []
    
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(np.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
            
    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def calc(D, L):
    if not os.path.exists('out'):
        os.makedirs('out')

    with open('out/output.txt', 'w') as f:
        for cls in [0,1]:
            print('Class', cls, file=f)
            DCls = D[:, L==cls]
            mu = DCls.mean(1).reshape(DCls.shape[0], 1)
            print('Mean:', file=f)
            print(mu, file=f)
            C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
            print('Covariance:', file=f)
            print(C, file=f)
            var = DCls.var(1)
            std = DCls.std(1)
            print('Variance:', var, file=f)
            print('Std. dev.:', std, file=f)
            print(file=f)
            
    with open('out/analysis.txt', 'w') as f:
        mu0 = D[:, L==0].mean(1)
        mu1 = D[:, L==1].mean(1)
        var0 = D[:, L==0].var(1)
        var1 = D[:, L==1].var(1)

        for indices in [range(2), range(2, 4)]:
            print('Analysis for features:', list(indices), file=f)
            for i in indices:
                print('Feature %d:' % i, file=f)
                print('  Mean0: %.4f, Mean1: %.4f, Diff: %.4f' % (mu0[i], mu1[i], abs(mu0[i]-mu1[i])), file=f)
                print('  Var0: %.4f, Var1: %.4f, Diff: %.4f' % (var0[i], var1[i], abs(var0[i]-var1[i])), file=f)
            print(file=f)