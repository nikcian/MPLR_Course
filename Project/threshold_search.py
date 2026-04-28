import numpy as np
import scripts.util as util
import lda
from classify import split_db_2to1

def search():
    fname = 'data/trainData.txt'
    D, L = util.load(fname)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)
    DTR_lda = lda.apply_lda(ULDA, DTR)

    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    mean_cls0 = DTR_lda[0, LTR==0].mean()
    mean_cls1 = DTR_lda[0, LTR==1].mean()
    base_threshold = (mean_cls0 + mean_cls1) / 2.0
    
    thresholds = np.linspace(DVAL_lda.min(), DVAL_lda.max(), 1000)
    best_err = len(LVAL)
    best_t = base_threshold

    for t in thresholds:
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= t] = 1
        PVAL[DVAL_lda[0] < t] = 0
        err = (PVAL != LVAL).sum()
        if err < best_err:
            best_err = err
            best_t = t

    print(f"Base Threshold: {base_threshold:.4f}")
    
    PVAL_base = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL_base[DVAL_lda[0] >= base_threshold] = 1
    PVAL_base[DVAL_lda[0] < base_threshold] = 0
    err_base = (PVAL_base != LVAL).sum()
    print(f"Base Error: {err_base} ({err_base/len(LVAL)*100:.2f}%)")
    
    print(f"Best Threshold found: {best_t:.4f}")
    print(f"Best Error: {best_err} ({best_err/len(LVAL)*100:.2f}%)")
    print(f"Improvement: {(err_base - best_err)/len(LVAL)*100:.2f}%")

if __name__ == '__main__':
    search()
