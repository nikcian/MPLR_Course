import numpy
import scipy.optimize

from utils.math_utils import vcol, vrow


def trainLogRegBinary(DTR, LTR, l):
    """Standard (non-weighted) regularized binary logistic regression.

    Minimizes  J(w, b) = l/2 ||w||^2 + (1/n) sum_i log(1 + exp(-z_i (w^T x_i + b))),
    with z_i = 2 c_i - 1.  Returns (w, b).
    """
    ZTR = LTR * 2.0 - 1.0

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2.0 * numpy.linalg.norm(w) ** 2, \
            numpy.hstack([GW, numpy.array(Gb)])

    vf, Jstar, _ = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj_with_grad, x0=numpy.zeros(DTR.shape[0] + 1)
    )
    return vf[:-1], vf[-1], Jstar


def trainWeightedLogRegBinary(DTR, LTR, l, pT):
    """Prior-weighted regularized binary logistic regression.

    Each sample contributes with weight  xi_i = pT/nT  if c_i = 1,  (1-pT)/nF  if c_i = 0,
    so the loss simulates a population with prior pT regardless of the empirical training prior.
    """
    ZTR = LTR * 2.0 - 1.0

    wTrue = pT / (ZTR > 0).sum()
    wFalse = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTrue
        loss[ZTR < 0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse

        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2.0 * numpy.linalg.norm(w) ** 2, \
            numpy.hstack([GW, numpy.array(Gb)])

    vf, Jstar, _ = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj_with_grad, x0=numpy.zeros(DTR.shape[0] + 1)
    )
    return vf[:-1], vf[-1], Jstar


def expand_features_quadratic(D):
    """Expand each column x of D into phi(x) = [vec(x x^T), x].

    For input shape (M, N) the output has shape (M*M + M, N).  The full outer product
    (not just the upper triangle) is kept: the LR optimizer is fine with the redundancy and the
    interface stays simple for both train and validation.
    """
    M, N = D.shape
    outer = (D[:, None, :] * D[None, :, :]).reshape(M * M, N)
    return numpy.vstack([outer, D])


def compute_scores(w, b, D):
    """Linear scoring s = w^T x + b on the columns of D, returned as a 1-D array."""
    return (vcol(w).T @ D + b).ravel()


def scores_to_llr(scores, prior_offset):
    """Subtract the prior log-odds  log(p / (1 - p))  from the raw LR scores to obtain LLR-like
    scores compatible with our DCF infrastructure.

    For the non-weighted model use prior_offset = pi_emp (the empirical training prior);
    for the prior-weighted model use prior_offset = pT (the target prior used during training).
    """
    return scores - numpy.log(prior_offset / (1.0 - prior_offset))
