import numpy
import scipy.optimize

from utils.math_utils import vcol, vrow


def train_dual_SVM_linear(DTR, LTR, C, K=1.0):
    """Dual linear SVM with the extended-feature trick (no equality constraint).

    Trains on DTR_EXT = [DTR; K * 1] so that the bias is implicitly regularized; b must be
    rescaled by K when extracting it back for raw-feature scoring.

    Returns (w, b, primal_loss, dual_loss).
    """
    ZTR = LTR * 2.0 - 1.0
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    # We use scipy L-BFGS-B default convergence (factr=1e7).  The strictest option used in the
    # lab (factr=nan, pgtol=1e-5) requires thousands of iterations on the project's 4000-sample
    # training set; the default still gives a small duality gap and the same minDCF/actDCF for
    # our purposes.  maxfun is raised because on a few configurations the optimizer hits the
    # 15000-call default before convergence.
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt,
        numpy.zeros(DTR_EXT.shape[1]),
        bounds=[(0, C) for _ in range(DTR_EXT.shape[1])],
        maxfun=50000,
    )

    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    w = w_hat[0:DTR.shape[0]]
    b = w_hat[-1] * K

    S_ext = (vrow(w_hat) @ DTR_EXT).ravel()
    primal_loss = 0.5 * numpy.linalg.norm(w_hat) ** 2 + C * numpy.maximum(0, 1 - ZTR * S_ext).sum()
    dual_loss = -fOpt(alphaStar)[0][0]

    return w, b, primal_loss, dual_loss


def polyKernel(degree, c):
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree
    return polyKernelFunc


def rbfKernel(gamma):
    def rbfKernelFunc(D1, D2):
        D1Norms = (D1 ** 2).sum(0)
        D2Norms = (D2 ** 2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)
    return rbfKernelFunc


def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0):
    """Dual kernel SVM. eps adds a constant to the kernel evaluations to provide a regularised
    bias term (eps = K^2 is the kernel analogue of the linear extended-feature trick).

    Returns (fScore, primal_loss, dual_loss) where fScore(DTE) computes scores on a test matrix.
    """
    ZTR = LTR * 2.0 - 1.0
    Kmat = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * Kmat

    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt,
        numpy.zeros(DTR.shape[1]),
        bounds=[(0, C) for _ in range(DTR.shape[1])],
        maxfun=50000,
    )

    # H[i,j] = z_i z_j k(x_i, x_j) so (H alpha)[i] = z_i * s(x_i) and the hinge becomes
    # max(0, 1 - z_i s(x_i)) = max(0, 1 - (H alpha)[i]).
    Ha = (H @ vcol(alphaStar)).ravel()
    primal_loss = 0.5 * (alphaStar * Ha).sum() + C * numpy.maximum(0, 1 - Ha).sum()
    dual_loss = -fOpt(alphaStar)[0][0]

    def fScore(DTE):
        Ktest = kernelFunc(DTR, DTE) + eps
        Hk = vcol(alphaStar) * vcol(ZTR) * Ktest
        return Hk.sum(0)

    return fScore, primal_loss, dual_loss
