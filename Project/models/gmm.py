import numpy
import scipy.special

from utils.math_utils import vcol, vrow, compute_mu_C, logpdf_GAU_ND


def logpdf_GMM(X, gmm):
    """Log-density of a GMM at every column of X.

    gmm is a list of (w, mu, C) triples - mu is a column (M, 1), C is (M, M).
    Returns a 1-D array of length N.
    """
    S = []
    for w, mu, C in gmm:
        S.append(logpdf_GAU_ND(X, mu, C) + numpy.log(w))
    S = numpy.vstack(S)
    return scipy.special.logsumexp(S, axis=0)


def smooth_covariance_matrix(C, psi):
    """Eigenvalue thresholding: replace any eigenvalue of C smaller than psi with psi."""
    U, s, _ = numpy.linalg.svd(C)
    s[s < psi] = psi
    return U @ (vcol(s) * U.T)


def _em_iteration(X, gmm, covType, psiEig):
    # E-step: joint log densities S[g, i] = log w_g + log N(x_i | mu_g, C_g) and responsibilities
    S = []
    for w, mu, C in gmm:
        S.append(logpdf_GAU_ND(X, mu, C) + numpy.log(w))
    S = numpy.vstack(S)
    logmarg = scipy.special.logsumexp(S, axis=0)
    gammas = numpy.exp(S - logmarg)

    # M-step: closed-form updates from the responsibilities
    gmmUpd = []
    for gIdx in range(len(gmm)):
        gamma = gammas[gIdx]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        Smat = (vrow(gamma) * X) @ X.T
        muUpd = F / Z
        CUpd = Smat / Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == "diagonal":
            CUpd = CUpd * numpy.eye(X.shape[0])
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == "tied":
        CTied = sum(w * C for (w, _, C) in gmmUpd)
        gmmUpd = [(w, mu, CTied) for (w, mu, _) in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for (w, mu, C) in gmmUpd]

    return gmmUpd


def train_GMM_EM(X, gmm, covType="Full", psiEig=None, epsLLAverage=1e-6, verbose=False):
    """Run EM until the *average* log-likelihood increases by less than epsLLAverage."""
    assert covType.lower() in ("full", "diagonal", "tied")
    llOld = logpdf_GMM(X, gmm).mean()
    if verbose:
        print(f"  EM it   0 - avg ll {llOld:.8e}")
    it = 1
    llDelta = None
    while llDelta is None or llDelta > epsLLAverage:
        gmmUpd = _em_iteration(X, gmm, covType=covType, psiEig=psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print(f"  EM it {it:3d} - avg ll {llUpd:.8e}")
        gmm = gmmUpd
        llOld = llUpd
        it += 1
    return gmm


def split_GMM_LBG(gmm, alpha=0.1):
    """Replace every component (w, mu, C) with two components displaced along the leading
    eigenvector of C, each receiving half of the original weight.
    """
    gmmOut = []
    for w, mu, C in gmm:
        U, s, _ = numpy.linalg.svd(C)
        d = U[:, 0:1] * (s[0] ** 0.5) * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut


def train_GMM_LBG_EM(X, numComponents, covType="Full", psiEig=None,
                     epsLLAverage=1e-6, lbgAlpha=0.1, verbose=False):
    """LBG initialisation followed by EM. Starts from a single-Gaussian ML fit, then doubles the
    number of components via split_GMM_LBG and re-runs EM until numComponents are reached.
    """
    mu, C = compute_mu_C(X)
    if covType.lower() == "diagonal":
        C = C * numpy.eye(X.shape[0])
    if psiEig is not None:
        C = smooth_covariance_matrix(C, psiEig)
    gmm = [(1.0, mu, C)]
    while len(gmm) < numComponents:
        if verbose:
            print(f"LBG: split {len(gmm)} -> {len(gmm) * 2}")
        gmm = split_GMM_LBG(gmm, alpha=lbgAlpha)
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig,
                           epsLLAverage=epsLLAverage, verbose=verbose)
    return gmm


def compute_llr_binary(DVAL, gmm1, gmm0):
    """Class 1 vs class 0 log-likelihood ratio (genuine - fake under our convention)."""
    return logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
