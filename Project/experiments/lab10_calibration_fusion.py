"""
Laboratory 10 - Project task: Score calibration and score-level fusion on the
fingerprint-spoofing dataset.

Target application: pi_T = 0.1 (same as labs 6-9).  We pick the best classifier from
each of the three families considered for the 01URTOV course - quadratic logistic
regression (lab 7), RBF SVM (lab 8) and diagonal-covariance GMM (lab 9) - and:

  1. K-fold calibrate (K=5) each of them on the validation scores at pT = 0.1.
  2. K-fold fuse the three calibrated systems at pT = 0.1.
  3. Choose the "delivered" model on actDCF (not minDCF), train the final calibration
     / fusion model on the full validation scores, and apply it to evalData.txt.
  4. Compare actDCF Bayes error plots of the three best systems and their fusion on
     the evaluation set.
  5. [Optional] Compute minDCF on the evaluation set for *all* GMM configurations
     (Full + Diagonal, the family of the delivered model) to check whether the
     selected hyper-parameters are still optimal on held-out data.

The calibration helpers live in models/calibration.py; the K-fold logic follows the
lab 10 reference solution but works in (M, N) format consistent with the rest of the
project.
"""

import os
import sys

import numpy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, evaluation, logger
from utils.math_utils import vrow
from models import calibration
from models import gmm as gmm_module
from models import logistic_regression as lr
from models import svm as svm_module


PLOT_DIR = "out/plots/lab10"
DATA_DIR = "out/lab10"

TARGET_PRIOR = 0.1
KFOLD = 5

# Best hyper-parameters selected in the previous labs (cf. CLAUDE.md sanity-check
# numbers and the lab07/08/09 logs).
BEST_LR_LAMBDA = 3.2e-2          # Quadratic LR, lab 7
BEST_SVM_GAMMA = numpy.exp(-2)   # RBF kernel, lab 8
BEST_SVM_C = 32.0                # RBF kernel, lab 8
BEST_SVM_EPS = 1.0               # RBF kernel xi, lab 8
BEST_GMM_K0 = 8                  # Diagonal GMM, lab 9
BEST_GMM_K1 = 16                 # Diagonal GMM, lab 9
GMM_PSI_EIG = 1e-2
GMM_LBG_ALPHA = 0.1


# ------------------------- generic plotting / metrics --------------------------------

def bayes_error_curves(scores, labels, lo=-4.0, hi=4.0, npts=21):
    logOdds = numpy.linspace(lo, hi, npts)
    effPriors = 1.0 / (1.0 + numpy.exp(-logOdds))
    actL = [evaluation.compute_actDCF_binary_fast(scores, labels, p, 1.0, 1.0) for p in effPriors]
    minL = [evaluation.compute_minDCF_binary_fast(scores, labels, p, 1.0, 1.0) for p in effPriors]
    return logOdds, numpy.asarray(actL), numpy.asarray(minL)


def plot_pre_post_calibration(name, pre_scores, pre_labels, cal_scores, cal_labels, fname):
    """Reproduce Figure 5/6 of the lab PDF: minDCF and actDCF for raw and calibrated
    scores on the same axes (raw values give the "pre-cal." curves)."""
    lo, act_pre, min_pre = bayes_error_curves(pre_scores, pre_labels)
    _, act_cal, _ = bayes_error_curves(cal_scores, cal_labels)
    plt.figure(figsize=(7, 4.5))
    plt.plot(lo, min_pre, "b--", label="minDCF (pre-cal.)")
    plt.plot(lo, act_pre, "b:", label="actDCF (pre-cal.)")
    plt.plot(lo, act_cal, "r-", label="actDCF (cal.)")
    plt.title(f"{name} - K-fold calibration ({'validation' if pre_labels is cal_labels else 'evaluation'})")
    plt.xlabel("prior log-odds")
    plt.ylabel(f"DCF (target pi_T = {TARGET_PRIOR})")
    plt.ylim(0.0, 0.8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()


def plot_eval_calibrated(name, raw_eval, eval_labels, cal_eval, fname):
    lo, act_pre, min_pre = bayes_error_curves(raw_eval, eval_labels)
    _, act_cal, _ = bayes_error_curves(cal_eval, eval_labels)
    plt.figure(figsize=(7, 4.5))
    plt.plot(lo, min_pre, "b--", label="minDCF")
    plt.plot(lo, act_pre, "b:", label="actDCF (pre-cal.)")
    plt.plot(lo, act_cal, "r-", label="actDCF (cal.)")
    plt.title(f"{name} - evaluation set")
    plt.xlabel("prior log-odds")
    plt.ylabel(f"DCF (target pi_T = {TARGET_PRIOR})")
    plt.ylim(0.0, 0.8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()


def plot_three_systems_plus_fusion(scores_dict, labels, fname, title):
    """One plot showing the actDCF/minDCF Bayes error curves for three calibrated
    systems plus their fusion.  scores_dict keys appear in the legend."""
    plt.figure(figsize=(7.5, 5))
    colors = ["C0", "C1", "C2", "C3"]
    for (name, scores), col in zip(scores_dict.items(), colors):
        lo, act, mn = bayes_error_curves(scores, labels)
        plt.plot(lo, mn, color=col, linestyle="--", label=f"{name} - minDCF")
        plt.plot(lo, act, color=col, linestyle="-", label=f"{name} - actDCF")
    plt.title(title)
    plt.xlabel("prior log-odds")
    plt.ylabel(f"DCF (target pi_T = {TARGET_PRIOR})")
    plt.ylim(0.0, 0.8)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()


def metrics(name, scores, labels):
    act = evaluation.compute_actDCF_binary_fast(scores, labels, TARGET_PRIOR, 1.0, 1.0)
    mn = evaluation.compute_minDCF_binary_fast(scores, labels, TARGET_PRIOR, 1.0, 1.0)
    print(f"  {name:<40} | actDCF = {act:.4f} | minDCF = {mn:.4f}")
    return act, mn


# ------------------------- per-model retraining --------------------------------------

def lr_quadratic_scores(DTR, LTR, DVAL, DEVAL):
    """Quadratic LR with the lab07 best lambda. Returns (val_sLLR, eval_sLLR)."""
    DTR_q = lr.expand_features_quadratic(DTR)
    DVAL_q = lr.expand_features_quadratic(DVAL)
    DEVAL_q = lr.expand_features_quadratic(DEVAL)
    pi_emp = (LTR == 1).sum() / LTR.size
    w, b, _ = lr.trainLogRegBinary(DTR_q, LTR, BEST_LR_LAMBDA)
    sVAL = lr.scores_to_llr(lr.compute_scores(w, b, DVAL_q), pi_emp)
    sEVAL = lr.scores_to_llr(lr.compute_scores(w, b, DEVAL_q), pi_emp)
    return sVAL, sEVAL, pi_emp


def svm_rbf_scores(DTR, LTR, DVAL, DEVAL):
    """RBF SVM with the lab08 best gamma/C. Returns (val_scores, eval_scores)."""
    fScore, _, _ = svm_module.train_dual_SVM_kernel(
        DTR, LTR, BEST_SVM_C, svm_module.rbfKernel(gamma=BEST_SVM_GAMMA), eps=BEST_SVM_EPS,
    )
    return fScore(DVAL), fScore(DEVAL)


def gmm_diagonal_scores(DTR, LTR, DVAL, DEVAL):
    """Diagonal GMM with the lab09 best (K0, K1). Returns (val_llr, eval_llr)."""
    gmm0 = gmm_module.train_GMM_LBG_EM(
        DTR[:, LTR == 0], BEST_GMM_K0, covType="Diagonal",
        psiEig=GMM_PSI_EIG, lbgAlpha=GMM_LBG_ALPHA,
    )
    gmm1 = gmm_module.train_GMM_LBG_EM(
        DTR[:, LTR == 1], BEST_GMM_K1, covType="Diagonal",
        psiEig=GMM_PSI_EIG, lbgAlpha=GMM_LBG_ALPHA,
    )
    sVAL = gmm_module.compute_llr_binary(DVAL, gmm1, gmm0)
    sEVAL = gmm_module.compute_llr_binary(DEVAL, gmm1, gmm0)
    return sVAL, sEVAL


# ------------------------- main --------------------------------------------------

if __name__ == "__main__":
    logger.setup_logger("out/lab10_calibration_fusion.txt")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    fname_train = "data/trainData.txt"
    fname_eval = "data/evalData.txt"
    if len(sys.argv) > 1:
        fname_train = sys.argv[1]
    if len(sys.argv) > 2:
        fname_eval = sys.argv[2]

    print(f"Loading training data from {fname_train}")
    D, L = data_utils.load(fname_train)
    (DTR, LTR), (DVAL, LVAL) = data_utils.split_db_2to1(D, L)
    print(f"  DTR shape: {DTR.shape}, DVAL shape: {DVAL.shape}")

    print(f"Loading evaluation data from {fname_eval}")
    DEVAL, LEVAL = data_utils.load(fname_eval)
    print(f"  DEVAL shape: {DEVAL.shape}")

    # ---------- (Re)train the three best-per-family models on DTR ------------------
    print("\n############### Retraining best per-family models on DTR ###############")
    print(f"  Quadratic LR  (lambda = {BEST_LR_LAMBDA:.1e})")
    sVAL_LR, sEVAL_LR, pi_emp = lr_quadratic_scores(DTR, LTR, DVAL, DEVAL)
    print(f"     empirical pi_1 on DTR = {pi_emp:.4f}")

    print(f"  RBF SVM       (gamma = e^-2 = {BEST_SVM_GAMMA:.4f}, C = {BEST_SVM_C:g}, xi = {BEST_SVM_EPS:g})")
    sVAL_SVM, sEVAL_SVM = svm_rbf_scores(DTR, LTR, DVAL, DEVAL)

    print(f"  Diagonal GMM  (K0 = {BEST_GMM_K0}, K1 = {BEST_GMM_K1})")
    sVAL_GMM, sEVAL_GMM = gmm_diagonal_scores(DTR, LTR, DVAL, DEVAL)

    # Sanity check: validation metrics should match the labs (LR/SVM/GMM logs).
    print("\n--- Sanity check: raw validation metrics at pi_T = 0.1 ---")
    metrics("Quadratic LR  (val raw, pi_emp offset)", sVAL_LR, LVAL)
    metrics("RBF SVM       (val raw)", sVAL_SVM, LVAL)
    metrics("Diagonal GMM  (val raw)", sVAL_GMM, LVAL)

    # ---------- K-fold calibration of each system --------------------------------
    print("\n############### Step 1 - K-fold calibration of each system ###############")
    print(f"K = {KFOLD}, pT (cal training prior) = {TARGET_PRIOR}, target pi_T = {TARGET_PRIOR}")

    cal_results = {}   # name -> dict(pooled_s, pooled_l, alpha, gamma, act, mn)
    for name, sVAL in [("LR_quadratic", sVAL_LR),
                       ("SVM_RBF", sVAL_SVM),
                       ("GMM_diagonal", sVAL_GMM)]:
        pooled_s, pooled_l = calibration.kfold_calibrate(sVAL, LVAL, TARGET_PRIOR, K=KFOLD)
        print(f"\n  -- {name} --")
        act_raw, mn_raw = metrics("raw scores (val)", sVAL, LVAL)
        act_cal, mn_cal = metrics("calibrated scores (val, K-fold pooled)", pooled_s, pooled_l)
        cal_results[name] = {
            "pooled_s": pooled_s, "pooled_l": pooled_l,
            "act_raw": act_raw, "mn_raw": mn_raw,
            "act_cal": act_cal, "mn_cal": mn_cal,
        }
        plot_pre_post_calibration(
            name, sVAL, LVAL, pooled_s, pooled_l,
            f"01_cal_kfold_{name}_val.png",
        )

    # ---------- K-fold fusion of the three systems --------------------------------
    print("\n############### Step 2 - K-fold fusion of the three systems ###############")
    S_val_matrix = numpy.vstack([sVAL_LR, sVAL_SVM, sVAL_GMM])
    pooled_fus_s, pooled_fus_l = calibration.kfold_fuse(S_val_matrix, LVAL, TARGET_PRIOR, K=KFOLD)
    print(f"\n  -- Fusion (LR + SVM + GMM) --")
    act_fus_val, mn_fus_val = metrics("fused calibrated scores (val, K-fold pooled)",
                                      pooled_fus_s, pooled_fus_l)

    # ---------- Bayes error plot: the three calibrated systems + fusion on val -----
    plot_three_systems_plus_fusion(
        {
            "LR (cal.)":   cal_results["LR_quadratic"]["pooled_s"],
            "SVM (cal.)":  cal_results["SVM_RBF"]["pooled_s"],
            "GMM (cal.)":  cal_results["GMM_diagonal"]["pooled_s"],
            "Fusion":      pooled_fus_s,
        },
        # All four pooled label arrays come from the same K-fold partition of LVAL,
        # so the pooling order is identical: any of them works as the label vector.
        pooled_fus_l,
        "02_bayes_three_plus_fusion_val.png",
        "Calibrated systems + fusion - validation (K-fold pooled)",
    )

    # ---------- Summary table for the validation set -------------------------------
    print("\n--- Validation summary (K-fold pooled) ---")
    print(f"{'system':>32} | {'minDCF':>7} | {'actDCF':>7}")
    print("-" * 54)
    for name in ["LR_quadratic", "SVM_RBF", "GMM_diagonal"]:
        r = cal_results[name]
        print(f"{name + ' (raw)':>32} | {r['mn_raw']:>7.4f} | {r['act_raw']:>7.4f}")
        print(f"{name + ' (cal.)':>32} | {r['mn_cal']:>7.4f} | {r['act_cal']:>7.4f}")
    print(f"{'Fusion (cal.)':>32} | {mn_fus_val:>7.4f} | {act_fus_val:>7.4f}")

    # ---------- Choose delivered model on actDCF -----------------------------------
    print("\n############### Step 3 - Delivered system choice ###############")
    candidates_actDCF = [
        ("LR_quadratic", cal_results["LR_quadratic"]["act_cal"]),
        ("SVM_RBF", cal_results["SVM_RBF"]["act_cal"]),
        ("GMM_diagonal", cal_results["GMM_diagonal"]["act_cal"]),
        ("Fusion", act_fus_val),
    ]
    candidates_actDCF.sort(key=lambda x: x[1])
    delivered_name, delivered_actDCF = candidates_actDCF[0]
    print(f"  Ranking by actDCF on validation (K-fold pooled):")
    for n, a in candidates_actDCF:
        marker = "  <- delivered" if n == delivered_name else ""
        print(f"     {n:>18}  actDCF = {a:.4f}{marker}")
    print(f"  Delivered system: {delivered_name}  (actDCF = {delivered_actDCF:.4f})")

    # ---------- Final calibration / fusion model trained on the WHOLE val set ------
    # For each system we still train a per-system calibrator (used to plot the
    # evaluation-set behaviour); the actual delivered system is then either one of the
    # calibrated singles or the fusion, depending on the ranking above.
    print("\n############### Step 4 - Apply final models to evaluation set ###############")

    final_cal_eval = {}
    for name, sVAL, sEVAL in [
        ("LR_quadratic", sVAL_LR, sEVAL_LR),
        ("SVM_RBF", sVAL_SVM, sEVAL_SVM),
        ("GMM_diagonal", sVAL_GMM, sEVAL_GMM),
    ]:
        alpha, gamma = calibration.train_calibration(sVAL, LVAL, TARGET_PRIOR)
        cal_eval = calibration.apply_calibration(sEVAL, alpha, gamma)
        final_cal_eval[name] = {
            "raw": sEVAL, "cal": cal_eval, "alpha": alpha, "gamma": gamma,
        }
        print(f"\n  -- {name} on EVALUATION set --")
        act_raw, mn_raw = metrics("raw scores (eval)", sEVAL, LEVAL)
        act_cal, mn_cal = metrics("calibrated scores (eval)", cal_eval, LEVAL)
        final_cal_eval[name].update({
            "act_raw": act_raw, "mn_raw": mn_raw,
            "act_cal": act_cal, "mn_cal": mn_cal,
        })
        plot_eval_calibrated(name, sEVAL, LEVAL, cal_eval, f"03_cal_{name}_eval.png")

    # Final fusion model trained on all validation scores at once
    alphas_fus, gamma_fus = calibration.train_fusion(S_val_matrix, LVAL, TARGET_PRIOR)
    S_eval_matrix = numpy.vstack([sEVAL_LR, sEVAL_SVM, sEVAL_GMM])
    fused_eval = calibration.apply_fusion(S_eval_matrix, alphas_fus, gamma_fus)
    print("\n  -- Fusion (LR + SVM + GMM) on EVALUATION set --")
    act_fus_eval, mn_fus_eval = metrics("fused calibrated scores (eval)", fused_eval, LEVAL)
    final_cal_eval["Fusion"] = {
        "cal": fused_eval, "act_cal": act_fus_eval, "mn_cal": mn_fus_eval,
    }

    # ---------- Bayes error plots on evaluation: 3 best + fusion -------------------
    plot_three_systems_plus_fusion(
        {
            "LR (cal.)":  final_cal_eval["LR_quadratic"]["cal"],
            "SVM (cal.)": final_cal_eval["SVM_RBF"]["cal"],
            "GMM (cal.)": final_cal_eval["GMM_diagonal"]["cal"],
            "Fusion":     fused_eval,
        },
        LEVAL,
        "04_bayes_three_plus_fusion_eval.png",
        "Calibrated systems + fusion - evaluation set",
    )

    # Dedicated Bayes error plot for the chosen delivered system on eval data
    # (this is the figure used in §10.1 of the report).
    delivered_scores = final_cal_eval[delivered_name]["cal"]
    lo, act_d, min_d = bayes_error_curves(delivered_scores, LEVAL)
    plt.figure(figsize=(7, 4.5))
    plt.plot(lo, min_d, "b--", label="minDCF")
    plt.plot(lo, act_d, "r-", label="actDCF")
    plt.title(f"Delivered system ({delivered_name}) - evaluation set")
    plt.xlabel("prior log-odds")
    plt.ylabel(f"DCF (target pi_T = {TARGET_PRIOR})")
    plt.ylim(0.0, 0.8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_delivered_bayes_eval.png")
    plt.close()

    # ---------- Evaluation summary table -------------------------------------------
    print("\n--- Evaluation summary ---")
    print(f"{'system':>32} | {'minDCF':>7} | {'actDCF':>7}")
    print("-" * 54)
    for name in ["LR_quadratic", "SVM_RBF", "GMM_diagonal"]:
        r = final_cal_eval[name]
        print(f"{name + ' (raw)':>32} | {r['mn_raw']:>7.4f} | {r['act_raw']:>7.4f}")
        print(f"{name + ' (cal.)':>32} | {r['mn_cal']:>7.4f} | {r['act_cal']:>7.4f}")
    print(f"{'Fusion (cal.)':>32} | {mn_fus_eval:>7.4f} | {act_fus_eval:>7.4f}")

    # ---------- [Optional] All GMM configs on eval ----------------------------------
    print("\n############### Step 5 [optional] - All GMM configs minDCF on eval ###############")
    print("Repeating the lab 9 sweep (Full + Diagonal) and reporting minDCF on the EVAL set.")
    component_grid = [1, 2, 4, 8, 16]
    eval_grid = {"Full": {}, "Diagonal": {}}
    for cov in ("Full", "Diagonal"):
        for K0 in component_grid:
            for K1 in component_grid:
                gmm0 = gmm_module.train_GMM_LBG_EM(
                    DTR[:, LTR == 0], K0, covType=cov,
                    psiEig=GMM_PSI_EIG, lbgAlpha=GMM_LBG_ALPHA,
                )
                gmm1 = gmm_module.train_GMM_LBG_EM(
                    DTR[:, LTR == 1], K1, covType=cov,
                    psiEig=GMM_PSI_EIG, lbgAlpha=GMM_LBG_ALPHA,
                )
                llr = gmm_module.compute_llr_binary(DEVAL, gmm1, gmm0)
                mn = evaluation.compute_minDCF_binary_fast(llr, LEVAL, TARGET_PRIOR, 1.0, 1.0)
                eval_grid[cov][(K0, K1)] = mn
                print(f"  cov={cov:>8}  K0={K0:>2}  K1={K1:>2}   minDCF(eval) = {mn:.4f}")
        # Pretty-print the matrix
        print(f"\n--- Eval minDCF, covType = {cov} ---")
        print("        " + " ".join(f"K1={K1:>2}" for K1 in component_grid))
        for K0 in component_grid:
            line = f"K0={K0:>2}   " + " ".join(
                f"{eval_grid[cov][(K0, K1)]:>5.4f}" for K1 in component_grid
            )
            print(line)
    best_eval_cfg = min(
        ((cov, K0, K1, mn) for cov, d in eval_grid.items() for (K0, K1), mn in d.items()),
        key=lambda x: x[3],
    )
    print(f"\nBest GMM config on EVAL: cov={best_eval_cfg[0]}, K0={best_eval_cfg[1]}, "
          f"K1={best_eval_cfg[2]}, minDCF = {best_eval_cfg[3]:.4f}")
    print(f"Selected config (on validation): cov=Diagonal, K0={BEST_GMM_K0}, K1={BEST_GMM_K1}, "
          f"minDCF(eval) = {eval_grid['Diagonal'][(BEST_GMM_K0, BEST_GMM_K1)]:.4f}")

    # ---------- Persist a few useful arrays for traceability ------------------------
    numpy.save(f"{DATA_DIR}/raw_val_LR.npy", sVAL_LR)
    numpy.save(f"{DATA_DIR}/raw_val_SVM.npy", sVAL_SVM)
    numpy.save(f"{DATA_DIR}/raw_val_GMM.npy", sVAL_GMM)
    numpy.save(f"{DATA_DIR}/raw_eval_LR.npy", sEVAL_LR)
    numpy.save(f"{DATA_DIR}/raw_eval_SVM.npy", sEVAL_SVM)
    numpy.save(f"{DATA_DIR}/raw_eval_GMM.npy", sEVAL_GMM)
    numpy.save(f"{DATA_DIR}/cal_eval_LR.npy", final_cal_eval["LR_quadratic"]["cal"])
    numpy.save(f"{DATA_DIR}/cal_eval_SVM.npy", final_cal_eval["SVM_RBF"]["cal"])
    numpy.save(f"{DATA_DIR}/cal_eval_GMM.npy", final_cal_eval["GMM_diagonal"]["cal"])
    numpy.save(f"{DATA_DIR}/cal_eval_Fusion.npy", fused_eval)
    print("\n  saved raw and calibrated eval scores under out/lab10/")
