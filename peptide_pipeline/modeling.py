"""Core modeling utilities: Logistic Regression fitting, coefficient-threshold
feature pruning, and classification metrics reporting.

The central idea reused across the ensemble and ablation pipelines is a
*coefficient-threshold* feature selection scheme: fit an L2-regularized
Logistic Regression on standardized features, then keep only the features
whose fitted coefficient magnitude exceeds a threshold, chosen to maximize
Matthews Correlation Coefficient (MCC) on a held-out validation split. MCC is
used as the primary selection metric because it is robust to class imbalance,
which is common in peptide bioactivity datasets.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

#: Default |coefficient| thresholds swept when pruning Logistic Regression features.
DEFAULT_COEF_GRID = (0.0, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)


def fit_lr(X: np.ndarray, y: np.ndarray, class_weight: Optional[Dict[int, float]] = None) -> LogisticRegression:
    """Fit an L2-regularized Logistic Regression classifier.

    Parameters
    ----------
    X, y:
        Standardized feature matrix and binary labels.
    class_weight:
        Optional per-class weight dict (e.g. from
        ``sklearn.utils.class_weight.compute_class_weight``) to counteract
        class imbalance. Pass ``None`` for uniform weighting.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted model (``lbfgs`` solver, up to 10,000 iterations).
    """
    return LogisticRegression(max_iter=10000, solver="lbfgs", class_weight=class_weight).fit(X, y)


def mcc_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    """Matthews Correlation Coefficient of predictions thresholded at ``thr``.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    probs:
        Predicted positive-class probabilities.
    thr:
        Decision threshold; ``probs > thr`` is predicted positive.

    Returns
    -------
    float
    """
    y_pred = (probs > thr).astype(int)
    return matthews_corrcoef(y_true, y_pred)


def tune_coef_threshold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    coef_grid: Sequence[float] = DEFAULT_COEF_GRID,
    decision_threshold: float = 0.4,
    class_weight: Optional[Dict[int, float]] = None,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """Select the |coefficient| threshold that maximizes validation MCC.

    Fits a base Logistic Regression on ``(X_tr, y_tr)``, then for each
    candidate threshold in ``coef_grid``: keeps features whose base-model
    coefficient magnitude is >= threshold, refits + rescales on just those
    features, and evaluates MCC on ``(X_val, y_val)`` at a fixed
    ``decision_threshold``. At least one feature (the single largest
    |coefficient|) is always kept, even if none clears the threshold.

    Parameters
    ----------
    X_tr, y_tr:
        Standardized training features and labels used to fit the base model
        and each pruned refit.
    X_val, y_val:
        Standardized validation features and labels used to score each
        candidate threshold.
    coef_grid:
        Candidate |coefficient| thresholds to sweep.
    decision_threshold:
        Probability threshold used when computing validation MCC for each
        candidate (this is a fixed scoring threshold, not itself tuned here;
        see the ensemble/ablation modules for separate decision-threshold
        tuning).
    class_weight:
        Optional per-class weight dict passed through to :func:`fit_lr`.

    Returns
    -------
    (best_thr, best_mask, best_mcc, base_coef)
        ``best_thr``: the chosen coefficient threshold.
        ``best_mask``: boolean feature-keep mask (length = X_tr.shape[1])
        for that threshold, computed on the base model's coefficients.
        ``best_mcc``: validation MCC achieved by the pruned refit.
        ``base_coef``: the base (all-features) model's coefficient vector.
    """
    base = fit_lr(X_tr, y_tr, class_weight)
    base_coef = base.coef_.ravel()
    best_thr, best_mcc, best_mask = None, -1.0, None

    for thr in coef_grid:
        keep = np.abs(base_coef) >= thr
        if not keep.any():
            keep[np.argmax(np.abs(base_coef))] = True  # always keep at least 1 feature

        X_tr_k = X_tr[:, keep]
        X_val_k = X_val[:, keep]
        scaler_k = StandardScaler().fit(X_tr_k)
        X_tr_k_s = scaler_k.transform(X_tr_k)
        X_val_k_s = scaler_k.transform(X_val_k)

        mdl = fit_lr(X_tr_k_s, y_tr, class_weight)
        p_val = mdl.predict_proba(X_val_k_s)[:, 1]
        mcc = mcc_at_threshold(y_val, p_val, thr=decision_threshold)

        if mcc > best_mcc + 1e-12:
            best_mcc, best_thr, best_mask = mcc, thr, keep

    return best_thr, best_mask, best_mcc, base_coef


def safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for binary labels ``{0, 1}``, even if a class is absent."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def metrics_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    thr: float = 0.4,
    name: str = "Model",
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute and (optionally) print a standard binary-classification report.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    probs:
        Predicted positive-class probabilities.
    thr:
        Decision threshold; ``probs > thr`` is predicted positive.
    name:
        Label used in the printed header (ignored if ``verbose=False``).
    verbose:
        If True, print a formatted summary in addition to returning it.

    Returns
    -------
    dict
        Keys: ``ACC``, ``AUC``, ``MCC``, ``F1``, ``PPV``, ``SN`` (sensitivity
        / recall), ``SP`` (specificity), ``tn``, ``fp``, ``fn``, ``tp``.
    """
    y_pred = (probs > thr).astype(int)
    tn, fp, fn, tp = safe_confusion(y_true, y_pred)
    sn = tp / (tp + fn + 1e-12)
    sp = tn / (tn + fp + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    mcc = matthews_corrcoef(y_true, y_pred)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = np.nan

    if verbose:
        print(f"\n{name} @ thr={thr:.2f}")
        print(f"  ACC: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f} | F1: {f1:.4f} | PPV: {ppv:.4f}")
        print(f"  SN:  {sn:.4f} | SP:  {sp:.4f}")
        print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    return dict(ACC=acc, AUC=auc, MCC=mcc, F1=f1, PPV=ppv, SN=sn, SP=sp, tn=tn, fp=fp, fn=fn, tp=tp)


def eval_if_labels_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thr: float,
    name: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compact one-line metrics report, used by the ablation leaderboards.

    Differs from :func:`metrics_report` only in output format: returns a
    flat dict (including ``name`` and capitalized ``TN``/``FP``/``FN``/``TP``
    keys as plain ``int``) directly usable as a row in a leaderboard
    DataFrame, and prints a single summary line instead of a multi-line
    block.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    probs:
        Predicted positive-class probabilities.
    thr:
        Decision threshold; ``probs > thr`` is predicted positive.
    name:
        Row label, e.g. a feature-combo or ensemble identifier.
    verbose:
        If True, print a one-line summary.

    Returns
    -------
    dict
        Keys: ``name``, ``ACC``, ``AUC``, ``MCC``, ``F1``, ``PPV``, ``TN``,
        ``FP``, ``FN``, ``TP``.
    """
    y_pred = (probs > thr).astype(int)
    tn, fp, fn, tp = safe_confusion(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = np.nan

    if verbose:
        sn = tp / (tp + fn + 1e-12)
        sp = tn / (tn + fp + 1e-12)
        print(
            f"{name}: ACC={acc:.4f} AUC={auc:.4f} MCC={mcc:.4f} F1={f1:.4f} "
            f"PPV={ppv:.4f} | SN={sn:.4f} SP={sp:.4f}"
        )

    return dict(name=name, ACC=acc, AUC=auc, MCC=mcc, F1=f1, PPV=ppv, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
