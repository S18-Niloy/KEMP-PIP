"""Feature-block ensemble ablation (Logistic Regression, coefficient-pruned).

This module trains one coefficient-pruned Logistic Regression per feature-
block combination (see ``DEFAULT_COMBOS`` below), reports single-model
metrics for each, and then tunes a simple convex-combination ("alpha
blend") ensemble for every pair of combos: ``p_ens = alpha * p1 + (1-alpha) * p2``,
choosing ``alpha`` and the decision threshold to maximize MCC.

This generalizes (and, notably, corrects a feature/name mismatch present in)
the original notebook's hand-written two-model ("Model A" / "Model B")
ensemble into a clean N-combo version. See ``README.md`` -> "Notes on the
original notebook" for details of what changed and why.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from .modeling import DEFAULT_COEF_GRID, eval_if_labels_metrics, fit_lr, tune_coef_threshold
from .preprocessing import build_matrix, variance_prune

#: The five feature-block combinations evaluated by default, matching the
#: notebook's "ENSEMBLE ABLATION: A-E" cell. Each combo names three or four
#: of the four feature blocks (kmer, physchem, esm, modlamp).
DEFAULT_COMBO_BLOCK_NAMES: Dict[str, Tuple[str, ...]] = {
    "A": ("kmer", "physchem", "esm"),
    "B": ("kmer", "physchem", "modlamp"),
    "C": ("kmer", "esm", "modlamp"),
    "D": ("physchem", "esm", "modlamp"),
    "E": ("kmer", "physchem", "esm", "modlamp"),
}


def train_single_combo(
    Xtr_raw: np.ndarray,
    Xte_raw: np.ndarray,
    names: Sequence[str],
    groups: Sequence[str],
    y: np.ndarray,
    combo_name: str,
    class_weight: Dict[int, float],
    val_size: float = 0.10,
    random_state: int = 42,
    coef_grid: Sequence[float] = DEFAULT_COEF_GRID,
    decision_threshold: float = 0.4,
) -> Dict[str, object]:
    """Train one coefficient-pruned Logistic Regression for a single feature combo.

    Pipeline: split off a validation slice -> tune the |coefficient| pruning
    threshold on it (:func:`peptide_pipeline.modeling.tune_coef_threshold`)
    -> refit on the *full* training set with the chosen threshold -> tune a
    per-model decision threshold on a second validation slice (for reporting
    only; ensemble decision thresholds are tuned separately, see
    :func:`tune_pairwise_ensemble`).

    Parameters
    ----------
    Xtr_raw, Xte_raw:
        Raw (unscaled), near-zero-variance-pruned train/test feature matrix
        for this combo.
    names, groups:
        Feature names and feature-block labels, aligned to the columns of
        ``Xtr_raw`` / ``Xte_raw``.
    y:
        Training labels.
    combo_name:
        Identifier used only for logging/bookkeeping.
    class_weight:
        Per-class weight dict, e.g. from ``compute_class_weight``.
    val_size:
        Fraction of training data held out (stratified) to tune the
        coefficient threshold.
    random_state:
        Seed for the coefficient-threshold-tuning split.
    coef_grid:
        Candidate |coefficient| thresholds to sweep.
    decision_threshold:
        Fixed probability threshold used while scoring MCC during
        coefficient-threshold tuning.

    Returns
    -------
    dict
        Keys: ``name``, ``scalers`` (``(scaler_full, scaler_keep)``),
        ``keep_mask``, ``final_model``, ``probs_test``, ``thr_single``
        (per-model decision threshold tuned for standalone reporting),
        ``kept_names``, ``kept_groups``.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    idx_tr, idx_val = next(sss.split(Xtr_raw, y))

    scaler_tune = StandardScaler().fit(Xtr_raw[idx_tr])
    Xtr = scaler_tune.transform(Xtr_raw[idx_tr])
    Xval = scaler_tune.transform(Xtr_raw[idx_val])
    ytr, yval = y[idx_tr], y[idx_val]

    best_thr, keep_mask, best_mcc, base_coef = tune_coef_threshold(
        Xtr, ytr, Xval, yval, coef_grid, decision_threshold=decision_threshold, class_weight=class_weight
    )

    scaler_full = StandardScaler().fit(Xtr_raw)
    Xfull = scaler_full.transform(Xtr_raw)
    Xtest = scaler_full.transform(Xte_raw)

    base_full = fit_lr(Xfull, y, class_weight)
    coef_full = base_full.coef_.ravel()
    keep_final = np.abs(coef_full) >= best_thr
    if not keep_final.any():
        keep_final[np.argmax(np.abs(coef_full))] = True

    Xfull_k = Xfull[:, keep_final]
    Xtest_k = Xtest[:, keep_final]
    scaler_keep = StandardScaler().fit(Xfull_k)
    Xfull_k_s = scaler_keep.transform(Xfull_k)
    Xtest_k_s = scaler_keep.transform(Xtest_k)

    final = fit_lr(Xfull_k_s, y, class_weight)
    probs_test = final.predict_proba(Xtest_k_s)[:, 1]

    # Separate 10% slice used only to pick a per-model reporting threshold.
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=123)
    it_tr, it_val = next(sss2.split(Xtr_raw, y))
    X_a_sub = scaler_keep.transform(scaler_full.transform(Xtr_raw)[it_tr][:, keep_final])
    X_v_sub = scaler_keep.transform(scaler_full.transform(Xtr_raw)[it_val][:, keep_final])
    ya, yv = y[it_tr], y[it_val]

    mdl_sub = LogisticRegression(max_iter=10000, solver="lbfgs", class_weight=class_weight).fit(X_a_sub, ya)
    p_val = mdl_sub.predict_proba(X_v_sub)[:, 1]
    ths = np.linspace(0.10, 0.90, 81)
    best_t, best_m = decision_threshold, -1.0
    for t in ths:
        m = matthews_corrcoef(yv, (p_val > t).astype(int))
        if m > best_m:
            best_m, best_t = m, t

    kept_names = [n for n, m in zip(names, keep_final) if m]
    kept_groups = [g for g, m in zip(groups, keep_final) if m]

    return {
        "name": combo_name,
        "scalers": (scaler_full, scaler_keep),
        "keep_mask": keep_final,
        "final_model": final,
        "probs_test": probs_test,
        "thr_single": float(best_t),
        "kept_names": kept_names,
        "kept_groups": kept_groups,
    }


def tune_pairwise_ensemble(
    probs1: np.ndarray,
    probs2: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Tune the blend weight ``alpha`` and decision threshold for two probability vectors.

    Grid-searches ``p_ens = alpha * probs1 + (1 - alpha) * probs2`` over
    ``alpha in [0, 1]`` (21 steps) and ``threshold in [0.10, 0.90]``
    (81 steps), maximizing MCC.

    Parameters
    ----------
    probs1, probs2:
        Predicted positive-class probabilities from two models, aligned
        row-for-row.
    y_true:
        Ground-truth labels to score against. If ``None`` (no labels
        available -- inference-only mode), returns a neutral 50/50 blend at
        threshold 0.5 without tuning.

    Returns
    -------
    dict
        ``{"mcc": float | None, "alpha": float, "thr": float}``.
    """
    if y_true is None:
        return {"mcc": None, "alpha": 0.5, "thr": 0.5}

    alphas = np.linspace(0.0, 1.0, 21)
    ths = np.linspace(0.10, 0.90, 81)
    best = {"mcc": -1.0, "alpha": 0.5, "thr": 0.5}
    for a in alphas:
        p = a * probs1 + (1 - a) * probs2
        for t in ths:
            m = matthews_corrcoef(y_true, (p > t).astype(int))
            if m > best["mcc"]:
                best = {"mcc": float(m), "alpha": float(a), "thr": float(t)}
    return best


def run_ensemble_ablation(
    feature_blocks: Dict[str, Tuple[np.ndarray, np.ndarray, Sequence[str]]],
    y_train: np.ndarray,
    class_weight: Dict[int, float],
    y_test: Optional[np.ndarray] = None,
    combos: Optional[Dict[str, Tuple[str, ...]]] = None,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """Run the full A-E feature-combo ensemble ablation end to end.

    For each combo in ``combos`` (default: :data:`DEFAULT_COMBO_BLOCK_NAMES`),
    concatenates the requested feature blocks, near-zero-variance-prunes
    them, trains a coefficient-pruned Logistic Regression
    (:func:`train_single_combo`), and reports single-model metrics. Then
    tunes and reports a pairwise alpha-blend ensemble
    (:func:`tune_pairwise_ensemble`) for every pair of combos.

    Parameters
    ----------
    feature_blocks:
        Mapping ``block_name -> (X_train, X_test, feature_names)`` for each
        of the four base feature blocks (``"kmer"``, ``"physchem"``,
        ``"esm"``, ``"modlamp"``). All ``X_train`` arrays must have the same
        number of rows (and likewise for ``X_test``).
    y_train:
        Training labels.
    class_weight:
        Per-class weight dict.
    y_test:
        Optional test labels. If omitted, per-combo probabilities are
        returned but no metrics/leaderboard rows are computed -- use this
        mode to generate predictions for an unlabeled test set.
    combos:
        Mapping ``combo_name -> tuple of block names`` to evaluate. Defaults
        to :data:`DEFAULT_COMBO_BLOCK_NAMES` (the notebook's A-E combos).
    val_size, random_state:
        Passed through to :func:`train_single_combo`.

    Returns
    -------
    (models, leaderboard)
        ``models``: dict of combo_name -> the :func:`train_single_combo`
        result dict, for downstream plotting/interpretation.
        ``leaderboard``: DataFrame of single-model and pairwise-ensemble
        metrics, sorted by MCC, AUC, ACC (empty if ``y_test`` is ``None``).
    """
    combos = combos or DEFAULT_COMBO_BLOCK_NAMES
    models: Dict[str, Dict[str, object]] = {}
    results: List[dict] = []

    for combo_name, block_names in combos.items():
        blocks = [
            (feature_blocks[b][0], feature_blocks[b][1], list(feature_blocks[b][2]), [b] * feature_blocks[b][0].shape[1])
            for b in block_names
        ]
        Xtr_raw, Xte_raw, names_all, groups_all = build_matrix(blocks)
        Xtr_nzv, Xte_nzv, names_nzv, groups_nzv, _ = variance_prune(Xtr_raw, Xte_raw, names_all, groups_all)

        info = train_single_combo(
            Xtr_nzv, Xte_nzv, names_nzv, groups_nzv, y_train,
            combo_name=combo_name, class_weight=class_weight,
            val_size=val_size, random_state=random_state,
        )
        models[combo_name] = info

        if y_test is not None:
            metrics = eval_if_labels_metrics(y_test, info["probs_test"], info["thr_single"], name=f"{combo_name} [single]")
            results.append(metrics)

    # Pairwise alpha-blend ensembles across all combo pairs.
    names_list = list(combos.keys())
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            n1, n2 = names_list[i], names_list[j]
            p1, p2 = models[n1]["probs_test"], models[n2]["probs_test"]
            best = tune_pairwise_ensemble(p1, p2, y_true=y_test)
            if y_test is not None:
                p_ens = best["alpha"] * p1 + (1 - best["alpha"]) * p2
                metrics = eval_if_labels_metrics(
                    y_test, p_ens, best["thr"],
                    name=f"Ensemble[{n1} + {n2}] (alpha={best['alpha']:.2f}, t={best['thr']:.2f})",
                )
                results.append(metrics)

    leaderboard = (
        pd.DataFrame([r for r in results if r is not None]).sort_values(
            by=["MCC", "AUC", "ACC"], ascending=False
        )
        if results
        else pd.DataFrame()
    )
    return models, leaderboard
