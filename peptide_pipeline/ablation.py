"""Full feature-block ablation across four classifiers.

Evaluates every non-empty combination of the four feature blocks (k-mer,
physchem, modlAMP, ESM -- ``2**4 - 1 = 15`` combos) with four classifiers
(XGBoost when available, Random Forest, SVM, MLP), on both a held-out
validation split and (optionally) the test set. Tree-based models
(XGBoost, Random Forest) use raw features; SVM and MLP use per-block
standardized features, since they are scale-sensitive.

This mirrors the notebook's "FIXED" ablation cell, which slices the
validation split from the *training* blocks (rather than mixing train/test
partitions), and is the recommended entry point over the single-model
Logistic-Regression-only ablation in earlier notebook drafts.
"""

from __future__ import annotations

from itertools import chain, combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import variance_prune

try:
    from xgboost import XGBClassifier

    HAVE_XGB = True
except ImportError:  # pragma: no cover - exercised only when xgboost is absent
    HAVE_XGB = False

ALL_BLOCK_NAMES = ("kmer", "physchem", "modlamp", "esm")


def report_metrics_fast(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute a compact classification-metrics dict at a fixed decision threshold.

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
    dict
        Keys: ``ACC``, ``AUC``, ``MCC``, ``F1``, ``PPV``, ``SN``, ``SP``,
        ``TN``, ``FP``, ``FN``, ``TP``.
    """
    y_pred = (probs > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    sn = tp / (tp + fn + 1e-12)
    sp = tn / (tn + fp + 1e-12)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = np.nan
    return dict(
        ACC=acc, AUC=auc, MCC=mcc, F1=f1v, PPV=ppv, SN=sn, SP=sp,
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


def make_models(
    scale_pos_weight: float,
    random_state: int = 42,
) -> List[Tuple[str, object, str]]:
    """Instantiate the four ablation classifiers with fixed hyperparameters.

    Parameters
    ----------
    scale_pos_weight:
        ``n_negative / n_positive`` in the training set, passed to XGBoost
        to counteract class imbalance (Random Forest and SVM use
        ``class_weight="balanced"`` instead; MLP has no built-in class
        weighting and is left unweighted, matching the original notebook).
    random_state:
        Seed shared by all four models.

    Returns
    -------
    list of (name, estimator, kind)
        ``kind`` is ``"raw"`` for tree-based models (use unscaled features)
        or ``"scaled"`` for SVM/MLP (use standardized features). XGBoost is
        omitted if the optional ``xgboost`` dependency is not installed.
    """
    models: List[Tuple[str, object, str]] = []
    if HAVE_XGB:
        models.append((
            "XGB",
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=random_state,
                n_jobs=-1, eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            ),
            "raw",
        ))
    models.append((
        "RF",
        RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        ),
        "raw",
    ))
    models.append((
        "SVM",
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=random_state),
        "scaled",
    ))
    models.append((
        "MLP",
        MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu",
            learning_rate_init=1e-3, batch_size=128,
            max_iter=100, early_stopping=True, random_state=random_state,
        ),
        "scaled",
    ))
    return models


def prepare_ablation_blocks(
    feature_blocks: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Near-zero-variance-prune each block, then produce raw and standardized versions.

    Parameters
    ----------
    feature_blocks:
        Mapping ``block_name -> (X_train, X_test)`` for each of the four
        base feature blocks. Keys should match :data:`ALL_BLOCK_NAMES`.

    Returns
    -------
    (blocks_raw_tr, blocks_raw_te, blocks_scl_tr, blocks_scl_te)
        Four dicts keyed by block name: raw (NZV-pruned only) and
        standardized (scaler fit on train, applied to both) train/test
        matrices, ready to be concatenated per combo.
    """
    blocks_raw_tr: Dict[str, np.ndarray] = {}
    blocks_raw_te: Dict[str, np.ndarray] = {}
    blocks_scl_tr: Dict[str, np.ndarray] = {}
    blocks_scl_te: Dict[str, np.ndarray] = {}

    for name, (Xtr, Xte) in feature_blocks.items():
        dummy_names = [f"{name}_{i}" for i in range(Xtr.shape[1])]
        Xtr_nzv, Xte_nzv, _, _, _ = variance_prune(Xtr, Xte, dummy_names, dummy_names)
        blocks_raw_tr[name] = Xtr_nzv
        blocks_raw_te[name] = Xte_nzv
        if Xtr_nzv.shape[1] > 0:
            sc = StandardScaler().fit(Xtr_nzv)
            blocks_scl_tr[name] = sc.transform(Xtr_nzv)
            blocks_scl_te[name] = sc.transform(Xte_nzv)
        else:
            blocks_scl_tr[name] = Xtr_nzv
            blocks_scl_te[name] = Xte_nzv

    return blocks_raw_tr, blocks_raw_te, blocks_scl_tr, blocks_scl_te


def concat_blocks(block_list: Sequence[np.ndarray]) -> np.ndarray:
    """Column-wise concatenate a list of same-row-count feature matrices (no-op for a single block)."""
    return np.concatenate(block_list, axis=1) if len(block_list) > 1 else block_list[0]


def all_nonempty_combos(block_names: Sequence[str] = ALL_BLOCK_NAMES) -> List[Tuple[str, ...]]:
    """Return every non-empty subset of ``block_names`` (``2**n - 1`` combos), smallest first."""
    return list(chain.from_iterable(combinations(block_names, r) for r in range(1, len(block_names) + 1)))


def run_block_ablation(
    feature_blocks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    val_size: float = 0.10,
    random_state: int = 42,
    decision_threshold: float = 0.4,
    block_names: Sequence[str] = ALL_BLOCK_NAMES,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run the full 15-combo x 4-model ablation and return validation/test leaderboards.

    Parameters
    ----------
    feature_blocks:
        Mapping ``block_name -> (X_train, X_test)`` for each base feature
        block (before NZV pruning/scaling -- this function does that
        internally via :func:`prepare_ablation_blocks`).
    y_train:
        Training labels.
    y_test:
        Optional test labels. If given, each combo/model is also refit on
        the full training set and evaluated on the test set.
    val_size, random_state:
        Stratified validation split configuration (used for the
        validation-set evaluation only; the full training set is used when
        evaluating on ``y_test``).
    decision_threshold:
        Fixed probability threshold used for all reported metrics
        (this ablation does not itself tune per-combo thresholds).
    block_names:
        Which blocks to combine; defaults to all four
        (:data:`ALL_BLOCK_NAMES`).

    Returns
    -------
    (val_df, test_df)
        Validation-set leaderboard (always returned, sorted by MCC, AUC,
        ACC) and test-set leaderboard (``None`` if ``y_test`` was not
        given).
    """
    blocks_raw_tr, blocks_raw_te, blocks_scl_tr, blocks_scl_te = prepare_ablation_blocks(feature_blocks)

    ref_block = blocks_raw_tr[block_names[0]]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    idx_tr, idx_val = next(sss.split(ref_block, y_train))
    y_tr, y_val = y_train[idx_tr], y_train[idx_val]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg) / float(max(pos, 1))

    combos = all_nonempty_combos(block_names)

    val_rows = []
    for combo in combos:
        combo = list(combo)
        combo_name = "+".join(combo)

        Xtr_raw = concat_blocks([blocks_raw_tr[b][idx_tr] for b in combo])
        Xva_raw = concat_blocks([blocks_raw_tr[b][idx_val] for b in combo])
        Xtr_scl = concat_blocks([blocks_scl_tr[b][idx_tr] for b in combo])
        Xva_scl = concat_blocks([blocks_scl_tr[b][idx_val] for b in combo])

        for mdl_name, mdl, kind in make_models(scale_pos_weight, random_state):
            Xtr_use, Xva_use = (Xtr_raw, Xva_raw) if kind == "raw" else (Xtr_scl, Xva_scl)
            mdl.fit(Xtr_use, y_tr)
            p_val = mdl.predict_proba(Xva_use)[:, 1]
            m = report_metrics_fast(y_val, p_val, thr=decision_threshold)
            m.update(dict(model=mdl_name, combo=combo_name, n_feat=Xtr_use.shape[1], set="val"))
            val_rows.append(m)

    val_df = pd.DataFrame(val_rows).sort_values(["MCC", "AUC", "ACC"], ascending=False).reset_index(drop=True)

    test_df = None
    if y_test is not None:
        test_rows = []
        for combo in combos:
            combo = list(combo)
            combo_name = "+".join(combo)

            Xtr_raw_full = concat_blocks([blocks_raw_tr[b] for b in combo])
            Xte_raw_full = concat_blocks([blocks_raw_te[b] for b in combo])
            Xtr_scl_full = concat_blocks([blocks_scl_tr[b] for b in combo])
            Xte_scl_full = concat_blocks([blocks_scl_te[b] for b in combo])

            for mdl_name, mdl, kind in make_models(scale_pos_weight, random_state):
                Xtr_use, Xte_use = (Xtr_raw_full, Xte_raw_full) if kind == "raw" else (Xtr_scl_full, Xte_scl_full)
                mdl.fit(Xtr_use, y_train)
                p_test = mdl.predict_proba(Xte_use)[:, 1]
                m = report_metrics_fast(y_test, p_test, thr=decision_threshold)
                m.update(dict(model=mdl_name, combo=combo_name, n_feat=Xtr_use.shape[1], set="test"))
                test_rows.append(m)

        test_df = pd.DataFrame(test_rows).sort_values(["MCC", "AUC", "ACC"], ascending=False).reset_index(drop=True)

    return val_df, test_df
