"""Shared preprocessing helpers used across the ablation and ensemble pipelines.

Includes near-zero-variance (constant column) pruning and small dataclasses
for keeping a feature block's train/test matrices, names, and group labels
together as a single object instead of four parallel variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_selection import VarianceThreshold


@dataclass
class FeatureBlock:
    """A named group of features with train/test matrices attached.

    Attributes
    ----------
    name:
        Short identifier for the block, e.g. ``"kmer"`` or ``"esm"``.
    X_train, X_test:
        Feature matrices, shape ``(n_train, n_features)`` / ``(n_test, n_features)``.
    feature_names:
        Column names, length ``n_features``.
    """

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: List[str]

    @property
    def group_names(self) -> List[str]:
        """Return ``[self.name] * n_features``, for bookkeeping when blocks are concatenated."""
        return [self.name] * len(self.feature_names)


def variance_prune(
    X_train: np.ndarray,
    X_test: np.ndarray,
    names: Sequence[str],
    groups: Sequence[str],
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
    """Drop near-zero-variance (e.g. constant) columns, fit only on train.

    Parameters
    ----------
    X_train, X_test:
        Feature matrices to prune. The variance threshold is fit on
        ``X_train`` only and then applied to both, to avoid leaking test-set
        statistics into feature selection.
    names, groups:
        Per-column feature names and feature-block group labels, in the same
        column order as ``X_train`` / ``X_test``.
    threshold:
        Minimum variance required to keep a column (``0.0`` drops exactly-
        constant columns, matching ``sklearn.feature_selection.VarianceThreshold``
        defaults).

    Returns
    -------
    (X_train_pruned, X_test_pruned, names_pruned, groups_pruned, keep_mask)
        Pruned matrices, the surviving names/groups, and the boolean mask
        (length = original number of columns) used to select them.
    """
    vt = VarianceThreshold(threshold=threshold)
    X_train_sel = vt.fit_transform(X_train)
    X_test_sel = vt.transform(X_test)
    mask = vt.get_support()
    names_sel = [n for n, m in zip(names, mask) if m]
    groups_sel = [g for g, m in zip(groups, mask) if m]
    return X_train_sel, X_test_sel, names_sel, groups_sel, mask


def build_matrix(
    blocks: Sequence[Tuple[np.ndarray, np.ndarray, Sequence[str], Sequence[str]]]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Concatenate several ``(X_train, X_test, names, groups)`` tuples column-wise.

    Parameters
    ----------
    blocks:
        Sequence of ``(X_train, X_test, feature_names, group_names)`` tuples,
        e.g. one per feature family, all sharing the same number of rows.

    Returns
    -------
    (X_train, X_test, names, groups)
        Concatenated matrices and combined name/group lists, in the order
        the blocks were given.
    """
    Xtr_parts, Xte_parts, names, groups = [], [], [], []
    for X_train, X_test, feat_names, grp_names in blocks:
        Xtr_parts.append(X_train)
        Xte_parts.append(X_test)
        names += list(feat_names)
        groups += list(grp_names)
    n_train = Xtr_parts[0].shape[0] if Xtr_parts else 0
    n_test = Xte_parts[0].shape[0] if Xte_parts else 0
    X_train_cat = np.concatenate(Xtr_parts, axis=1) if Xtr_parts else np.zeros((n_train, 0), dtype=np.float32)
    X_test_cat = np.concatenate(Xte_parts, axis=1) if Xte_parts else np.zeros((n_test, 0), dtype=np.float32)
    return X_train_cat, X_test_cat, names, groups
