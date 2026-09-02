"""Post-hoc interpretation of a fitted Logistic Regression's coefficients.

Provides a simple "positive-quadrant" (eigenvector-style) projection that
L2-normalizes a coefficient vector and takes its absolute value, turning
signed coefficients into a non-negative "importance" score comparable across
features regardless of the direction of their association with the positive
class. This is an interpretation layer only: it never changes or refits the
underlying trained model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def positive_quadrant_projection(coef_vec: Sequence[float]) -> np.ndarray:
    """L2-normalize a coefficient vector, then take its absolute value.

    Parameters
    ----------
    coef_vec:
        Raw fitted coefficients (any sign) for one model's kept features.

    Returns
    -------
    numpy.ndarray
        Non-negative importance scores, same length as ``coef_vec``, with
        L2 norm 1 before the absolute value is taken (so
        ``sum(result**2) == 1`` up to floating point).
    """
    coef_vec = np.asarray(coef_vec, dtype=float)
    norm = np.linalg.norm(coef_vec) + 1e-12
    return np.abs(coef_vec / norm)


def per_group_projection(
    df_in: pd.DataFrame,
    group_col: str = "group",
    coef_col: str = "coef",
    out_col: str = "coef_pos_eig_group",
) -> pd.DataFrame:
    """Apply :func:`positive_quadrant_projection` independently within each feature group.

    Unlike a single global projection over all features, this normalizes
    each feature-block's coefficients (e.g. all "kmer" features, all "esm"
    features, ...) to its own unit L2 norm before taking the absolute value,
    so importance scores are comparable *within* a block even if the blocks
    have very different coefficient scales.

    Parameters
    ----------
    df_in:
        DataFrame with at least ``group_col`` and ``coef_col`` columns (one
        row per feature).
    group_col:
        Column identifying each feature's block/group.
    coef_col:
        Column holding the raw fitted coefficient for each feature.
    out_col:
        Name of the new column to add with the per-group projected scores.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df_in`` with ``out_col`` added.
    """
    df = df_in.copy()
    pos_vals = np.zeros(len(df))
    for _, idx in df.groupby(group_col).groups.items():
        v = df.loc[idx, coef_col].values
        pos_vals[idx] = positive_quadrant_projection(v)
    df[out_col] = pos_vals
    return df


def build_coefficient_table(kept_names: Sequence[str], kept_groups: Sequence[str], coef: np.ndarray) -> pd.DataFrame:
    """Assemble a tidy per-feature coefficient table with both projections.

    Parameters
    ----------
    kept_names, kept_groups:
        Feature names and group labels surviving coefficient-threshold
        pruning (e.g. ``model_info["kept_names"]`` / ``["kept_groups"]``
        from :func:`peptide_pipeline.ensemble.train_single_combo`).
    coef:
        The fitted model's coefficient vector, aligned to ``kept_names``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``group``, ``coef``, ``coef_pos_eig`` (global
        positive-quadrant projection), ``coef_pos_eig_group`` (per-group
        projection).
    """
    df = pd.DataFrame({"feature": kept_names, "group": kept_groups, "coef": coef})
    df["coef_pos_eig"] = positive_quadrant_projection(df["coef"].values)
    df = per_group_projection(df, group_col="group", coef_col="coef", out_col="coef_pos_eig_group")
    return df


def summarize_positive_quadrant(df: pd.DataFrame, name: str, col: str = "coef_pos_eig") -> pd.DataFrame:
    """Print and return a per-group summary (count/mean/sum) of a projected-importance column.

    Parameters
    ----------
    df:
        Output of :func:`build_coefficient_table` (or any DataFrame with
        ``group`` and ``col`` columns).
    name:
        Label used in the printed header.
    col:
        Which projected-importance column to summarize.

    Returns
    -------
    pandas.DataFrame
        Columns: ``group``, ``count``, ``mean``, ``sum``.
    """
    s = df.groupby("group")[col].agg(["count", "mean", "sum"]).reset_index()
    print(f"\n[{name}] Positive-orthant (unit-L2 then |.|) summary by group:")
    print(s.to_string(index=False))
    return s
