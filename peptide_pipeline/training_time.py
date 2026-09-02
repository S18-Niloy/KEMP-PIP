"""Benchmark ``model.fit()`` wall-clock time across the same combos/models
used in :mod:`peptide_pipeline.ablation`.

Useful for reporting compute-cost trade-offs alongside predictive
performance (e.g. in a paper's methods/discussion section).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
import pandas as pd

from .ablation import ALL_BLOCK_NAMES, concat_blocks, all_nonempty_combos, make_models


def format_time(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS.ss``."""
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def measure_training_time(
    blocks_raw_tr: Dict[str, np.ndarray],
    blocks_scl_tr: Dict[str, np.ndarray],
    blocks_raw_te: Optional[Dict[str, np.ndarray]],
    blocks_scl_te: Optional[Dict[str, np.ndarray]],
    y_train: np.ndarray,
    idx_tr: np.ndarray,
    idx_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    random_state: int = 42,
    block_names: Sequence[str] = ALL_BLOCK_NAMES,
    include_test: bool = True,
) -> pd.DataFrame:
    """Time ``model.fit()`` for every (combo, model) pair, on validation and (optionally) test data.

    Reuses the same feature blocks, combos, and model configurations as
    :func:`peptide_pipeline.ablation.run_block_ablation` so timing numbers
    are directly comparable to that function's performance metrics. Only
    the fit call is timed -- data preparation and prediction are excluded.

    Parameters
    ----------
    blocks_raw_tr, blocks_scl_tr:
        Raw and standardized training feature blocks, as returned by
        :func:`peptide_pipeline.ablation.prepare_ablation_blocks`.
    blocks_raw_te, blocks_scl_te:
        Raw and standardized test feature blocks (only required if
        ``include_test=True`` and ``y_test`` is given).
    y_train:
        Full training labels.
    idx_tr:
        Row indices (into the training blocks) used for the validation-
        split model fits, e.g. from the same
        ``StratifiedShuffleSplit`` used in :func:`run_block_ablation`.
    idx_val:
        Unused for timing (kept for symmetry with the ablation split); pass
        the same value you used to create ``idx_tr`` if you have it.
    y_test:
        Optional test labels; if given (and ``include_test=True``), full-
        training-set fits are also timed.
    random_state:
        Seed passed to :func:`peptide_pipeline.ablation.make_models`.
    block_names:
        Which feature blocks to combine; defaults to all four.
    include_test:
        Whether to additionally time full-training-set fits (mirrors the
        notebook's "TEST" timing section).

    Returns
    -------
    pandas.DataFrame
        One row per (combo, model, set) with columns ``set``, ``combo``,
        ``model``, ``n_feat``, ``training_time_sec``, ``training_time_min``,
        ``training_time_hms``.
    """
    del idx_val  # accepted for API symmetry with the ablation split; not needed for timing

    y_tr = y_train[idx_tr]
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg) / float(max(pos, 1))

    combos = all_nonempty_combos(block_names)
    rows: List[dict] = []

    for combo in combos:
        combo = list(combo)
        combo_name = "+".join(combo)

        Xtr_raw = concat_blocks([blocks_raw_tr[b][idx_tr] for b in combo])
        Xtr_scl = concat_blocks([blocks_scl_tr[b][idx_tr] for b in combo])

        for mdl_name, mdl, kind in make_models(scale_pos_weight, random_state):
            Xtr_use = Xtr_raw if kind == "raw" else Xtr_scl

            start = time.perf_counter()
            mdl.fit(Xtr_use, y_tr)
            elapsed = time.perf_counter() - start

            rows.append({
                "set": "validation",
                "combo": combo_name,
                "model": mdl_name,
                "n_feat": Xtr_use.shape[1],
                "training_time_sec": elapsed,
                "training_time_min": elapsed / 60,
            })

    if include_test and y_test is not None and blocks_raw_te is not None and blocks_scl_te is not None:
        for combo in combos:
            combo = list(combo)
            combo_name = "+".join(combo)

            Xtr_raw_full = concat_blocks([blocks_raw_tr[b] for b in combo])
            Xtr_scl_full = concat_blocks([blocks_scl_tr[b] for b in combo])

            for mdl_name, mdl, kind in make_models(scale_pos_weight, random_state):
                Xtr_use = Xtr_raw_full if kind == "raw" else Xtr_scl_full

                start = time.perf_counter()
                mdl.fit(Xtr_use, y_train)
                elapsed = time.perf_counter() - start

                rows.append({
                    "set": "test",
                    "combo": combo_name,
                    "model": mdl_name,
                    "n_feat": Xtr_use.shape[1],
                    "training_time_sec": elapsed,
                    "training_time_min": elapsed / 60,
                })

    df = pd.DataFrame(rows)
    df["training_time_hms"] = df["training_time_sec"].apply(format_time)
    return df


def summarize_training_time(training_time_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce pivot and summary tables from a :func:`measure_training_time` result.

    Parameters
    ----------
    training_time_df:
        Output of :func:`measure_training_time`.

    Returns
    -------
    (val_pivot, test_pivot, model_summary)
        ``val_pivot`` / ``test_pivot``: combo x model tables of
        ``training_time_sec`` (``test_pivot`` is empty if there is no
        ``"test"`` rows). ``model_summary``: mean/median/min/max/total
        training time per ``(set, model)``.
    """
    val_time_df = training_time_df[training_time_df["set"] == "validation"]
    val_pivot = val_time_df.pivot(index="combo", columns="model", values="training_time_sec").reset_index()

    test_time_df = training_time_df[training_time_df["set"] == "test"]
    test_pivot = (
        test_time_df.pivot(index="combo", columns="model", values="training_time_sec").reset_index()
        if len(test_time_df) > 0
        else pd.DataFrame()
    )

    model_summary = (
        training_time_df.groupby(["set", "model"])["training_time_sec"]
        .agg(Mean_seconds="mean", Median_seconds="median", Min_seconds="min", Max_seconds="max", Total_seconds="sum")
        .reset_index()
    )
    model_summary["Mean_minutes"] = model_summary["Mean_seconds"] / 60
    model_summary["Total_minutes"] = model_summary["Total_seconds"] / 60

    return val_pivot, test_pivot, model_summary
