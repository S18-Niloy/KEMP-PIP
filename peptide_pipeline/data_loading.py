"""
Dataset loading utilities.

The original notebook hard-coded Kaggle input paths
(``/kaggle/input/datasets/...``). This module replaces that with an explicit,
reusable function so the pipeline can be pointed at any local CSV files.

Expected input format
----------------------
Each of the four CSVs is a single-column, header-less file:

* ``X_train.csv`` / ``X_test.csv``       -> one peptide sequence per row
* ``label_train.csv`` / ``label_test.csv`` -> one binary label (0/1) per row,
  in the same row order as the corresponding sequence file.

Test labels are optional: if ``label_test`` is not provided (or the file is
missing), the pipeline falls back to an inference-only mode and writes
prediction CSVs instead of computing test-set metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _read_single_column_csv(path: str | Path) -> pd.Series:
    """Read a header-less, single-column CSV file into a pandas Series.

    Parameters
    ----------
    path:
        Path to a CSV file with no header row and exactly one column.

    Returns
    -------
    pandas.Series
        The column contents, as strings (for sequences) or whatever dtype
        pandas infers (for labels, later coerced to int by the caller).
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 1:
        raise ValueError(
            f"Expected a single-column CSV at {path}, found {df.shape[1]} columns."
        )
    return df.iloc[:, 0]


def load_dataset(
    sequences_path: str | Path,
    labels_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load one split (train or test) into a tidy DataFrame.

    Parameters
    ----------
    sequences_path:
        Path to a header-less CSV of peptide sequences.
    labels_path:
        Optional path to a header-less CSV of 0/1 labels, aligned row-for-row
        with ``sequences_path``. Pass ``None`` for an unlabeled test set.

    Returns
    -------
    pandas.DataFrame
        Columns: ``peptide_sequence`` (str) and, if ``labels_path`` was
        given, ``label`` (int).
    """
    sequences = _read_single_column_csv(sequences_path)
    data = pd.DataFrame({"peptide_sequence": sequences.astype(str)})

    if labels_path is not None:
        labels = _read_single_column_csv(labels_path)
        data["label"] = labels.astype(int)

    return data


def load_train_test(
    train_sequences_path: str | Path,
    train_labels_path: str | Path,
    test_sequences_path: str | Path,
    test_labels_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the standard train/test split used throughout the pipeline.

    This mirrors the first cell of the original notebook, but takes explicit
    paths instead of a hard-coded Kaggle dataset location.

    Parameters
    ----------
    train_sequences_path, train_labels_path:
        Paths to the training sequences and labels (both required).
    test_sequences_path:
        Path to the test sequences (required).
    test_labels_path:
        Path to the test labels. Optional -- omit this for a held-out test
        set you don't have ground truth for; the downstream pipeline will
        then run in "predict only" mode.

    Returns
    -------
    (train_data, test_data):
        Two DataFrames with columns ``peptide_sequence`` and ``label``
        (``test_data`` only has ``label`` if ``test_labels_path`` was given).
    """
    train_data = load_dataset(train_sequences_path, train_labels_path)
    test_data = load_dataset(test_sequences_path, test_labels_path)
    return train_data, test_data
