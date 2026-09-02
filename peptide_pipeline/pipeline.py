"""High-level orchestration: turn raw sequences into the four feature blocks.

This is the glue between :mod:`peptide_pipeline.data_loading` and the
per-feature-family extractors in :mod:`peptide_pipeline.features`, matching
step "1-4" of the original notebook (ESM embeddings, k-mer frequencies,
physchem descriptors, modlAMP descriptors) plus class-weight computation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from .features.esm import DEFAULT_ESM_MODEL, esm_feature_names, get_esm_embeddings, prepare_esm
from .features.kmer import DEFAULT_KS, get_kmer_freqs, kmer_feature_names
from .features.modlamp_features import get_modlamp_features, modlamp_feature_names
from .features.physchem import PHYSCHEM_FEATURE_NAMES, get_physchem_core


@dataclass
class FeatureSet:
    """Container for all four extracted feature blocks, train and test.

    Attributes
    ----------
    train, test:
        Dicts mapping block name (``"kmer"``, ``"physchem"``, ``"modlamp"``,
        ``"esm"``) to that block's feature matrix.
    feature_names:
        Dict mapping block name to its list of column names.
    """

    train: Dict[str, np.ndarray] = field(default_factory=dict)
    test: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: Dict[str, List[str]] = field(default_factory=dict)

    def as_block_tuples_train_test(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return ``{block_name: (X_train, X_test)}``, for :mod:`peptide_pipeline.ablation`."""
        return {name: (self.train[name], self.test[name]) for name in self.train}

    def as_block_tuples_with_names(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Return ``{block_name: (X_train, X_test, feature_names)}``, for :mod:`peptide_pipeline.ensemble`."""
        return {
            name: (self.train[name], self.test[name], self.feature_names[name]) for name in self.train
        }


def extract_all_features(
    train_sequences: List[str],
    test_sequences: List[str],
    ks: Tuple[int, ...] = DEFAULT_KS,
    esm_model_name: str = DEFAULT_ESM_MODEL,
    esm_batch_size: int = 16,
    pH: float = 7.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> FeatureSet:
    """Extract k-mer, physchem, modlAMP, and ESM-2 features for train and test sequences.

    Parameters
    ----------
    train_sequences, test_sequences:
        Lists of peptide sequence strings.
    ks:
        k-mer lengths for the k-mer block (default ``(2, 3, 4)``).
    esm_model_name:
        HuggingFace Hub identifier for the ESM-2 checkpoint.
    esm_batch_size:
        Batch size used during ESM-2 embedding extraction.
    pH:
        pH used for the physchem charge features.
    device:
        Torch device for the ESM-2 model. Defaults to CUDA if available.
    verbose:
        Print progress and shape information as each block is extracted.

    Returns
    -------
    FeatureSet
        All four feature blocks for both splits, plus their feature names.
    """
    fs = FeatureSet()

    if verbose:
        print("Loading ESM-2 model & tokenizer...")
    tok, esm_model = prepare_esm(esm_model_name, device=device)
    if verbose:
        print("Extracting ESM-2 embeddings...")
    X_esm_train = get_esm_embeddings(train_sequences, tok, esm_model, batch_size=esm_batch_size)
    X_esm_test = get_esm_embeddings(test_sequences, tok, esm_model, batch_size=esm_batch_size)
    fs.train["esm"], fs.test["esm"] = X_esm_train, X_esm_test
    fs.feature_names["esm"] = esm_feature_names(X_esm_train.shape[1])
    if verbose:
        print("  esm:", X_esm_train.shape, "|", X_esm_test.shape)

    if verbose:
        print("Extracting multi-scale k-mer features...")
    X_kmer_train = get_kmer_freqs(train_sequences, ks=ks)
    X_kmer_test = get_kmer_freqs(test_sequences, ks=ks)
    fs.train["kmer"], fs.test["kmer"] = X_kmer_train, X_kmer_test
    fs.feature_names["kmer"] = kmer_feature_names(ks=ks)
    if verbose:
        print("  kmer:", X_kmer_train.shape, "|", X_kmer_test.shape)

    if verbose:
        print("Extracting physchem (core) features...")
    X_phys_train = get_physchem_core(train_sequences, pH=pH)
    X_phys_test = get_physchem_core(test_sequences, pH=pH)
    fs.train["physchem"], fs.test["physchem"] = X_phys_train, X_phys_test
    fs.feature_names["physchem"] = list(PHYSCHEM_FEATURE_NAMES)
    if verbose:
        print("  physchem:", X_phys_train.shape, "|", X_phys_test.shape)

    if verbose:
        print("Extracting modlAMP features...")
    X_modl_train = get_modlamp_features(train_sequences)
    X_modl_test = get_modlamp_features(test_sequences)
    fs.train["modlamp"], fs.test["modlamp"] = X_modl_train, X_modl_test
    fs.feature_names["modlamp"] = modlamp_feature_names(X_modl_train.shape[1])
    if verbose:
        print("  modlamp:", X_modl_train.shape, "|", X_modl_test.shape)

    return fs


def compute_class_weights(y_train: np.ndarray, verbose: bool = True) -> Dict[int, float]:
    """Compute balanced per-class weights for the training labels.

    Parameters
    ----------
    y_train:
        Training labels (binary, 0/1).
    verbose:
        Print class counts and computed weights.

    Returns
    -------
    dict
        ``{class_label: weight}``, suitable for scikit-learn's
        ``class_weight`` argument.
    """
    if verbose:
        print("Class counts (train):", Counter(y_train))
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, weights)}
    if verbose:
        print("Class weights:", class_weight_dict)
    return class_weight_dict
