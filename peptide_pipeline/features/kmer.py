"""Multi-scale amino-acid k-mer frequency features.

For each k in ``ks`` (default ``[2, 3, 4]``), every peptide is represented by
the normalized frequency of each possible k-length amino-acid subsequence
("k-mer"). Frequencies are computed per-sequence (counts divided by the
number of valid k-mers in that sequence), and the blocks for each k are
concatenated together.

Note this uses the standard 20-letter amino-acid alphabet; any k-mer
containing a character outside that alphabet (e.g. ``X`` for an unknown
residue) is skipped when counting, matching the notebook's original
behaviour.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable, List, Sequence

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_KS = (2, 3, 4)


def get_kmer_freqs(sequences: Iterable[str], ks: Sequence[int] = DEFAULT_KS) -> np.ndarray:
    """Compute normalized multi-scale k-mer frequencies for each sequence.

    Parameters
    ----------
    sequences:
        Iterable of peptide sequence strings.
    ks:
        k-mer lengths to compute, e.g. ``[2, 3, 4]`` for di-, tri- and
        tetra-peptide frequencies. The resulting feature blocks are
        concatenated in the order given.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, sum(20**k for k in ks))``, dtype
        ``float32``. Row i sums to ``len(ks)`` (each k-block sums to 1 per
        sequence, unless a sequence has zero valid k-mers for that k, in
        which case that block is all zeros).
    """
    sequences = list(sequences)
    all_features = []
    for k in ks:
        vocab = ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]
        vocab_idx = {kmer: idx for idx, kmer in enumerate(vocab)}
        feat = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
        for i, seq in enumerate(sequences):
            seq = seq.upper()
            kmers = [
                seq[j : j + k]
                for j in range(len(seq) - k + 1)
                if all(ch in AMINO_ACIDS for ch in seq[j : j + k])
            ]
            counts = Counter(kmers)
            total = float(sum(counts.values()))
            if total > 0:
                for kmer, c in counts.items():
                    feat[i, vocab_idx[kmer]] = c / total
        all_features.append(feat)
    return (
        np.concatenate(all_features, axis=1)
        if all_features
        else np.zeros((len(sequences), 0), dtype=np.float32)
    )


def kmer_feature_names(ks: Sequence[int] = DEFAULT_KS) -> List[str]:
    """Return the feature names matching the columns of :func:`get_kmer_freqs`.

    Parameters
    ----------
    ks:
        Same k-mer lengths passed to :func:`get_kmer_freqs`.

    Returns
    -------
    list of str
        Names of the form ``kmer_{k}_{index}``, where ``index`` is the
        position of that k-mer in the lexicographic vocabulary for that k.
    """
    names = []
    for k in ks:
        kmers = ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]
        names += [f"kmer_{k}_{i}" for i in range(len(kmers))]
    return names
