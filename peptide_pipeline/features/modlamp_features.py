"""Global peptide descriptors from the ``modlamp`` package.

This is a thin, documented wrapper around
``modlamp.descriptors.GlobalDescriptor.calculate_all``, which computes a
standard set of global sequence descriptors (length, charge, hydrophobicity,
aromaticity, instability index, aliphatic index, Boman index, etc.) commonly
used in antimicrobial-peptide (AMP) prediction literature.

Requires the optional dependency ``modlamp`` (see ``requirements.txt``).
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def get_modlamp_features(sequences: Iterable[str]) -> np.ndarray:
    """Compute modlAMP's full battery of global descriptors.

    Parameters
    ----------
    sequences:
        Iterable of peptide sequence strings.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, n_modlamp_descriptors)``, dtype
        ``float32``. The number and order of descriptor columns is
        determined by ``modlamp.descriptors.GlobalDescriptor`` and is
        stable for a given ``modlamp`` version, but is not otherwise named
        here -- use :func:`modlamp_feature_names` to generate matching
        placeholder names.
    """
    # Imported lazily so the rest of the package can be used without the
    # optional `modlamp` dependency installed.
    from modlamp.descriptors import GlobalDescriptor

    sequences = list(sequences)
    desc = GlobalDescriptor(sequences)
    desc.calculate_all()
    return np.array(desc.descriptor, dtype=np.float32)


def modlamp_feature_names(n_features: int) -> List[str]:
    """Generate placeholder names ``modlamp_0 .. modlamp_{n-1}`` for the descriptor matrix.

    modlAMP does not expose descriptor names directly through
    ``GlobalDescriptor``, so downstream code (interpretability tables,
    feature-group bookkeeping) refers to columns positionally with these
    generic names, matching the original notebook's convention.

    Parameters
    ----------
    n_features:
        Number of columns returned by :func:`get_modlamp_features` (i.e.
        ``X_modlamp.shape[1]``).

    Returns
    -------
    list of str
    """
    return [f"modlamp_{i}" for i in range(n_features)]
