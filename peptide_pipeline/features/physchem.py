"""Hand-crafted physicochemical descriptors for peptide sequences.

This module computes a small, interpretable set of "core" physicochemical
features per peptide:

* sequence length
* mean Kyte-Doolittle hydrophobicity
* hydrophobic moment (helix and sheet periodicities)
* net positive charge at a given pH (Henderson-Hasselbalch, K/R/H side
  chains + N-terminus)
* positive-charge density (charge / length)
* fraction of helix-favouring and sheet-favouring residues
* predicted proteolytic cleavage-site density for trypsin, chymotrypsin and
  elastase

These are intentionally independent of, and complementary to, the modlAMP
descriptor block in :mod:`peptide_pipeline.features.modlamp_features`.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

#: Kyte-Doolittle hydrophobicity scale.
HYDRO_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}

#: pKa values for basic (positively-charged) side chains.
PKA_BASIC = {"K": 10.5, "R": 12.5, "H": 6.0}
#: pKa of the free N-terminal amine.
PKA_N_TERM = 9.69

#: Residues that favour alpha-helix formation.
HELIX_PREF = set("AEHKLMQR")
#: Residues that favour beta-sheet formation.
SHEET_PREF = set("VIYFWTC")

PHYSCHEM_FEATURE_NAMES: List[str] = [
    "length", "KD_mean", "muH_helix", "muH_sheet",
    "pos_charge", "pos_charge_density",
    "f_helix", "f_sheet",
    "dens_trypsin", "dens_chymo", "dens_elastase",
]


def hydrophobic_moment(seq: str, radians_per_res: float) -> float:
    """Compute the (unnormalized) hydrophobic moment of a sequence.

    The hydrophobic moment measures amphipathicity: how strongly
    hydrophobic residues cluster on one face of an idealized secondary
    structure with a given per-residue turn angle.

    Parameters
    ----------
    seq:
        Peptide sequence.
    radians_per_res:
        Turn angle per residue, in radians (e.g. ``100 deg`` for an
        alpha-helix, ``180 deg`` for a beta-strand).

    Returns
    -------
    float
        The hydrophobic moment, normalized by sequence length. Returns
        ``0.0`` for an empty sequence.
    """
    s = seq.upper()
    if not s:
        return 0.0
    angles = np.arange(len(s)) * radians_per_res
    h = np.array([HYDRO_SCALE.get(a, 0.0) for a in s], dtype=float)
    x = np.sum(h * np.cos(angles))
    y = np.sum(h * np.sin(angles))
    mu = np.sqrt(x * x + y * y) / len(s)
    return float(mu)


def positive_charge_at_pH(seq: str, pH: float = 7.0, include_Nterm: bool = True) -> float:
    """Estimate net positive charge at a given pH via Henderson-Hasselbalch.

    Parameters
    ----------
    seq:
        Peptide sequence.
    pH:
        pH at which to evaluate protonation state. Defaults to physiological
        pH 7.0.
    include_Nterm:
        Whether to include the contribution of the free N-terminal amine
        (only meaningful for a linear, non-blocked peptide).

    Returns
    -------
    float
        Estimated net positive charge (fractional protonation summed over
        K, R, H side chains and, optionally, the N-terminus).
    """
    s = seq.upper()
    chg = 0.0
    for aa, pKa in PKA_BASIC.items():
        n = s.count(aa)
        chg += n * (1.0 / (1.0 + 10.0 ** (pH - pKa)))
    if include_Nterm and s:
        chg += 1.0 / (1.0 + 10.0 ** (pH - PKA_N_TERM))
    return float(chg)


def _cleavage_density(seq: str, cut_after: str) -> float:
    """Shared implementation for the protease cleavage-density functions.

    A cleavage site is counted after any residue in ``cut_after`` that is
    not immediately followed by a proline (standard specificity rule for
    trypsin/chymotrypsin/elastase-like proteases).
    """
    s = seq.upper()
    if not s:
        return 0.0
    L, sites = len(s), 0
    for i in range(L - 1):
        if s[i] in cut_after and s[i + 1] != "P":
            sites += 1
    if s[-1] in cut_after:
        sites += 1
    return sites / L


def cleavage_density_trypsin(seq: str) -> float:
    """Predicted trypsin cleavage-site density (cuts after K/R, not before P)."""
    return _cleavage_density(seq, "KR")


def cleavage_density_chymo(seq: str) -> float:
    """Predicted chymotrypsin cleavage-site density (cuts after F/Y/W/L, not before P)."""
    return _cleavage_density(seq, "FYWL")


def cleavage_density_elastase(seq: str) -> float:
    """Predicted elastase cleavage-site density (cuts after A/V/I/L, not before P)."""
    return _cleavage_density(seq, "AVIL")


def get_physchem_core(sequences: Iterable[str], pH: float = 7.0) -> np.ndarray:
    """Compute the core physicochemical feature matrix for a set of sequences.

    Parameters
    ----------
    sequences:
        Iterable of peptide sequence strings.
    pH:
        pH used for the net-charge calculation. Defaults to 7.0.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, 11)``, dtype ``float32``. Column
        order matches :data:`PHYSCHEM_FEATURE_NAMES`.
    """
    feats = []
    for seq in sequences:
        s = seq.upper()
        L = len(s)
        L_safe = max(L, 1)
        kd_vals = [HYDRO_SCALE.get(a, 0.0) for a in s]
        kd_mean = float(np.mean(kd_vals)) if kd_vals else 0.0
        muH_helix = hydrophobic_moment(s, np.deg2rad(100.0))
        muH_sheet = hydrophobic_moment(s, np.deg2rad(180.0))
        pos_charge = positive_charge_at_pH(s, pH=pH, include_Nterm=True)
        pos_charge_density = pos_charge / L_safe
        f_helix = sum(1 for a in s if a in HELIX_PREF) / L_safe
        f_sheet = sum(1 for a in s if a in SHEET_PREF) / L_safe
        dens_trypsin = cleavage_density_trypsin(s)
        dens_chymo = cleavage_density_chymo(s)
        dens_elastase = cleavage_density_elastase(s)
        feats.append(
            [
                float(L), kd_mean, muH_helix, muH_sheet,
                pos_charge, pos_charge_density,
                f_helix, f_sheet,
                dens_trypsin, dens_chymo, dens_elastase,
            ]
        )
    return np.array(feats, dtype=np.float32)
