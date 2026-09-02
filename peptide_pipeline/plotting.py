"""ROC-curve plotting for pairwise ensembles."""

from __future__ import annotations

import itertools
import os
import re
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def _slug(s: str) -> str:
    """Sanitize a string into a filesystem-safe slug (letters, digits, ``._-`` only)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def plot_ensemble_rocs_separately(
    models: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    tuned_params: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    out_dir: Optional[str] = None,
    dpi: int = 140,
    close_after_save: bool = True,
) -> None:
    """Plot one ROC curve per pairwise alpha-blend ensemble of the given models.

    Parameters
    ----------
    models:
        Mapping ``model_name -> {"probs_test": array}`` of test-set
        positive-class probabilities for each base model/combo.
    y_true:
        Ground-truth test labels.
    tuned_params:
        Optional mapping ``(name1, name2) -> {"alpha": float, "thr": float}``
        with the tuned blend weight for each pair (order-insensitive: also
        checked as ``(name2, name1)``). Pairs missing from this dict default
        to a neutral 50/50 blend. Pass ``None`` to use 50/50 for all pairs.
    out_dir:
        If given, save each figure as a PNG there (directory created if
        needed) instead of displaying it inline.
    dpi:
        PNG resolution, used only when ``out_dir`` is given.
    close_after_save:
        Close each figure after saving (recommended in notebooks/scripts to
        avoid accumulating open figures).
    """

    def get_params(n1: str, n2: str) -> Tuple[float, float]:
        if tuned_params is None:
            return 0.5, 0.5
        params = tuned_params.get((n1, n2), tuned_params.get((n2, n1), {}))
        return params.get("alpha", 0.5), params.get("thr", 0.5)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    names = list(models.keys())
    for n1, n2 in itertools.combinations(names, 2):
        p1 = models[n1]["probs_test"]
        p2 = models[n2]["probs_test"]
        alpha, _thr = get_params(n1, n2)
        p_ens = alpha * p1 + (1 - alpha) * p2

        fpr, tpr, _ = roc_curve(y_true, p_ens)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, lw=2, linestyle="--", label=f"AUC = {roc_auc:.3f}  |  alpha = {alpha:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC — Ensemble [{n1} + {n2}]")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if out_dir:
            fname = f"ROC_ensemble_{_slug(n1)}__{_slug(n2)}.png"
            path = os.path.join(out_dir, fname)
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {path}")
            if close_after_save:
                plt.close()
        else:
            plt.show()
