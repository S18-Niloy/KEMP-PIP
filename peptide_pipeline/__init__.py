"""
peptide_pipeline
=================

A small research pipeline for peptide bioactivity classification that combines
four families of sequence-derived features:

* **k-mer** composition frequencies (multi-scale amino-acid n-grams)
* **physicochemical** descriptors (hydrophobicity, charge, secondary-structure
  propensity, protease-cleavage-site density, ...)
* **modlAMP** global descriptors (via the ``modlamp`` package)
* **ESM-2** protein language model embeddings (via HuggingFace ``transformers``)

and evaluates several downstream classifiers (Logistic Regression, XGBoost,
Random Forest, SVM, MLP) across every combination ("ablation") of the four
feature blocks, plus a coefficient-pruned Logistic Regression ensemble.

This package is a refactor of an exploratory Jupyter notebook into importable,
documented modules so the pipeline can be reproduced, unit-tested, and reused
outside of a single notebook session. See ``examples/run_pipeline.py`` for an
end-to-end, command-line-driven example.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("peptide_pipeline")
except PackageNotFoundError:  # pragma: no cover - package not installed, e.g. running from source
    __version__ = "0.1.0"

__all__ = ["__version__"]
