"""Feature-extraction sub-package.

Each module implements one independent feature "block":

* :mod:`peptide_pipeline.features.kmer` -- multi-scale k-mer frequencies
* :mod:`peptide_pipeline.features.physchem` -- physicochemical descriptors
* :mod:`peptide_pipeline.features.modlamp_features` -- modlAMP global descriptors
* :mod:`peptide_pipeline.features.esm` -- ESM-2 protein language model embeddings

All extraction functions share the same signature convention: they accept a
list/iterable of peptide sequence strings and return a 2D ``numpy.ndarray``
of shape ``(n_sequences, n_features)``, plus a companion function that
returns the corresponding feature names.
"""
