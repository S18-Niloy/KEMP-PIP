import numpy as np

from peptide_pipeline.features.kmer import get_kmer_freqs, kmer_feature_names


def test_kmer_shape_matches_names():
    seqs = ["ACDE", "KLMNPQ", ""]
    X = get_kmer_freqs(seqs, ks=[2, 3])
    names = kmer_feature_names(ks=[2, 3])
    assert X.shape == (3, len(names))


def test_kmer_row_sums_to_number_of_ks_when_all_valid():
    # A sequence long enough that every k in ks has at least one valid k-mer
    # should have each per-k block sum to 1.0, so the whole row sums to len(ks).
    seqs = ["ACDEFGHIK"]
    ks = [2, 3]
    X = get_kmer_freqs(seqs, ks=ks)
    assert np.isclose(X.sum(), len(ks), atol=1e-5)


def test_empty_sequence_gives_all_zero_row():
    X = get_kmer_freqs([""], ks=[2, 3])
    assert np.allclose(X, 0.0)


def test_unknown_residues_are_skipped_not_errored():
    # 'X' and 'B' are not in the 20-letter alphabet; should not raise,
    # and should simply not be counted as part of any k-mer.
    X = get_kmer_freqs(["ACXDE"], ks=[2])
    assert X.shape[0] == 1
    assert np.isfinite(X).all()
