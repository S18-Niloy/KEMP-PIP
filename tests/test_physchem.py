import numpy as np

from peptide_pipeline.features.physchem import (
    PHYSCHEM_FEATURE_NAMES,
    cleavage_density_chymo,
    cleavage_density_elastase,
    cleavage_density_trypsin,
    get_physchem_core,
    hydrophobic_moment,
    positive_charge_at_pH,
)


def test_output_shape_matches_feature_names():
    seqs = ["ACDEFGHIK", "KRKRKR", ""]
    X = get_physchem_core(seqs)
    assert X.shape == (3, len(PHYSCHEM_FEATURE_NAMES))


def test_length_column_is_correct():
    seqs = ["ACDE", "KRKRKRKR"]
    X = get_physchem_core(seqs)
    length_idx = PHYSCHEM_FEATURE_NAMES.index("length")
    assert list(X[:, length_idx]) == [4.0, 8.0]


def test_empty_sequence_does_not_crash():
    X = get_physchem_core([""])
    assert X.shape == (1, len(PHYSCHEM_FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_highly_basic_sequence_has_high_positive_charge():
    charge_krk = positive_charge_at_pH("KRK", pH=7.0)
    charge_aaa = positive_charge_at_pH("AAA", pH=7.0)
    assert charge_krk > charge_aaa


def test_hydrophobic_moment_nonnegative():
    assert hydrophobic_moment("ACDEFGHIK", np.deg2rad(100.0)) >= 0.0
    assert hydrophobic_moment("", np.deg2rad(100.0)) == 0.0


def test_cleavage_densities_in_valid_range():
    seq = "AKPRLVKAR"
    for fn in (cleavage_density_trypsin, cleavage_density_chymo, cleavage_density_elastase):
        d = fn(seq)
        assert 0.0 <= d <= 1.0


def test_trypsin_respects_proline_exception():
    # K followed by P should NOT count as a cleavage site.
    no_site = cleavage_density_trypsin("AKPA")
    with_site = cleavage_density_trypsin("AKA")
    assert no_site == 0.0
    assert with_site > 0.0
