import numpy as np

from peptide_pipeline.modeling import fit_lr, mcc_at_threshold, metrics_report, tune_coef_threshold
from peptide_pipeline.preprocessing import build_matrix, variance_prune


def test_variance_prune_drops_constant_columns():
    X_train = np.array([[1.0, 5.0, 0.0], [2.0, 5.0, 0.0], [3.0, 5.0, 0.0]])
    X_test = np.array([[4.0, 5.0, 0.0]])
    names = ["a", "const_b", "const_c"]
    groups = ["g1", "g1", "g1"]

    Xtr_p, Xte_p, names_p, groups_p, mask = variance_prune(X_train, X_test, names, groups)

    assert Xtr_p.shape == (3, 1)
    assert Xte_p.shape == (1, 1)
    assert names_p == ["a"]
    assert groups_p == ["g1"]
    assert mask.tolist() == [True, False, False]


def test_build_matrix_concatenates_blocks():
    Xtr1 = np.ones((4, 2))
    Xte1 = np.ones((2, 2))
    Xtr2 = np.zeros((4, 3))
    Xte2 = np.zeros((2, 3))
    blocks = [
        (Xtr1, Xte1, ["a", "b"], ["g1", "g1"]),
        (Xtr2, Xte2, ["c", "d", "e"], ["g2", "g2", "g2"]),
    ]
    Xtr, Xte, names, groups = build_matrix(blocks)
    assert Xtr.shape == (4, 5)
    assert Xte.shape == (2, 5)
    assert names == ["a", "b", "c", "d", "e"]
    assert groups == ["g1", "g1", "g2", "g2", "g2"]


def _make_toy_classification_data(n=200, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    # Feature 0 is informative, the rest are noise.
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_fit_lr_and_mcc_at_threshold_are_consistent():
    X, y = _make_toy_classification_data()
    model = fit_lr(X, y)
    probs = model.predict_proba(X)[:, 1]
    mcc = mcc_at_threshold(y, probs, thr=0.5)
    # A reasonably separable toy problem should not score near-zero MCC.
    assert mcc > 0.2


def test_tune_coef_threshold_keeps_at_least_one_feature():
    X, y = _make_toy_classification_data(n=300, n_features=20)
    X_tr, X_val = X[:200], X[200:]
    y_tr, y_val = y[:200], y[200:]

    best_thr, keep_mask, best_mcc, base_coef = tune_coef_threshold(
        X_tr, y_tr, X_val, y_val, coef_grid=[0.0, 0.5, 1.0, 5.0, 100.0]
    )
    assert keep_mask.sum() >= 1
    assert base_coef.shape == (20,)


def test_metrics_report_returns_expected_keys():
    X, y = _make_toy_classification_data()
    model = fit_lr(X, y)
    probs = model.predict_proba(X)[:, 1]
    report = metrics_report(y, probs, thr=0.5, verbose=False)
    for key in ("ACC", "AUC", "MCC", "F1", "PPV", "SN", "SP", "tn", "fp", "fn", "tp"):
        assert key in report
