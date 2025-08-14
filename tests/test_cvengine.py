import numpy as np
import os
import tempfile
import pytest

from src.feateval.data.splits import CVEngine, CVConfig
from sklearn.utils.multiclass import type_of_target


def _toy_classification(n=60, n_features=3, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, n_classes, size=n)
    return X, y


def _toy_regression(n=50, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=n)
    return X, y


def _toy_groups(n=60, n_groups=6, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    rng.shuffle(groups)
    return groups


# ---------- stratified ----------
def test_stratified_basic_shapes_and_class_balance():
    X, y = _toy_classification(n=60, n_classes=3, seed=1)
    cfg = CVConfig(cv_type="stratified", n_splits=5, n_repeats=1, shuffle=True, seed=42)
    cv = CVEngine(cfg)
    splits = cv.split(X, y)


    assert len(splits) == 5
    all_idx = np.concatenate([np.concatenate(s) for s in splits])
    assert set(all_idx) == set(range(len(y)))

    for _, va in splits:
        classes_in_val = set(y[va])
        assert classes_in_val.issubset(set(np.unique(y)))
        assert len(classes_in_val) >= 2  


def test_stratified_repeats_change_splits_when_shuffled():
    X, y = _toy_classification(n=50, n_classes=2, seed=2)
    cfg = CVConfig(cv_type="stratified", n_splits=5, n_repeats=2, shuffle=True, seed=10)
    cv = CVEngine(cfg)
    s1 = cv.split(X, y)

    cfg2 = CVConfig(cv_type="stratified", n_splits=5, n_repeats=2, shuffle=True, seed=11)
    cv2 = CVEngine(cfg2)
    s2 = cv2.split(X, y)

    different = any(
    not (np.array_equal(tr1, tr2) and np.array_equal(va1, va2))
    for (tr1, va1), (tr2, va2) in zip(s1, s2)
    )


    assert different


def test_stratified_no_shuffle_seed_is_ignored_but_repeats_still_return_same():
    X, y = _toy_classification(n=40, n_classes=2, seed=3)
    cfg = CVConfig(cv_type="stratified", n_splits=4, n_repeats=3, shuffle=False, seed=123)
    cv = CVEngine(cfg)
    splits = cv.split(X, y)
    assert len(splits) == cfg.n_splits * cfg.n_repeats
    block1 = splits[:cfg.n_splits]
    block2 = splits[cfg.n_splits:2*cfg.n_splits]

    different = any(
    not (np.array_equal(tr1, tr2) and np.array_equal(va1, va2))
    for (tr1, va1), (tr2, va2) in zip(block1, block2)
    )

    assert not(different)


# ---------- kfold ----------
def test_kfold_basic():
    X, y = _toy_regression(n=30, seed=4)
    cfg = CVConfig(cv_type="kfold", n_splits=3, shuffle=True, seed=7)
    cv = CVEngine(cfg)
    splits = cv.split(X, y)
    assert len(splits) == 3
    # покрытие индексов
    seen = np.concatenate([np.concatenate(s) for s in splits])
    assert set(seen) == set(range(len(y)))


# ---------- group ----------
def test_group_requires_groups_and_respects_group_boundaries():
    X, y = _toy_classification(n=60, n_classes=2, seed=5)
    groups = _toy_groups(n=60, n_groups=6, seed=5)
    cfg = CVConfig(cv_type="group", n_splits=3)

    cv = CVEngine(cfg)
    with pytest.raises(ValueError):
        cv.split(X, y, groups=None)

    splits = cv.split(X, y, groups=groups)
    assert len(splits) == 3
    for tr, va in splits:
        g_tr, g_va = set(groups[tr]), set(groups[va])
        assert g_tr.isdisjoint(g_va)


# ---------- time ----------
def test_time_series_split_monotonic_increasing_val_indices():
    n = 25
    X = np.arange(n).reshape(n, 1)
    y = np.arange(n)
    cfg = CVConfig(cv_type="time", n_splits=5)
    cv = CVEngine(cfg)
    splits = cv.split(X, y)

    last_max = -1
    for _, va in splits:
        assert va.min() > last_max
        last_max = va.max()


# ---------- save/load ----------
def test_save_and_load_roundtrip_returns_same_splits(tmp_path):
    X, y = _toy_classification(n=40, n_classes=2, seed=6)
    cfg = CVConfig(cv_type="stratified", n_splits=4, n_repeats=2, shuffle=True, seed=99)
    cv = CVEngine(cfg)
    s1 = cv.split(X, y)

    p = tmp_path / "splits.pkl"
    cv.save(str(p))
    cv2 = CVEngine.load(str(p))
    s2 = cv2.split(X, y)

    different = any(
    not (np.array_equal(tr1, tr2) and np.array_equal(va1, va2))
    for (tr1, va1), (tr2, va2) in zip(s1, s2)
    )


    assert not(different)


def test_stratified_warns_when_min_class_lt_n_splits():
    y = np.array([0]*19 + [1])   
    X = np.random.randn(20, 2)
    cfg = CVConfig(cv_type="stratified", n_splits=10, shuffle=True, seed=0)
    cv = CVEngine(cfg)

    with pytest.warns(UserWarning):
        splits = cv.split(X, y)
    one_class_folds = 0
    for _, va in splits:
        if len(np.unique(y[va])) == 1:
            one_class_folds += 1
    assert one_class_folds >= 1
