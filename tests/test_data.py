import pandas as pd
import pytest

from uplift.data import load_hillstrom, make_uplift_frame, split_data


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_hillstrom()


def test_raw_data_has_expected_shape_and_arms(raw):
    assert len(raw) == 64_000
    assert raw["segment"].nunique() == 3
    assert raw.isna().sum().sum() == 0


def test_treatment_is_any_email_and_features_are_numeric(raw):
    X, y, w = make_uplift_frame(raw, "conversion")
    assert set(w.unique()) == {0, 1}
    assert w.sum() == (raw["segment"] != "No E-Mail").sum()
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    # the outcome must never leak into the features
    assert not {"visit", "conversion", "spend", "segment"} & set(X.columns)


def test_unknown_outcome_raises(raw):
    with pytest.raises(ValueError):
        make_uplift_frame(raw, "revenue")


def test_split_is_disjoint_and_preserves_treatment_share_and_rate(raw):
    X, y, w = make_uplift_frame(raw, "conversion")
    X_tr, X_te, y_tr, y_te, w_tr, w_te = split_data(X, y, w)
    assert len(X_tr) + len(X_te) == len(X)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert w_tr.mean() == pytest.approx(w_te.mean(), abs=0.002)
    assert y_tr.mean() == pytest.approx(y_te.mean(), abs=0.0005)
