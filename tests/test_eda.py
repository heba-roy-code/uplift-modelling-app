import numpy as np
import pandas as pd
import pytest

from uplift.eda import ate_bootstrap_ci, ate_wald_ci, standardised_mean_differences


@pytest.fixture()
def experiment() -> tuple[pd.Series, pd.Series]:
    """Simulated randomised experiment with a known ATE of +0.05."""
    rng = np.random.default_rng(1)
    n = 20_000
    w = pd.Series(rng.integers(0, 2, n))
    y = pd.Series(rng.binomial(1, 0.10 + 0.05 * w))
    return y, w


def test_wald_ci_contains_true_effect_and_orders_bounds(experiment):
    y, w = experiment
    res = ate_wald_ci(y, w)
    assert res.lower < res.ate < res.upper
    assert res.lower < 0.05 < res.upper
    assert res.n_treated + res.n_control == len(y)


def test_bootstrap_agrees_with_wald_for_large_samples(experiment):
    y, w = experiment
    wald = ate_wald_ci(y, w)
    lo, hi = ate_bootstrap_ci(y, w, n_boot=1_000)
    assert lo == pytest.approx(wald.lower, abs=0.002)
    assert hi == pytest.approx(wald.upper, abs=0.002)


def test_no_effect_gives_interval_covering_zero():
    rng = np.random.default_rng(2)
    w = pd.Series(rng.integers(0, 2, 10_000))
    y = pd.Series(rng.binomial(1, 0.1, 10_000))
    res = ate_wald_ci(y, w)
    assert res.lower < 0 < res.upper


def test_smd_flags_an_unbalanced_feature():
    w = pd.Series([0] * 500 + [1] * 500)
    X = pd.DataFrame(
        {"balanced": np.tile([0, 1], 500), "unbalanced": w * 2.0 + np.tile([0, 1], 500)}
    )
    smd = standardised_mean_differences(X, w)
    assert smd["balanced"] < 0.01
    assert smd["unbalanced"] > 1
