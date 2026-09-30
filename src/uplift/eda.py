"""Exploratory statistics: balance checks and the average treatment effect (ATE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ATEResult:
    """Average treatment effect with a confidence interval."""

    ate: float
    lower: float
    upper: float
    mean_treated: float
    mean_control: float
    n_treated: int
    n_control: int

    @property
    def relative_lift(self) -> float:
        """ATE as a fraction of the control mean."""
        return self.ate / self.mean_control


def ate_wald_ci(y: pd.Series, w: pd.Series, confidence: float = 0.95) -> ATEResult:
    """Difference in means with a normal-approximation (Wald) confidence interval.

    Valid for a randomised experiment. The standard error is
    ``sqrt(var_1 / n_1 + var_0 / n_0)`` (unpooled, so it also works for non-binary outcomes).
    """
    y1, y0 = y[w == 1].to_numpy(float), y[w == 0].to_numpy(float)
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(0.5 + confidence / 2)
    return ATEResult(ate, ate - z * se, ate + z * se, y1.mean(), y0.mean(), len(y1), len(y0))


def ate_bootstrap_ci(
    y: pd.Series,
    w: pd.Series,
    confidence: float = 0.95,
    n_boot: int = 5_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the ATE (each arm resampled separately)."""
    rng = np.random.default_rng(seed)
    y1, y0 = y[w == 1].to_numpy(float), y[w == 0].to_numpy(float)
    diffs = np.array(
        [rng.choice(y1, len(y1)).mean() - rng.choice(y0, len(y0)).mean() for _ in range(n_boot)]
    )
    tail = (1 - confidence) / 2 * 100
    lower, upper = np.percentile(diffs, [tail, 100 - tail])
    return float(lower), float(upper)


def standardised_mean_differences(X: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Absolute standardised mean difference of each feature between treated and control.

    Values below about 0.1 are conventionally read as "balanced".
    """
    treated, control = X[w == 1], X[w == 0]
    pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
    return ((treated.mean() - control.mean()) / pooled_sd).abs().fillna(0.0)
