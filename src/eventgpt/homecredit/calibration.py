"""Calibration metrics for a binary outcome c and a stated probability q.

Sources of the less common ones:

    Brier index and adjusted Brier index   bradley-terry-models/scripts/calibration_run.py (also author-voice
                                           voice/metrics.py): 100 (1 - sqrt(Brier)), 50 for always predicting 0.5,
                                           and 100 - 50 sqrt(Brier / (r (1 - r))), 50 for the base-rate forecast r
    ECE and MCE                            woe-paper/catboost-calibration (CalibrationModule.calc_metrics): the
                                           count-weighted and the largest gap between mean q and the rate of c over
                                           equal-width bins
    ICE and the Gini gap                   woe-paper/gini-bayes-paper, "Calibration in cumulative coordinates":
                                           sort worst-first by q, accumulate the outcomes (empirical CAP) and the
                                           claims (model-implied CAP, each q counted as q expected events); ICE is
                                           the area between the two curves, and the model-implied Gini above the
                                           empirical Gini signals overconfidence, below it compression

and the standard ones: calibration-in-the-large (mean q minus the rate of c) with observed over expected, the
calibration slope and intercept of the logistic recalibration logit P(c) = a + b logit q (b < 1 overconfident,
b > 1 compressed), and Spiegelhalter's z, which is approximately standard normal under calibration.
"""

from __future__ import annotations

import math

import numpy as np


def brier(c: np.ndarray, q: np.ndarray) -> float:
    """Mean squared error of q against the 0/1 outcomes."""
    return float(np.mean((q - c) ** 2))


def brier_index(c: np.ndarray, q: np.ndarray) -> float:
    """Brier index in percent: 100 (1 - sqrt(Brier)), 50 for always predicting 0.5."""
    return float(100.0 * (1.0 - math.sqrt(brier(c, q))))


def adjusted_brier_index(c: np.ndarray, q: np.ndarray, base_rate: float | None = None) -> float:
    """Brier index against the base-rate forecast, in percent: 50 for always predicting the base rate."""
    r = float(np.mean(c)) if base_rate is None else base_rate
    return float(100.0 - 50.0 * math.sqrt(brier(c, q) / (r * (1.0 - r))))


def brier_skill(c: np.ndarray, q: np.ndarray, base_rate: float | None = None) -> float:
    """1 - Brier / Brier of the base-rate forecast."""
    r = float(np.mean(c)) if base_rate is None else base_rate
    return float(1.0 - brier(c, q) / (r * (1.0 - r)))


def ece_mce(c: np.ndarray, q: np.ndarray, bins: int = 10) -> tuple[float, float]:
    """Expected (count-weighted) and maximum calibration error over equal-width bins of q."""
    b = np.clip((q * bins).astype(int), 0, bins - 1)
    ece, mce = 0.0, 0.0
    for k in range(bins):
        sel = b == k
        if sel.any():
            gap = abs(float(q[sel].mean()) - float(c[sel].mean()))
            ece += gap * float(sel.mean())
            mce = max(mce, gap)
    return ece, mce


def caps(c: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical and model-implied CAPs, worst-first by q (highest q first), at the population fractions i / n."""
    order = np.argsort(-q, kind="stable")
    cs, qs = c[order].astype(np.float64), q[order].astype(np.float64)
    return np.cumsum(cs) / cs.sum(), np.cumsum(qs) / qs.sum()


def cap_gini(f: np.ndarray, rate: float) -> float:
    """Accuracy ratio of a CAP: area between the curve and the diagonal over that of the perfect model."""
    return float((2.0 * f.mean() - 1.0) / (1.0 - rate))


def ice_and_ginis(c: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """ICE, empirical Gini, model-implied Gini and their gap (implied minus empirical)."""
    f_emp, f_model = caps(c, q)
    g_emp, g_model = cap_gini(f_emp, float(c.mean())), cap_gini(f_model, float(q.mean()))
    return {"ice": float(np.mean(np.abs(f_model - f_emp))), "gini_emp": g_emp, "gini_implied": g_model, "gini_gap": g_model - g_emp}


def calibration_slope(c: np.ndarray, q: np.ndarray, steps: int = 25) -> tuple[float, float]:
    """Intercept a and slope b of logit P(c) = a + b logit q, by Newton-Raphson on the log-likelihood; nan for a constant q."""
    if float(np.std(q)) < 1e-9:
        return float("nan"), float("nan")
    x = np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1 - 1e-6))
    xm = np.stack([np.ones_like(x), x], 1)
    w = np.array([0.0, 1.0])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(xm @ w)))
        g = xm.T @ (c - p)
        h = (xm * (p * (1 - p))[:, None]).T @ xm
        step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def spiegelhalter_z(c: np.ndarray, q: np.ndarray) -> float:
    """Spiegelhalter's z: sum (c - q)(1 - 2q) / sqrt(sum (1 - 2q)^2 q (1 - q))."""
    num = np.sum((c - q) * (1 - 2 * q))
    den = math.sqrt(float(np.sum((1 - 2 * q) ** 2 * q * (1 - q))))
    return float(num / den) if den > 0 else float("nan")


def summary(c: np.ndarray, q: np.ndarray, base_rate: float | None = None) -> dict[str, float]:
    """Every metric above for one set of outcomes and stated probabilities."""
    c, q = np.asarray(c, dtype=np.float64), np.asarray(q, dtype=np.float64)
    ece, mce = ece_mce(c, q)
    a, b = calibration_slope(c, q)
    return {
        "brier": brier(c, q),
        "brier_index": brier_index(c, q),
        "adj_brier_index": adjusted_brier_index(c, q, base_rate),
        "brier_skill": brier_skill(c, q, base_rate),
        "ece": ece,
        "mce": mce,
        "citl": float(q.mean() - c.mean()),
        "o_e": float(c.sum() / q.sum()),
        "cal_intercept": a,
        "cal_slope": b,
        "spiegelhalter_z": spiegelhalter_z(c, q),
        **ice_and_ginis(c, q),
    }
