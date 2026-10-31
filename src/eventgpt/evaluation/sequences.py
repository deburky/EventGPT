"""Sequence metrics: compare a sampled month of transactions with the month that followed, and two baselines.

A month is a list of (time, kind) pairs, time in days from the as-of date and kind = etype * 1000 + subtype. For
each row and each sampled month we record:
- n_pred, n_real: the number of transactions.
- overlap: matched by kind over the larger of the two counts, sum_k min(p_k, r_k) / sum_k max(p_k, r_k), where
  p_k and r_k count kind k; 1 when both months are empty. Timing is ignored, and extra events are penalized.
- type_recall, type_precision: the share of the real month's kinds the sampled month contains, and the reverse.
- otd: the optimal transport distance of Mei et al. (2019), per kind: match events of one kind across the two
  months at cost |t - s| days, or delete or insert an event at cost C days, then sum over kinds. Divided by
  C (n_pred + n_real), so 0 is a perfect month and 1 a month with no event matched.
The baselines write the month without the model:
- repeat: the client's transactions of the month before the as-of date, shifted forward by the month's length.
- poisson: a homogeneous Poisson process at the client's transaction rate of the last 90 days, kinds drawn from
  those 90 days' transactions and times uniform in the month.
"""

from __future__ import annotations

import numpy as np

DELETE_COST = 3.0  # days; two events of one kind are matched when they are less than 2C apart
POISSON_DAYS = 90.0
METRICS = ("n_pred", "n_real", "overlap", "type_recall", "type_precision", "otd")


# -------------------------------------------------------------------------------
# Distances between two months
# -------------------------------------------------------------------------------
def transport(a: np.ndarray, b: np.ndarray, c: float) -> float:
    """Optimal transport between two sorted 1-D time lists with deletion and insertion cost c."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return c * (n + m)
    d = np.arange(m + 1, dtype=np.float64) * c
    for i in range(1, n + 1):
        prev, d[0] = d.copy(), i * c
        for j in range(1, m + 1):
            d[j] = min(prev[j] + c, d[j - 1] + c, prev[j - 1] + abs(a[i - 1] - b[j - 1]))
    return float(d[m])


def compare(pred: tuple[np.ndarray, np.ndarray], real: tuple[np.ndarray, np.ndarray], c: float = DELETE_COST) -> dict[str, float]:
    """The metrics of one sampled month (times, kinds) against the real month."""
    (tp, kp), (tr, kr) = pred, real
    kinds_p, cnt_p = np.unique(kp, return_counts=True)
    kinds_r, cnt_r = np.unique(kr, return_counts=True)
    p = dict(zip(kinds_p.tolist(), cnt_p.tolist(), strict=True))
    r = dict(zip(kinds_r.tolist(), cnt_r.tolist(), strict=True))
    both = set(p) | set(r)
    hi = sum(max(p.get(k, 0), r.get(k, 0)) for k in both)
    shared = set(p) & set(r)
    otd = sum(transport(np.sort(tp[kp == k]), np.sort(tr[kr == k]), c) for k in both)
    n = len(kp) + len(kr)
    return {
        "n_pred": float(len(kp)),
        "n_real": float(len(kr)),
        "overlap": sum(min(p.get(k, 0), r.get(k, 0)) for k in both) / hi if hi else 1.0,
        "type_recall": len(shared) / len(r) if r else float("nan"),
        "type_precision": len(shared) / len(p) if p else float("nan"),
        "otd": otd / (c * n) if n else 0.0,
    }


def mean_compare(paths: list[tuple[np.ndarray, np.ndarray]], real: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """The metrics averaged over sampled months, NaN entries left out."""
    rows = [compare(p, real) for p in paths]
    return {m: float(np.nanmean([r[m] for r in rows])) if any(not np.isnan(r[m]) for r in rows) else float("nan") for m in METRICS}


# -------------------------------------------------------------------------------
# Baselines
# -------------------------------------------------------------------------------
def repeat_month(times: np.ndarray, kinds: np.ndarray, month: float) -> tuple[np.ndarray, np.ndarray]:
    """The month before the as-of date, shifted forward by the month's length; times in days before the as-of date are negative."""
    last = times >= -month
    return times[last] + month, kinds[last]


def poisson_months(
    times: np.ndarray, kinds: np.ndarray, month: float, k: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Months from a Poisson process (k of them) at the client's rate of the last POISSON_DAYS, kinds resampled from that period."""
    recent = kinds[times >= -POISSON_DAYS]
    out = []
    for _ in range(k):
        n = int(rng.poisson(len(recent) / POISSON_DAYS * month)) if len(recent) else 0
        out.append((rng.uniform(0, month, n), rng.choice(recent, n) if n else np.zeros(0, dtype=np.int64)))
    return out


# -------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------
def clustered_mean(x: np.ndarray, clients: np.ndarray) -> tuple[float, float]:
    """Mean of x and its standard error with rows clustered by client; NaN rows dropped."""
    ok = ~np.isnan(x)
    x, clients = x[ok], clients[ok]
    mu = float(x.mean())
    _, inv = np.unique(clients, return_inverse=True)
    sums = np.bincount(inv, weights=x - mu)
    g = len(sums)
    return mu, float(np.sqrt(g / max(g - 1, 1) * (sums**2).sum()) / len(x))
