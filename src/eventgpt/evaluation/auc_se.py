"""Standard errors of ROC-AUC, of differences between models and of the official-target mean, clustered by client.

DeLong's method (DeLong, DeLong and Clarke-Pearson 1988) writes the AUC as a mean of placement values: the
share of negatives a positive outranks, and the share of positives that outrank a negative. Each row's
deviation of its placement from the AUC, divided by its class count, is its influence on the AUC, and the
variance is the sum of squared influences. The test fold has twelve monthly rows per client, which are not
independent, so the influences are summed within each client first and the variance is taken over clients
(Obuchowski 1997). Without clustering this is DeLong's variance up to the n/(n-1) factors.

Several AUCs on the same rows (two models, or four targets) get a covariance matrix from the same client
sums, which gives the SE of a difference between models and of a mean over targets. A paired difference has
a much smaller SE than the two separate SEs suggest, because both models rank the same clients.

Predictions are read from csv or parquet files with client_id, mon and one column per label, the preds.parquet
that eventgpt.baselines.catboost and eventgpt.evaluation.profile_readout write, and joined on (client_id, mon).

Example: `uv run python -m eventgpt.evaluation.auc_se data/baseline/preds.parquet data/runs/hier-alq-s42/probe-now-2k/preds_profile.parquet`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from eventgpt.evaluation.probe import OFFICIAL, UNDEFINED, labels, mark_undefined


def influence(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """AUC and each row's influence on it, ties counted as half; rows with an UNDEFINED label get 0."""
    ok = y != UNDEFINED
    yk, pk = y[ok], p[ok]
    pos = yk == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    r_all = rankdata(pk)
    # placement of a positive: negatives below it; of a negative: positives above it
    v10 = (r_all[pos] - rankdata(pk[pos])) / n0
    v01 = 1.0 - (r_all[~pos] - rankdata(pk[~pos])) / n1
    auc = float(v10.mean())
    inf = np.empty(len(yk))
    inf[pos] = (v10 - auc) / n1
    inf[~pos] = (v01 - auc) / n0
    out = np.zeros(len(y))
    out[ok] = inf
    return auc, out


def covariance(infl: np.ndarray, cluster: np.ndarray | None) -> np.ndarray:
    """Covariance of the AUCs whose influences are the columns of infl (rows, k), clustered when cluster is given."""
    if cluster is not None:
        infl = pd.DataFrame(infl).groupby(cluster).sum().to_numpy()
    g = len(infl)
    return infl.T @ infl * g / (g - 1)


def compare(test: pd.DataFrame, preds: dict[str, pd.DataFrame], labs: list[str], cluster: bool = True) -> pd.DataFrame:
    """AUC and SE per model and label, the official mean, and each model's paired difference from the first.

    Args:
        test: test rows with client_id, mon and the labels
        preds: model name -> predictions with client_id, mon and a column per label
        labs: labels to score
        cluster: cluster the variance by client
    """
    names = list(preds)
    keys = test[["client_id", "mon"]]
    aucs, cols, index = [], [], []
    for name in names:
        p = keys.merge(preds[name], on=["client_id", "mon"], how="left", validate="one_to_one")
        for lab in labs:
            a, f = influence(test[lab].to_numpy(), p[lab].to_numpy())
            aucs.append(a)
            cols.append(f)
            index.append((name, lab))
    auc, infl = np.array(aucs), np.stack(cols, axis=1)
    cov = covariance(infl, test["client_id"].to_numpy() if cluster else None)

    # linear combinations of the AUCs: each one alone, the official mean, differences from the first model
    rows, weights = [], []
    k = len(labs)
    official = [j for j, lab in enumerate(labs) if lab in OFFICIAL]
    for m, name in enumerate(names):
        for j, lab in enumerate(labs):
            w = np.zeros(len(auc))
            w[m * k + j] = 1.0
            rows.append((name, lab))
            weights.append(w)
        w = np.zeros(len(auc))
        w[[m * k + j for j in official]] = 1.0 / len(official)
        rows.append((name, "official_mean"))
        weights.append(w)
    n_each = k + 1
    for m, name in enumerate(names[1:], start=1):
        for j in range(n_each):
            rows.append((f"{name} - {names[0]}", rows[j][1]))
            weights.append(weights[m * n_each + j] - weights[j])
    w = np.stack(weights)
    est = w @ auc
    se = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    out = pd.DataFrame(rows, columns=["model", "label"])
    out["auc"], out["se"] = est, se
    out["lo"], out["hi"] = est - 1.96 * se, est + 1.96 * se
    out["p"] = np.where(out["model"].str.contains(" - "), 2 * norm.sf(np.abs(est / se)), np.nan)
    return out


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    """AUC with clustered DeLong SEs for each prediction file, and paired differences from the first."""
    p = argparse.ArgumentParser()
    p.add_argument("preds", type=Path, nargs="+", help="prediction files; the first is the reference for differences")
    p.add_argument("--names", default="", help="comma-separated model names; default the file paths")
    p.add_argument("--test-rows", type=Path, default=None, help="test rows with labels; default the test_rows next to the second file")
    p.add_argument("--no-cluster", action="store_true", help="treat rows as independent (plain DeLong)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    names = a.names.split(",") if a.names else [str(f) for f in a.preds]
    test_path = a.test_rows or a.preds[-1].parent / "test_rows.parquet"
    test = mark_undefined(pd.read_parquet(test_path)).reset_index(drop=True)
    preds = {n: pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f) for n, f in zip(names, a.preds, strict=True)}
    labs = [lab for lab in labels() if (test[lab] == 1).any() and (test[lab] == 0).any()]
    r = compare(test, preds, labs, cluster=not a.no_cluster)
    print(r.round(4).to_string(index=False))
    if a.out:
        r.to_csv(a.out, index=False)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
