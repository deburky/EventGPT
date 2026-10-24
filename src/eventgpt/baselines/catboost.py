"""Aggregation baseline: hand-built features of the history before each cut, CatBoost, ROC-AUC and PR-AUC on the test fold.

The MBD paper's "Aggregation" row, rebuilt on MBD-mini, which has no published numbers of its own. Rows,
labels, the training sample and the test fold are the probe's own (eventgpt.evaluation.probe.load_rows and
sample_train with the same seed), so the two AUC tables compare row for row.

Features per (client, month), all from events strictly before the cut, over three windows (all history,
the last month, the last three months):

    transactions   count, amount sum, mean and max; count and amount sum per event type for the TOP_TYPES
                   most frequent types
    geo            pings and active days
    timing         days since the last and since the first transaction
    profile        src_type21, src_type22, src_type31, src_type32 as categorical features, which CatBoost
                   encodes with ordered target statistics, so the 2,785-value src_type21 needs no bucketing

Months are calendar months in Moscow time, which is what makes the windows exact: the cut is the first of a
month, so "before the cut" is a whole number of months.

Example: `uv run python -m eventgpt.baselines.catboost --max-clients 2000`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eventgpt.data.mbd import N_FOLDS, OUT, RAW
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import UNDEFINED, Fold, labels, load_rows, report, sample_train, score

ROOT = Path(__file__).resolve().parents[3]
TOP_TYPES = 15
WINDOWS = {"all": 10_000, "m1": 1, "m3": 3}  # months before the cut
PROFILE = ["src_type21", "src_type22", "src_type31", "src_type32"]


def features(rows: pd.DataFrame, amount_bins: int = 0) -> pd.DataFrame:
    """Aggregate features for each (client_id, mon) in rows, computed in DuckDB from the raw tables.

    With amount_bins > 0 every amount is replaced by the mean amount of its quantile bin, which is as much of
    the amount as a model reading bin tokens can recover.
    """
    con = duckdb.connect()
    con.register("rows_df", rows[["client_id", "mon"]].drop_duplicates())
    con.execute(f"create view trx as select * from read_parquet('{RAW}/detail/trx/*/*.parquet')")
    if amount_bins:
        qs = [float(q) for q in np.linspace(0, 1, amount_bins + 1)[1:-1]]
        edges = con.execute("select quantile_cont(amount, ?::double[]) from trx", [qs]).fetchone()
        assert edges is not None
        lo = np.r_[-np.inf, edges[0]]
        con.register("bins", pd.DataFrame({"lo": lo, "hi": np.r_[lo[1:], np.inf]}))
        con.execute(f"""create temp table trx_binned as
            with b as (select t.*, bins.lo from read_parquet('{RAW}/detail/trx/*/*.parquet') t
                       join bins on t.amount >= bins.lo and t.amount < bins.hi)
            select * exclude (lo) replace (avg(amount) over (partition by lo) as amount) from b""")
        con.execute("create or replace view trx as select * from trx_binned")
    con.execute(f"create view geo as select * from read_parquet('{RAW}/detail/geo/*/*.parquet')")
    con.execute("""create temp table r as
        select client_id, mon, date_trunc('month', mon::date)::date m from rows_df""")
    types = [t for (t,) in con.execute(f"select event_type from trx group by 1 order by count(*) desc limit {TOP_TYPES}").fetchall()]
    per_type = ", ".join(
        f"sum((event_type = {t})::int) n_t{t}, sum(case when event_type = {t} then amount else 0 end) a_t{t}" for t in types
    )
    con.execute(f"""create temp table cm as
        select client_id, date_trunc('month', event_time + interval 3 hour)::date m, count(*) n, sum(amount) a_sum,
               max(amount) a_max, min(event_time) t_first, max(event_time) t_last, {per_type}
        from trx where client_id in (select client_id from r) group by all""")
    con.execute("""create temp table gm as
        select client_id, date_trunc('month', event_time + interval 3 hour)::date m, count(*) pings,
               count(distinct date_trunc('day', event_time + interval 3 hour)) active_days
        from geo where client_id in (select client_id from r) group by all""")
    sums = ["n", "a_sum"] + [f"n_t{t}" for t in types] + [f"a_t{t}" for t in types]
    cols = []
    for w, months in WINDOWS.items():
        inside = f"cm.m >= r.m - interval {months} month"
        cols += [f"sum(case when {inside} then cm.{c} else 0 end) {c}_{w}" for c in sums]
        cols.append(f"max(case when {inside} then cm.a_max end) a_max_{w}")
    trx_feats = con.execute(f"""
        select r.client_id, r.mon, {", ".join(cols)},
               date_diff('day', max(cm.t_last), r.m - interval 3 hour) days_since_last,
               date_diff('day', min(cm.t_first), r.m - interval 3 hour) days_since_first
        from r left join cm on cm.client_id = r.client_id and cm.m < r.m
        group by r.client_id, r.mon, r.m""").df()
    geo_cols = [
        f"sum(case when gm.m >= r.m - interval {months} month then gm.{c} else 0 end) geo_{c}_{w}"
        for w, months in WINDOWS.items()
        for c in ("pings", "active_days")
    ]
    geo_feats = con.execute(f"""
        select r.client_id, r.mon, {", ".join(geo_cols)}
        from r left join gm on gm.client_id = r.client_id and gm.m < r.m
        group by r.client_id, r.mon""").df()
    prof = con.execute(f"""
        select client_id, {", ".join(f"max({c}) {c}" for c in PROFILE)}
        from trx where client_id in (select client_id from r) group by 1""").df()
    out = trx_feats.merge(geo_feats, on=["client_id", "mon"]).merge(prof, on="client_id", how="left")
    for w in WINDOWS:
        out[f"a_mean_{w}"] = out[f"a_sum_{w}"] / out[f"n_{w}"].where(out[f"n_{w}"] > 0)
    return out


def main() -> None:
    """Same rows as the probe, aggregate features, one boosted model per label, ROC-AUC on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("--n-neg", type=int, default=60_000, help="negative training rows sampled from the pretraining folds")
    p.add_argument("--max-clients", type=int, default=0, help="clients per fold, for a quick run; 0 is every client")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amount-bins", type=int, default=0, help="replace amounts by their quantile-bin mean; 0 keeps them exact")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "baseline")
    a = p.parse_args()

    t0 = time.time()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    for k in range(N_FOLDS):
        rows = load_rows(Fold(k, vocab), k, a.max_clients or None, a.seed)
        if k == test_fold:
            rows["weight"] = 1.0
            frames["test"].append(rows)
        else:
            frames["train"].append(sample_train(rows, a.n_neg // (N_FOLDS - 1), a.seed + k))
        print(f"fold {k}: {len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)
    train, test = pd.concat(frames["train"], ignore_index=True), pd.concat(frames["test"], ignore_index=True)
    feats = features(pd.concat([train, test]), a.amount_bins)
    print(f"{feats.shape[1] - 2} features ({time.time() - t0:.0f}s)", flush=True)
    xtr = train[["client_id", "mon"]].merge(feats, on=["client_id", "mon"], how="left").drop(columns=["client_id", "mon"])
    xte = test[["client_id", "mon"]].merge(feats, on=["client_id", "mon"], how="left").drop(columns=["client_id", "mon"])
    for c in PROFILE:
        xtr[c] = xtr[c].fillna(-1).astype("int64").astype(str)
        xte[c] = xte[c].fillna(-1).astype("int64").astype(str)
    # a tenth of the training clients is held out for early stopping, by client so no client sits on both sides
    rng = np.random.default_rng(a.seed)
    clients = train["client_id"].unique()
    stop_clients = set(rng.choice(clients, size=len(clients) // 10, replace=False))
    is_stop = train["client_id"].isin(stop_clients).to_numpy()
    w = train["weight"].to_numpy()

    res: dict[str, dict[str, float]] = {}
    preds = test[["client_id", "mon"]].copy()
    for lab in labels():
        ytr, yte = train[lab].to_numpy(), test[lab].to_numpy()
        fit_rows, stop_rows = ~is_stop & (ytr != UNDEFINED), is_stop & (ytr != UNDEFINED)
        defined = yte[yte != UNDEFINED]
        if ytr[fit_rows].min() == ytr[fit_rows].max() or ytr[stop_rows].min() == ytr[stop_rows].max() or defined.min() == defined.max():
            res[lab] = score(yte, None)
            continue
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            eval_metric="AUC",
            early_stopping_rounds=100,
            random_seed=a.seed,
            verbose=False,
            allow_writing_files=False,
        )
        fit = Pool(xtr[fit_rows], ytr[fit_rows], cat_features=PROFILE, weight=w[fit_rows])
        stop = Pool(xtr[stop_rows], ytr[stop_rows], cat_features=PROFILE, weight=w[stop_rows])
        model.fit(fit, eval_set=stop, use_best_model=True)
        preds[lab] = model.predict_proba(Pool(xte, cat_features=PROFILE))[:, 1]
        res[lab] = score(yte, preds[lab].to_numpy())
    a.out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(a.out / "preds.parquet")  # test predictions, for eventgpt.evaluation.auc_se
    report({"catboost": res}, test, a.out, {"n_features": int(xtr.shape[1]), "elapsed_s": round(time.time() - t0)})


if __name__ == "__main__":
    main()
