"""Token-count baseline: counts of the most frequent tokens over time windows, no knowledge of the schema, then CatBoost.

The only inputs are the token stream and event timestamps, which any tokenized event history has, so this is the
table an institution can build when no one has written features for its events. For each (client, cut):

    counts     of the TOP most frequent non-special tokens (any key=value, amount bin or time bucket) over all
               history, the last 91 days and the last 30 days
    events     the number of events in each window
    timing     days since the client's first and last event

The profile event (the client's first) stays in every window, as it does in EventGPT's input. With --window the
history is cut to the most recent tokens exactly as the probe cuts it, so the table sees what EventGPT sees. Rows,
labels, weights and the train/test split are a probe run's own, read from its output directory.

Example: `uv run python -m eventgpt.baselines.token_counts --probe-dir data/runs/hier-rope-s42/probe-now-2k --window 2046`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import SPECIALS, Vocab
from eventgpt.evaluation.probe import UNDEFINED, Fold, labels, mark_undefined, score

WINDOWS = {"all": np.inf, "d91": 91 * 86400, "d30": 30 * 86400}


def fold_features(fold: Fold, rows: pd.DataFrame, col: np.ndarray, top: int, window: int) -> np.ndarray:
    """Feature rows for the (client, cut) rows of one fold; col maps a token id to its count column or -1."""
    tok = np.asarray(fold.tokens)
    es = fold.event_start
    out = np.full((len(rows), len(WINDOWS) * (top + 1) + 2), np.nan, dtype=np.float32)
    ci = rows["client_id"].map(fold.index).to_numpy()
    for r, (c, n, cut) in enumerate(zip(ci, rows["n_hist"].to_numpy(), rows["cut"].to_numpy(), strict=True)):
        if n == 0:
            continue
        a = fold.client_events[c]
        end_ev = a + n
        end_tok = es[end_ev] if end_ev < fold.client_events[c + 1] else fold.end_of_client(c)
        k0 = a + 1
        # the probe's cut: [USR] and the profile event, then the most recent events that fit
        if window and end_tok - (es[a] - 1) > window:
            budget = window - (es[a + 1] - (es[a] - 1))
            k0 = a + 1 + int(np.searchsorted(es[a + 1 : end_ev], end_tok - budget, side="left"))
        evs = np.r_[a, np.arange(k0, end_ev)]
        starts = es[evs]
        lens = np.r_[es[evs[:-1] + 1], end_tok] - starts
        idx = np.arange(lens.sum()) + np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens)
        age_ev = (cut - fold.event_time[evs]).astype(np.float64)
        age_tok = np.repeat(age_ev, lens)
        cols = col[tok[idx]]
        for j, span in enumerate(WINDOWS.values()):
            m = (age_tok < span) & (cols >= 0)
            out[r, j * (top + 1) : j * (top + 1) + top] = np.bincount(cols[m], minlength=top)
            out[r, j * (top + 1) + top] = (age_ev < span).sum()
        out[r, -2] = age_ev[1:].min() / 86400 if len(evs) > 1 else np.nan
        out[r, -1] = age_ev.max() / 86400
    return out


def build(rows: pd.DataFrame, vocab: Vocab, col: np.ndarray, top: int, window: int, t0: float) -> pd.DataFrame:
    """Features for rows from any folds, in the order of rows."""
    parts = []
    for k in np.unique(rows["fold"].to_numpy()).tolist():
        g = rows[rows["fold"] == k]
        parts.append(pd.DataFrame(fold_features(Fold(int(k), vocab), g, col, top, window), index=g.index))
        print(f"fold {k}: {len(g):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts).loc[rows.index]


def main() -> None:
    """Token-count table on the probe's rows, one CatBoost model per label, ROC-AUC and PR-AUC on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows and test_rows")
    p.add_argument("--window", type=int, default=0, help="history window in tokens as the probe cuts it; 0 is full history")
    p.add_argument("--top", type=int, default=200, help="most frequent tokens to count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="csv path; default <probe-dir>/token_counts_<window>.csv")
    a = p.parse_args()

    t0 = time.time()
    vocab = Vocab.load(OUT / "vocab")
    tr = mark_undefined(pd.read_parquet(a.probe_dir / "train_rows.parquet"))
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet"))
    # count columns: the most frequent tokens of one training fold
    freq = np.bincount(np.asarray(Fold(int(tr["fold"].iloc[0]), vocab).tokens), minlength=vocab.size)
    freq[: len(SPECIALS)] = 0
    kept = np.argsort(freq)[::-1][: a.top]
    col = np.full(vocab.size, -1)
    col[kept] = np.arange(a.top)
    print(f"{a.top} tokens cover {freq[kept].sum() / freq.sum():.1%} of non-special tokens", flush=True)
    inv = {i: t for t, i in vocab.structured.items()}
    names = [inv.get(int(i), str(i)) for i in kept]
    xtr, xte = build(tr, vocab, col, a.top, a.window, t0), build(te, vocab, col, a.top, a.window, t0)
    xtr.columns = xte.columns = [f"{w}:{t}" for w in WINDOWS for t in [*names, "n_events"]] + ["days_since_last", "days_since_first"]

    # a tenth of the training clients is held out for early stopping, by client, as in eventgpt.baselines.catboost
    rng = np.random.default_rng(a.seed)
    clients = tr["client_id"].unique()
    stop = tr["client_id"].isin(set(rng.choice(clients, size=len(clients) // 10, replace=False))).to_numpy()
    w = tr["weight"].to_numpy()
    res = []
    for lab in labels():
        ytr, yte = tr[lab].to_numpy(), te[lab].to_numpy()
        fit_rows, stop_rows = ~stop & (ytr != UNDEFINED), stop & (ytr != UNDEFINED)
        defined = yte[yte != UNDEFINED]
        if ytr[fit_rows].min() == ytr[fit_rows].max() or ytr[stop_rows].min() == ytr[stop_rows].max() or defined.min() == defined.max():
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
        fit, held = Pool(xtr[fit_rows], ytr[fit_rows], weight=w[fit_rows]), Pool(xtr[stop_rows], ytr[stop_rows], weight=w[stop_rows])
        model.fit(fit, eval_set=held, use_best_model=True)
        s = score(yte, model.predict_proba(xte)[:, 1])
        res.append((lab, s["roc_auc"], s["pr_auc"]))
        print(f"{lab} {s['roc_auc']:.4f} {s['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(res, columns=["label", "roc", "pr"]).set_index("label")
    print(r.round(4).to_string())
    print("official mean", r.iloc[:4].mean().round(4).to_dict())
    out = a.out or a.probe_dir / f"token_counts_{a.window}.csv"
    r.to_csv(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
