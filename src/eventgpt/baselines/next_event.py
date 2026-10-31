"""Next-event baselines without labels: score the event targets from event statistics, as EventGPT zero-shot does.

Each baseline estimates how likely the client is to make a transaction of kind (X, Y) from the events alone,
fitted on the event streams of the four pretraining folds and never on the labels. On the rows EventGPT's
zero-shot scores were saved for (zero_shot_scores.parquet), with the same rule for earlier occurrences:
- markov: P(next event is (X, Y) | the last event's kind), a first-order Markov chain over event kinds.
- markov_recent: the mean of that probability over the last RECENT events, a smoothed bag of recent kinds.
- activity: the number of transactions in the 30 days before the as-of date.
- markov_activity: markov_recent times the number of events in those 30 days, the expected count of the kind in a
  month if the client keeps its pace.
- type_share: the share of the client's transactions of type X, any subtype.
A kind is etype * 1000 + subtype for a transaction, and minus the name token for other events.

Example: `uv run python -m eventgpt.baselines.next_event data/runs/hier-alq-s42/zero_shot_scores.parquet`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.auc_se import covariance, influence
from eventgpt.evaluation.probe import FIRST_KINDS, Fold

RECENT = 10
WINDOW_DAYS = 30
SMOOTH = 0.5  # added to every transition count


# -------------------------------------------------------------------------------
# Kinds and transitions
# -------------------------------------------------------------------------------
def kinds(fold: Fold) -> np.ndarray:
    """Kind code of every event of the fold."""
    name = np.asarray(fold.tokens[fold.event_start + 1]).astype(np.int64)
    return np.where(fold.is_trx, fold.etype.astype(np.int64) * 1000 + fold.esub, -name)


def client_of_event(fold: Fold) -> np.ndarray:
    """Client index of every event."""
    return np.repeat(np.arange(len(fold.client_events) - 1), np.diff(fold.client_events))


def transitions(folds: list[Fold], codes: np.ndarray) -> np.ndarray:
    """Row-normalized transition matrix over the kinds in codes, counted within clients, with SMOOTH added."""
    k = len(codes)
    counts = np.full((k, k), SMOOTH)
    for f in folds:
        kk = np.searchsorted(codes, kinds(f))
        same = np.diff(client_of_event(f)) == 0
        pair = kk[:-1][same] * k + kk[1:][same]
        counts += np.bincount(pair, minlength=k * k).reshape(k, k)
    return counts / counts.sum(1, keepdims=True)


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    """Fit the transitions on folds 1 to 4 and score the saved zero-shot rows of fold 0."""
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path, help="zero_shot_scores.parquet from eventgpt.evaluation.zero_shot --save-scores")
    p.add_argument("--out", type=Path, default=None, help="csv path; default next_event_baselines.csv beside the scores")
    a = p.parse_args()

    vocab = Vocab.load(OUT / "vocab")
    test = Fold(0, vocab)
    train = [Fold(k, vocab) for k in (1, 2, 3, 4)]
    codes = np.unique(np.concatenate([kinds(f) for f in [test, *train]]))
    trans = transitions(train, codes)
    print(f"{len(codes)} kinds, transitions from {sum(len(f.event_time) for f in train):,} events", flush=True)

    rows = pd.read_parquet(a.scores)
    ci = rows["client_id"].map(test.index).to_numpy()
    first, n_hist, cut = test.client_events[ci], rows["n_hist"].to_numpy(), rows["cut"].to_numpy()
    test_kinds = np.searchsorted(codes, kinds(test))
    target_idx = np.searchsorted(codes, [x * 1000 + y for x, y in FIRST_KINDS])
    n = len(rows)
    scores = {m: np.zeros((n, len(FIRST_KINDS))) for m in ("markov", "markov_recent", "markov_activity", "type_share")}
    activity = np.zeros(n)
    for r in range(n):
        a0, e = int(first[r]), int(first[r] + n_hist[r])
        if n_hist[r] == 0:
            continue
        recent = test_kinds[max(a0, e - RECENT) : e]
        p_recent = trans[recent][:, target_idx].mean(0)
        scores["markov"][r] = trans[test_kinds[e - 1], target_idx]
        scores["markov_recent"][r] = p_recent
        in_window = test.event_time[a0:e] >= cut[r] - WINDOW_DAYS * 86_400
        activity[r] = (in_window & test.is_trx[a0:e]).sum()
        scores["markov_activity"][r] = p_recent * in_window.sum()
        trx = test.is_trx[a0:e]
        if trx.any():
            et = test.etype[a0:e][trx]
            scores["type_share"][r] = [(et == x).mean() for x, _ in FIRST_KINDS]
    scores["activity"] = np.repeat(activity[:, None], len(FIRST_KINDS), 1)
    scores["eventgpt"] = rows[[f"zs_first_{x}_{y}" for x, y in FIRST_KINDS]].to_numpy()

    clients = rows["client_id"].to_numpy()
    out = []
    for k, (x, y) in enumerate(FIRST_KINDS):
        lab = f"first_{x}_{y}"
        ev = (rows["in_rand"] | (rows[lab] == 1)).to_numpy()
        yk = rows[lab].to_numpy()[ev]
        unseen = rows[f"unseen_{lab}"].to_numpy()[ev]
        aucs, infl = {}, {}
        for m, s in scores.items():
            aucs[m], infl[m] = influence(yk, s[ev, k] * unseen)
        names = list(scores)
        cov = covariance(np.stack([infl[m] for m in names], 1), clients[ev])
        j = names.index("eventgpt")
        for i, m in enumerate(names):
            diff_se = float(np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 0.0)))
            line = {"label": lab, "model": m, "auc": aucs[m], "se": float(np.sqrt(cov[i, i]))}
            line |= {"eventgpt_minus": aucs["eventgpt"] - aucs[m], "diff_se": diff_se}
            out.append(line)
    res = pd.DataFrame(out)
    print(res.pivot(index="model", columns="label", values="auc").round(3).to_string())
    print(res.pivot(index="model", columns="label", values="eventgpt_minus").round(3).to_string())
    dest = a.out or a.scores.with_name("next_event_baselines.csv")
    res.to_csv(dest, index=False)
    print(f"written to {dest}")


if __name__ == "__main__":
    main()
