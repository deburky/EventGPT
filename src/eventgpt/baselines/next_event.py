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


def baseline_scores(
    fold: Fold, trans: np.ndarray, codes: np.ndarray, ci: np.ndarray, n_hist: np.ndarray, cut: np.ndarray, kinds_xy: list[tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Each baseline's score (rows, kinds) for the transaction kinds kinds_xy."""
    first = fold.client_events[ci]
    fold_kinds = np.searchsorted(codes, kinds(fold))
    target_idx = np.searchsorted(codes, [x * 1000 + y for x, y in kinds_xy])
    n, k = len(ci), len(kinds_xy)
    out = {m: np.zeros((n, k)) for m in ("markov", "markov_recent", "markov_activity", "type_share", "activity")}
    for r in range(n):
        a0, e = int(first[r]), int(first[r] + n_hist[r])
        if n_hist[r] == 0:
            continue
        out["markov"][r] = trans[fold_kinds[e - 1], target_idx]
        p_recent = trans[fold_kinds[max(a0, e - RECENT) : e]][:, target_idx].mean(0)
        out["markov_recent"][r] = p_recent
        in_window = fold.event_time[a0:e] >= cut[r] - WINDOW_DAYS * 86_400
        out["activity"][r] = (in_window & fold.is_trx[a0:e]).sum()
        out["markov_activity"][r] = p_recent * in_window.sum()
        trx = fold.is_trx[a0:e]
        if trx.any():
            et = fold.etype[a0:e][trx]
            out["type_share"][r] = [(et == x).mean() for x, _ in kinds_xy]
    return out


def events_before(fold: Fold, ci: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Number of each client's events strictly before its cut."""
    lo, hi = fold.client_events[ci], fold.client_events[ci + 1]
    return np.array([np.searchsorted(fold.event_time[a:b], c, side="left") for a, b, c in zip(lo, hi, cut, strict=True)])


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def all_kinds(rows: pd.DataFrame, test: Fold, trans: np.ndarray, codes: np.ndarray, ci: np.ndarray, cut: np.ndarray, dest: Path) -> None:
    """Compare EventGPT zero-shot with Markov x events on every kind of zero_shot_all_scores.parquet, paired SEs by client."""
    pairs = [c[3:].split("_") for c in rows.columns if c.startswith("zs_")]
    kinds_xy = [(int(x), int(y)) for x, y in pairs]
    base = baseline_scores(test, trans, codes, ci, events_before(test, ci, cut), cut, kinds_xy)["markov_activity"]
    clients, out = rows["client_id"].to_numpy(), []
    for k, (x, y) in enumerate(kinds_xy):
        pos = rows[f"first_{x}_{y}"].to_numpy().astype(bool)
        ev = rows["in_rand"].to_numpy() | pos
        unseen = ~rows[f"seen_{x}_{y}"].to_numpy()[ev]
        a_m, i_m = influence(pos[ev].astype(np.int64), rows[f"zs_{x}_{y}"].to_numpy()[ev] * unseen)
        a_b, i_b = influence(pos[ev].astype(np.int64), base[ev, k] * unseen)
        cov = covariance(np.stack([i_m, i_b], 1), clients[ev])
        se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
        out.append(
            {
                "type": x,
                "subtype": y,
                "positives": int(pos[ev].sum()),
                "eventgpt": a_m,
                "markov_activity": a_b,
                "diff": a_m - a_b,
                "diff_se": se,
            }
        )
    r = pd.DataFrame(out)
    r["z"] = r["diff"] / r["diff_se"]
    print(r.sort_values("diff").round(3).to_string(index=False))
    q = r[["eventgpt", "markov_activity", "diff"]].median().round(3).to_dict()
    print(f"{len(r)} kinds, medians {q}; EventGPT higher by 2 SE on {int((r.z > 2).sum())}, lower on {int((r.z < -2).sum())}")
    r.to_csv(dest, index=False)
    print(f"written to {dest}")


def main() -> None:
    """Fit the transitions on folds 1 to 4 and score the saved zero-shot rows of fold 0."""
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path, help="zero_shot_scores.parquet from eventgpt.evaluation.zero_shot --save-scores")
    p.add_argument("--out", type=Path, default=None, help="csv path; default next_event_baselines.csv beside the scores")
    p.add_argument("--all-kinds", action="store_true", help="scores is zero_shot_all_scores.parquet: compare on every kind in it")
    a = p.parse_args()

    vocab = Vocab.load(OUT / "vocab")
    test = Fold(0, vocab)
    train = [Fold(k, vocab) for k in (1, 2, 3, 4)]
    codes = np.unique(np.concatenate([kinds(f) for f in [test, *train]]))
    trans = transitions(train, codes)
    print(f"{len(codes)} kinds, transitions from {sum(len(f.event_time) for f in train):,} events", flush=True)

    rows = pd.read_parquet(a.scores)
    ci = rows["client_id"].map(test.index).to_numpy()
    cut = rows["cut"].to_numpy()
    if a.all_kinds:
        all_kinds(rows, test, trans, codes, ci, cut, a.out or a.scores.with_name("next_event_all.csv"))
        return
    n_hist = rows["n_hist"].to_numpy()
    scores = baseline_scores(test, trans, codes, ci, n_hist, cut, FIRST_KINDS)
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
