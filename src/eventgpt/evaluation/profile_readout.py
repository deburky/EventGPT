"""Customer embedding with a static path: the history state plus the learned embeddings of the profile event's tokens.

The profile event (the client's first, holding the static attributes) reaches the history state only after
mean pooling and the causal layers, and next-event prediction gives the model little reason to carry it to the
last event. Here the input embeddings of its field tokens, one learned vector per category, are appended to
the history state at readout, so a head sees every attribute directly. The tokens are read off the client's
own stream, so the readout needs no knowledge of what the attributes are.

With --timing, five numbers read off the event timestamps join as well: days since the last and since the
first event, events in the last 30 and 90 days, and all events, each as log(1 + x). The hand-built baseline's
timing features are its strongest group after the profile, and the 2,048-token window can hold less than
a client's whole history, so these test what the history state is missing on recency and tenure.

With --counts, the token-count table of eventgpt.baselines.token_counts over the full history (counts of the
200 most frequent tokens over all history, 91 and 30 days), as log(1 + x), joins as well: it holds what lies
before the 2,048-token window.

Rows, labels, weights and the history states are a probe run's own (its output directory); the token
embeddings come from the same checkpoint.

Example with R=data/runs/hier-alq-s42: `uv run egpt profile-readout $R/hier-alq-s42.pt --probe-dir $R/probe-now-2k`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eventgpt.baselines.token_counts import build
from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import SPECIALS, Vocab
from eventgpt.evaluation.probe import OFFICIAL, UNDEFINED, Fold, labels, mark_undefined, score
from eventgpt.models.flat import load_checkpoint

FIELDS = 4  # tokens after [EVT] name=profile dt=<b>: s21 s22 s31 s32


def profile_tokens(rows: pd.DataFrame, vocab: Vocab) -> np.ndarray:
    """(rows, FIELDS) token ids of the profile event's fields, 0 for a client whose first event is not a profile."""
    out = np.zeros((len(rows), FIELDS), dtype=np.int64)
    profile = vocab.id("name=profile")
    for k in np.unique(rows["fold"].to_numpy()).tolist():
        fold = Fold(int(k), vocab)
        m = (rows["fold"] == k).to_numpy()
        start = fold.event_start[fold.client_events[rows.loc[m, "client_id"].map(fold.index).to_numpy()]]
        tok = np.asarray(fold.tokens)
        ids = np.stack([tok[start + 3 + j] for j in range(FIELDS)], axis=1).astype(np.int64)
        out[m] = np.where((tok[start + 1] == profile)[:, None], ids, 0)
    return out


def timing(rows: pd.DataFrame, vocab: Vocab) -> np.ndarray:
    """(rows, 5) log(1 + x) of days since the last and first event, events in the last 30 and 90 days, and all events."""
    out = np.zeros((len(rows), 5))
    day = 86_400
    for k in np.unique(rows["fold"].to_numpy()).tolist():
        fold = Fold(int(k), vocab)
        m = np.flatnonzero((rows["fold"] == k).to_numpy())
        start = fold.client_events[rows["client_id"].iloc[m].map(fold.index).to_numpy()]
        n = rows["n_hist"].to_numpy()[m]
        cut = rows["cut"].to_numpy()[m]
        for j, (a, h, c) in enumerate(zip(start, n, cut, strict=True)):
            if h == 0:
                continue
            t = fold.event_time[a : a + h]
            out[m[j]] = [
                (c - t[-1]) / day,
                (c - t[0]) / day,
                h - np.searchsorted(t, c - 30 * day),
                h - np.searchsorted(t, c - 90 * day),
                h,
            ]
    return np.log1p(np.maximum(out, 0))


def counts(rows: pd.DataFrame, vocab: Vocab, ref_fold: int, top: int = 200) -> np.ndarray:
    """log(1 + x) of eventgpt.baselines.token_counts' full-history table, counting the most frequent tokens of ref_fold."""
    freq = np.bincount(np.asarray(Fold(ref_fold, vocab).tokens), minlength=vocab.size)
    freq[: len(SPECIALS)] = 0
    col = np.full(vocab.size, -1)
    col[np.argsort(freq)[::-1][:top]] = np.arange(top)
    x = build(rows, vocab, col, top, 0, time.time()).to_numpy()
    return np.log1p(np.nan_to_num(np.maximum(x, 0)))


def main() -> None:
    """Logistic heads on the history state alone and with the profile path, ROC-AUC on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows, test_rows and embeddings")
    p.add_argument("--readout", default="last", choices=["last", "mean", "recency"])
    p.add_argument("--timing", action="store_true", help="also fit heads with the five timing numbers")
    p.add_argument("--counts", action="store_true", help="also fit heads with the full-history token counts")
    p.add_argument("--out", type=Path, default=None, help="csv path; default <probe-dir>/profile_readout.csv")
    a = p.parse_args()

    vocab = Vocab.load(OUT / "vocab")
    emb = load_checkpoint(str(a.ckpt)).tok_emb.weight.detach().numpy()
    tr = mark_undefined(pd.read_parquet(a.probe_dir / "train_rows.parquet"))
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet"))
    z = np.load(a.probe_dir / "embeddings.npz")
    ptr = emb[profile_tokens(tr, vocab)].reshape(len(tr), -1)
    pte = emb[profile_tokens(te, vocab)].reshape(len(te), -1)
    variants = {
        "history": (z[f"train_{a.readout}"], z[f"test_{a.readout}"]),
        "history + profile": (np.hstack([z[f"train_{a.readout}"], ptr]), np.hstack([z[f"test_{a.readout}"], pte])),
    }
    if a.timing:
        ttr, tte = timing(tr, vocab), timing(te, vocab)
        for name in list(variants):
            xtr, xte = variants[name]
            variants[f"{name} + timing"] = (np.hstack([xtr, ttr]), np.hstack([xte, tte]))
    if a.counts:
        ref = int(tr["fold"].iloc[0])  # the fold eventgpt.baselines.token_counts takes its tokens from
        ctr, cte = counts(tr, vocab, ref), counts(te, vocab, ref)
        for name in ["history", "history + profile"]:
            xtr, xte = variants[name]
            variants[f"{name} + counts"] = (np.hstack([xtr, ctr]), np.hstack([xte, cte]))
    w = tr["weight"].to_numpy()
    res = []
    preds = {name: te[["client_id", "mon"]].copy() for name in variants}
    for name, (xtr, xte) in variants.items():
        sc = StandardScaler().fit(xtr)
        xtr, xte = sc.transform(xtr), sc.transform(xte)
        for lab in labels():
            ok = (tr[lab] != UNDEFINED).to_numpy()
            head = LogisticRegression(C=0.1, max_iter=3000).fit(xtr[ok], tr[lab].to_numpy()[ok], sample_weight=w[ok])
            preds[name][lab] = head.predict_proba(xte)[:, 1]
            s = score(te[lab].to_numpy(), preds[name][lab].to_numpy())
            res.append((name, lab, s["roc_auc"], s["pr_auc"]))
    r = pd.DataFrame(res, columns=["variant", "label", "roc", "pr"])
    print(r.pivot(index="label", columns="variant", values="roc").loc[labels()].round(4).to_string())
    print("official mean", r[r["label"].isin(OFFICIAL)].groupby("variant")["roc"].mean().round(4).to_dict())
    out = a.out or a.probe_dir / "profile_readout.csv"
    r.to_csv(out, index=False)
    # test predictions, for eventgpt.evaluation.auc_se
    for name, pr in preds.items():
        slug = name.removeprefix("history + ").replace(" + ", "_")
        pr.to_parquet(out.with_name(f"preds_{slug}.parquet"))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
