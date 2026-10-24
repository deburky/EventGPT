"""Customer embedding with a static path: the history state plus the learned embeddings of the profile event's tokens.

The profile event (the client's first, holding the static attributes) reaches the history state only after
mean pooling and the causal layers, and next-event prediction gives the model little reason to carry it to the
last event. Here the input embeddings of its field tokens, one learned vector per category, are appended to
the history state at readout, so a head sees every attribute directly. The tokens are read off the client's
own stream, so the readout needs no knowledge of what the attributes are.

Rows, labels, weights and the history states are a probe run's own (its output directory); the token
embeddings come from the same checkpoint.

Example with R=data/runs/hier-alq-s42: `uv run egpt profile-readout $R/hier-alq-s42.pt --probe-dir $R/probe-now-2k`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
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


def main() -> None:
    """Logistic heads on the history state alone and with the profile path, ROC-AUC on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows, test_rows and embeddings")
    p.add_argument("--readout", default="last", choices=["last", "mean", "recency"])
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
    preds["history"].to_parquet(out.with_name("preds_history.parquet"))
    preds["history + profile"].to_parquet(out.with_name("preds_profile.parquet"))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
