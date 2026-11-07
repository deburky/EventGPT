"""Probe on Home Credit: frozen EventGPT embeddings at the decision date, a logistic head for default, ROC-AUC on the test fold.

A row is one application. Its history is every event of its stream (all of them precede the decision date by
construction, see eventgpt.homecredit.prepare), the cut is the decision date, and with --now the history ends
in a now event whose gap runs from the last event to the decision date. The embedding is read off the history
decoder after the last event, pooled three ways (last, mean, recency), as in eventgpt.evaluation.probe.

Heads are fitted on folds 1 to 4 (decisions before week 70) and scored on fold 0 (weeks 70 to 91), which the
model never saw. The training rows are every default plus a uniform sample of the rest, weighted back to its
share; the test fold is scored in full. The probe writes, next to the checkpoint under probe/:

    embeddings.npz        train_<readout> and test_<readout> arrays
    train_rows.parquet    case_id, week, target, weight, n_hist of the training rows
    test_rows.parquet     the same for the test rows
    preds.parquet         case_id and the head's score per readout on the test rows
    results.json          ROC-AUC and PR-AUC per readout

eventgpt.homecredit.compare then puts the feature table beside these embeddings on the same rows.

Example: `MBD_OUT=data/home-credit/tokens uv run python -m eventgpt.homecredit.probe data/ckpt/hc-alq-s42.pt --now --batch 32`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import READOUTS, Fold, embed_rows, score
from eventgpt.homecredit.prepare import N_FOLDS
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

LABEL = "target"


def load_rows(fold: Fold, k: int, max_cases: int | None, seed: int) -> pd.DataFrame:
    """One row per case of fold k with its cut, history length and label."""
    t = pd.read_parquet(OUT / "labels.parquet")
    t = t[t["fold"] == k].reset_index(drop=True)
    if max_cases:
        t = t.sample(min(max_cases, len(t)), random_state=seed).reset_index(drop=True)
    t["case_id"] = t["case_id"].astype(str)
    t = t[t["case_id"].isin(fold.index.keys())].reset_index(drop=True)
    ci = t["case_id"].map(fold.index).to_numpy()
    t["client_idx"] = ci
    t["n_hist"] = fold.client_events[ci + 1] - fold.client_events[ci]
    t["cut"] = t["cut"].astype(np.int64)
    return t


def sample_train(rows: pd.DataFrame, n_neg: int, seed: int) -> pd.DataFrame:
    """Every default plus a uniform sample of the other rows, weighted back to their share."""
    pos = rows[LABEL].to_numpy() == 1
    neg = rows.index[~pos]
    take = np.random.default_rng(seed).choice(neg, size=min(n_neg, len(neg)), replace=False)
    out = pd.concat([rows[pos], rows.loc[take]])
    out["weight"] = np.where(out.index.isin(rows.index[pos]), 1.0, len(neg) / max(len(take), 1))
    return out


def fit_predict(xtr: np.ndarray, ytr: np.ndarray, wtr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    """Test scores of a standardised logistic head fitted on the weighted training rows."""
    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    head.fit(xtr, ytr, logisticregression__sample_weight=wtr)
    return np.asarray(head.predict_proba(xte)[:, 1])


def main() -> None:
    """Embed the sampled training rows and the full test fold, fit heads, report ROC-AUC and PR-AUC."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-neg", type=int, default=120_000, help="non-default training rows sampled from folds 1 to 4 together")
    p.add_argument("--max-cases", type=int, default=0, help="cases per fold, for a quick run; 0 is every case")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--now", action="store_true", help="end each history in a now event at the decision date")
    p.add_argument("--out", type=Path, default=None, help="results directory (default: probe/ next to the checkpoint)")
    a = p.parse_args()

    t0 = time.time()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("vocab_sha256") != vocab.fingerprint():
        raise SystemExit(f"{a.ckpt} was trained on vocabulary {str(ck.get('vocab_sha256'))[:12]}, {OUT} is {vocab.fingerprint()[:12]}")
    model = load_checkpoint(str(a.ckpt)).to(dev)
    out_dir = a.out or a.ckpt.parent / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.ckpt.name} step {ck.get('step')} on {dev}, test fold {test_fold}", flush=True)

    emb: dict[str, dict[str, list[np.ndarray]]] = {"train": {r: [] for r in READOUTS}, "test": {r: [] for r in READOUTS}}
    frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    for k in range(N_FOLDS):
        split = "test" if k == test_fold else "train"
        fold = Fold(k, vocab)
        rows = load_rows(fold, k, a.max_cases or None, a.seed)
        if split == "train":
            rows = sample_train(rows, a.n_neg // (N_FOLDS - 1), a.seed + k)
        else:
            rows["weight"] = 1.0
        got = embed_rows(model, fold, rows, dev, a.block, a.batch, vocab if a.now else None, out_dir / "cache" / f"fold{k}")
        for r in READOUTS:
            emb[split][r].append(got[r])
        frames[split].append(rows.drop(columns=["client_idx"]))
        print(f"fold {k} ({split}): {len(rows):,} rows embedded ({time.time() - t0:.0f}s)", flush=True)

    train, test = pd.concat(frames["train"], ignore_index=True), pd.concat(frames["test"], ignore_index=True)
    np.savez(out_dir / "embeddings.npz", **{f"{s}_{r}": np.concatenate(emb[s][r]) for s in ("train", "test") for r in READOUTS})
    train.to_parquet(out_dir / "train_rows.parquet")
    test.to_parquet(out_dir / "test_rows.parquet")

    results: dict[str, dict[str, float]] = {}
    preds = test[["case_id"]].copy()
    for r in READOUTS:
        xtr, xte = np.concatenate(emb["train"][r]), np.concatenate(emb["test"][r])
        preds[r] = fit_predict(xtr, train[LABEL].to_numpy(), train["weight"].to_numpy(), xte)
        results[r] = score(test[LABEL].to_numpy(), preds[r].to_numpy())
    preds.to_parquet(out_dir / "preds.parquet")
    meta = {
        "ckpt": str(a.ckpt),
        "step": ck.get("step"),
        "block": a.block,
        "now": a.now,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_positives": int(test[LABEL].sum()),
        "elapsed_s": round(time.time() - t0),
    }
    (out_dir / "results.json").write_text(json.dumps({**meta, "metrics": results}, indent=1))
    print(pd.DataFrame(results).T.round(4).to_string(), flush=True)
    print(f"written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
