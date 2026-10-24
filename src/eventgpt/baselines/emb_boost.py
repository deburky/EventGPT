"""CatBoost on a frozen EventGPT embedding: as one embedding feature, as plain numeric columns, and beside the aggregates.

CatBoost's embedding feature type takes the vector as a whole and derives its own features from it (LDA
projections and distances to the nearest training rows of each class), where numeric columns give the trees
one coordinate at a time. Rows, labels, weights and the train/test split are a probe run's own, read from its
output directory with the embeddings it saved, and the hand-built aggregates are eventgpt.baselines.catboost's.

    emb          the embedding as one embedding feature
    emb_numeric  the embedding as d numeric columns
    agg_emb      the aggregates plus the embedding feature

Example: `uv run python -m eventgpt.baselines.emb_boost --probe-dir data/runs/hier-al-s42/probe-now-2k`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eventgpt.baselines.catboost import PROFILE, features
from eventgpt.evaluation.probe import UNDEFINED, labels, mark_undefined, score

VARIANTS = ["emb", "emb_numeric", "agg_emb"]


def tables(rows: pd.DataFrame, emb: np.ndarray, agg: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Feature table for one variant, with its categorical and embedding columns."""
    if variant == "emb_numeric":
        return pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])]), [], []
    x = pd.DataFrame({"emb": list(emb)})
    if variant == "emb":
        return x, [], ["emb"]
    a = rows[["client_id", "mon"]].merge(agg, on=["client_id", "mon"], how="left").drop(columns=["client_id", "mon"])
    for c in PROFILE:
        a[c] = a[c].fillna(-1).astype("int64").astype(str)
    return pd.concat([a.reset_index(drop=True), x], axis=1), PROFILE, ["emb"]


def main() -> None:
    """One CatBoost model per label and variant, ROC-AUC and PR-AUC on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows, test_rows and embeddings")
    p.add_argument("--readout", default="last", choices=["last", "mean", "recency"])
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="csv path; default <probe-dir>/emb_boost.csv")
    a = p.parse_args()

    t0 = time.time()
    tr = mark_undefined(pd.read_parquet(a.probe_dir / "train_rows.parquet"))
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet"))
    z = np.load(a.probe_dir / "embeddings.npz")
    etr, ete = z[f"train_{a.readout}"], z[f"test_{a.readout}"]
    variants = a.variants.split(",")
    agg = features(pd.concat([tr, te])) if "agg_emb" in variants else pd.DataFrame()
    print(f"{len(tr):,} train rows, {len(te):,} test rows, embedding {etr.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    # a tenth of the training clients is held out for early stopping, by client, as in eventgpt.baselines.catboost
    rng = np.random.default_rng(a.seed)
    clients = tr["client_id"].unique()
    stop = tr["client_id"].isin(set(rng.choice(clients, size=len(clients) // 10, replace=False))).to_numpy()
    w = tr["weight"].to_numpy()
    res = []
    for v in variants:
        xtr, cat, embf = tables(tr, etr, agg, v)
        xte, _, _ = tables(te, ete, agg, v)
        for lab in labels():
            ytr, yte = tr[lab].to_numpy(), te[lab].to_numpy()
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
            fit_rows, stop_rows = ~stop & (ytr != UNDEFINED), stop & (ytr != UNDEFINED)
            fit = Pool(xtr[fit_rows], ytr[fit_rows], cat_features=cat, embedding_features=embf or None, weight=w[fit_rows])
            held = Pool(xtr[stop_rows], ytr[stop_rows], cat_features=cat, embedding_features=embf or None, weight=w[stop_rows])
            model.fit(fit, eval_set=held, use_best_model=True)
            s = score(yte, model.predict_proba(Pool(xte, cat_features=cat, embedding_features=embf or None))[:, 1])
            res.append((v, lab, s["roc_auc"], s["pr_auc"]))
            print(f"{v} {lab} {s['roc_auc']:.4f} {s['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(res, columns=["variant", "label", "roc", "pr"])
    print(r.pivot(index="label", columns="variant", values="roc").loc[labels()].round(4).to_string())
    print("official mean", r[r["label"].isin(labels()[:4])].groupby("variant")["roc"].mean().round(4).to_dict())
    out = a.out or a.probe_dir / "emb_boost.csv"
    r.to_csv(out, index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
