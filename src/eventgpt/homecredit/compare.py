"""The feature table beside the frozen embedding, on the probe's rows: ROC-AUC with standard errors and paired differences.

Reads what eventgpt.homecredit.probe wrote (embeddings, train and test rows, head scores) and the Home Credit
feature table (home_credit_full.parquet, 208 hand-built predictors per case, from the rwds-submission build),
and fits on the same training rows:

    embedding       the probe's logistic head on the last-state embedding (read back, not refitted); the
                    recency-pooled readout's head is listed beside it
    table           CatBoost on the feature table
    table+embedding CatBoost on the table with the embedding appended as 384 more columns
    rank average    the mean of the ranks of embedding and table scores, weights fixed in advance
    table+embedding feature   with --catboost-embedding, the embedding as a CatBoost embedding feature (boost_embedding)
    table (all)     CatBoost on every case of weeks 0 to 69, the strongest table model, as a reference

Standard errors follow eventgpt.evaluation.auc_se (DeLong), one row per case so no clustering; the paired
difference of each column against `table` uses the covariance of the two influence vectors.

Example: `uv run python -m eventgpt.homecredit.compare data/runs/hc-alq-s42/ckpt/probe --features <home_credit_full.parquet>`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import norm, rankdata

from eventgpt.evaluation.auc_se import covariance, influence

LABEL = "target"
KEYS = ["case_id", "date_decision", "MONTH", "WEEK_NUM", LABEL]
READOUT = "last"


def table(features: Path, case_ids: pd.Series) -> pd.DataFrame:
    """The feature table rows of the given cases, in their order."""
    df = pd.read_parquet(features)
    df["case_id"] = df["case_id"].astype(str)
    df = df.set_index("case_id").loc[case_ids.to_numpy()].reset_index(drop=True)
    return df.drop(columns=[c for c in KEYS if c in df.columns])


def boost(xtr: pd.DataFrame, ytr: np.ndarray, wtr: np.ndarray, xte: pd.DataFrame, seed: int) -> np.ndarray:
    """Test scores of CatBoost fitted on the weighted training rows; string columns are its categorical features."""
    cats = [c for c in xtr.columns if str(xtr[c].dtype) in ("object", "category")]
    xtr, xte = xtr.copy(), xte.copy()
    for c in cats:
        xtr[c] = xtr[c].astype(str).fillna("NA")
        xte[c] = xte[c].astype(str).fillna("NA")
    m = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=8, allow_writing_files=False
    )
    m.fit(xtr, ytr, sample_weight=wtr, cat_features=cats)
    return np.asarray(m.predict_proba(xte)[:, 1])


def boost_embedding(
    xtr: pd.DataFrame, etr: np.ndarray, ytr: np.ndarray, wtr: np.ndarray, xte: pd.DataFrame, ete: np.ndarray, seed: int
) -> np.ndarray:
    """Test scores of CatBoost with the table columns and the embedding passed as one embedding feature.

    CatBoost derives its own features from an embedding column (linear discriminant projections and
    nearest-neighbour class statistics, fitted on the training rows) instead of splitting on the raw
    coordinates, which is what boost() does when the embedding is appended as plain columns.
    """
    cats = [c for c in xtr.columns if str(xtr[c].dtype) in ("object", "category")]
    xtr, xte = xtr.copy(), xte.copy()
    for c in cats:
        xtr[c] = xtr[c].astype(str).fillna("NA")
        xte[c] = xte[c].astype(str).fillna("NA")
    xtr["emb"] = list(etr.astype(np.float32))
    xte["emb"] = list(ete.astype(np.float32))
    train = Pool(xtr, ytr, weight=wtr, cat_features=cats, embedding_features=["emb"])
    test = Pool(xte, cat_features=cats, embedding_features=["emb"])
    m = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=8, allow_writing_files=False
    )
    m.fit(train)
    return np.asarray(m.predict_proba(test)[:, 1])


def main() -> None:
    """Fit the table models on the probe's rows and report ROC-AUC with standard errors against the table."""
    p = argparse.ArgumentParser()
    p.add_argument("probe_dir", type=Path)
    p.add_argument("--features", type=Path, required=True, help="home_credit_full.parquet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-all", action="store_true", help="skip the table model on every case of weeks 0 to 69")
    p.add_argument("--no-columns", action="store_true", help="skip the table model with the embedding appended as columns")
    p.add_argument("--catboost-embedding", action="store_true", help="add the table with the embedding as a CatBoost embedding feature")
    a = p.parse_args()
    t0 = time.time()
    train = pd.read_parquet(a.probe_dir / "train_rows.parquet")
    test = pd.read_parquet(a.probe_dir / "test_rows.parquet")
    preds = pd.read_parquet(a.probe_dir / "preds.parquet")
    emb = np.load(a.probe_dir / "embeddings.npz")
    ytr, wtr, yte = train[LABEL].to_numpy(), train["weight"].to_numpy(), test[LABEL].to_numpy()
    xtr, xte = table(a.features, train["case_id"]), table(a.features, test["case_id"])
    n_tr, n_te = int(ytr.sum()), int(yte.sum())
    print(
        f"train {len(train):,} rows ({n_tr:,} defaults), test {len(test):,} rows ({n_te:,} defaults), {xtr.shape[1]} table columns",
        flush=True,
    )

    scores: dict[str, np.ndarray] = {"embedding": preds[READOUT].to_numpy(), "embedding (recency)": preds["recency"].to_numpy()}
    scores["table"] = boost(xtr, ytr, wtr, xte, a.seed)
    print(f"table fitted ({time.time() - t0:.0f}s)", flush=True)
    etr, ete = emb[f"train_{READOUT}"], emb[f"test_{READOUT}"]
    if not a.no_columns:
        cols = [f"e{j}" for j in range(etr.shape[1])]
        xtr_e = pd.concat([xtr, pd.DataFrame(etr, columns=cols)], axis=1)
        xte_e = pd.concat([xte, pd.DataFrame(ete, columns=cols)], axis=1)
        scores["table+embedding"] = boost(xtr_e, ytr, wtr, xte_e, a.seed)
        print(f"table+embedding fitted ({time.time() - t0:.0f}s)", flush=True)
    if a.catboost_embedding:
        for r in (READOUT, "recency"):
            scores[f"table+embedding feature ({r})"] = boost_embedding(xtr, emb[f"train_{r}"], ytr, wtr, xte, emb[f"test_{r}"], a.seed)
            print(f"table+embedding feature ({r}) fitted ({time.time() - t0:.0f}s)", flush=True)
    scores["rank average"] = 0.5 * (rankdata(scores["embedding"]) + rankdata(scores["table"])) / len(yte)
    if not a.no_all:
        full = pd.read_parquet(a.features, columns=["case_id", "WEEK_NUM", LABEL])
        full = full[full["WEEK_NUM"] < 70]
        xall = table(a.features, full["case_id"].astype(str))
        scores["table (all weeks 0-69)"] = boost(xall, full[LABEL].to_numpy(), np.ones(len(full)), xte, a.seed)
        print(f"table (all) fitted on {len(full):,} rows ({time.time() - t0:.0f}s)", flush=True)

    names = list(scores)
    aucs, infl = {}, []
    for n in names:
        auc, inf = influence(yte, scores[n])
        aucs[n] = auc
        infl.append(inf)
    cov = covariance(np.stack(infl, axis=1), None)
    base = names.index("table")
    rows = []
    for i, n in enumerate(names):
        se = float(np.sqrt(cov[i, i]))
        diff = aucs[n] - aucs["table"]
        se_diff = float(np.sqrt(max(cov[i, i] + cov[base, base] - 2 * cov[i, base], 0.0)))
        z = diff / se_diff if se_diff > 0 else float("nan")
        rows.append(
            {
                "model": n,
                "roc_auc": aucs[n],
                "se": se,
                "gini": 2 * aucs[n] - 1,
                "diff_vs_table": diff,
                "se_diff": se_diff,
                "p": 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False), flush=True)
    tag = "_embfeat" if a.catboost_embedding else ""
    out.to_csv(a.probe_dir / f"compare{tag}.csv", index=False)
    (a.probe_dir / f"compare{tag}.json").write_text(
        json.dumps({"rows": rows, "test_rows": len(test), "test_positives": int(yte.sum())}, indent=1)
    )
    pd.DataFrame({"case_id": test["case_id"], **scores}).to_parquet(a.probe_dir / f"compare{tag}_preds.parquet")
    print(f"written to {a.probe_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
