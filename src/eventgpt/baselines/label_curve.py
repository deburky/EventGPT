"""Labels against zero-shot: the aggregates baseline fitted on n labelled rows, beside the zero-shot score on the event targets.

Zero-shot scoring (eventgpt.evaluation.zero_shot) uses no labels; the aggregates need a labelled training set per
target. For each budget n we draw n client-months uniformly from the pretraining folds, as a team labelling
rows at random would, fit CatBoost on their hand-built features (eventgpt.baselines.catboost.features) for each
event target and score the whole test fold. A draw with no positive gives no model and is scored 0.5. Each
budget is drawn several times. CatBoost's scores are also reported under the zero-shot rule, which sets
clients who already had the kind to zero and needs no labels either.

Example: `uv run egpt label-curve --zero-shot data/runs/hier-rope-s42/zero_shot.csv data/runs/hier-rope-s43/zero_shot.csv`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eventgpt.baselines.catboost import PROFILE, features
from eventgpt.data.mbd import N_FOLDS, OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import Fold, load_rows, score
from eventgpt.evaluation.zero_shot import FIRST, seen_before

ROOT = Path(__file__).resolve().parents[3]
BUDGETS = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]


def draws(n: int) -> int:
    """Draws per budget: more where one draw's positives are few and its AUC noisy."""
    return 5 if n <= 30_000 else 3 if n <= 100_000 else 2


def fit_predict(x: pd.DataFrame, y: np.ndarray, clients: np.ndarray, xte: pd.DataFrame, seed: int) -> np.ndarray | None:
    """Fit CatBoost as in the aggregates baseline and score xte; None without a positive.

    A tenth of the clients is held out for early stopping when both sides have a positive; otherwise the
    model is fitted on every row for a fixed 300 iterations.
    """
    if y.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(clients)
    stop = np.isin(clients, rng.choice(uniq, size=max(len(uniq) // 10, 1), replace=False))
    early = y[stop].sum() > 0 and y[~stop].sum() > 0
    model = CatBoostClassifier(
        iterations=2000 if early else 300,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        early_stopping_rounds=100 if early else None,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    if early:
        held = Pool(x[stop], y[stop], cat_features=PROFILE)
        model.fit(Pool(x[~stop], y[~stop], cat_features=PROFILE), eval_set=held, use_best_model=True)
    else:
        model.fit(Pool(x, y, cat_features=PROFILE))
    return model.predict_proba(Pool(xte, cat_features=PROFILE))[:, 1]


def main() -> None:
    """ROC-AUC of the aggregates per label budget and draw, written beside the zero-shot reference."""
    p = argparse.ArgumentParser()
    p.add_argument("--zero-shot", type=Path, nargs="+", required=True, help="zero_shot.csv files, averaged as the reference")
    p.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "baseline" / "label_curve.csv")
    a = p.parse_args()

    t0 = time.time()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    pool_parts, test = [], pd.DataFrame()
    for k in range(N_FOLDS):
        fold = Fold(k, vocab)
        rows = load_rows(fold, k, None, a.seed)
        if k == test_fold:
            test = rows
            unseen = ~seen_before(fold, rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy())
        else:
            pool_parts.append(rows)
        print(f"fold {k}: {len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)
    pool = pd.concat(pool_parts, ignore_index=True)
    rng = np.random.default_rng(a.seed)
    picks = {(n, d): rng.choice(len(pool), size=min(n, len(pool)), replace=False) for n in a.budgets for d in range(draws(n))}
    used = pool.iloc[np.unique(np.concatenate(list(picks.values())))]
    feats = features(pd.concat([used, test]))
    print(f"{feats.shape[1] - 2} features for {len(used):,} pool rows ({time.time() - t0:.0f}s)", flush=True)

    def design(rows: pd.DataFrame) -> pd.DataFrame:
        x = rows[["client_id", "mon"]].merge(feats, on=["client_id", "mon"], how="left").drop(columns=["client_id", "mon"])
        for c in PROFILE:
            x[c] = x[c].fillna(-1).astype("int64").astype(str)
        return x

    xte = design(test)
    res = []
    for (n, d), idx in picks.items():
        tr = pool.iloc[idx]
        xtr = design(tr)
        for k, lab in enumerate(FIRST):
            y, yte = tr[lab].to_numpy(), test[lab].to_numpy()
            q = fit_predict(xtr, y, tr["client_id"].to_numpy(), xte, a.seed + d)
            raw = score(yte, q)["roc_auc"] if q is not None else 0.5
            rule = score(yte, q * unseen[:, k])["roc_auc"] if q is not None else 0.5
            res.append({"budget": n, "draw": d, "label": lab, "positives": int(y.sum()), "catboost": raw, "catboost_rule": rule})
        print(f"budget {n:,} draw {d} ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(a.out, index=False)

    zs = pd.concat([pd.read_csv(f, index_col="label")["zero_shot_unseen"] for f in a.zero_shot], axis=1).mean(axis=1)
    summary = r.groupby(["label", "budget"])[["positives", "catboost", "catboost_rule"]].mean().unstack("budget")
    print(summary.round(3).to_string())
    print(f"zero-shot reference (mean of {len(a.zero_shot)}):\n{zs.round(3).to_string()}")
    print(f"written to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
