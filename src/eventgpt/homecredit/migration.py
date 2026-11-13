"""DPD migration on Home Credit: EventGPT's softmax over next month's days-past-due bucket against the roll-rate matrix.

A `cb_pay` event is one bureau payment month of a case, and its `pdpd` field is the worst days past due over the
case's contracts that month, in the buckets

    b0 0   b1 1-5   b2 6-30   b3 31-60   b4 61-90   b5 91-180   b6 over 180   na nothing due

A row is a case with a payment month before the cut (the first day of the month before the decision month, as in
eventgpt.homecredit.zero_shot) and one inside the held-out month. The current bucket is the `pdpd` of the last payment
month before the cut and the target is the `pdpd` of the first payment month after the cut, a migration over about
30 days.

The zero-shot row of the transition matrix comes from the token decoder with no labels: after the history the scorer
appends `[EVT] name=cb_pay dt=<g> n late`, and P(pdpd) is summed over the n and late values the decoder puts before
it, given that the next event is a payment month,

    P(pdpd = j) = sum_v sum_w P(n = v) P(late = w | n = v) P(pdpd = j | n = v, late = w).

The baseline is the empirical roll-rate matrix of the training folds (1 to 4): one row per current bucket, the same
for every case in the bucket, with add-one smoothing. Test rows are every case of the test fold in arrears now
(bucket b1 or worse) plus a uniform sample of the others; population metrics use the uniform rows and the metrics
within a current bucket use every row of the bucket. Example:

    MBD_OUT=data/home-credit/tokens uv run python -m eventgpt.homecredit.migration data/runs/hc-alq-s42/ckpt/hc-alq-s42.pt \
        --out data/runs/hc-mig-s42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.homecredit.prepare import N_FOLDS
from eventgpt.homecredit.zero_shot import Events, auc, load_rows
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

BUCKETS = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "na"]
ORDERED = BUCKETS[:-1]  # na (nothing due) has no place in the order


# -------------------------------------------------------------------------------
# Rows
# -------------------------------------------------------------------------------
def pay_bucket(fold: Fold, e: int, pdpd: dict[int, int]) -> int:
    """Bucket index of the pdpd field of event e, read by field name; -1 if the field is missing."""
    s = int(fold.event_start[e])
    t = int(fold.event_start[e + 1]) if e + 1 < len(fold.event_start) else len(fold.tokens)
    for z in fold.tokens[s + 3 : t]:
        j = pdpd.get(int(z))
        if j is not None:
            return j
    return -1


def migration_rows(fold: Fold, vocab: Vocab, k: int, max_cases: int | None, seed: int) -> pd.DataFrame:
    """Cases of fold k with a payment month before the cut and one in the held-out month: current and next bucket."""
    ev = Events(fold)
    rows = load_rows(fold, ev, k, max_cases, seed)
    pdpd = {vocab.id(f"pdpd={b}"): j for j, b in enumerate(BUCKETS)}
    a = fold.client_events[rows["client_idx"].to_numpy()]
    cur, nxt = np.full(len(rows), -1), np.full(len(rows), -1)
    for r, (s, h, m) in enumerate(zip(a, rows["n_hist"].to_numpy(), rows["n_month"].to_numpy(), strict=True)):
        before = np.flatnonzero(ev.is_pay[s : s + h])
        inside = np.flatnonzero(ev.is_pay[s + h : s + m])
        if len(before) and len(inside):
            cur[r] = pay_bucket(fold, int(s + before[-1]), pdpd)
            nxt[r] = pay_bucket(fold, int(s + h + inside[0]), pdpd)
    rows["cur"], rows["next"] = cur, nxt
    return rows[(rows["cur"] >= 0) & (rows["next"] >= 0)].reset_index(drop=True)


# -------------------------------------------------------------------------------
# Zero-shot softmax
# -------------------------------------------------------------------------------
@torch.no_grad()
def bucket_probs(model: Any, seqs: list[np.ndarray], vocab: Vocab, dev: torch.device) -> np.ndarray:
    """(len(seqs), 8) P(next pdpd bucket | the next event is a payment month); sequences end in `[EVT] cb_pay dt n late`."""
    v = vocab
    lengths = np.array([len(s) for s in seqs])
    width = -(-int(lengths.max()) // 256) * 256  # few distinct shapes, so the MPS graph cache stays small
    x = np.zeros((len(seqs), width), dtype=np.int64)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = s
    xt = torch.from_numpy(x).to(dev)
    h, event, _, pos_in_event, tok = model.stages(xt, xt != 0)
    rows = torch.arange(len(seqs), device=dev)
    at_evt = torch.from_numpy(lengths - 5).to(dev)
    at_dt, at_n, at_late = at_evt + 2, at_evt + 3, at_evt + 4
    n_ids = torch.tensor([i for t, i in v.structured.items() if t.startswith("n=")], device=dev)
    late_ids = torch.tensor([i for t, i in v.structured.items() if t.startswith("late=")], device=dev)
    pdpd_ids = torch.tensor([v.id(f"pdpd={b}") for b in BUCKETS], device=dev)

    def field(y: torch.Tensor, at: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(model.head(y[rows, at]).float(), -1)[:, ids]
        return p / p.sum(1, keepdim=True).clamp(min=1e-12)

    y = model.decode_tokens(xt, h, event, pos_in_event, tok)
    p_n = field(y, at_dt, n_ids)
    out = torch.zeros(len(seqs), len(BUCKETS), device=dev)
    for j, nid in enumerate(n_ids):
        xt[rows, at_n] = nid
        p_late = field(model.decode_tokens(xt, h, event, pos_in_event, tok), at_n, late_ids)
        for k, lid in enumerate(late_ids):
            xt[rows, at_late] = lid
            p_pdpd = field(model.decode_tokens(xt, h, event, pos_in_event, tok), at_late, pdpd_ids)
            out += (p_n[:, j] * p_late[:, k]).unsqueeze(1) * p_pdpd
    return out.cpu().numpy()


def score(model: Any, fold: Fold, rows: pd.DataFrame, vocab: Vocab, dev: torch.device, block: int, bsz: int, cache: Path) -> np.ndarray:
    """Zero-shot bucket probabilities for every row, batched by length and token budget, resumable from `cache`."""
    ci, n_hist = rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy()
    last = fold.event_time[fold.client_events[ci] + n_hist - 1]
    gap = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
    head = np.array([vocab.id("[EVT]"), vocab.id("name=cb_pay")])
    tail = np.array([vocab.id("n=1"), vocab.id("late=0")])
    aligned = bool(getattr(model.cfg, "aligned", False))
    seqs = [
        history_tokens(fold, int(c), int(n), block, np.r_[head, g, tail], aligned).astype(np.int64)
        for c, n, g in zip(ci, n_hist, gap, strict=True)
    ]
    lengths = np.array([len(s) for s in seqs])
    order = np.argsort(lengths)
    out = np.zeros((len(seqs), len(BUCKETS)), dtype=np.float32)
    seen = np.zeros(len(seqs), dtype=bool)
    if cache.exists():
        z = np.load(cache)
        if len(z["seen"]) == len(seqs):
            out, seen = z["out"], z["seen"]
            print(f"    resumed {int(seen.sum()):,} scored rows from {cache}", flush=True)
    batches, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and j - i < bsz and (j + 1 - i) * lengths[order[j]] <= bsz * 256:
            j += 1
        batches.append(order[i:j])
        i = j
    t0, done = time.time(), 0
    for b_i, idx in enumerate(batches):
        if seen[idx].all():
            done += len(idx)
            continue
        out[idx] = bucket_probs(model, [seqs[j] for j in idx], vocab, dev)
        seen[idx] = True
        if b_i % 50 == 0:
            np.savez(cache, out=out, seen=seen)
        if dev.type == "mps":
            torch.mps.empty_cache()
        done += len(idx)
        if b_i % 100 == 0:
            print(f"    {done:,}/{len(order):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return out


# -------------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------------
def multiclass(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Log loss, Brier (summed over buckets) and top-1 accuracy."""
    onehot = np.eye(p.shape[1])[y]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))),
        "brier": float(np.mean(((p - onehot) ** 2).sum(1))),
        "accuracy": float(np.mean(p.argmax(1) == y)),
    }


def binary_events(cur: np.ndarray, nxt: np.ndarray, p: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(label, score) of the early-warning events, on the rows where each is defined.

    roll  the bucket worsens (both buckets ordered, current below b6): score P(next bucket above the current one)
    cure  a case in arrears returns to b0: score P(b0)
    dpd30 the next bucket is b3 or worse: score P(b3..b6)
    """
    ordered = (cur < 7) & (nxt < 7)
    above = np.array([p[r, cur[r] + 1 : 7].sum() for r in range(len(cur))])
    out = {
        "roll": ((nxt > cur)[ordered & (cur < 6)], above[ordered & (cur < 6)]),
        "cure": ((nxt == 0)[(cur >= 1) & (cur < 7)], p[(cur >= 1) & (cur < 7), 0]),
        "dpd30": (((nxt >= 3) & (nxt < 7)), p[:, 3:7].sum(1)),
    }
    return {k: (y.astype(np.int64), s) for k, (y, s) in out.items()}


def main() -> None:
    """Score the test fold's next payment month zero-shot and compare with the training folds' roll-rate matrix."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-uniform", type=int, default=15_000, help="test rows drawn uniformly beside every row in arrears")
    p.add_argument("--max-cases", type=int, default=0, help="cases read per fold, for a quick run; 0 is every case")
    p.add_argument("--train-cases", type=int, default=60_000, help="cases read per training fold for the roll-rate matrix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    t0 = time.time()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    nb = len(BUCKETS)

    counts = np.ones((nb, nb))  # add-one smoothing
    test = pd.DataFrame()
    fold_test = None
    for k in range(N_FOLDS):
        fold = Fold(k, vocab)
        cap = (a.max_cases or None) if k == test_fold else min(a.max_cases or a.train_cases, a.train_cases)
        rows = migration_rows(fold, vocab, k, cap, a.seed + k)
        print(f"  fold {k}: {len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)
        if k == test_fold:
            rng = np.random.default_rng(a.seed)
            uniform = np.zeros(len(rows), dtype=bool)
            uniform[rng.choice(len(rows), min(a.n_uniform, len(rows)), replace=False)] = True
            arrears = (rows["cur"].to_numpy() >= 1) & (rows["cur"].to_numpy() < 7)
            test = rows[uniform | arrears].copy()
            test["uniform"] = uniform[uniform | arrears]
            test = test.reset_index(drop=True)
            fold_test = fold
        else:
            np.add.at(counts, (rows["cur"].to_numpy(), rows["next"].to_numpy()), 1)
    matrix = counts / counts.sum(1, keepdims=True)
    assert fold_test is not None
    print(f"{len(test):,} test rows, {int(test['uniform'].sum()):,} uniform ({time.time() - t0:.0f}s)", flush=True)

    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    p_zs = score(model, fold_test, test, vocab, dev, a.block, a.batch, a.out / "zs_cache.npz")
    assert np.isfinite(p_zs).all()
    cur, nxt = test["cur"].to_numpy(), test["next"].to_numpy()
    preds = {"roll_rate": matrix[cur], "zero_shot": p_zs}
    for j, b in enumerate(BUCKETS):
        test[f"zs_{b}"] = p_zs[:, j]

    uni = test["uniform"].to_numpy()
    results: dict[str, Any] = {"population": {}, "by_bucket": {}, "events": {}, "mean_row": {}}
    for name, pr in preds.items():
        results["population"][name] = multiclass(nxt[uni], pr[uni])
    for j, b in enumerate(BUCKETS):
        sel = cur == j
        if sel.sum() < 50:
            continue
        results["by_bucket"][b] = {"rows": int(sel.sum())} | {name: multiclass(nxt[sel], pr[sel]) for name, pr in preds.items()}
        results["mean_row"][b] = {
            "observed": (np.bincount(nxt[sel], minlength=nb) / sel.sum()).round(4).tolist(),
            **{name: pr[sel].mean(0).round(4).tolist() for name, pr in preds.items()},
        }
    for name, pr in preds.items():
        for ev_name, (y, s) in binary_events(cur, nxt, pr).items():
            if 0 < y.sum() < len(y):
                a_, se = auc(y, s)
                results["events"].setdefault(ev_name, {"rows": len(y), "positives": int(y.sum())})[name] = {"auc": a_, "se": se}

    test.drop(columns=["client_idx"]).to_parquet(a.out / "migration_scores.parquet")
    pd.DataFrame(matrix, index=BUCKETS, columns=BUCKETS).to_csv(a.out / "roll_rate_matrix.csv")
    (a.out / "results.json").write_text(json.dumps({"ckpt": str(a.ckpt), "elapsed_s": round(time.time() - t0)} | results, indent=1))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(results["population"]).T.round(4).to_string())
    print(
        pd.DataFrame(
            {
                b: {f"{n}_{m}": r[n][m] for n in preds for m in ("log_loss", "brier")} | {"rows": r["rows"]}
                for b, r in results["by_bucket"].items()
            }
        )
        .T.round(4)
        .to_string()
    )
    print(json.dumps(results["events"], indent=1))
    print(f"written to {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
