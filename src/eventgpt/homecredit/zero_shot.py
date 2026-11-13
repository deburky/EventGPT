"""Zero-shot event targets on Home Credit: score what happens in the last month before the decision, with no labels.

The streams end at the decision date, so the future is made by holding out the last full calendar month before
the decision month: the history is every event before the first day of that month (the cut), and the targets
are read off the events inside it. A case needs at least one real event before the cut to be a row.

Targets, each 1 if the event occurs inside the held-out month:

    late        a bureau payment summary with at least one contract late (cb_pay with late >= 1)
    first_late  the same, for a case with no late month before the cut
    dpd30       a bureau payment summary with a contract more than 30 days past due (pdpd >= b3)
    new_credit  a bureau contract starts (cb_open)
    new_app     the case applies to the lender (app)

Zero-shot scores come from the token decoder, as in eventgpt.evaluation.zero_shot: after the history we append
`[EVT] name=<kind> dt=<g>` with g the bucketed gap from the last event to the cut, read P(name) at [EVT] and the
field distributions after it. The cb_pay fields are n, late, pdpd in that order, so P(late >= 1) sums over the
n values and P(pdpd >= b3) over the n and late values, each a rerun of the token decoder only. The score for
first_late is set to zero for a case that was late before the cut, a rule that needs no labels.

Two baselines use no labels either: `share`, the share of the case's past payment months with the event (or the
count of cb_open and app events in the year before the cut), and `last`, whether the last payment month had it.
For reference, a logistic head on the history state at the cut is fitted on rows from folds 1 to 4.

Test rows are every case positive for a target plus 20,000 uniform rows; each target's ROC-AUC uses its own positives
and the uniform rows, with DeLong standard errors. Rows and scores go to zero_shot_scores.parquet. Example:

    MBD_OUT=data/home-credit/tokens uv run python -m eventgpt.homecredit.zero_shot data/runs/hc-alq-s42/ckpt/hc-alq-s42.pt --batch 64
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.auc_se import influence
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.homecredit.prepare import N_FOLDS
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

TARGETS = ["late", "first_late", "dpd30", "new_credit", "new_app"]
DAY = 86400
YEAR = 365 * DAY
LATE_POS, PDPD_POS = 4, 5  # token offsets inside a cb_pay event: [EVT] name dt n late pdpd pov


# -------------------------------------------------------------------------------
# Events read back off a fold's stream
# -------------------------------------------------------------------------------
class Events:
    """Per event of a fold: kind, whether a payment month was late, and whether it had 30+ days past due."""

    def __init__(self, fold: Fold) -> None:
        v = fold.vocab
        tokens = fold.tokens
        n = len(fold.event_start)
        name = np.asarray(tokens[fold.event_start + 1])
        self.is_pay = name == v.id("name=cb_pay")
        self.is_open = name == v.id("name=cb_open")
        self.is_app = name == v.id("name=app")
        late_tok = np.asarray(tokens[np.minimum(fold.event_start + LATE_POS, len(tokens) - 1)])
        pdpd_tok = np.asarray(tokens[np.minimum(fold.event_start + PDPD_POS, len(tokens) - 1)])
        self.late = self.is_pay & (late_tok != v.id("late=0"))
        dpd30_ids = {v.id(f"pdpd=b{j}") for j in range(3, 7)}
        self.dpd30 = self.is_pay & np.isin(pdpd_tok, list(dpd30_ids))
        self.time = fold.event_time
        self.n = n


def month_cut(dd: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Epoch seconds of the first day of the month before the decision month, and of the decision month."""
    first = pd.to_datetime(dd).dt.to_period("M").dt.to_timestamp()
    end = first.to_numpy().astype("datetime64[s]").astype(np.int64)
    start = (first - pd.offsets.MonthBegin(1)).to_numpy().astype("datetime64[s]").astype(np.int64)
    return start, end


def load_rows(fold: Fold, ev: Events, k: int, max_cases: int | None, seed: int) -> pd.DataFrame:
    """One row per case of fold k with a real event before its cut: cut, history length, labels and baseline scores."""
    t = pd.read_parquet(OUT / "labels.parquet")
    t = t[t["fold"] == k].reset_index(drop=True)
    if max_cases:
        t = t.sample(min(max_cases, len(t)), random_state=seed).reset_index(drop=True)
    t["case_id"] = t["case_id"].astype(str)
    t = t[t["case_id"].isin(fold.index.keys())].reset_index(drop=True)
    ci = t["case_id"].map(fold.index).to_numpy()
    cut, end = month_cut(t["date_decision"])
    a, b = fold.client_events[ci], fold.client_events[ci + 1]
    n_hist = np.empty(len(t), dtype=np.int64)
    n_month = np.empty(len(t), dtype=np.int64)
    labels = {name: np.zeros(len(t), dtype=np.int64) for name in TARGETS}
    base: dict[str, np.ndarray] = {f"{name}_share": np.zeros(len(t)) for name in TARGETS} | {
        f"{name}_last": np.zeros(len(t)) for name in ("late", "dpd30")
    }
    for r in range(len(t)):
        times = ev.time[a[r] : b[r]]
        h = int(np.searchsorted(times, cut[r], side="left"))
        m = int(np.searchsorted(times, end[r], side="left"))
        n_hist[r], n_month[r] = h, m
        s, hh, mm = a[r], a[r] + h, a[r] + m
        late_before = ev.late[s:hh]
        labels["late"][r] = int(ev.late[hh:mm].any())
        labels["first_late"][r] = int(ev.late[hh:mm].any() and not late_before.any())
        labels["dpd30"][r] = int(ev.dpd30[hh:mm].any())
        labels["new_credit"][r] = int(ev.is_open[hh:mm].any())
        labels["new_app"][r] = int(ev.is_app[hh:mm].any())
        pay = ev.is_pay[s:hh]
        n_pay = int(pay.sum())
        if n_pay:
            base["late_share"][r] = late_before.sum() / n_pay
            base["first_late_share"][r] = 0.0 if late_before.any() else late_before.sum() / n_pay
            base["dpd30_share"][r] = ev.dpd30[s:hh].sum() / n_pay
            last_pay = s + int(np.flatnonzero(pay)[-1])
            base["late_last"][r] = float(ev.late[last_pay])
            base["dpd30_last"][r] = float(ev.dpd30[last_pay])
        recent = times[:h] >= cut[r] - YEAR
        base["new_credit_share"][r] = float((ev.is_open[s:hh] & recent).sum())
        base["new_app_share"][r] = float((ev.is_app[s:hh] & recent).sum())
    t["client_idx"], t["cut"], t["cut_end"], t["n_hist"], t["n_month"] = ci, cut, end, n_hist, n_month
    for name, col in labels.items():
        t[name] = col
    for name, col in base.items():
        t[name] = col
    t["seen_late"] = (t["late_share"] > 0).astype(np.int64) | ((t["first_late"] == 0) & (t["late"] == 1)).astype(np.int64)
    return t[t["n_hist"] >= 2].reset_index(drop=True)


def select_test_rows(rows: pd.DataFrame, n_random: int, seed: int) -> pd.DataFrame:
    """Every row positive for a target plus a uniform sample of n_random rows, flagged in `uniform`.

    Each target's ROC-AUC then uses its own positives and the uniform sample, as in the MBD zero-shot protocol:
    positives of other targets are left out rather than counted as negatives, which keeps the AUC unbiased.
    """
    rand = np.random.default_rng(seed).choice(len(rows), min(n_random, len(rows)), replace=False)
    uniform = np.zeros(len(rows), dtype=bool)
    uniform[rand] = True
    pos = rows[TARGETS].to_numpy().any(axis=1)
    out = rows[uniform | pos].copy()
    out["uniform"] = uniform[uniform | pos]
    return out.reset_index(drop=True)


# -------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------
@torch.no_grad()
def next_event_scores(model: Any, seqs: list[np.ndarray], vocab: Vocab, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(len(seqs), 4) zero-shot scores for late, dpd30, new_credit, new_app, and the history state at the cut.

    Every sequence ends in `[EVT] name=cb_pay dt=<g> n=<p> late=<p>` with placeholders p; the token decoder reruns
    with each n value and each (n, late) pair in turn.
    """
    v = vocab
    lengths = np.array([len(s) for s in seqs])
    x = np.zeros((len(seqs), lengths.max()), dtype=np.int64)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = s
    xt = torch.from_numpy(x).to(dev)
    real = xt != 0
    h, event, _, pos_in_event, tok = model.stages(xt, real)
    rows = torch.arange(len(seqs), device=dev)
    at_evt = torch.from_numpy(lengths - 5).to(dev)
    at_dt, at_n, at_late = at_evt + 2, at_evt + 3, at_evt + 4
    emb = h[rows, event[rows, at_evt] - 1].float().cpu().numpy()
    y = model.decode_tokens(xt, h, event, pos_in_event, tok)
    p_name = torch.softmax(model.head(y[rows, at_evt]).float(), -1)
    name_pay, name_open, name_app = v.id("name=cb_pay"), v.id("name=cb_open"), v.id("name=app")
    p_pay = p_name[:, name_pay]
    n_ids = torch.tensor([i for t, i in v.structured.items() if t.startswith("n=")], device=dev)
    late_ids = torch.tensor([i for t, i in v.structured.items() if t.startswith("late=")], device=dev)
    late_zero = v.id("late=0")
    dpd_ids = torch.tensor([v.id(f"pdpd=b{j}") for j in range(3, 7)], device=dev)
    p_n = torch.softmax(model.head(y[rows, at_dt]).float(), -1)[:, n_ids]
    p_n = p_n / p_n.sum(1, keepdim=True).clamp(min=1e-12)
    p_late_any = torch.zeros(len(seqs), device=dev)
    p_dpd30 = torch.zeros(len(seqs), device=dev)
    for j, nid in enumerate(n_ids):
        xt[rows, at_n] = nid
        y = model.decode_tokens(xt, h, event, pos_in_event, tok)
        p_late = torch.softmax(model.head(y[rows, at_n]).float(), -1)[:, late_ids]
        p_late = p_late / p_late.sum(1, keepdim=True).clamp(min=1e-12)
        p_late_any += p_n[:, j] * (1.0 - p_late[:, (late_ids == late_zero).nonzero()[0, 0]])
        for k, lid in enumerate(late_ids):
            xt[rows, at_late] = lid
            y = model.decode_tokens(xt, h, event, pos_in_event, tok)
            p_pdpd = torch.softmax(model.head(y[rows, at_late]).float(), -1)
            p_dpd30 += p_n[:, j] * p_late[:, k] * p_pdpd[:, dpd_ids].sum(1)
    out = torch.stack([p_pay * p_late_any, p_pay * p_dpd30, p_name[:, name_open], p_name[:, name_app]], dim=1)
    return out.cpu().numpy(), emb


def score_rows(
    model: Any, fold: Fold, rows: pd.DataFrame, vocab: Vocab, dev: torch.device, block: int, bsz: int, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-shot scores and history states for every row, in length order, batched."""
    ci, n_hist = rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy()
    last = fold.event_time[fold.client_events[ci] + n_hist - 1]
    gap = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
    head = np.array([vocab.id("[EVT]"), vocab.id("name=cb_pay")])
    tail = np.array([vocab.id("n=1"), vocab.id("late=0")])
    aligned = bool(getattr(model.cfg, "aligned", False))
    seqs = [history_tokens(fold, int(c), int(n), block, np.r_[head, g, tail], aligned) for c, n, g in zip(ci, n_hist, gap, strict=True)]
    order = np.argsort([len(s) for s in seqs])
    zs = np.zeros((len(seqs), 4), dtype=np.float32)
    emb = np.zeros((len(seqs), model.cfg.d_model), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(order), bsz):
        idx = order[i : i + bsz]
        zs[idx], emb[idx] = next_event_scores(model, [seqs[j] for j in idx], vocab, dev)
        if (i // bsz) % 200 == 0:
            print(f"  {label}: {i:,}/{len(order):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return zs, emb


def auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and its DeLong standard error."""
    a, inf = influence(y, p)
    return a, float(np.sqrt((inf**2).sum() * len(inf) / max(len(inf) - 1, 1)))


def main() -> None:
    """Score the test fold's held-out months zero-shot, against label-free baselines and a supervised head."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-cases", type=int, default=0, help="test cases drawn before the row selection, for a quick run; 0 is every case")
    p.add_argument("--n-random", type=int, default=20_000, help="test rows sampled uniformly beside every row positive for a target")
    p.add_argument("--n-train", type=int, default=20_000, help="rows from folds 1 to 4 for the supervised head")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="results directory (default: zero_shot/ next to the checkpoint)")
    a = p.parse_args()
    t0 = time.time()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("vocab_sha256") != vocab.fingerprint():
        raise SystemExit(f"{a.ckpt} was trained on vocabulary {str(ck.get('vocab_sha256'))[:12]}, {OUT} is {vocab.fingerprint()[:12]}")
    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    out_dir = a.out or a.ckpt.parent / "zero_shot"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    embs: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    for k in range(N_FOLDS):
        split = "test" if k == test_fold else "train"
        fold = Fold(k, vocab)
        ev = Events(fold)
        rows = load_rows(fold, ev, k, (a.max_cases or None) if split == "test" else max(a.n_train // (N_FOLDS - 1), 1), a.seed + k)
        if split == "test":
            rows = select_test_rows(rows, a.n_random, a.seed)
        zs, emb = score_rows(model, fold, rows, vocab, dev, a.block, a.batch, f"fold {k}")
        for j, name in enumerate(("late", "dpd30", "new_credit", "new_app")):
            rows[f"zs_{name}"] = zs[:, j]
        rows["zs_first_late"] = np.where(rows["seen_late"] == 1, 0.0, rows["zs_late"])
        frames[split].append(rows.drop(columns=["client_idx"]))
        embs[split].append(emb)
        print(f"fold {k} ({split}): {len(rows):,} rows scored ({time.time() - t0:.0f}s)", flush=True)
    train, test = pd.concat(frames["train"], ignore_index=True), pd.concat(frames["test"], ignore_index=True)
    etr, ete = np.concatenate(embs["train"]), np.concatenate(embs["test"])

    results = []
    for name in TARGETS:
        ev_rows = test["uniform"].to_numpy() | (test[name].to_numpy() == 1)
        sub = test[ev_rows]
        y = sub[name].to_numpy()
        prevalence = float(test.loc[test["uniform"], name].mean())
        row: dict[str, Any] = {"target": name, "rows": len(y), "positives": int(y.sum()), "prevalence": prevalence}
        cols = {"zero_shot": sub[f"zs_{name}"].to_numpy(), "share": sub[f"{name}_share"].to_numpy()}
        if f"{name}_last" in test:
            cols["last"] = sub[f"{name}_last"].to_numpy()
        ytr = train[name].to_numpy()
        if 0 < ytr.sum() < len(ytr):
            head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(etr, ytr)
            pr = head.predict_proba(ete)[:, 1]
            pr = np.where(test["seen_late"] == 1, 0.0, pr) if name == "first_late" else pr
            test[f"head_{name}"] = pr
            cols["head"] = pr[ev_rows]
        for col, pcol in cols.items():
            if 0 < y.sum() < len(y):
                row[col], row[f"{col}_se"] = auc(y, pcol)
        results.append(row)
    res = pd.DataFrame(results)
    test.to_parquet(out_dir / "zero_shot_scores.parquet")
    res.to_csv(out_dir / "zero_shot.csv", index=False)
    (out_dir / "results.json").write_text(
        json.dumps({"ckpt": str(a.ckpt), "rows": len(test), "elapsed_s": round(time.time() - t0), "results": results}, indent=1)
    )
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False), flush=True)
    print(f"written to {out_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
