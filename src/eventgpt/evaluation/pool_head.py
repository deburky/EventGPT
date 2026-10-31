"""A trained attention-pooling head over a client's history states, with the backbone frozen.

The probe reads one vector per row (the last history state, or a fixed mean or recency pool of them) and fits
a logistic head on it. Here each label gets one learned query that attends over the row's history states
instead, and a logistic output on the pooled vector and the last state. The backbone stays frozen, so the
history states of the training rows are computed once and cached.

Histories are built exactly as in eventgpt.evaluation.probe (the probe run's window, the now event when the
probe used one, the aligned rule), so the pooled head and the probe's readouts see the same states. Two limits
keep the cache in memory: the training rows are a subsample of the probe's (every row positive for a label
other than churn, plus a uniform share of the rest, reweighted back), and each row keeps its last
--max-events valid states in float16. The test fold is not cached: it streams through the backbone once after
training and is scored in full.

Beside the pooled head, a logistic head on the last state is refitted on the same training subsample, so the
comparison does not depend on the subsample; the probe's own results.json readouts are reported as well.

Example with R=data/runs/hier-alqp-s42: `uv run python -m eventgpt.evaluation.pool_head $R/hier-alqp-s42.pt --probe-dir $R/probe-now-2k`
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
import torch.nn.functional as F
from torch import nn

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.auc_se import compare
from eventgpt.evaluation.probe import OFFICIAL, UNDEFINED, Fold, fit_score, history_tokens, labels, mark_undefined, score
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device


# -------------------------------------------------------------------------------
# Rows and histories
# -------------------------------------------------------------------------------
def subsample(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Every row positive for a label other than churn, plus a uniform share of the rest, reweighted to keep the sample unbiased."""
    if n <= 0 or n >= len(rows):
        return rows.reset_index(drop=True)
    rare = [lab for lab in labels() if lab != "churn"]
    keep = (rows[rare].to_numpy() == 1).any(axis=1)
    rest = np.flatnonzero(~keep)
    k = max(min(n - int(keep.sum()), len(rest)), 1)
    take = np.random.default_rng(seed).choice(rest, size=k, replace=False)
    out = pd.concat([rows[keep], rows.iloc[take].assign(weight=rows["weight"].iloc[take] * len(rest) / k)])
    return out.reset_index(drop=True)


def histories(rows: pd.DataFrame, folds: dict[int, Fold], vocab: Vocab, block: int, now: bool, aligned: bool) -> list[np.ndarray]:
    """Token sequence of each row's history, as the probe builds it; rows with no history get an empty sequence."""
    evt = vocab.id("[EVT]")
    out = []
    for k, c, n, cut in zip(rows["fold"], rows["client_id"], rows["n_hist"], rows["cut"], strict=True):
        fold = folds[int(k)]
        if n == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        ci = fold.index[c]
        tail = None
        if now:
            last = fold.event_time[fold.client_events[ci] + n - 1]
            tail = np.array([evt, vocab.dt_tokens(np.array([max(cut - last, 0)], dtype=np.float64))[0]])
        out.append(history_tokens(fold, ci, int(n), block, tail, aligned))
    return out


@torch.no_grad()
def states(model: Any, seqs: list[np.ndarray], idx: np.ndarray, dev: torch.device, max_events: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The last max_events valid history states of seqs[idx] (b, max_events, d) and their mask, valid events right-aligned.

    Valid events are those of probe._readouts: events holding a real token, without event 0.
    """
    x = np.zeros((len(idx), max(max(len(seqs[j]) for j in idx), 1)), dtype=np.int64)
    for r, j in enumerate(idx):
        x[r, : len(seqs[j])] = seqs[j]
    xt = torch.from_numpy(x).to(dev)
    real = xt != 0
    h, event, _ = model.history(xt, key_mask=real)
    valid = torch.zeros(h.shape[:2], dtype=torch.bool, device=dev)
    valid.scatter_(1, torch.where(real, event, torch.zeros_like(event)), True)
    valid[:, 0] = False
    # rank from the end: 0 for the last valid event; keep ranks below max_events, right-aligned
    rank = valid.flip(1).cumsum(1).flip(1) - 1
    keep = valid & (rank < max_events)
    slot = (max_events - 1 - rank).clamp(min=0)
    out = torch.zeros(len(idx), max_events, h.shape[-1], dtype=torch.float16, device=dev)
    mask = torch.zeros(len(idx), max_events, dtype=torch.bool, device=dev)
    b, e = torch.nonzero(keep, as_tuple=True)
    out[b, slot[b, e]] = h[b, e].to(torch.float16)
    mask[b, slot[b, e]] = True
    return out, mask


def cache_states(model: Any, seqs: list[np.ndarray], dev: torch.device, bsz: int, max_events: int) -> tuple[torch.Tensor, torch.Tensor]:
    """States and masks of every sequence on the CPU, in input order, computed in length-sorted batches."""
    d = model.cfg.d_model
    out = torch.zeros(len(seqs), max_events, d, dtype=torch.float16)
    mask = torch.zeros(len(seqs), max_events, dtype=torch.bool)
    order = np.argsort([len(s) for s in seqs])
    order = order[[len(seqs[j]) > 0 for j in order]]
    for i in range(0, len(order), bsz):
        idx = order[i : i + bsz]
        s, m = states(model, seqs, idx, dev, max_events)
        out[idx], mask[idx] = s.cpu(), m.cpu()
    return out, mask


# -------------------------------------------------------------------------------
# Head
# -------------------------------------------------------------------------------
class PoolHead(nn.Module):
    """Per label: a learned query over the states, then a logistic output on the pooled vector and the last state."""

    def __init__(self, d: int, n_labels: int, use_last: bool = True) -> None:
        """Zero-initialized queries and weights for n_labels labels over states of width d."""
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_labels, d))
        self.w_pool = nn.Parameter(torch.zeros(n_labels, d))
        self.w_last = nn.Parameter(torch.zeros(n_labels, d)) if use_last else None
        self.bias = nn.Parameter(torch.zeros(n_labels))
        self.scale = d**-0.5

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Logits (b, labels) from states h (b, n, d) and their mask (b, n); a row with no state gets the bias alone."""
        att = torch.einsum("bnd,ld->bln", h, self.query) * self.scale
        att = att.masked_fill(~mask[:, None, :], float("-inf"))
        w = torch.nan_to_num(torch.softmax(att, -1))
        pooled = torch.einsum("bln,bnd->bld", w, h)
        logit = (pooled * self.w_pool).sum(-1) + self.bias
        if self.w_last is not None:
            logit = logit + h[:, -1] @ self.w_last.T  # the last valid state sits in the last slot
        return logit


def loss_fn(logit: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted binary cross-entropy, UNDEFINED labels left out, summed over labels and averaged over weight."""
    ok = (y != UNDEFINED).float()
    bce = F.binary_cross_entropy_with_logits(logit, y.clamp(min=0).float(), reduction="none")
    return (bce * ok * w[:, None]).sum() / w.sum().clamp(min=1e-9)


def train_head(
    h: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, w: torch.Tensor, val: np.ndarray, a: argparse.Namespace, dev: torch.device
) -> PoolHead:
    """Fit the head with AdamW, keeping the epoch with the lowest validation loss; states move to dev batch by batch."""
    head = PoolHead(h.shape[-1], y.shape[1], use_last=not a.no_last).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tr = np.flatnonzero(~val)
    va = np.flatnonzero(val)
    gen = np.random.default_rng(a.seed)

    def batches(ix: np.ndarray) -> Any:
        for i in range(0, len(ix), a.head_batch):
            j = torch.from_numpy(ix[i : i + a.head_batch])
            yield h[j].to(dev).float(), mask[j].to(dev), y[j].to(dev), w[j].to(dev)

    best, best_state, stale = float("inf"), {k: v.clone() for k, v in head.state_dict().items()}, 0
    for epoch in range(a.epochs):
        head.train()
        for hb, mb, yb, wb in batches(gen.permutation(tr)):
            opt.zero_grad()
            loss_fn(head(hb, mb), yb, wb).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            num = sum(float(loss_fn(head(hb, mb), yb, wb)) * float(wb.sum()) for hb, mb, yb, wb in batches(va))
        v = num / float(w[torch.from_numpy(va)].sum())
        print(f"  epoch {epoch + 1}: validation loss {v:.5f}", flush=True)
        if v < best - 1e-6:
            best, best_state, stale = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= a.patience:
                break
    head.load_state_dict(best_state)
    return head


@torch.no_grad()
def score_stream(
    model: Any, head: PoolHead, seqs: list[np.ndarray], dev: torch.device, bsz: int, max_events: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled-head probabilities (rows, labels) and last states (rows, d) of every sequence, without caching the states."""
    head.eval()
    d = model.cfg.d_model
    empty = torch.sigmoid(head(torch.zeros(1, max_events, d, device=dev), torch.zeros(1, max_events, dtype=torch.bool, device=dev)))
    probs = np.tile(empty.cpu().numpy(), (len(seqs), 1))
    last = np.zeros((len(seqs), d), dtype=np.float32)
    order = np.argsort([len(s) for s in seqs])
    order = order[[len(seqs[j]) > 0 for j in order]]
    for i in range(0, len(order), bsz):
        idx = order[i : i + bsz]
        s, m = states(model, seqs, idx, dev, max_events)
        probs[idx] = torch.sigmoid(head(s.float(), m)).cpu().numpy()
        last[idx] = s[:, -1].float().cpu().numpy()
    return probs, last


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    """Fit the pooled head on cached training states, score it and a last-state logistic head on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows, test_rows and results.json")
    p.add_argument("--out", type=Path, default=None, help="json path; default <probe-dir>/pool_head.json")
    p.add_argument("--train-rows", type=int, default=60_000, help="training rows kept (positives of the rare labels always); 0 keeps all")
    p.add_argument("--max-events", type=int, default=128, help="last valid history states kept per row")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3, help="epochs without a lower validation loss before stopping")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--head-batch", type=int, default=512)
    p.add_argument("--batch", type=int, default=32, help="histories per backbone forward pass")
    p.add_argument("--no-last", action="store_true", help="logistic output on the pooled vector alone")
    p.add_argument("--limit", type=int, default=0, help="rows per split, for a smoke test; 0 is every row")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="", help="cpu, cuda or mps; default the training device")
    a = p.parse_args()

    t0 = time.time()
    dev = torch.device(a.device) if a.device else device()
    vocab = Vocab.load(OUT / "vocab")
    meta = json.loads((a.probe_dir / "results.json").read_text())
    block, now = int(meta["block"]), bool(meta.get("now", False))
    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    aligned = bool(getattr(model.cfg, "aligned", False))
    tr = mark_undefined(pd.read_parquet(a.probe_dir / "train_rows.parquet"))
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet")).reset_index(drop=True)
    tr = subsample(tr, a.train_rows, a.seed)
    if a.limit:
        tr = tr.sample(min(a.limit, len(tr)), random_state=a.seed).reset_index(drop=True)
        te = te.sample(min(a.limit, len(te)), random_state=a.seed).reset_index(drop=True)
    folds = {int(k): Fold(int(k), vocab) for k in pd.concat([tr["fold"], te["fold"]]).unique()}
    labs = labels()
    gb = len(tr) * a.max_events * model.cfg.d_model * 2 / 1e9
    print(f"{a.ckpt.name} on {dev}: {len(tr):,} training rows ({gb:.1f} GB of float16 states), {len(te):,} test rows", flush=True)

    h, mask = cache_states(model, histories(tr, folds, vocab, block, now, aligned), dev, a.batch, a.max_events)
    print(f"training states cached ({time.time() - t0:.0f}s)", flush=True)
    clients = tr["client_id"].unique()
    held = np.random.default_rng(a.seed).choice(clients, size=max(len(clients) // 10, 1), replace=False)
    val = tr["client_id"].isin(held).to_numpy()
    y = torch.from_numpy(tr[labs].to_numpy().astype(np.int64))
    w = torch.from_numpy(tr["weight"].to_numpy().astype(np.float32))
    head = train_head(h, mask, y, w, val, a, dev)
    print(f"head trained ({time.time() - t0:.0f}s)", flush=True)

    probs, last_te = score_stream(model, head, histories(te, folds, vocab, block, now, aligned), dev, a.batch, a.max_events)
    print(f"test fold scored ({time.time() - t0:.0f}s)", flush=True)
    last_tr = h[:, -1].float().numpy()
    results: dict[str, dict[str, dict[str, float]]] = {"pool": {}, "last_same_rows": {}}
    for k, lab in enumerate(labs):
        yte = te[lab].to_numpy()
        results["pool"][lab] = score(yte, probs[:, k]) if (yte == 1).any() else score(yte, None)
        results["last_same_rows"][lab] = fit_score(last_tr, tr[lab].to_numpy(), tr["weight"].to_numpy(), last_te, yte)
    for col in results.values():
        col["official_mean"] = {m: float(np.nanmean([col[lab][m] for lab in OFFICIAL])) for m in ("roc_auc", "pr_auc")}
    probe = {r: v for r, v in meta.get("metrics", {}).items()} if not a.limit else {}
    table = pd.DataFrame({c: {lab: v["roc_auc"] for lab, v in res.items()} for c, res in {**results, **probe}.items()})
    print(f"\nroc_auc\n{table.round(4).to_string()}", flush=True)
    out = a.out or a.probe_dir / "pool_head.json"
    # test predictions for eventgpt.evaluation.auc_se, and the pooled head's clustered standard errors
    preds = te[["client_id", "mon"]].copy()
    for k, lab in enumerate(labs):
        preds[lab] = probs[:, k]
    preds.to_parquet(out.with_name("preds_pool.parquet"))
    scorable = [lab for lab in labs if (te[lab] == 1).any() and (te[lab] == 0).any()]
    se = compare(te, {"pool": preds}, scorable)
    settings = {k: v for k, v in vars(a).items() if k not in ("ckpt", "probe_dir", "out")}
    record = {
        "ckpt": str(a.ckpt),
        "settings": settings,
        "train_rows": len(tr),
        "test_rows": len(te),
        "metrics": results,
        "se": {str(lab): {"auc": float(v), "se": float(e)} for lab, v, e in zip(se["label"], se["auc"], se["se"], strict=True)},
        "elapsed_s": round(time.time() - t0),
    }
    out.write_text(json.dumps(record, indent=1))
    print(f"written to {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
