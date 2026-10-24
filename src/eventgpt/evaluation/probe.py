"""Downstream probe: frozen EventGPT embeddings at monthly cuts, a logistic head per target, ROC-AUC and PR-AUC on the test fold.

A row is a (client, month). The cut is the start of the month in Moscow time, and the client's history is
every event strictly before it, which is the MBD protocol: "to predict a purchase in a certain month, take
events strictly before the beginning of this month". The embedding is read off the history decoder after the
last event before the cut, pooled three ways (last, mean, recency).

Two kinds of label share the rows and the embeddings:

    official   target_1..4 from the MBD targets table: the client takes product k during the month. These are
               external labels. No event type marks a purchase (the best single type covers 10% of positives),
               so they cannot be read off the stream.
    event      read off the stream, as the targets of a deployed propensity model usually are. `first_<type>_<subtype>` is that
               transaction kind appearing in the client's history for the first time during the month. The
               kinds are the ones most enriched among product takers: 48/31, 23/49, 26/28 and 48/10 among
               target_2 positives (56x to 262x), 50/19 among target_3 (27x). `churn` is no transaction during
               the month from a client who had one before the cut.

Heads are fitted on the pretraining folds and scored on the test fold, which the model never saw. Positives
are rare (target_2 is 0.05% of rows), so the training rows are every row positive for any label plus a
uniform sample of the rest; the test fold is scored in full. Rows with no history before the cut are kept,
with a zero embedding, since the MBD protocol scores every client every month.

Example: `uv run egpt probe data/runs/smoke-l4/smoke-l4.pt --max-clients 500`
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eventgpt.data.mbd import N_FOLDS, OUT, RAW
from eventgpt.data.vocab import Vocab
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

READOUTS = ("last", "mean", "recency")
RECENCY_EVENTS = 32.0  # e-folding length of the recency-weighted pool, in events
TOKENS_PER_EVENT = 8  # mean on MBD, for the flat decoder's token-level recency pool
CACHE_CHUNK = 40_000  # histories per cached chunk of embeddings, about five minutes on an L4
OFFICIAL = ["target_1", "target_2", "target_3", "target_4"]
METRICS = ("roc_auc", "pr_auc")
FIRST_KINDS = [(48, 31), (23, 49), (26, 28), (48, 10), (50, 19)]
MSK_OFFSET = 3 * 3600
UNDEFINED = -1  # a label the data cannot settle: churn in a month after the last event in the data


# -------------------------------------------------------------------------------
# Rows and labels
# -------------------------------------------------------------------------------
class Fold:
    """The arrays mbd.py wrote for one fold, plus each event's transaction kind read back off the stream."""

    def __init__(self, k: int, vocab: Vocab) -> None:
        d = OUT / f"fold={k}"
        self.tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self.ids = np.load(d / "client_ids.npy")
        self.client_events = np.load(d / "client_events.npy")
        self.event_start = np.load(d / "event_start.npy")
        self.event_time = np.load(d / "event_time.npy")
        self.vocab = vocab
        self.index = {c: i for i, c in enumerate(self.ids)}
        names = {i: t for t, i in vocab.structured.items()}
        name_tok = np.asarray(self.tokens[self.event_start + 1])
        sub_tok = np.asarray(self.tokens[np.minimum(self.event_start + 3, len(self.tokens) - 1)])
        is_trx = np.array([names[int(t)].startswith("name=trx_") for t in range(vocab.size)])[name_tok]
        self.is_trx = is_trx
        kind_of_name = {i: int(n.split("_", 1)[1]) for n, i in vocab.structured.items() if n.startswith("name=trx_") and n[9:].isdigit()}
        kind_of_sub = {
            i: int(n.split("=", 1)[1]) for n, i in vocab.structured.items() if n.startswith("sub=") and n[4:].lstrip("-").isdigit()
        }
        self.etype = np.array([kind_of_name.get(int(t), -1) for t in name_tok]) * is_trx + (~is_trx) * -1
        self.esub = np.array([kind_of_sub.get(int(t), -1) for t in sub_tok]) * is_trx + (~is_trx) * -1

    def end_of_client(self, i: int) -> int:
        """Token offset one past the client's last token."""
        nxt = self.client_events[i + 1]
        return int(self.event_start[nxt] - 1) if nxt < len(self.event_start) else len(self.tokens)


def month_cut(mon: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """UTC epoch seconds of the start of the month named by `mon` and of the month after, in Moscow time."""
    month = pd.to_datetime(mon).to_numpy().astype("datetime64[M]")
    start = month.astype("datetime64[s]").astype(np.int64)
    end = (month + np.timedelta64(1, "M")).astype("datetime64[s]").astype(np.int64)
    return start - MSK_OFFSET, end - MSK_OFFSET


def load_rows(fold: Fold, k: int, max_clients: int | None, seed: int) -> pd.DataFrame:
    """One row per (client, month) of fold k with its cut, history length and every label."""
    t = duckdb.sql(f"select * from read_parquet('{RAW}/targets/fold={k}/*.parquet')").df()
    if max_clients:
        keep = pd.Series(t["client_id"].unique()).sample(min(max_clients, t["client_id"].nunique()), random_state=seed)
        t = t[t["client_id"].isin(set(keep))]
    t = t[t["client_id"].isin(fold.index.keys())].reset_index(drop=True)
    t["cut"], t["cut_end"] = month_cut(t["mon"])
    ci = t["client_id"].map(fold.index).to_numpy()
    a, b = fold.client_events[ci], fold.client_events[ci + 1]
    cut, cut_end = t["cut"].to_numpy(dtype=np.int64), t["cut_end"].to_numpy(dtype=np.int64)
    n_hist = np.empty(len(t), dtype=np.int64)
    n_month = np.empty(len(t), dtype=np.int64)
    for r in range(len(t)):
        times = fold.event_time[a[r] : b[r]]
        n_hist[r] = np.searchsorted(times, cut[r], side="left")
        n_month[r] = np.searchsorted(times, cut_end[r], side="left")
    t["client_idx"], t["n_hist"], t["n_month"] = ci, n_hist, n_month
    add_event_labels(t, fold)
    return t


@functools.cache
def data_end() -> int:
    """UTC epoch seconds of the last event in any fold."""
    return max(int(np.load(OUT / f"fold={k}" / "event_time.npy", mmap_mode="r").max()) for k in range(N_FOLDS))


def mark_undefined(t: pd.DataFrame) -> pd.DataFrame:
    """Set churn to UNDEFINED for rows whose month starts after the last event in the data.

    MBD-mini's transactions end on 31 December 2022, so every client looks churned in January 2023: 99% of
    that month's rows would be positive, and a model that can tell the month of the cut, as the time since a
    client's first event does, would score them without reading any behavior. Applied to rows read back from
    a probe run as well, so outputs written before the rule existed are scored the same way.
    """
    t.loc[t["cut"].to_numpy() > data_end(), "churn"] = UNDEFINED
    return t


def add_event_labels(t: pd.DataFrame, fold: Fold) -> None:
    """Event targets for each row, from the events between the cut and the end of the month."""
    first = {f"first_{x}_{y}": np.zeros(len(t), dtype=np.int64) for x, y in FIRST_KINDS}
    churn = np.zeros(len(t), dtype=np.int64)
    a = fold.client_events[t["client_idx"].to_numpy()]
    for r in range(len(t)):
        s, h, m = a[r], a[r] + t["n_hist"].iat[r], a[r] + t["n_month"].iat[r]
        trx_before = fold.is_trx[s:h].any()
        churn[r] = int(trx_before and not fold.is_trx[h:m].any())
        for x, y in FIRST_KINDS:
            seen = ((fold.etype[s:h] == x) & (fold.esub[s:h] == y)).any()
            now = ((fold.etype[h:m] == x) & (fold.esub[h:m] == y)).any()
            first[f"first_{x}_{y}"][r] = int(now and not seen)
    for name, col in first.items():
        t[name] = col
    t["churn"] = churn
    mark_undefined(t)


def labels() -> list[str]:
    """Every label column the probe scores."""
    return OFFICIAL + [f"first_{x}_{y}" for x, y in FIRST_KINDS] + ["churn"]


# -------------------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------------------
def history_tokens(fold: Fold, ci: int, n: int, block: int, now: np.ndarray | None = None, aligned: bool = False) -> np.ndarray:
    """[USR], then the client's first n events, cut to the most recent `block` tokens at an event boundary.

    The profile event is the client's first, so a truncated window gets [USR] and the profile event put back in
    front: static attributes would otherwise drop out of exactly the clients with the longest histories. `now`,
    when given, is appended after the window and counts against `block`. With `aligned` (a checkpoint trained on
    eventgpt.data.windows.AlignedStream), the first event after the profile carries the gap from the profile event.
    """
    if now is not None:
        return np.concatenate([history_tokens(fold, ci, n, block - len(now), aligned=aligned), now])
    a = fold.client_events[ci]
    usr = int(fold.event_start[a] - 1)
    end = int(fold.event_start[a + n]) if a + n < fold.client_events[ci + 1] else fold.end_of_client(ci)
    if end - usr <= block:
        out, kept, at = np.array(fold.tokens[usr:end]), a + 1, int(fold.event_start[min(a + 1, len(fold.event_start) - 1)]) - usr
    else:
        has_profile = n > 1
        head_end = int(fold.event_start[a + 1]) if has_profile else usr + 1
        budget = block - (head_end - usr)
        starts = fold.event_start[a + 1 : a + n] if has_profile else fold.event_start[a : a + n]
        j = int(np.searchsorted(starts, end - budget, side="left"))
        out, kept, at = np.concatenate([fold.tokens[usr:head_end], fold.tokens[int(starts[j]) : end]]), a + 1 + j, head_end - usr
    if aligned and n > 1:
        out[at + 2] = fold.vocab.dt_tokens(np.array([fold.event_time[kept] - fold.event_time[a]], dtype=np.float64))[0]
    return out


@torch.no_grad()
def _readouts(model: Any, seqs: list[np.ndarray], idx: np.ndarray, dev: torch.device) -> list[np.ndarray]:
    """Last, mean and recency readouts for the sequences seqs[idx], padded into one batch."""
    x = np.zeros((len(idx), max(len(seqs[j]) for j in idx)), dtype=np.int64)
    for r, j in enumerate(idx):
        x[r, : len(seqs[j])] = seqs[j]
    xt = torch.from_numpy(x).to(dev)
    real = xt != 0
    if hasattr(model, "history"):
        h, event, _ = model.history(xt, key_mask=real)
        valid = torch.zeros(h.shape[:2], dtype=torch.float32, device=dev)
        valid.scatter_(1, torch.where(real, event, torch.zeros_like(event)), 1.0)
        valid[:, 0] = 0.0  # event 0 holds no tokens: every sequence starts with [USR], which opens event 1
        span = RECENCY_EVENTS
    else:
        # the flat decoder has no event states, so its readouts run over tokens: the last real token, the
        # mean over tokens, and a recency weight over about as many tokens as RECENCY_EVENTS events hold
        h = model.hidden(xt, doc_mask=True, key_mask=real)
        valid = real.float()
        span = RECENCY_EVENTS * TOKENS_PER_EVENT
    h = h.float()
    rank = torch.cumsum(valid, dim=1)
    n = valid.sum(1, keepdim=True).clamp(min=1.0)
    last = (rank * valid).argmax(1)
    w = torch.exp(-(n - rank) / span) * valid
    rows = torch.arange(len(idx), device=dev)
    return [
        h[rows, last].cpu().numpy(),
        ((h * valid.unsqueeze(-1)).sum(1) / n).cpu().numpy(),
        ((h * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-9)).cpu().numpy(),
    ]


def embed(model: Any, seqs: list[np.ndarray], dev: torch.device, bsz: int) -> dict[str, np.ndarray]:
    """Last, mean and recency-weighted states per sequence: history states for EventGPT, token states for a flat GPT.

    Sequences go in length order, so batch memory grows along the run. A batch that runs out of GPU memory
    is retried at half the size rather than ending the run; the first probe of hier-rope-s42 died that way
    on an L4 with its traceback lost, after training had finished.
    """
    model.eval()
    d = model.cfg.d_model
    out = {r: np.zeros((len(seqs), d), dtype=np.float32) for r in READOUTS}
    order = np.argsort([len(s) for s in seqs])
    i, size, done = 0, bsz, 0
    t0 = time.time()
    while i < len(seqs):
        idx = order[i : i + size]
        try:
            got = _readouts(model, seqs, idx, dev)
        except (torch.OutOfMemoryError, RuntimeError) as err:
            # CUDA raises OutOfMemoryError; MPS raises a plain RuntimeError that says so
            if size == 1 or "out of memory" not in str(err):
                raise
            size //= 2
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            elif dev.type == "mps":
                torch.mps.empty_cache()
            print(f"out of memory at {len(idx)} sequences of up to {len(seqs[idx[-1]])} tokens; batch now {size}", flush=True)
            continue
        for r, e in zip(READOUTS, got, strict=True):
            out[r][idx] = e
        i += len(idx)
        done += 1
        if done % 500 == 0:
            mem = f", GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak" if dev.type == "cuda" else ""
            print(f"  {i:,}/{len(seqs):,} sequences ({time.time() - t0:.0f}s{mem})", flush=True)
    return out


def embed_rows(
    model: Any,
    fold: Fold,
    rows: pd.DataFrame,
    dev: torch.device,
    block: int,
    bsz: int,
    vocab: Vocab | None = None,
    cache: Path | None = None,
) -> dict[str, np.ndarray]:
    """Embeddings for every row; rows of one client with the same history share one forward pass.

    With `cache` given, the forward passes run in chunks of CACHE_CHUNK histories and each chunk's readouts are
    saved there, so a run cut off part way (a spot VM preempted) resumes at the first missing chunk.

    With `vocab` given, each history ends in a "now" event, `[EVT] dt=<gap>`, where the gap runs from the last
    event to the cut. The history decoder then places the readout at the cut time instead of at the last event,
    so two clients with the same history but different silences before the cut get different embeddings. Rows
    of one client with the same history then share a pass only when their cuts match too.
    """
    n_hist = rows["n_hist"].to_numpy()
    ci = rows["client_idx"].to_numpy()
    if vocab is None:
        gap_tok = np.zeros(len(rows), dtype=np.int64)
    else:
        last = fold.event_time[np.maximum(fold.client_events[ci] + n_hist - 1, 0)]
        gap_tok = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
    key = list(zip(ci, n_hist, gap_tok, strict=True))
    uniq = sorted({k for k in key if k[1] > 0})
    pos = {k: i for i, k in enumerate(uniq)}
    evt = vocab.id("[EVT]") if vocab is not None else 0
    aligned = bool(getattr(model.cfg, "aligned", False))
    seqs = [history_tokens(fold, int(c), int(n), block, None if vocab is None else np.array([evt, g]), aligned) for c, n, g in uniq]
    d = model.cfg.d_model
    if cache is None:
        got = embed(model, seqs, dev, bsz) if uniq else {}
    else:
        cache.mkdir(parents=True, exist_ok=True)
        got = {r: np.zeros((len(seqs), d), dtype=np.float32) for r in READOUTS}
        for j in range(0, len(seqs), CACHE_CHUNK):
            f = cache / f"{len(seqs)}_{j}.npz"
            if f.exists():
                part = dict(np.load(f))
            else:
                part = embed(model, seqs[j : j + CACHE_CHUNK], dev, bsz)
                tmp = f.with_suffix(".tmp")
                with tmp.open("wb") as fh:
                    np.savez(fh, **part)
                tmp.rename(f)
            for r in READOUTS:
                got[r][j : j + CACHE_CHUNK] = part[r]
    out = {}
    has = n_hist > 0
    for r in READOUTS:
        e = np.zeros((len(rows), d), dtype=np.float32)
        e[has] = got[r][[pos[k] for k, h in zip(key, has, strict=True) if h]]
        out[r] = e
    return out


# -------------------------------------------------------------------------------
# Heads
# -------------------------------------------------------------------------------
def sample_train(rows: pd.DataFrame, n_neg: int, seed: int) -> pd.DataFrame:
    """Every row positive for any label, plus a uniform sample of the rest, weighted back to its share.

    A row with an UNDEFINED label is kept whole too, at weight 1, which leaves the sample unbiased and draws
    the same rows as before the label existed.
    """
    pos = rows[labels()].to_numpy().any(axis=1)
    neg = rows.index[~pos]
    take = np.random.default_rng(seed).choice(neg, size=min(n_neg, len(neg)), replace=False)
    out = pd.concat([rows[pos], rows.loc[take]])
    out["weight"] = np.where(out.index.isin(rows.index[pos]), 1.0, len(neg) / max(len(take), 1))
    return out


def score(y: np.ndarray, p: np.ndarray | None) -> dict[str, float]:
    """ROC-AUC and PR-AUC (average precision) of scores p against labels y; NaN without scores.

    PR-AUC is reported beside ROC-AUC because positives are rare, from 0.05% to 0.7% of rows: ROC-AUC can be
    high while the top of the ranking is still mostly negatives. A random ranking's PR-AUC is the prevalence,
    which report() prints alongside. Rows with an UNDEFINED label are left out.
    """
    if p is None:
        return {m: float("nan") for m in METRICS}
    ok = y != UNDEFINED
    y, p = y[ok], p[ok]
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def report(results: dict[str, dict[str, dict[str, float]]], test: pd.DataFrame, out_dir: Path, meta: dict[str, object]) -> None:
    """Add the official-target means, write results.json and print one table per metric.

    Args:
        results: column (a readout or a model) -> label -> metric -> value
        test: the test rows, for prevalence and positive counts
        out_dir: where results.json goes
        meta: extra fields for results.json
    """
    for col in results.values():
        col["official_mean"] = {m: float(np.nanmean([col[lab][m] for lab in OFFICIAL])) for m in METRICS}
    prevalence = {lab: float(test.loc[test[lab] != UNDEFINED, lab].mean()) for lab in labels()}
    positives = {lab: int((test[lab] == 1).sum()) for lab in labels()}
    out = {**meta, "metrics": results, "test_prevalence": prevalence, "test_positives": positives}
    (out_dir / "results.json").write_text(json.dumps(out, indent=1))
    for m in METRICS:
        table = pd.DataFrame({col: {lab: v[m] for lab, v in res.items()} for col, res in results.items()})
        if m == "pr_auc":
            table["prevalence"] = pd.Series(prevalence)
            table.loc["official_mean", "prevalence"] = float(np.mean([prevalence[lab] for lab in OFFICIAL]))
        table["test_positives"] = pd.Series(positives)
        print(f"\n{m}\n{table.round(4).to_string()}", flush=True)
    print(f"written to {out_dir}", flush=True)


def fit_score(xtr: np.ndarray, ytr: np.ndarray, wtr: np.ndarray, xte: np.ndarray, yte: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC on the test rows of a standardised logistic head; NaN when a split has one class.

    Rows with an UNDEFINED label are left out of the fit and the score.
    """
    ok = ytr != UNDEFINED
    xtr, ytr, wtr = xtr[ok], ytr[ok], wtr[ok]
    defined = yte[yte != UNDEFINED]
    if ytr.min() == ytr.max() or defined.min() == defined.max():
        return score(yte, None)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    head.fit(xtr, ytr, logisticregression__sample_weight=wtr)
    return score(yte, head.predict_proba(xte)[:, 1])


def main() -> None:
    """Embed the pretraining folds' sampled rows and the full test fold, fit heads, report ROC-AUC and PR-AUC per label."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-neg", type=int, default=60_000, help="negative training rows sampled from the pretraining folds")
    p.add_argument("--max-clients", type=int, default=0, help="clients per fold, for a quick run; 0 is every client")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--now", action="store_true", help="end each history in a now event at the cut (see embed_rows)")
    p.add_argument("--out", type=Path, default=None, help="results directory (default: next to the checkpoint)")
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
        rows = load_rows(fold, k, a.max_clients or None, a.seed)
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

    results: dict[str, dict[str, dict[str, float]]] = {}
    for r in READOUTS:
        xtr, xte = np.concatenate(emb["train"][r]), np.concatenate(emb["test"][r])
        results[r] = {lab: fit_score(xtr, train[lab].to_numpy(), train["weight"].to_numpy(), xte, test[lab].to_numpy()) for lab in labels()}
    meta = {"ckpt": str(a.ckpt), "step": ck.get("step"), "block": a.block, "now": a.now, "elapsed_s": round(time.time() - t0)}
    report(results, test, out_dir, meta)


if __name__ == "__main__":
    main()
