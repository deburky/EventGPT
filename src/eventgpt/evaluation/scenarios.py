"""Conditioned months: insert a hypothetical transaction at the as-of date, sample the month after it, and compare.

For each uniformly sampled test row we sample K months as the rollout does (eventgpt.evaluation.rollout), once
from the real history and once per scenario from the history with one more transaction of kind X/Y appended at the
as-of date. The appended event is the commonest token sequence of that kind in the fold (name, subtype and the
other fields), with its gap token set to the time from the client's last event to the as-of date. Every scenario of
a batch draws from the same random seed, so the differences between scenarios come from the history and not from
the draw.

Per row and scenario, averaged over the K months: the expected number of transactions, the share of months with
none (churn), the probability of each event target in the month (the rollout compensator), and the share of months
in which each transaction kind occurs. The summary gives each scenario's mean change from the unconditioned months
with standard errors clustered by client, and the kinds whose occurrence moves most.

These are the model's associations: clients whose history held the event went on to behave differently. They are
not the effect of causing the event.

Example with R=data/runs/hier-alq-s42:
`uv run python -m eventgpt.evaluation.scenarios $R/hier-alq-s42.pt --probe-dir $R/probe-now-2k --event 50/19 --event 1/12`
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import FIRST_KINDS, Fold, history_tokens, mark_undefined
from eventgpt.evaluation.rollout import MAX_EVENT_TOKENS, pad, path_mean, prefill, sample_month, tables
from eventgpt.evaluation.sequences import clustered_mean
from eventgpt.models.flat import load_checkpoint
from eventgpt.models.hierarchy import EventGPT

TEMPLATE_SAMPLE = 20_000  # events of the kind read to find its commonest token sequence
TOP_KINDS = 10


# -------------------------------------------------------------------------------
# The appended event
# -------------------------------------------------------------------------------
def template(fold: Fold, x: int, y: int) -> np.ndarray:
    """The commonest token sequence of a transaction of kind x/y in the fold; its gap token (position 2) is set later."""
    found = np.flatnonzero((fold.etype == x) & (fold.esub == y))
    if not len(found):
        raise SystemExit(f"no transaction of kind {x}/{y} in the fold")
    found = np.random.default_rng(0).choice(found, min(TEMPLATE_SAMPLE, len(found)), replace=False)
    usr, counts = fold.vocab.id("[USR]"), Counter[tuple[int, ...]]()
    for e in found:
        end = int(fold.event_start[e + 1]) if e + 1 < len(fold.event_start) else len(fold.tokens)
        toks = [int(t) for t in fold.tokens[int(fold.event_start[e]) : end][:MAX_EVENT_TOKENS]]
        if toks[-1] == usr:
            toks = toks[:-1]
        toks[2] = 0
        counts[tuple(toks)] += 1
    return np.array(counts.most_common(1)[0][0], dtype=np.int64)


def occurrence(trace: list[dict[str, np.ndarray]], etype_of_tok: dict[int, int], k: int) -> list[dict[int, float]]:
    """Per row, the share of its k months in which each transaction kind (etype * 1000 + subtype) occurs."""
    inside = np.stack([s["inside"] for s in trace], 1)
    etype = np.vectorize(lambda tok: etype_of_tok.get(int(tok), -1))(np.stack([s["name"] for s in trace], 1))
    kinds = etype.astype(np.int64) * 1000 + np.stack([s["sub"] for s in trace], 1)
    keep = inside & (etype >= 0)
    out = []
    for r in range(len(inside) // k):
        seen: Counter[int] = Counter()
        for p in range(r * k, (r + 1) * k):
            seen.update(set(kinds[p][keep[p]].tolist()))
        out.append({kind: c / k for kind, c in seen.items()})
    return out


def kind_name(kind: int) -> str:
    """X/Y for kind = X * 1000 + Y."""
    x, y = divmod(kind, 1000)
    return f"{x}/{y}"


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    """Sample months with and without each hypothetical event and summarize the changes."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with test_rows")
    p.add_argument("--event", action="append", required=True, help="transaction kind X/Y to append at the as-of date; repeatable")
    p.add_argument("--n-rows", type=int, default=2_000, help="test rows sampled uniformly")
    p.add_argument("--k", type=int, default=16, help="sampled months per row and scenario")
    p.add_argument("--max-events", type=int, default=120)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--block", type=int, default=0, help="history window in tokens; 0 takes the probe run's window")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--out", type=Path, default=None, help="csv path; default <ckpt dir>/scenarios.csv")
    a = p.parse_args()

    t0 = time.time()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = Vocab.load(OUT / "vocab")
    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    if not isinstance(model, EventGPT):
        raise SystemExit("scenarios need an EventGPT checkpoint")
    block = a.block or int(json.loads((a.probe_dir / "results.json").read_text())["block"])
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet"))
    te = te[te["n_hist"] > 1].reset_index(drop=True)
    fold = Fold(int(te["fold"].iloc[0]), vocab)
    rows = te.iloc[np.sort(np.random.default_rng(a.seed).choice(len(te), min(a.n_rows, len(te)), replace=False))].reset_index(drop=True)
    events = {e: template(fold, *map(int, e.split("/"))) for e in a.event}
    print(f"{a.ckpt.name}: {len(rows):,} rows, k={a.k}, scenarios {list(events)}, on {dev}", flush=True)

    ci, n_hist = rows["client_id"].map(fold.index).to_numpy(), rows["n_hist"].to_numpy()
    last = fold.event_time[fold.client_events[ci] + n_hist - 1]
    silence = np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64)
    month = (rows["cut_end"].to_numpy() - rows["cut"].to_numpy()).astype(np.float64)
    aligned = bool(getattr(model.cfg, "aligned", False))
    gap_tok = vocab.dt_tokens(silence)
    tb = tables(vocab, dev)
    etype_of_tok = {i: int(t.removeprefix("name=trx_")) for t, i in vocab.structured.items() if t.startswith("name=trx_")}
    names = ["none", *events]
    per: dict[str, dict[str, np.ndarray]] = {
        s: {"n_trx": np.zeros(len(rows)), "churn": np.zeros(len(rows)), "comp": np.zeros((len(rows), len(FIRST_KINDS)))} for s in names
    }
    occ: dict[str, list[dict[int, float]]] = {s: [{} for _ in range(len(rows))] for s in names}
    for i in range(0, len(rows), a.batch):
        sel = np.arange(i, min(i + a.batch, len(rows)))
        for s in names:
            if s == "none":
                seqs = [history_tokens(fold, int(ci[j]), int(n_hist[j]), block, aligned=aligned) for j in sel]
                s_t = torch.tensor(silence[sel], dtype=torch.float32, device=dev)
            else:
                now = [np.r_[events[s][:2], gap_tok[j], events[s][3:]] for j in sel]
                seqs = [history_tokens(fold, int(ci[j]), int(n_hist[j]), block, now[r], aligned) for r, j in enumerate(sel)]
                s_t = torch.zeros(len(sel), device=dev)
            xt, real = pad(seqs, dev)
            st, _ = prefill(model, xt, real, a.max_events)
            st = st.repeat(a.k)
            m_t = torch.tensor(month[sel], dtype=torch.float32, device=dev)
            gen = torch.Generator(device=dev).manual_seed(a.seed * 100_003 + i)
            trace: list[dict[str, np.ndarray]] = []
            r = sample_month(model, st, tb, s_t.repeat_interleave(a.k), m_t.repeat_interleave(a.k), a.max_events, gen, trace, a.temperature)
            per[s]["n_trx"][sel] = path_mean(r["n_trx"], a.k)
            per[s]["churn"][sel] = 1.0 - path_mean(r["any_trx"], a.k)
            per[s]["comp"][sel] = path_mean(r["comp"], a.k)
            for row, o in zip(sel, occurrence(trace, etype_of_tok, a.k), strict=True):
                occ[s][row] = o
        if (i // a.batch) % 10 == 0:
            print(f"  {sel[-1] + 1:,}/{len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)

    clients = rows["client_id"].to_numpy()
    lines = []
    for s in names:
        cols: dict[str, np.ndarray] = {"n_trx": per[s]["n_trx"], "churn": per[s]["churn"]}
        cols |= {f"p_{x}_{y}": per[s]["comp"][:, k] for k, (x, y) in enumerate(FIRST_KINDS)}
        for metric, v in cols.items():
            line: dict[str, object] = {"scenario": s, "metric": metric}
            line["mean"], line["se"] = clustered_mean(v, clients)
            if s != "none":
                base = per["none"]["n_trx"] if metric == "n_trx" else per["none"]["churn"] if metric == "churn" else None
                if base is None:
                    k = [f"p_{x}_{y}" for x, y in FIRST_KINDS].index(metric)
                    base = per["none"]["comp"][:, k]
                line["change"], line["change_se"] = clustered_mean(v - base, clients)
            lines.append(line)
    summary = pd.DataFrame(lines)
    print(summary.round(4).to_string(index=False))
    moves = []
    all_kinds = sorted({k for s in names for o in occ[s] for k in o})
    base_occ = np.array([[o.get(k, 0.0) for k in all_kinds] for o in occ["none"]])
    for s in events:
        cond = np.array([[o.get(k, 0.0) for k in all_kinds] for o in occ[s]])
        diff = cond - base_occ
        for j in np.argsort(-np.abs(diff.mean(0)))[:TOP_KINDS]:
            mu, se = clustered_mean(diff[:, j], clients)
            moves.append({"scenario": s, "kind": kind_name(all_kinds[j]), "base": base_occ[:, j].mean(), "change": mu, "se": se})
    moves_df = pd.DataFrame(moves)
    print(moves_df.round(4).to_string(index=False))
    dest = a.out or a.ckpt.parent / "scenarios.csv"
    summary.to_csv(dest, index=False)
    moves_df.to_csv(dest.with_name(dest.stem + "_kinds.csv"), index=False)
    print(f"written to {dest} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
