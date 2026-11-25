"""Features from EventGPT's history heads at the decision date of Home Credit applications.

For each application in a table of case_id and fold, the history up to the decision date (point-in-time build) runs
through a pretrained EventGPT, and eventgpt.feature_engineering summarizes what each history head reads from the state at
the last event before the cut, the state the token decoder predicts from:

    late    attention on payment months with a late contract
    dpd30   attention on payment months 30 or more days past due
    pay, tax, app, open, close   attention on each kind of event
    age     attention-weighted months from the event to the decision date
    recent  attention on events within 3 months of the decision date
    focus   entropy of the head's weights, in nats

The [USR] slot and the profile event take part in the attention with every attribute at 0. The output parquet has
case_id, one column per layer, head and summary (L{layer}H{head}_{name}), and base_{name}, each history's share of
events with the attribute, for HeadFeatures.enrichment.

    MBD_OUT=data/home-credit/tokens-pit uv run python -m eventgpt.homecredit.attention_features
        --ckpt data/runs/hc-pit-s42/ckpt/hc-pit-s42.pt --cases cases.parquet --out attention.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.feature_engineering import AttentionRecorder, HeadFeatures
from eventgpt.models.flat import load_checkpoint
from eventgpt.models.hierarchy import EventGPT
from eventgpt.training.train import device

MONTH = 30.44 * 86400
NAMES = ["late", "dpd30", "pay", "tax", "app", "open", "close", "age", "recent"]
DPD30 = {"pdpd=b3", "pdpd=b4", "pdpd=b5", "pdpd=b6"}


def event_attributes(fold: Fold, a: int, n: int, slots: int, last: int, cut: int, inv: dict[int, str]) -> np.ndarray:
    """(slots, 9) attributes of the event in each history slot.

    Slot 0 is empty, slot 1 [USR] and slot 2 the profile (all attributes 0); slot j from 3 to `last` is the client's
    event a + n - 1 - (last - j), so the last slot is its last event before the cut, also when the window is truncated.
    """
    out = np.zeros((slots, len(NAMES)))
    for j in range(3, last + 1):
        e = a + n - 1 - (last - j)
        s = int(fold.event_start[e])
        t = int(fold.event_start[e + 1]) if e + 1 < len(fold.event_start) else len(fold.tokens)
        toks = {inv.get(int(z), "") for z in fold.tokens[s:t]}
        kind = next((x[5:] for x in toks if x.startswith("name=")), "")
        age = (cut - int(fold.event_time[e])) / MONTH
        late = kind == "cb_pay" and any(x.startswith("late=") and x != "late=0" for x in toks)
        dpd30 = kind == "cb_pay" and bool(toks & DPD30)
        kinds = [kind == k for k in ("cb_pay", "tax", "app", "cb_open", "cb_close")]
        out[j] = [late, dpd30, *kinds, age, age <= 3]
    return out


def batches(lengths: list[int], max_rows: int = 16, budget: int = 8192) -> list[list[int]]:
    """Positions grouped into batches whose padded token count stays under the budget; lengths must be sorted."""
    out, i = [], 0
    while i < len(lengths):
        j = i + 1
        while j < len(lengths) and j - i < max_rows and (j + 1 - i) * lengths[j] <= budget:
            j += 1
        out.append(list(range(i, j)))
        i = j
    return out


@torch.no_grad()
def extract(model: EventGPT, vocab: Vocab, cases: pd.DataFrame, dev: torch.device) -> pd.DataFrame:
    """Head features of every case, in the order of the cases table."""
    inv = {i: s for s, i in vocab.structured.items()}
    lab = pd.read_parquet(OUT / "labels.parquet")
    cut_of: dict[str, int] = {str(c): int(t) for c, t in zip(lab["case_id"], lab["cut"], strict=True)}
    feats: list[pd.DataFrame | None] = [None] * len(cases)
    bases = np.zeros((len(cases), len(NAMES)))
    t0 = time.time()
    features = HeadFeatures(NAMES)
    with AttentionRecorder.flat_blocks(model.hist_blocks) as rec:
        for k in sorted(cases["fold"].unique()):
            fold = Fold(int(k), vocab)
            index: dict[str, int] = {str(c): i for i, c in enumerate(fold.ids)}
            info = []
            for r in np.flatnonzero(cases["fold"].to_numpy() == k):
                cid = str(cases["case_id"].iloc[r])
                ci = index[cid]
                a = int(fold.client_events[ci])
                n = int(fold.client_events[ci + 1] - a)
                info.append((r, a, n, cut_of[cid], history_tokens(fold, ci, n, 2048, aligned=True)))
            info.sort(key=lambda z: len(z[4]))
            for group in batches([len(z[4]) for z in info]):
                batch = [info[g] for g in group]
                x = torch.zeros(len(batch), max(len(z[4]) for z in batch), dtype=torch.long)
                for b, z in enumerate(batch):
                    x[b, : len(z[4])] = torch.from_numpy(z[4].astype(np.int64))
                x = x.to(dev)
                model.stages(x, x != 0)
                event, _, _, n_ev = model.segmenter(x, x != 0)
                attn = rec.weights()
                last = np.array([int(event[b, : len(z[4])].max()) for b, z in enumerate(batch)])
                attrs = np.stack(
                    [event_attributes(fold, z[1], z[2], n_ev, int(lst), z[3], inv) for z, lst in zip(batch, last, strict=True)]
                )
                frame = features.transform(attn, last, attrs)
                for b, z in enumerate(batch):
                    feats[z[0]] = frame.iloc[[b]]
                    bases[z[0]] = attrs[b, 3 : last[b] + 1].mean(0) if last[b] >= 3 else 0.0
                del x, event
                if dev.type == "mps":
                    torch.mps.empty_cache()
            print(f"  fold {k}: {int((cases['fold'] == k).sum()):,} cases ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat([f for f in feats if f is not None], ignore_index=True)
    out.insert(0, "case_id", cases["case_id"].to_numpy())
    return pd.concat([out, pd.DataFrame(bases, columns=[f"base_{s}" for s in NAMES])], axis=1)


def main() -> None:
    """Extract the head features of the cases in a table and write them."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True, help="parquet with case_id and fold")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    model = load_checkpoint(str(a.ckpt)).eval().to(dev)
    assert isinstance(model, EventGPT), "head features read EventGPT's history blocks"
    cases = pd.read_parquet(a.cases, columns=["case_id", "fold"])
    out = extract(model, vocab, cases, dev)
    out.to_parquet(a.out, index=False)
    print(f"written {a.out}: {len(out):,} cases, {sum(c.startswith('L') for c in out.columns)} head features", flush=True)


if __name__ == "__main__":
    main()
