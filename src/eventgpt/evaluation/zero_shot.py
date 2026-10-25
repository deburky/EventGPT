"""Zero-shot event targets: score first_<type>_<subtype> from a pretrained model's next-event distribution, no labels, no head.

After the history at a cut (the probe's window) we append the start of a next event, `[EVT] name=trx_X dt=<g>`,
with g the bucketed gap from the last event to the cut. The token decoder gives P(name=trx_X) at [EVT] and
P(sub=Y | name=trx_X, dt=g) at the dt token, and the score is their product: the probability that the next event
is of kind (X, Y). A client who already had the kind before the cut cannot be positive under the label's
definition, so that score is also reported with those clients set to zero; the rule needs no labels.

For EventGPT the history stages run once per row and only the token decoder reruns per type X; a flat decoder
runs whole once per type. Rows are every test-fold row
positive for an event target plus a uniform sample, read from a probe run's output directory, so the zero-shot
scores and the probe's logistic head (refitted on its saved embeddings) are compared on the same rows. Each
target's ROC-AUC uses its own positives and the uniform sample, which leaves it unbiased.

Example with R=data/runs/hier-rope-s42: `uv run egpt zero-shot $R/hier-rope-s42.pt --probe-dir $R/probe-now-2k`
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
from eventgpt.evaluation.probe import FIRST_KINDS, Fold, history_tokens, score
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

FIRST = [f"first_{x}_{y}" for x, y in FIRST_KINDS]


# -------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------
@torch.no_grad()
def next_event_scores(model: Any, seqs: list[np.ndarray], vocab: Vocab, dev: torch.device) -> np.ndarray:
    """(len(seqs), len(FIRST_KINDS)) probabilities that the next event is each kind, for one padded batch.

    Every sequence ends in `[EVT] name=<placeholder> dt=<g>`; the placeholder is swapped for each type in turn.
    """
    types = sorted({x for x, _ in FIRST_KINDS})
    name_id = {x: vocab.id(f"name=trx_{x}") for x in types}
    lengths = np.array([len(s) for s in seqs])
    x = np.zeros((len(seqs), lengths.max()), dtype=np.int64)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = s
    xt = torch.from_numpy(x).to(dev)
    real = xt != 0
    hier = hasattr(model, "stages")
    if hier:
        h, event, _, pos_in_event, tok = model.stages(xt, real)
    rows = torch.arange(len(seqs), device=dev)
    at_evt, at_dt = torch.from_numpy(lengths - 3).to(dev), torch.from_numpy(lengths - 1).to(dev)
    out = np.zeros((len(seqs), len(FIRST_KINDS)))
    for t in types:
        xt[rows, at_evt + 1] = name_id[t]
        # EventGPT reruns only its token decoder; a flat decoder has no history states to reuse and runs whole
        y = model.decode_tokens(xt, h, event, pos_in_event, tok) if hier else model.hidden(xt, doc_mask=True, key_mask=real)
        logp_name = torch.log_softmax(model.head(y[rows, at_evt]).float(), -1)
        logp_sub = torch.log_softmax(model.head(y[rows, at_dt]).float(), -1)
        for k, (xk, yk) in enumerate(FIRST_KINDS):
            if xk == t:
                out[:, k] = (logp_name[:, name_id[xk]] + logp_sub[:, vocab.id(f"sub={yk}")]).exp().cpu().numpy()
    return out


def seen_before(fold: Fold, ci: np.ndarray, n_hist: np.ndarray) -> np.ndarray:
    """(rows, len(FIRST_KINDS)) True where the client had the kind before the cut."""
    a = fold.client_events[ci]
    seen = np.zeros((len(ci), len(FIRST_KINDS)), dtype=bool)
    for r in range(len(ci)):
        s, e = a[r], a[r] + n_hist[r]
        for k, (x, y) in enumerate(FIRST_KINDS):
            seen[r, k] = ((fold.etype[s:e] == x) & (fold.esub[s:e] == y)).any()
    return seen


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    """Zero-shot and supervised ROC-AUC on the event targets, on the same sampled test rows."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with test_rows, train_rows and embeddings")
    p.add_argument("--n-random", type=int, default=20_000, help="test rows sampled uniformly beside the positives")
    p.add_argument("--block", type=int, default=0, help="history window in tokens; 0 takes the probe run's window")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="csv path; default <ckpt dir>/zero_shot.csv")
    p.add_argument("--save-scores", action="store_true", help="also write each row's scores to zero_shot_scores.parquet")
    a = p.parse_args()

    t0 = time.time()
    block = a.block or int(json.loads((a.probe_dir / "results.json").read_text())["block"])
    vocab = Vocab.load(OUT / "vocab")
    for t in [*(f"name=trx_{x}" for x, _ in FIRST_KINDS), *(f"sub={y}" for _, y in FIRST_KINDS)]:
        if t not in vocab.structured:
            raise SystemExit(f"{t} is not in the vocabulary")
    te = pd.read_parquet(a.probe_dir / "test_rows.parquet")
    rand = np.random.default_rng(a.seed).choice(len(te), a.n_random, replace=False)
    pick = np.union1d(np.flatnonzero(te[FIRST].to_numpy().any(1)), rand)
    in_rand = np.isin(pick, rand)
    rows = te.iloc[pick]
    fold = Fold(int(rows["fold"].iloc[0]), vocab)
    dev = device()
    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    print(f"{a.ckpt.name}: {len(rows):,} rows on {dev}", flush=True)

    # each history ends in the start of a next event, at the gap from its last event to the cut
    ci, n_hist = rows["client_id"].map(fold.index).to_numpy(), rows["n_hist"].to_numpy()
    last = fold.event_time[np.maximum(fold.client_events[ci] + n_hist - 1, 0)]
    gap = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
    head = np.array([vocab.id("[EVT]"), vocab.id(f"name=trx_{FIRST_KINDS[0][0]}")])
    seqs = [history_tokens(fold, int(c), int(n), block, np.r_[head, g], model.cfg.aligned) for c, n, g in zip(ci, n_hist, gap, strict=True)]
    order = np.argsort([len(s) for s in seqs])
    zs = np.zeros((len(seqs), len(FIRST_KINDS)))
    for i in range(0, len(order), a.batch):
        idx = order[i : i + a.batch]
        zs[idx] = next_event_scores(model, [seqs[j] for j in idx], vocab, dev)
        if (i // a.batch) % 200 == 0:
            print(f"  {i:,}/{len(order):,} ({time.time() - t0:.0f}s)", flush=True)
    unseen = ~seen_before(fold, ci, n_hist)

    tr = pd.read_parquet(a.probe_dir / "train_rows.parquet")
    emb = np.load(a.probe_dir / "embeddings.npz")
    res = []
    for k, lab in enumerate(FIRST):
        ev = in_rand | (rows[lab].to_numpy() == 1)  # this target's positives and the uniform sample
        y = rows[lab].to_numpy()[ev]
        head_k = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        head_k.fit(emb["train_last"], tr[lab].to_numpy(), logisticregression__sample_weight=tr["weight"].to_numpy())
        sup = head_k.predict_proba(emb["test_last"][pick][ev])[:, 1]
        z, u = zs[ev, k], unseen[ev, k]
        res.append((lab, int(y.sum()), *(score(y, q)["roc_auc"] for q in (z, z * u, sup, sup * u))))
    cols = ["positives", "zero_shot", "zero_shot_unseen", "supervised_head", "supervised_head_unseen"]
    r = pd.DataFrame(res, columns=["label", *cols]).set_index("label")
    print(r.round(4).to_string())
    out = a.out or a.ckpt.parent / "zero_shot.csv"
    if a.save_scores:
        saved = rows[["client_id", "mon", "cut", "n_hist", *FIRST]].assign(in_rand=in_rand)
        for k, lab in enumerate(FIRST):
            saved[f"zs_{lab}"], saved[f"unseen_{lab}"] = zs[:, k], unseen[:, k]
        saved.to_parquet(out.with_name("zero_shot_scores.parquet"))
    r.to_csv(out)
    print(f"written to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
