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

With --gaps the token decoder also runs once per transaction name, at the name token, for the distribution of
the next event's gap. Mixed over the names, it gives the probability that the next transaction comes after the
end of the label month, given that none came between the last event and the cut: a churn score with no labels.
It holds for a model trained on transactions alone, whose next event is always a transaction or the profile;
with daily geolocation events the next transaction usually follows a geolocation event, and only sampling a
month of events would answer it. The same pass gives each kind's score divided by the probability that the
next event is a transaction, P(kind | the next event is a transaction).

With --all-kinds MIN, every transaction kind (type, subtype) with at least MIN first-occurrence positives in the
test fold is scored the same way, with the rule for earlier occurrences, and written to zero_shot_all.csv: the
distribution of zero-shot ROC-AUC over kinds, of which the five event targets are one hand-picked part.

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
from eventgpt.data.vocab import MAX_TIME_BUCKET, Vocab, time_bucket
from eventgpt.evaluation.probe import FIRST_KINDS, MSK_OFFSET, UNDEFINED, Fold, history_tokens, mark_undefined, score
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

FIRST = [f"first_{x}_{y}" for x, y in FIRST_KINDS]


# -------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------
@torch.no_grad()
def next_event_scores(
    model: Any, seqs: list[np.ndarray], vocab: Vocab, dev: torch.device, kinds: list[tuple[int, int]] = FIRST_KINDS
) -> np.ndarray:
    """(len(seqs), len(kinds)) probabilities that the next event is each (type, subtype) kind, for one padded batch.

    Every sequence ends in `[EVT] name=<placeholder> dt=<g>`; the placeholder is swapped for each type in turn.
    """
    types = sorted({x for x, _ in kinds})
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
    out = np.zeros((len(seqs), len(kinds)))
    for t in types:
        xt[rows, at_evt + 1] = name_id[t]
        # EventGPT reruns only its token decoder; a flat decoder has no history states to reuse and runs whole
        y = model.decode_tokens(xt, h, event, pos_in_event, tok) if hier else model.hidden(xt, doc_mask=True, key_mask=real)
        logp_name = torch.log_softmax(model.head(y[rows, at_evt]).float(), -1)
        logp_sub = torch.log_softmax(model.head(y[rows, at_dt]).float(), -1)
        for k, (xk, yk) in enumerate(kinds):
            if xk == t:
                out[:, k] = (logp_name[:, name_id[xk]] + logp_sub[:, vocab.id(f"sub={yk}")]).exp().cpu().numpy()
    return out


@torch.no_grad()
def next_gap(model: Any, seqs: list[np.ndarray], vocab: Vocab, dev: torch.device, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """P(next event's name) for each of names, and P(gap bucket, name among names) summed over them, for one padded batch.

    Every sequence ends in `[EVT] name=<placeholder> dt=<g>`; the placeholder is swapped for each name in turn and
    the token decoder's output at the name token gives the distribution of the gap.
    """
    name_ids = [vocab.id(n) for n in names]
    dt_ids = torch.tensor([vocab.id(f"dt={b}") for b in range(MAX_TIME_BUCKET + 1)], device=dev)
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
    at_evt = torch.from_numpy(lengths - 3).to(dev)
    p_name = np.zeros((len(seqs), len(names)))
    p_gap = torch.zeros(len(seqs), MAX_TIME_BUCKET + 1, device=dev)
    for k, nid in enumerate(name_ids):
        xt[rows, at_evt + 1] = nid
        y = model.decode_tokens(xt, h, event, pos_in_event, tok) if hier else model.hidden(xt, doc_mask=True, key_mask=real)
        pn = torch.softmax(model.head(y[rows, at_evt]).float(), -1)[:, nid]
        pg = torch.softmax(model.head(y[rows, at_evt + 1]).float(), -1)[:, dt_ids]
        p_gap += pn[:, None] * pg / pg.sum(1, keepdim=True).clamp(min=1e-12)
        p_name[:, k] = pn.cpu().numpy()
    return p_name, p_gap.cpu().numpy()


def churn_scores(p_gap: np.ndarray, gap: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """P(the next transaction comes after the label month | none before the cut), from the gap distribution.

    gap is the silence from the last event to the cut and cut the start of the label month, both in seconds; the
    month runs to the start of the next calendar month in Moscow time.
    """
    start = pd.to_datetime(cut + MSK_OFFSET, unit="s")
    month = ((start + pd.offsets.MonthBegin(1)) - start).total_seconds().to_numpy()
    survival = np.cumsum(p_gap[:, ::-1], axis=1)[:, ::-1]  # P(gap bucket >= b)
    r = np.arange(len(gap))
    b0, b1 = time_bucket(gap), time_bucket(gap + month)
    return survival[r, b1] / np.maximum(survival[r, b0], 1e-12)


def first_times(fold: Fold, ci: np.ndarray) -> pd.DataFrame:
    """Time of each client's first transaction of each kind, for the clients ci: columns client, etype, esub, first."""
    clients = np.unique(ci)
    start, n = fold.client_events[clients], fold.client_events[clients + 1] - fold.client_events[clients]
    offset = np.repeat(np.cumsum(n) - n, n)
    idx = np.arange(int(n.sum())) - offset + np.repeat(start, n)  # every event of the clients, client by client
    ev = pd.DataFrame({"client": np.repeat(clients, n), "etype": fold.etype[idx], "esub": fold.esub[idx], "first": fold.event_time[idx]})
    return ev[ev["etype"] >= 0].groupby(["client", "etype", "esub"], as_index=False).agg(first=("first", "min"))


def kind_labels(rows: pd.DataFrame, ci: np.ndarray, first: pd.DataFrame, kinds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """(rows, kinds) first-occurrence labels for the month after each cut, and whether the kind came before the cut."""
    pos = np.zeros((len(rows), len(kinds)), dtype=bool)
    seen = np.zeros((len(rows), len(kinds)), dtype=bool)
    col = {k: j for j, k in enumerate(kinds)}
    r = pd.DataFrame({"row": np.arange(len(rows)), "client": ci, "cut": rows["cut"].to_numpy(), "cut_end": rows["cut_end"].to_numpy()})
    m = r.merge(first, on="client")
    m = m[[(x, y) in col for x, y in zip(m["etype"], m["esub"], strict=True)]]
    j = np.array([col[(x, y)] for x, y in zip(m["etype"], m["esub"], strict=True)], dtype=np.int64)
    i = m["row"].to_numpy()
    before = (m["first"] < m["cut"]).to_numpy()
    inside = ((m["first"] >= m["cut"]) & (m["first"] < m["cut_end"])).to_numpy()
    seen[i[before], j[before]] = True
    pos[i[inside], j[inside]] = True
    return pos, seen


def all_kinds(a: argparse.Namespace, model: Any, te: pd.DataFrame, vocab: Vocab, block: int, dev: torch.device) -> None:
    """Zero-shot ROC-AUC for every kind with at least a.all_kinds first-occurrence positives in the test rows."""
    t0 = time.time()
    fold = Fold(int(te["fold"].iloc[0]), vocab)
    ci_all = te["client_id"].map(fold.index).to_numpy()
    first = first_times(fold, ci_all)
    known = {(x, y) for x, y in zip(first["etype"], first["esub"], strict=True)}
    candidates = sorted(k for k in known if f"name=trx_{k[0]}" in vocab.structured and f"sub={k[1]}" in vocab.structured)
    pos_all, _ = kind_labels(te, ci_all, first, candidates)
    counts = pos_all.sum(0)
    kinds = [k for k, n in zip(candidates, counts, strict=True) if n >= a.all_kinds]
    print(f"{len(kinds)} of {len(candidates)} kinds have at least {a.all_kinds} positives ({time.time() - t0:.0f}s)", flush=True)
    keep = [candidates.index(k) for k in kinds]
    rand = np.random.default_rng(a.seed).choice(len(te), a.n_random, replace=False)
    pick = np.union1d(np.flatnonzero(pos_all[:, keep].any(1)), rand)
    in_rand = np.isin(pick, rand)
    rows = te.iloc[pick]
    ci, n_hist = ci_all[pick], rows["n_hist"].to_numpy()
    pos, seen = kind_labels(rows, ci, first, kinds)
    last = fold.event_time[np.maximum(fold.client_events[ci] + n_hist - 1, 0)]
    gap = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
    head = np.array([vocab.id("[EVT]"), vocab.id(f"name=trx_{kinds[0][0]}")])
    seqs = [history_tokens(fold, int(c), int(n), block, np.r_[head, g], model.cfg.aligned) for c, n, g in zip(ci, n_hist, gap, strict=True)]
    order = np.argsort([len(s) for s in seqs])
    zs = np.zeros((len(seqs), len(kinds)))
    print(f"{len(rows):,} rows, {len({x for x, _ in kinds})} types", flush=True)
    for i in range(0, len(order), a.batch):
        idx = order[i : i + a.batch]
        zs[idx] = next_event_scores(model, [seqs[j] for j in idx], vocab, dev, kinds)
        if (i // a.batch) % 200 == 0:
            print(f"  {i:,}/{len(order):,} ({time.time() - t0:.0f}s)", flush=True)
    res = []
    for k, (x, y) in enumerate(kinds):
        ev = in_rand | pos[:, k]
        yk = pos[ev, k].astype(np.int64)
        unseen_auc = score(yk, (zs[:, k] * ~seen[:, k])[ev])["roc_auc"]
        res.append((x, y, int(counts[keep[k]]), int(yk.sum()), unseen_auc, score(yk, zs[ev, k])["roc_auc"]))
    r = pd.DataFrame(res, columns=["type", "subtype", "test_positives", "positives", "zero_shot_unseen", "zero_shot"])
    r["target"] = [(x, y) in FIRST_KINDS for x, y in zip(r["type"], r["subtype"], strict=True)]
    out = a.out or a.ckpt.parent / "zero_shot_all.csv"
    r.to_csv(out, index=False)
    q = r["zero_shot_unseen"].quantile([0.25, 0.5, 0.75]).round(3).tolist()
    print(r.sort_values("zero_shot_unseen").round(4).to_string(index=False))
    print(f"{len(r)} kinds: quartiles {q}; targets {r.loc[r['target'], 'zero_shot_unseen'].round(3).tolist()}")
    print(f"written to {out} ({time.time() - t0:.0f}s)")


def seen_before(fold: Fold, ci: np.ndarray, n_hist: np.ndarray, kinds: list[tuple[int, int]] = FIRST_KINDS) -> np.ndarray:
    """(rows, len(kinds)) True where the client had the kind before the cut."""
    a = fold.client_events[ci]
    seen = np.zeros((len(ci), len(kinds)), dtype=bool)
    for r in range(len(ci)):
        s, e = a[r], a[r] + n_hist[r]
        for k, (x, y) in enumerate(kinds):
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
    p.add_argument("--fold", type=int, default=None, help="score this training fold's probe rows, without the supervised head")
    p.add_argument("--all-kinds", type=int, default=0, help="score every kind with at least this many test positives instead")
    p.add_argument("--gaps", action="store_true", help="also score churn and the kinds given a transaction, from the gap pass")
    a = p.parse_args()

    t0 = time.time()
    block = a.block or int(json.loads((a.probe_dir / "results.json").read_text())["block"])
    vocab = Vocab.load(OUT / "vocab")
    for t in [*(f"name=trx_{x}" for x, _ in FIRST_KINDS), *(f"sub={y}" for _, y in FIRST_KINDS)]:
        if t not in vocab.structured:
            raise SystemExit(f"{t} is not in the vocabulary")
    te = pd.read_parquet(a.probe_dir / "test_rows.parquet")
    if a.all_kinds:
        dev = device()
        model = load_checkpoint(str(a.ckpt)).to(dev).eval()
        all_kinds(a, model, te, vocab, block, dev)
        return
    if a.fold is not None:
        te = pd.read_parquet(a.probe_dir / "train_rows.parquet").query("fold == @a.fold").reset_index(drop=True)
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
    if a.gaps:
        trx_names = [t for t in vocab.structured if t.startswith("name=trx_")]
        p_trx, churn = np.zeros(len(seqs)), np.zeros(len(seqs))
        gap_s = np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64)
        for i in range(0, len(order), a.batch):
            idx = order[i : i + a.batch]
            p_name, p_gap = next_gap(model, [seqs[j] for j in idx], vocab, dev, trx_names)
            p_trx[idx] = p_name.sum(1)
            churn[idx] = churn_scores(p_gap, gap_s[idx], rows["cut"].to_numpy()[idx])
            if (i // a.batch) % 200 == 0:
                print(f"  gaps {i:,}/{len(order):,} ({time.time() - t0:.0f}s)", flush=True)

    tr = pd.read_parquet(a.probe_dir / "train_rows.parquet")
    emb = np.load(a.probe_dir / "embeddings.npz")
    res = []
    heads = np.full((len(rows), len(FIRST)), 0.5)  # the head is fitted on the training folds, so it is scored on the test rows only
    for k, lab in enumerate(FIRST):
        ev = in_rand | (rows[lab].to_numpy() == 1)  # this target's positives and the uniform sample
        y = rows[lab].to_numpy()[ev]
        if a.fold is None:
            head_k = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            head_k.fit(emb["train_last"], tr[lab].to_numpy(), logisticregression__sample_weight=tr["weight"].to_numpy())
            heads[:, k] = head_k.predict_proba(emb["test_last"][pick])[:, 1]
        sup = heads[ev, k]
        z, u = zs[ev, k], unseen[ev, k]
        res.append((lab, int(y.sum()), *(score(y, q)["roc_auc"] for q in (z, z * u, sup, sup * u))))
    cols = ["positives", "zero_shot", "zero_shot_unseen", "supervised_head", "supervised_head_unseen"]
    r = pd.DataFrame(res, columns=["label", *cols]).set_index("label")
    if a.gaps:
        given_trx = zs / np.maximum(p_trx, 1e-12)[:, None]
        r["zero_shot_given_trx_unseen"] = [
            score(rows[lab].to_numpy()[ev], (given_trx[:, k] * unseen[:, k])[ev])["roc_auc"]
            for k, lab in enumerate(FIRST)
            for ev in [in_rand | (rows[lab].to_numpy() == 1)]
        ]
        # churn on the uniform sample, without the rows whose month the data cannot settle
        y_churn = mark_undefined(rows.copy())["churn"].to_numpy()
        ev = in_rand & (y_churn != UNDEFINED)
        head_c = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        tr_c = mark_undefined(tr.copy())
        ok = (tr_c["churn"] != UNDEFINED).to_numpy()
        head_c.fit(emb["train_last"][ok], tr_c["churn"].to_numpy()[ok], logisticregression__sample_weight=tr_c["weight"].to_numpy()[ok])
        sup_c = head_c.predict_proba(emb["test_last"][pick])[:, 1]
        r.loc["churn", ["positives", "zero_shot", "supervised_head"]] = [
            int(y_churn[ev].sum()),
            score(y_churn[ev], churn[ev])["roc_auc"],
            score(y_churn[ev], sup_c[ev])["roc_auc"],
        ]
    print(r.round(4).to_string())
    out = a.out or a.ckpt.parent / "zero_shot.csv"
    if a.save_scores:
        saved = rows[["client_id", "mon", "cut", "n_hist", *FIRST]].assign(in_rand=in_rand)
        for k, lab in enumerate(FIRST):
            saved[f"zs_{lab}"], saved[f"unseen_{lab}"], saved[f"head_{lab}"] = zs[:, k], unseen[:, k], heads[:, k]
        if a.gaps:
            saved["p_trx"], saved["zs_churn"] = p_trx, churn
        saved.to_parquet(out.with_name("zero_shot_scores.parquet"))
    r.to_csv(out)
    print(f"written to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
