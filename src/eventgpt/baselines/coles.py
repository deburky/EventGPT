"""CoLES baseline (Babaev et al. 2022) with the MBD benchmark's transaction configuration, scored on the probe's rows.

CoLES trains a sequence encoder contrastively: random contiguous slices of one client's transactions are
positives, slices of other clients negatives. The MBD paper (Mollaev et al. 2024, arXiv:2409.17587) reports it
as a transactions-only baseline; this module reimplements pytorch-lifestream 0.6.0 with the settings of the
paper's own config (github.com/Dzhambo/MBD, scenario_mbd/conf/trx_coles.yaml):

    encoder    ptls TrxEncoder: an embedding per categorical field (sizes in EMBEDDINGS, padding index 0,
               Gaussian noise of scale 0.003 in training) and log amount, sign(x) log(1 + |x|), through a
               BatchNorm, concatenated (181 wide); ptls RnnSeqEncoder: a one-layer GRU of 256 with a trainable
               starting state, the last step's state as the embedding
    slicing    SampleSlices(split_count=5, cnt_min=32, cnt_max=180) on the last 4,096 transactions of each
               client with at least 32
    loss       ContrastiveLoss(margin=0.5) with HardNegativePairSelector(neg_count=5) on L2-normalised
               embeddings, the CoLESModule defaults
    training   batches of 512 clients, AdamW (lr 1e-3, weight decay 0.01), StepLR(step 3, gamma 0.9025) per
               epoch, gradient clipping 0.5, 15 epochs, the last epoch's weights

The encoder trains on the pretraining folds only (every fold but the test fold), with VAL_SHARE of those
clients held out to report BatchRecallTopK(K=4, cosine) per epoch. The MBD dataset ships its fields already
integer-coded; MBD-mini has raw codes, so each field is coded by frequency rank on the pretraining folds
(1 the most frequent, a missing value a code of its own) and clipped to the field's size as ptls clips.

Each probe row is then embedded point in time: the GRU state after the client's last transaction strictly
before the cut, over at most the last 4,096 such transactions. A row with no transaction before the cut gets
a zero embedding, as the probe does. Heads are the probe's: logistic regression (C=0.1) on standardised
embeddings with the training rows' weights, scored on every test row, with client-clustered SEs and paired
differences from eventgpt.evaluation.auc_se.

Example: `uv run egpt coles --probe-dir data/runs/hier-alq-s42/probe-now-2k --out data/runs/coles-s42`
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from eventgpt.data.mbd import OUT, RAW
from eventgpt.evaluation.auc_se import compare
from eventgpt.evaluation.probe import OFFICIAL, UNDEFINED, labels, mark_undefined, score
from eventgpt.training.train import device

EMBEDDINGS = {  # field -> (codes including padding 0, embedding size), from scenario_mbd/conf/trx_coles.yaml
    "event_type": (58, 24),
    "event_subtype": (59, 24),
    "src_type11": (85, 24),
    "src_type12": (349, 24),
    "dst_type11": (84, 24),
    "dst_type12": (417, 12),
    "src_type22": (90, 24),
    "src_type32": (91, 24),
}
HIDDEN = 256
NOISE = 0.003
SPLIT_COUNT, CNT_MIN, CNT_MAX = 5, 32, 180
MIN_LEN, MAX_LEN = 32, 4096
BATCH = 512
VALID_BATCH = 128
EPOCHS = 15
LR, WEIGHT_DECAY = 1e-3, 0.01
STEP_SIZE, GAMMA = 3, 0.9025
CLIP = 0.5
MARGIN, NEG_COUNT = 0.5, 5
RECALL_K = 4
VAL_SHARE = 0.02
EMBED_TOKENS = 1 << 18  # transactions per inference batch


# -------------------------------------------------------------------------------
# Transactions
# -------------------------------------------------------------------------------
@dataclass
class Trx:
    """Every client's transactions, client after client, with the fields coded for the encoder."""

    fold: np.ndarray  # fold per client
    ids: np.ndarray  # client id per client
    offsets: np.ndarray  # transaction range per client, length n_clients + 1
    time_us: np.ndarray  # UTC epoch microseconds per transaction
    amount: np.ndarray  # sign(x) log(1 + |x|) per transaction
    cats: np.ndarray  # (n, len(EMBEDDINGS)) codes per transaction

    def index(self) -> dict[tuple[int, str], int]:
        """Client position per (fold, client id)."""
        return {(int(f), str(c)): i for i, (f, c) in enumerate(zip(self.fold, self.ids, strict=True))}


def load_trx(test_fold: int) -> Trx:
    """Read the transactions of every fold, in client and time order, and code the fields on the pretraining folds."""
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"create view trx as select * from read_parquet('{RAW}/detail/trx/*/*.parquet', hive_partitioning=true)")
    order = "fold, client_id, event_time, event_type, event_subtype, amount"
    cl = con.sql("select fold, client_id, count(*) as n from trx group by fold, client_id order by fold, client_id").df()
    cols = ", ".join(f"coalesce({c}, -1)::integer as {c}" for c in EMBEDDINGS)
    d = con.sql(f"select epoch_us(event_time) as t, amount, {cols} from trx order by {order}").fetchnumpy()
    n = cl["n"].to_numpy()
    fold = cl["fold"].to_numpy()
    train = np.repeat(fold != test_fold, n)
    cats = np.empty((len(train), len(EMBEDDINGS)), dtype=np.int16)
    for j, (name, (size, _)) in enumerate(EMBEDDINGS.items()):
        v = np.asarray(d[name], dtype=np.int64)
        vals, cnt = np.unique(v[train], return_counts=True)
        rank = np.empty(len(vals), dtype=np.int64)
        rank[np.lexsort((vals, -cnt))] = np.arange(1, len(vals) + 1)
        pos = np.minimum(np.searchsorted(vals, v), len(vals) - 1)
        cats[:, j] = np.minimum(np.where(vals[pos] == v, rank[pos], size - 1), size - 1)
    amount = np.asarray(d["amount"], dtype=np.float64)
    print(f"{len(train):,} transactions of {len(cl):,} clients ({time.time() - t0:.0f}s)", flush=True)
    return Trx(
        fold=fold,
        ids=cl["client_id"].to_numpy(),
        offsets=np.r_[0, np.cumsum(n)],
        time_us=np.asarray(d["t"], dtype=np.int64),
        amount=(np.sign(amount) * np.log1p(np.abs(amount))).astype(np.float32),
        cats=cats,
    )


def padded(trx: Trx, starts: np.ndarray, lens: np.ndarray, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Codes, amounts and lengths of the transaction ranges [start, start + len), right-padded with zeros."""
    steps = np.arange(int(lens.max()))
    real = steps[None, :] < lens[:, None]
    idx = np.where(real, starts[:, None] + steps[None, :], 0)
    cats = np.where(real[..., None], trx.cats[idx], 0).astype(np.int64)
    amount = np.where(real, trx.amount[idx], 0.0).astype(np.float32)
    return torch.from_numpy(cats).to(dev), torch.from_numpy(amount).to(dev), torch.from_numpy(lens.astype(np.int64)).to(dev)


def sample_slices(trx: Trx, clients: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five random slices of each client's last MAX_LEN transactions as ptls SampleSlices cuts them, with client labels."""
    starts, lens = [], []
    for c in clients:
        b = int(trx.offsets[c + 1])
        a = max(int(trx.offsets[c]), b - MAX_LEN)
        n = b - a
        if n <= CNT_MIN:
            s, ln = np.zeros(SPLIT_COUNT, dtype=np.int64), np.full(SPLIT_COUNT, n)
        else:
            ln = rng.integers(CNT_MIN, min(CNT_MAX, n) + 1, SPLIT_COUNT)
            s = (rng.random(SPLIT_COUNT) * ((n - ln).clip(0, None) + 1 - 1e-9)).astype(np.int64)
        starts.append(a + s)
        lens.append(ln)
    return np.concatenate(starts), np.concatenate(lens), np.repeat(np.arange(len(clients)), SPLIT_COUNT)


# -------------------------------------------------------------------------------
# Model and loss
# -------------------------------------------------------------------------------
class CoLES(nn.Module):
    """ptls TrxEncoder and RnnSeqEncoder (GRU, static trainable starter, last-step readout)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(n, d, padding_idx=0) for n, d in EMBEDDINGS.values())
        self.amount_bn = nn.BatchNorm1d(1)
        width = sum(d for _, d in EMBEDDINGS.values()) + 1
        self.rnn = nn.GRU(width, HIDDEN, batch_first=True)
        self.starter = nn.Parameter(torch.randn(1, 1, HIDDEN))

    def states(self, cats: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
        """GRU state after every transaction (b, t, HIDDEN)."""
        parts = []
        for j, e in enumerate(self.emb):
            x = e(cats[..., j])
            if self.training:
                # ptls NoisyEmbedding: one noise vector per forward pass, shared by every position
                x = x + NOISE * torch.randn(x.shape[-1], device=x.device)
            parts.append(x)
        b, t = amount.shape
        # ptls RBatchNorm normalises over every position of the padded batch, padding included
        parts.append(self.amount_bn(amount.reshape(b * t, 1)).reshape(b, t, 1))
        h0 = torch.tanh(self.starter).expand(1, b, HIDDEN).contiguous()
        out, _ = self.rnn(torch.cat(parts, dim=-1), h0)
        return out

    def forward(self, cats: torch.Tensor, amount: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """State after each sequence's last real transaction (b, HIDDEN)."""
        return self.states(cats, amount)[torch.arange(len(lens), device=lens.device), lens - 1]


def contrastive_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Contrastive loss as in ptls: every positive pair, and the NEG_COUNT hardest negatives per row (HardNegativePairSelector)."""
    n = len(y)
    same = y[:, None] == y[None, :]
    pos = torch.triu(same.int(), diagonal=1).nonzero()
    d = torch.cdist(z.detach(), z.detach())
    upper = int((2 * n) ** 0.5) + 1
    hard = ((upper - d) * (~same).to(d.dtype)).topk(NEG_COUNT, dim=0).indices
    neg = torch.stack([torch.arange(n, device=z.device).repeat(NEG_COUNT), hard.reshape(-1)], dim=1)
    pos_loss = F.pairwise_distance(z[pos[:, 0]], z[pos[:, 1]]).pow(2)
    neg_loss = F.relu(MARGIN - F.pairwise_distance(z[neg[:, 0]], z[neg[:, 1]])).pow(2)
    return torch.cat([pos_loss, neg_loss]).sum()


def recall_top_k(z: torch.Tensor, y: torch.Tensor) -> float:
    """Share of each slice's K nearest others (cosine) that come from its own client, ptls BatchRecallTopK."""
    zn = F.normalize(z, dim=1)
    sim = zn @ zn.T
    sim.fill_diagonal_(-float("inf"))
    top = sim.topk(RECALL_K, dim=1).indices
    return float((y[top] == y[:, None]).float().sum(1).div(SPLIT_COUNT - 1).mean())


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
def train(trx: Trx, test_fold: int, dev: torch.device, out: Path, seed: int, epochs: int, max_clients: int) -> tuple[CoLES, float]:
    """Train CoLES on the pretraining folds' clients, resuming from out/coles.pt; the model and training seconds."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lens = np.diff(trx.offsets)
    clients = np.flatnonzero((trx.fold != test_fold) & (lens >= MIN_LEN))
    if max_clients:
        clients = rng.choice(clients, size=min(max_clients, len(clients)), replace=False)
    held = rng.random(len(clients)) < VAL_SHARE
    fit, valid = clients[~held], clients[held]
    print(f"CoLES: {len(fit):,} training clients, {len(valid):,} held out, on {dev}", flush=True)

    model = CoLES().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    ckpt = out / "coles.pt"
    start, spent, history = 0, 0.0, []
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, spent, history = state["epoch"], state["train_s"], state["history"]
        rng = np.random.default_rng([seed, start])
        print(f"resumed after epoch {start} ({spent:.0f}s)", flush=True)
    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        order = rng.permutation(fit)
        losses = []
        for i in range(0, len(order), BATCH):
            s, ln, y = sample_slices(trx, order[i : i + BATCH], rng)
            cats, amount, lt = padded(trx, s, ln, dev)
            loss = contrastive_loss(F.normalize(model(cats, amount, lt), dim=1), torch.from_numpy(y).to(dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            losses.append(loss.item())
        sched.step()
        recall = validate(model, trx, valid, dev, np.random.default_rng(seed + 1))
        spent += time.time() - t0
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "recall_top4": recall, "seconds": round(time.time() - t0)})
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.2f}, valid recall@4 {recall:.3f} ({time.time() - t0:.0f}s)", flush=True)
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1}
        torch.save({**state, "train_s": spent, "history": history}, ckpt.with_suffix(".tmp"))
        ckpt.with_suffix(".tmp").rename(ckpt)
    (out / "train_history.json").write_text(json.dumps(history, indent=1))
    return model, spent


@torch.no_grad()
def validate(model: CoLES, trx: Trx, clients: np.ndarray, dev: torch.device, rng: np.random.Generator) -> float:
    """Mean BatchRecallTopK over batches of VALID_BATCH held-out clients."""
    if len(clients) == 0:
        return float("nan")
    model.eval()
    got = []
    for i in range(0, len(clients), VALID_BATCH):
        s, ln, y = sample_slices(trx, clients[i : i + VALID_BATCH], rng)
        z = model(*padded(trx, s, ln, dev))
        got.append(recall_top_k(z, torch.from_numpy(y).to(dev)))
    return float(np.mean(got))


# -------------------------------------------------------------------------------
# Embedding the probe rows
# -------------------------------------------------------------------------------
def history_length(trx: Trx, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Client position (-1 without transactions) and transactions strictly before the cut, per row."""
    index = trx.index()
    ci = np.array([index.get((int(f), str(c)), -1) for f, c in zip(rows["fold"], rows["client_id"], strict=True)])
    n = np.zeros(len(rows), dtype=np.int64)
    cut_us = rows["cut"].to_numpy(dtype=np.int64) * 1_000_000
    for r in np.flatnonzero(ci >= 0):
        a, b = trx.offsets[ci[r]], trx.offsets[ci[r] + 1]
        n[r] = np.searchsorted(trx.time_us[a:b], cut_us[r], side="left")
    return ci, n


@torch.no_grad()
def embed_rows(model: CoLES, trx: Trx, rows: pd.DataFrame, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """GRU state after each row's last transaction before the cut, over at most MAX_LEN of them; and the counts.

    The GRU reads forward only, so one pass over a client's first transactions gives the state after each of
    them: a history of at most MAX_LEN transactions is read off that pass. A longer one is cut to its last
    MAX_LEN and gets a pass of its own.
    """
    model.eval()
    ci, n = history_length(trx, rows)
    out = np.zeros((len(rows), HIDDEN), dtype=np.float32)
    # one job per pass: (start, length, rows, step to read per row)
    short = (n > 0) & (n <= MAX_LEN)
    jobs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    r_short = np.flatnonzero(short)[np.argsort(ci[short], kind="stable")]
    for g in np.split(r_short, np.flatnonzero(np.diff(ci[r_short])) + 1):
        if len(g):
            jobs.append((int(trx.offsets[ci[g[0]]]), int(n[g].max()), g, n[g] - 1))
    for r in np.flatnonzero(n > MAX_LEN):
        jobs.append((int(trx.offsets[ci[r]] + n[r] - MAX_LEN), MAX_LEN, np.array([r]), np.array([MAX_LEN - 1])))
    jobs.sort(key=lambda j: j[1])
    t0, i, done = time.time(), 0, 0
    while i < len(jobs):
        # jobs run shortest first, so a batch is as many as fit EMBED_TOKENS at its longest job's length
        j = i + 1
        while j < len(jobs) and (j - i + 1) * jobs[j][1] <= EMBED_TOKENS:
            j += 1
        batch = jobs[i:j]
        size = len(batch)
        cats, amount, _ = padded(trx, np.array([j[0] for j in batch]), np.array([j[1] for j in batch]), dev)
        h = model.states(cats, amount)
        b = torch.from_numpy(np.concatenate([np.full(len(j[2]), k) for k, j in enumerate(batch)])).to(dev)
        step = torch.from_numpy(np.concatenate([j[3] for j in batch])).to(dev)
        out[np.concatenate([j[2] for j in batch])] = h[b, step].float().cpu().numpy()
        i += size
        done += 1
        if done % 50 == 0:
            print(f"  {i:,}/{len(jobs):,} passes ({time.time() - t0:.0f}s)", flush=True)
    return out, n


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def heads(xtr: np.ndarray, tr: pd.DataFrame, xte: np.ndarray, te: pd.DataFrame) -> pd.DataFrame:
    """Test predictions of the probe's logistic head (C=0.1 on standardised inputs, row weights) per label."""
    sc = StandardScaler().fit(xtr)
    xtr, xte = sc.transform(xtr), sc.transform(xte)
    w = tr["weight"].to_numpy()
    preds = te[["client_id", "mon"]].copy()
    for lab in labels():
        ok = (tr[lab] != UNDEFINED).to_numpy()
        head = LogisticRegression(C=0.1, max_iter=3000).fit(xtr[ok], tr[lab].to_numpy()[ok], sample_weight=w[ok])
        preds[lab] = head.predict_proba(xte)[:, 1]
    return preds


def main() -> None:
    """Train CoLES on the pretraining folds, embed the probe's rows point in time, fit heads, compare on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows and test_rows")
    p.add_argument("--eventgpt", type=Path, default=None, help="EventGPT test predictions; default <probe-dir>/preds_history.parquet")
    p.add_argument("--engineered", type=Path, default=Path("data/baseline/preds.parquet"), help="engineered-feature test predictions")
    p.add_argument("--out", type=Path, default=Path("data/runs/coles-s42"))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--max-clients", type=int, default=0, help="training clients, for a quick run; 0 is every client")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="", help="torch device; default MPS, else CUDA, else CPU")
    a = p.parse_args()

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device) if a.device else device()
    test_fold = int((OUT / "test_fold.txt").read_text())
    trx = load_trx(test_fold)
    model, train_s = train(trx, test_fold, dev, a.out, a.seed, a.epochs, a.max_clients)

    tr = mark_undefined(pd.read_parquet(a.probe_dir / "train_rows.parquet"))
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet")).reset_index(drop=True)
    xtr, ntr = embed_rows(model, trx, tr, dev)
    xte, nte = embed_rows(model, trx, te, dev)
    np.savez(a.out / "embeddings.npz", train=xtr, test=xte, train_n_trx=ntr, test_n_trx=nte)
    empty = f"{(ntr == 0).mean():.1%} train, {(nte == 0).mean():.1%} test"
    print(f"rows embedded ({time.time() - t0:.0f}s); without a transaction before the cut: {empty}", flush=True)

    preds = heads(xtr, tr, xte, te)
    preds.to_parquet(a.out / "preds_coles.parquet")
    metrics = {lab: score(te[lab].to_numpy(), preds[lab].to_numpy()) for lab in labels()}
    metrics["official_mean"] = {m: float(np.mean([metrics[lab][m] for lab in OFFICIAL])) for m in ("roc_auc", "pr_auc")}

    eventgpt = pd.read_parquet(a.eventgpt or a.probe_dir / "preds_history.parquet")
    engineered = pd.read_parquet(a.engineered)
    labs = [lab for lab in labels() if (te[lab] == 1).any() and (te[lab] == 0).any()]
    vs_egpt = compare(te, {"eventgpt": eventgpt, "coles": preds}, labs)
    vs_eng = compare(te, {"engineered": engineered, "coles": preds}, labs)
    se = pd.concat([vs_egpt, vs_eng[vs_eng["model"] != "coles"]], ignore_index=True)
    se.to_csv(a.out / "auc_se.csv", index=False)
    print(se.round(4).to_string(index=False))

    config = {
        "embeddings": EMBEDDINGS,
        "amount": "sign(x) log(1 + |x|), BatchNorm",
        "hidden": HIDDEN,
        "noise": NOISE,
        "slices": {"split_count": SPLIT_COUNT, "cnt_min": CNT_MIN, "cnt_max": CNT_MAX},
        "seq_len": {"min": MIN_LEN, "max": MAX_LEN},
        "batch": BATCH,
        "epochs": a.epochs,
        "optimizer": {"lr": LR, "weight_decay": WEIGHT_DECAY, "step_size": STEP_SIZE, "gamma": GAMMA, "clip": CLIP},
        "loss": {"margin": MARGIN, "neg_count": NEG_COUNT},
        "seed": a.seed,
        "max_clients": a.max_clients,
    }
    out = {"config": config, "device": str(dev), "train_s": round(train_s), "elapsed_s": round(time.time() - t0), "metrics": metrics}
    (a.out / "results.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(metrics).T.round(4).to_string())
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
