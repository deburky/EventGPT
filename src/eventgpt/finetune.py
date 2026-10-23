"""Fine-tune a pretrained checkpoint end to end on the probe's labels, one output per label on the last state.

The rows, weights and labels are a probe run's own (train_rows and test_rows in its output directory), and each
history is built as the probe builds it: the most recent `block` tokens before the cut, ending in the now event.
A tenth of the training clients is held out to pick the best step; the test fold is scored once, at that step.
The head trains alone for the first --head-steps steps, then the pretrained weights join at a lower rate, so the
untrained head does not disturb them. The state is saved at every evaluation next to the output, so a run cut
off part way (a spot VM preempted) continues from its last evaluation.

Example: `uv run python -m eventgpt.finetune data/runs/hier-al-s42/hier-al-s42.pt --probe-dir data/runs/hier-al-s42/probe-now-2k`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from eventgpt.flat import load_checkpoint
from eventgpt.mbd import OUT
from eventgpt.probe import OFFICIAL, Fold, history_tokens, labels, score
from eventgpt.train import device
from eventgpt.vocab import Vocab


class Scorer(nn.Module):
    """Pretrained model plus a linear head on the state of the last token, one logit per label."""

    def __init__(self, model: Any, n_labels: int) -> None:
        super().__init__()
        self.model = model
        self.head = nn.Linear(model.cfg.d_model, n_labels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits (b, n_labels) for padded token ids x and the number of real tokens per row."""
        h = self.model.hidden(x, doc_mask=True, key_mask=x != 0)
        return self.head(h[torch.arange(len(x), device=x.device), lengths - 1].float())


class Histories:
    """Token sequences of probe rows, built on demand from the fold files."""

    def __init__(self, rows: pd.DataFrame, vocab: Vocab, block: int, aligned: bool) -> None:
        self.folds = {k: Fold(k, vocab) for k in np.unique(rows["fold"].to_numpy()).tolist()}
        self.rows = rows
        self.block, self.aligned = block, aligned
        self.evt = vocab.id("[EVT]")
        ci = np.empty(len(rows), dtype=np.int64)
        gap = np.empty(len(rows), dtype=np.int64)
        for k, fold in self.folds.items():
            m = (rows["fold"] == k).to_numpy()
            ci[m] = rows.loc[m, "client_id"].map(fold.index).to_numpy()
            last = fold.event_time[np.maximum(fold.client_events[ci[m]] + rows.loc[m, "n_hist"].to_numpy() - 1, 0)]
            gap[m] = vocab.dt_tokens(np.maximum(rows.loc[m, "cut"].to_numpy() - last, 0).astype(np.float64))
        self.ci, self.gap = ci, gap
        self.fold_of, self.n_hist = rows["fold"].to_numpy(), rows["n_hist"].to_numpy()

    def batch(self, idx: np.ndarray, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded token ids and lengths for rows idx."""
        seqs = [
            history_tokens(
                self.folds[int(self.fold_of[i])],
                int(self.ci[i]),
                int(self.n_hist[i]),
                self.block,
                np.array([self.evt, self.gap[i]]),
                self.aligned,
            )
            for i in idx
        ]
        lengths = np.array([len(s) for s in seqs])
        x = np.zeros((len(seqs), lengths.max()), dtype=np.int64)
        for r, s in enumerate(seqs):
            x[r, : len(s)] = s
        return torch.from_numpy(x).to(dev), torch.from_numpy(lengths).to(dev)


@torch.no_grad()
def predict(scorer: Scorer, hist: Histories, idx: np.ndarray, bsz: int, dev: torch.device) -> np.ndarray:
    """Probabilities (len(idx), n_labels), in length order for less padding, returned in the order of idx."""
    scorer.eval()
    order = idx[np.argsort(hist.n_hist[idx])]
    out = np.zeros((len(hist.rows), scorer.head.out_features), dtype=np.float32)
    for i in range(0, len(order), bsz):
        j = order[i : i + bsz]
        x, n = hist.batch(j, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out[j] = torch.sigmoid(scorer(x, n)).float().cpu().numpy()
    scorer.train()
    return out[idx]


def official_mean(y: np.ndarray, p: np.ndarray, labs: list[str]) -> float:
    """Mean ROC-AUC over the official targets that have both classes in y."""
    aucs = [score(y[:, k], p[:, k])["roc_auc"] for k, lab in enumerate(labs) if lab in OFFICIAL and 0 < y[:, k].sum() < len(y)]
    return float(np.mean(aucs))


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    """Fine-tune, keep the step with the best validation mean over the official targets, score the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with train_rows, test_rows and results.json")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5, help="peak rate of the pretrained weights")
    p.add_argument("--head-steps", type=int, default=2000, help="steps that train the head alone before the pretrained weights move")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=20_000, help="validation rows scored at each evaluation")
    p.add_argument("--max-rows", type=int, default=0, help="training and test rows, for a quick run; 0 is all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="csv path; default <ckpt dir>/finetune.csv")
    a = p.parse_args()

    t0 = time.time()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    block = int(json.loads((a.probe_dir / "results.json").read_text())["block"])
    tr = pd.read_parquet(a.probe_dir / "train_rows.parquet")
    te = pd.read_parquet(a.probe_dir / "test_rows.parquet")
    if a.max_rows:
        tr = tr.sample(a.max_rows, random_state=a.seed).reset_index(drop=True)
        te = te.sample(a.max_rows, random_state=a.seed).reset_index(drop=True)
    labs = labels()
    model = load_checkpoint(str(a.ckpt)).to(dev)
    aligned = bool(getattr(model.cfg, "aligned", False))
    scorer = Scorer(model, len(labs)).to(dev).train()
    h_tr, h_te = Histories(tr, vocab, block, aligned), Histories(te, vocab, block, aligned)
    y_tr = torch.from_numpy(tr[labs].to_numpy(dtype=np.float32))
    w_tr = torch.from_numpy(tr["weight"].to_numpy(dtype=np.float32))

    # a tenth of the training clients picks the step, by client as in eventgpt.baseline
    clients = tr["client_id"].unique()
    held = tr["client_id"].isin(set(rng.choice(clients, size=len(clients) // 10, replace=False))).to_numpy()
    fit_idx = np.flatnonzero(~held)
    val_idx = rng.choice(np.flatnonzero(held), size=min(a.n_val, int(held.sum())), replace=False)
    steps = int(a.epochs * len(fit_idx) / a.batch)
    warmup = min(500, steps // 10)
    opt = torch.optim.AdamW(
        [{"params": scorer.model.parameters(), "lr": a.lr}, {"params": scorer.head.parameters(), "lr": a.head_lr}],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    peaks = [a.lr, a.head_lr]
    print(f"{a.ckpt.name}: {len(fit_idx):,} fit rows, {len(val_idx):,} validation rows, {steps:,} steps, block {block}, aligned={aligned}")

    out = a.out or a.ckpt.parent / "finetune.csv"
    state_path = out.with_suffix(".state.pt")
    best = (-1.0, 0, copy.deepcopy(scorer.state_dict()))
    order = rng.permutation(np.resize(fit_idx, steps * a.batch))
    first = 1
    if state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        scorer.load_state_dict(st["scorer"])
        opt.load_state_dict(st["opt"])
        best, first = st["best"], st["step"] + 1
        print(f"resuming at step {first}", flush=True)
    for step in range(first, steps + 1):
        f = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for g, peak in zip(opt.param_groups, peaks, strict=True):
            g["lr"] = peak * f
        if step <= a.head_steps:
            # a head that starts at zero sends large gradients into the pretrained weights; train it alone first
            opt.param_groups[0]["lr"] = 0.0
        idx = order[(step - 1) * a.batch : step * a.batch]
        x, n = h_tr.batch(idx, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = scorer(x, n)
        y, w = y_tr[idx].to(dev), w_tr[idx].to(dev)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == steps:
            v = official_mean(y_tr[val_idx].numpy(), predict(scorer, h_tr, val_idx, a.batch * 2, dev), labs)
            print(f"eval step {step} validation official mean {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if v > best[0]:
                best = (v, step, copy.deepcopy(scorer.state_dict()))
            tmp = state_path.with_suffix(".tmp")
            torch.save({"scorer": scorer.state_dict(), "opt": opt.state_dict(), "best": best, "step": step}, tmp)
            tmp.rename(state_path)

    scorer.load_state_dict(best[2])
    pr = predict(scorer, h_te, np.arange(len(te)), a.batch * 2, dev)
    res = [(lab, *score(te[lab].to_numpy(), pr[:, k]).values()) for k, lab in enumerate(labs) if 0 < te[lab].sum() < len(te)]
    r = pd.DataFrame(res, columns=["label", "roc", "pr"]).set_index("label")
    print(f"best step {best[1]} (validation {best[0]:.4f})")
    print(r.round(4).to_string())
    print("official mean", r.loc[[lab for lab in OFFICIAL if lab in r.index]].mean().round(4).to_dict())
    r.to_csv(out)
    print(f"written to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
