"""A verifier head on Home Credit: P(an event matching a fixed pattern occurs in the held-out month) in one pass.

The zero-shot scorer of eventgpt.homecredit.zero_shot reads the probability that the *next* event is of a kind,
and for the payment targets reruns the token decoder up to 28 times to sum out the fields in front of the one
asked about. The verifier asks the question directly. A pattern is an event kind with an optional set of field
values, written as a partial event: `[EVT] name=cb_pay late=1` (or the union late in {1, 2, 3+}). The frozen event
encoder turns the pattern into a vector e_q, the frozen history decoder gives the state h at the cut (the history
ends in a now event `[EVT] dt=<gap>`), and a small head scores

    P(match in month | h, q) = sigmoid(<A h, B e_q> + c . e_q + g . h + b)

The labels come from the stream itself: for every training row and every pattern, whether an event inside the
held-out month matches it. The five targets of the zero-shot module are patterns too, so the head can be trained
with them (a multi-task supervised head) or without them, in which case the question is whether the head
generalizes to patterns it never saw. The arms:

    all         every pattern, binary cross-entropy
    brier_loss   every pattern, Brier loss (the RLCR reward y - (p - y)^2 once the event is fixed)
    target_out  the target patterns and their member values left out, plus 20% of the other patterns at random
    kind_out    also every cb_open and app pattern and every cb_pay pattern on late or pdpd

Rows: training rows are uniform samples of folds 1 to 4 built as in the zero-shot module; test rows are the 53,929
rows of the zero-shot run (zero_shot_scores.parquet), so the zero-shot and supervised-head scores sit beside the
verifier row for row. ROC-AUC per target on its positives plus the uniform rows; Brier score and a ten-bin
reliability table on the uniform rows only, since only they are a sample of the population. Example:

    MBD_OUT=data/home-credit/tokens uv run python -m eventgpt.homecredit.verifier data/runs/hc-alq-s42/ckpt/hc-alq-s42.pt \
        --zero-shot data/runs/hc-zs-s42/ckpt/zero_shot/zero_shot_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.homecredit.prepare import FIELDS, N_FOLDS
from eventgpt.homecredit.zero_shot import TARGETS, Events, auc, load_rows
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

TARGET_PATTERNS = {"late": "late", "dpd30": "dpd30", "new_credit": "new_credit", "new_app": "new_app"}
ARMS = ["all", "brier_loss", "target_out", "kind_out"]


# -------------------------------------------------------------------------------
# Patterns
# -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pattern:
    """An event kind with an optional set of field-value tokens; an event matches if it has the kind and one value."""

    label: str
    kind: str
    name_id: int
    values: tuple[int, ...] = ()
    field: str = ""


def catalogue(vocab: Vocab) -> list[Pattern]:
    """Name-only patterns for every kind but profile, one pattern per field value, and the four target unions."""
    out: list[Pattern] = []
    for kind, fields in FIELDS.items():
        if kind == "profile":
            continue
        nid = vocab.id(f"name={kind}")
        out.append(Pattern(kind, kind, nid))
        for f in fields:
            for tok, i in sorted(vocab.structured.items(), key=lambda kv: kv[1]):
                if tok.startswith(f"{f}="):
                    out.append(Pattern(f"{kind}:{tok}", kind, nid, (i,), f))
    pay = vocab.id("name=cb_pay")
    late = tuple(vocab.id(t) for t in ("late=1", "late=2", "late=3+"))
    dpd = tuple(vocab.id(f"pdpd=b{j}") for j in range(3, 7))
    out.append(Pattern("late", "cb_pay", pay, late, "late"))
    out.append(Pattern("dpd30", "cb_pay", pay, dpd, "pdpd"))
    out.append(Pattern("new_credit", "cb_open", vocab.id("name=cb_open")))
    out.append(Pattern("new_app", "app", vocab.id("name=app")))
    return out


def held_out(patterns: list[Pattern], arm: str, vocab: Vocab, seed: int) -> np.ndarray:
    """Boolean mask of the patterns an arm leaves out of training."""
    out = np.zeros(len(patterns), dtype=bool)
    if arm in ("all", "brier_loss"):
        return out
    late = {vocab.id(t) for t in ("late=1", "late=2", "late=3+")}
    dpd = {vocab.id(f"pdpd=b{j}") for j in range(3, 7)}
    for i, p in enumerate(patterns):
        target = p.label in TARGET_PATTERNS
        member = p.kind == "cb_pay" and bool(set(p.values) & (late | dpd))
        name_only = p.kind in ("cb_open", "app") and not p.values
        if target or member or name_only:
            out[i] = True
        if arm == "kind_out" and (p.kind in ("cb_open", "app") or (p.kind == "cb_pay" and p.field in ("late", "pdpd"))):
            out[i] = True
    rest = np.flatnonzero(~out & np.array([p.label not in TARGET_PATTERNS for p in patterns]))
    pick = np.random.default_rng(seed).choice(rest, size=len(rest) // 5, replace=False)
    out[pick] = True
    return out


def month_labels(fold: Fold, rows: pd.DataFrame, patterns: list[Pattern]) -> np.ndarray:
    """(rows, patterns) uint8: 1 if an event inside the row's held-out month matches the pattern."""
    by_name: dict[int, list[int]] = {}
    by_value: dict[tuple[int, int], list[int]] = {}
    for j, p in enumerate(patterns):
        if p.values:
            for v in p.values:
                by_value.setdefault((p.name_id, v), []).append(j)
        else:
            by_name.setdefault(p.name_id, []).append(j)
    ci = rows["client_idx"].to_numpy()
    a = fold.client_events[ci]
    lo, hi = a + rows["n_hist"].to_numpy(), a + rows["n_month"].to_numpy()
    starts = fold.event_start
    n_tok = len(fold.tokens)
    y = np.zeros((len(rows), len(patterns)), dtype=np.uint8)
    for r in range(len(rows)):
        for e in range(int(lo[r]), int(hi[r])):
            s = int(starts[e])
            end = int(starts[e + 1]) if e + 1 < len(starts) else n_tok
            span = np.asarray(fold.tokens[s:end])
            name = int(span[1])
            y[r, by_name.get(name, [])] = 1
            for v in span[3:]:
                hit = by_value.get((name, int(v)))
                if hit:
                    y[r, hit] = 1
    return y


# -------------------------------------------------------------------------------
# Frozen encodings
# -------------------------------------------------------------------------------
@torch.no_grad()
def history_states(model: Any, fold: Fold, rows: pd.DataFrame, vocab: Vocab, dev: torch.device, block: int, bsz: int) -> np.ndarray:
    """(rows, d) history state of a now event `[EVT] dt=<gap to the cut>` appended to the history before the cut."""
    ci, n_hist = rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy()
    last = fold.event_time[fold.client_events[ci] + n_hist - 1]
    gap = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
    evt = vocab.id("[EVT]")
    aligned = bool(getattr(model.cfg, "aligned", False))
    seqs = [history_tokens(fold, int(c), int(n), block, np.array([evt, g]), aligned) for c, n, g in zip(ci, n_hist, gap, strict=True)]
    lengths = np.array([len(s) for s in seqs])
    order = np.argsort(lengths)
    out = np.zeros((len(seqs), model.cfg.d_model), dtype=np.float32)
    t0 = time.time()
    # batches of at most bsz rows and bsz * 256 tokens: sorted by length, the long histories come last in small batches
    batches, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and j - i < bsz and (j + 1 - i) * lengths[order[j]] <= bsz * 256:
            j += 1
        batches.append(order[i:j])
        i = j
    done = 0
    for n_done, idx in enumerate(batches):
        x = np.zeros((len(idx), lengths[idx].max()), dtype=np.int64)
        for r, j in enumerate(idx):
            x[r, : lengths[j]] = seqs[j]
        xt = torch.from_numpy(x).to(dev)
        h, event, _, _, _ = model.stages(xt, xt != 0)
        at = torch.from_numpy(lengths[idx] - 1).to(dev)
        rr = torch.arange(len(idx), device=dev)
        out[idx] = h[rr, event[rr, at]].float().cpu().numpy()
        del xt, h, event
        if dev.type == "mps":  # the MPS allocator otherwise keeps every batch's buffers and runs out on long histories
            torch.mps.empty_cache()
        done += len(idx)
        if n_done % 100 == 0:
            print(f"    {done:,}/{len(order):,} histories ({time.time() - t0:.0f}s)", flush=True)
    return out


@torch.no_grad()
def pattern_vectors(model: Any, patterns: list[Pattern], vocab: Vocab, dev: torch.device) -> np.ndarray:
    """(patterns, d) event-encoder vector of `[USR] [EVT] name=<kind> <value>`, averaged over a union's values."""
    usr, evt = vocab.id("[USR]"), vocab.id("[EVT]")
    seqs, owner = [], []
    for j, p in enumerate(patterns):
        for v in p.values or (None,):
            seqs.append([usr, evt, p.name_id] + ([v] if v is not None else []))
            owner.append(j)
    x = np.zeros((len(seqs), 4), dtype=np.int64)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = s
    xt = torch.from_numpy(x).to(dev)
    _, _, ev, _, _ = model.stages(xt, xt != 0)
    vec = ev[:, 2].float().cpu().numpy()  # event 0 is empty, 1 is [USR], 2 the pattern
    out = np.zeros((len(patterns), vec.shape[1]), dtype=np.float32)
    np.add.at(out, np.array(owner), vec)
    return (out / np.bincount(owner, minlength=len(patterns))[:, None]).astype(np.float32)


# -------------------------------------------------------------------------------
# Head
# -------------------------------------------------------------------------------
class Verifier(nn.Module):
    """Low-rank bilinear score of a history state against a pattern vector, with a pattern and a history bias."""

    shift: torch.Tensor
    center: torch.Tensor

    def __init__(self, d: int, rank: int, n_patterns: int, head: str) -> None:
        super().__init__()
        self.head = head
        self.h_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, rank))
        self.q_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, rank))
        self.q_bias = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.h_bias = nn.Linear(rank, 1)
        # a free bias per pattern; a pattern left out of training keeps its zero
        self.free = nn.Parameter(torch.zeros(n_patterns))
        # log-odds correction per pattern, set by `calibrate` after training and kept in the state dict
        self.register_buffer("shift", torch.zeros(n_patterns))
        # mean learned logit per pattern over the training states, used by the centered head at inference
        self.register_buffer("center", torch.zeros(n_patterns))

    def forward(self, h: torch.Tensor, q: torch.Tensor, idx: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """(rows, patterns) logits for the patterns `idx`, whose vectors are `q` and base-rate logits `offset`."""
        u, v = self.h_proj(h), self.q_proj(q)
        if self.head == "prior":
            # posterior log-odds = prior log-odds of the pattern (its counted base rate, a fixed bias) + evidence from h
            return offset + u @ v.T / u.shape[1] ** 0.5 + self.h_bias(u)
        if self.head == "centered":
            # base-rate logit plus a history term with zero mean per pattern: the batch mean in training, the mean
            # over the training states at inference; no pattern can drift from its counted rate in log-odds
            g = u @ v.T / u.shape[1] ** 0.5 + self.h_bias(u)
            c = g.mean(0, keepdim=True) if self.training else self.center[idx]
            return offset + g - c + self.shift[idx]
        z = u @ v.T / u.shape[1] ** 0.5 + self.q_bias(q).T + self.h_bias(u)
        if "free" in self.head:
            z = z + self.free[idx]
        if "offset" in self.head:
            z = z + offset
        return z + self.shift[idx]

    @torch.no_grad()
    def calibrate(self, h: torch.Tensor, q: torch.Tensor, offset: torch.Tensor, rate: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Set the log-odds correction so the mean probability over the history states `h` equals each pattern's `rate`.

        The rate is counted from the event stream, so the correction needs no labels and applies to patterns the head
        never trained on. Returns the mean probability per pattern before the correction.
        """
        self.shift.zero_()
        idx = torch.arange(len(q))
        z = torch.cat([self(h[i : i + 4096], q, idx, offset) for i in range(0, len(h), 4096)])
        before = torch.sigmoid(z).mean(0)
        lo, hi = torch.full((len(q),), -30.0), torch.full((len(q),), 30.0)
        for _ in range(steps):  # bisection on mean(sigmoid(z + b)) = rate, which rises with b
            b = (lo + hi) / 2
            above = torch.sigmoid(z + b).mean(0) > rate
            hi, lo = torch.where(above, b, hi), torch.where(above, lo, b)
        self.shift.copy_((lo + hi) / 2)
        return before

    @torch.no_grad()
    def set_center(self, h: torch.Tensor, q: torch.Tensor) -> None:
        """Mean history term per pattern over the training states `h`; needs no labels, so unseen patterns get one."""
        u, v = self.h_proj(h), self.q_proj(q)
        self.center.copy_((u.mean(0) @ v.T / u.shape[1] ** 0.5 + self.h_bias(u).mean()).reshape(-1))


def rlcr_policy(logits: torch.Tensor, target: torch.Tensor, group: int = 8) -> torch.Tensor:
    """REINFORCE loss of the RLCR reward, the verifier stating a verdict and its confidence on each (row, pattern).

    The verdict is drawn from the head's own probability, its confidence is the probability it gave that verdict,
    and the reward is correctness minus the Brier penalty, r = c - (q - c)^2. Advantages are taken against the mean
    reward of `group` draws of the same (row, pattern), as in the RLCR notebook.
    """
    p = torch.sigmoid(logits).unsqueeze(0).expand(group, *logits.shape)
    verdict = torch.bernoulli(p.detach())
    q = torch.where(verdict == 1, p, 1 - p).detach()
    c = (verdict == target).float()
    r = c - (q - c).square()
    adv = r - r.mean(0, keepdim=True)
    logp = -nn.functional.binary_cross_entropy_with_logits(logits.unsqueeze(0).expand_as(verdict), verdict, reduction="none")
    return -(adv * logp).mean()


def fit(
    h: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    train_cols: np.ndarray,
    loss: str,
    dev: torch.device,
    seed: int,
    epochs: int,
    rank: int,
    head: str,
    offset: np.ndarray,
    fisher: str = "",
) -> Verifier:
    """Fit the head on the training columns; the epoch with the lowest validation loss on 10% of the rows is kept.

    With `fisher` set to "empirical" or "expected", each pattern's intercept (the `shift` buffer) is moved by Fisher
    scoring after every epoch and run to convergence at the end, as in fisher-scoring's LogisticRegression restricted
    to the intercept: b += sum(y - p) / I, with I = sum((y - p)^2) (empirical) or sum(p (1 - p)) (expected). At its
    fixed point the intercept's score is zero, so each trained pattern's mean prediction on the training rows equals
    its rate, which SGD with early stopping and weight decay does not guarantee.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    val = rng.random(len(h)) < 0.1
    ht, qt = torch.from_numpy(h).to(dev), torch.from_numpy(q[train_cols]).to(dev)
    it = torch.from_numpy(np.flatnonzero(train_cols)).to(dev)
    ot = torch.from_numpy(offset[train_cols].astype(np.float32)).to(dev)
    yt = torch.from_numpy(y[:, train_cols].astype(np.float32)).to(dev)
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    net = Verifier(h.shape[1], rank, q.shape[0], head).to(dev)
    rest = [t for n, t in net.named_parameters() if n != "free"]
    groups: list[dict[str, Any]] = [{"params": rest}, {"params": [net.free], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=1e-2)

    def fisher_scoring(steps: int) -> None:
        with torch.no_grad():
            tt = torch.from_numpy(tr_idx).to(dev)
            z = torch.cat([net(ht[tt[i : i + 4096]], qt, it, ot) for i in range(0, len(tt), 4096)])
            d = torch.zeros(z.shape[1], device=dev)
            for _ in range(steps):
                p = torch.sigmoid(z + d)
                r = yt[tt] - p
                info = r.square().sum(0) if fisher == "empirical" else (p * (1 - p)).sum(0)
                d += (r.sum(0) / info.clamp_min(1e-6)).clamp(-2.0, 2.0)
            net.shift[it] += d

    def objective(logits: torch.Tensor, target: torch.Tensor, train: bool = True) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        if loss == "brier":
            return (torch.sigmoid(logits) - target).square().mean()
        if loss == "rlcr" and train:
            return 0.5 * bce + 0.25 * rlcr_policy(logits, target)
        return bce

    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr_idx), max(len(tr_idx) // 256, 1)):
            bt = torch.from_numpy(b).to(dev)
            l_ = objective(net(ht[bt], qt, it, ot), yt[bt])
            opt.zero_grad()
            l_.backward()
            opt.step()
        net.eval()
        if fisher:
            fisher_scoring(3)
        with torch.no_grad():
            vt = torch.from_numpy(va_idx).to(dev)
            v_loss = objective(net(ht[vt], qt, it, ot), yt[vt], train=False).item()
        if v_loss < best:
            best, best_state = v_loss, {k: t.detach().clone() for k, t in net.state_dict().items()}
        print(f"    epoch {ep + 1}: validation {loss} {v_loss:.5f}", flush=True)
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    if fisher:
        fisher_scoring(25)
    return net


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Mean prediction and observed rate per quantile bin of the prediction."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [
        {"mean_p": float(p[which == b].mean()), "rate": float(y[which == b].mean()), "n": int((which == b).sum())}
        for b in range(len(edges) - 1)
        if (which == b).any()
    ]


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def encode_split(
    model: Any, vocab: Vocab, patterns: list[Pattern], dev: torch.device, a: argparse.Namespace, test_fold: int, cache: Path
) -> dict[str, Any]:
    """History states and pattern labels of the training rows (folds 1 to 4) and the zero-shot test rows, cached."""
    path = cache / "encoded.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files} | {"test_rows": pd.read_parquet(cache / "test_rows.parquet")}
    t0 = time.time()
    hs, ys = [], []
    for k in range(N_FOLDS):
        if k == test_fold:
            continue
        part = cache / f"fold{k}.npz"
        if part.exists():
            z = np.load(part)
            ys.append(z["y"])
            hs.append(z["h"])
            continue
        fold = Fold(k, vocab)
        rows = load_rows(fold, Events(fold), k, a.n_train // (N_FOLDS - 1), a.seed + k)
        print(f"  fold {k}: {len(rows):,} training rows ({time.time() - t0:.0f}s)", flush=True)
        ys.append(month_labels(fold, rows, patterns))
        hs.append(history_states(model, fold, rows, vocab, dev, a.block, a.batch))
        np.savez(part, y=ys[-1], h=hs[-1])
    fold = Fold(test_fold, vocab)
    test = pd.read_parquet(a.zero_shot)
    if a.max_test:
        test = test.sample(a.max_test, random_state=a.seed).reset_index(drop=True)
    test["client_idx"] = test["case_id"].astype(str).map(fold.index).to_numpy()
    print(f"  fold {test_fold}: {len(test):,} test rows ({time.time() - t0:.0f}s)", flush=True)
    y_te = month_labels(fold, test, patterns)
    h_te = history_states(model, fold, test, vocab, dev, a.block, a.batch)
    out = {"h_train": np.concatenate(hs), "y_train": np.concatenate(ys), "h_test": h_te, "y_test": y_te}
    np.savez(path, **out)
    test.to_parquet(cache / "test_rows.parquet")
    return out | {"test_rows": test}


def main() -> None:
    """Encode rows and patterns once, fit one verifier per arm, score the targets and the held-out patterns."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--zero-shot", type=Path, required=True, help="zero_shot_scores.parquet of the zero-shot run")
    p.add_argument("--n-train", type=int, default=40_000)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--min-pos", type=int, default=50, help="training positives a non-target pattern needs to be kept")
    p.add_argument("--max-test", type=int, default=0, help="test rows drawn for a quick run; 0 is every row")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None, help="folder of encoded.npz from an earlier run; defaults to --out")
    p.add_argument(
        "--head",
        default="base",
        choices=["base", "free", "offset", "offset_free", "centered", "prior"],
        help="free: per-pattern bias; offset: training base-rate logit added; centered: that logit plus a zero-mean history term",
    )
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--loss", default="bce", choices=["bce", "rlcr"], help="rlcr: 0.5 BCE + 0.25 REINFORCE on c - (q - c)^2")
    p.add_argument("--fisher", default="", choices=["", "empirical", "expected"], help="Fisher scoring of each pattern's intercept")
    p.add_argument("--calibrate", action="store_true", help="bake in a log-odds correction matching counted base rates")
    a = p.parse_args()
    t0 = time.time()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    out_dir = a.out or a.ckpt.parent / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = catalogue(vocab)
    enc = encode_split(model, vocab, patterns, dev, a, test_fold, a.cache or out_dir)
    keep = (enc["y_train"].sum(0) >= a.min_pos) | np.array([p_.label in TARGET_PATTERNS for p_ in patterns])
    patterns = [p_ for p_, k in zip(patterns, keep, strict=True) if k]
    y_tr, y_te = enc["y_train"][:, keep], enc["y_test"][:, keep]
    test = enc["test_rows"]
    q = pattern_vectors(model, patterns, vocab, dev)
    labels = [p_.label for p_ in patterns]
    # base rate of every pattern, counted from the training stream; needs no labels beyond the events themselves
    rate = (y_tr.sum(0) + 1.0) / (len(y_tr) + 2.0)
    offset = np.log(rate / (1 - rate)).astype(np.float32)
    col = {t: labels.index(t) for t in TARGET_PATTERNS}
    for t, j in col.items():  # the verifier's labels must agree with the zero-shot module's
        assert (y_te[:, j] == test[t].to_numpy()).all(), t
    print(f"{len(patterns)} patterns, {len(y_tr):,} training rows, {len(y_te):,} test rows ({time.time() - t0:.0f}s)", flush=True)

    uniform = test["uniform"].to_numpy()
    seen_late = test["seen_late"].to_numpy() == 1
    results: list[dict[str, Any]] = []
    calib: dict[str, Any] = {}
    pattern_rows: list[dict[str, Any]] = []
    cpu = torch.device("cpu")  # the head is small; on CPU its training is deterministic and free of MPS faults
    ht = torch.from_numpy(enc["h_test"])
    arms = a.arms
    for arm in arms:
        out_mask = held_out(patterns, arm, vocab, a.seed)
        print(f"  {arm}: training on {int((~out_mask).sum())} patterns, {int(out_mask.sum())} held out", flush=True)
        net = fit(
            enc["h_train"],
            q,
            y_tr,
            ~out_mask,
            "brier" if arm == "brier_loss" else a.loss,
            cpu,
            a.seed,
            a.epochs,
            a.rank,
            a.head,
            offset,
            a.fisher,
        )
        if a.head == "centered":
            net.set_center(torch.from_numpy(enc["h_train"]), torch.from_numpy(q))
        with torch.no_grad():  # the maximum-likelihood check: mean probability on the training rows against their rate
            h_tr, all_idx = torch.from_numpy(enc["h_train"]), torch.arange(len(q))
            fitted = torch.cat(
                [
                    torch.sigmoid(net(h_tr[i : i + 4096], torch.from_numpy(q), all_idx, torch.from_numpy(offset)))
                    for i in range(0, len(h_tr), 4096)
                ]
            ).mean(0)
        for t, j in col.items():
            print(f"    {t}: training-row mean {fitted[j]:.4f} against counted rate {rate[j]:.4f}", flush=True)
        if a.calibrate:
            net.calibrate(h_tr, torch.from_numpy(q), torch.from_numpy(offset), torch.from_numpy(rate.astype(np.float32)))
        with torch.no_grad():
            prob = torch.sigmoid(net(ht, torch.from_numpy(q), torch.arange(len(q)), torch.from_numpy(offset))).numpy()
        assert np.isfinite(prob).all(), arm
        for t, j in col.items():
            test[f"ver_{arm}_{t}"] = prob[:, j]
        test[f"ver_{arm}_first_late"] = np.where(seen_late, 0.0, prob[:, col["late"]])
        for j, pat in enumerate(patterns):  # every pattern on the uniform rows: seen against held-out
            yu = y_te[uniform, j]
            if pat.label not in TARGET_PATTERNS and 20 <= yu.sum() < len(yu):
                pattern_rows.append(
                    {
                        "arm": arm,
                        "pattern": pat.label,
                        "held_out": bool(out_mask[j]),
                        "positives": int(yu.sum()),
                        "roc_auc": auc(yu, prob[uniform, j])[0],
                        "rate": float(yu.mean()),
                        "train_rate": float(rate[j]),
                        "mean_p": float(prob[uniform, j].mean()),
                        "brier": float(np.mean((prob[uniform, j] - yu) ** 2)),
                    }
                )

    for name in TARGETS:
        ev_rows = uniform | (test[name].to_numpy() == 1)
        y = test.loc[ev_rows, name].to_numpy()
        row: dict[str, Any] = {"target": name, "positives": int(y.sum())}
        yu = test.loc[uniform, name].to_numpy()
        cols = {"zero_shot": f"zs_{name}", "head": f"head_{name}"} | {arm: f"ver_{arm}_{name}" for arm in arms}
        for label, c in cols.items():
            row[label], row[f"{label}_se"] = auc(y, test.loc[ev_rows, c].to_numpy())
            pu = test.loc[uniform, c].to_numpy()
            row[f"{label}_brier"] = float(np.mean((pu - yu) ** 2))
            row[f"{label}_mean_p"] = float(pu.mean())
            calib[f"{name}/{label}"] = reliability(yu, pu)
        row["rate"] = float(yu.mean())
        results.append(row)

    res = pd.DataFrame(results)
    pats = pd.DataFrame(pattern_rows)
    pats["abs_gap"] = (pats["mean_p"] - pats["rate"]).abs()
    pats["log_ratio"] = np.log(pats["mean_p"] / pats["rate"])
    summary = (
        pats.groupby(["arm", "held_out"])
        .agg(
            count=("roc_auc", "size"),
            mean=("roc_auc", "mean"),
            median=("roc_auc", "median"),
            brier=("brier", "mean"),
            abs_gap=("abs_gap", "mean"),
            med_abs_log_ratio=("log_ratio", lambda x: float(np.median(np.abs(x)))),
        )
        .reset_index()
    )
    res.to_csv(out_dir / "verifier.csv", index=False)
    pats.to_csv(out_dir / "patterns.csv", index=False)
    test.drop(columns=["client_idx"]).to_parquet(out_dir / "verifier_scores.parquet")
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "ckpt": str(a.ckpt),
                "head": a.head,
                "calibrate": a.calibrate,
                "loss": a.loss,
                "fisher": a.fisher,
                "patterns": labels,
                "elapsed_s": round(time.time() - t0),
                "results": results,
                "pattern_summary": summary.to_dict("records"),
                "reliability": calib,
            },
            indent=1,
        )
    )
    pd.set_option("display.width", 250)
    show = ["target", "positives", "rate", "zero_shot", "head", *arms]
    print(res[show].round(4).to_string(index=False))
    print(res[["target", *[f"{c}_brier" for c in ("zero_shot", "head", *arms)]]].round(5).to_string(index=False))
    print(summary.round(4).to_string(index=False))
    print(f"written to {out_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
