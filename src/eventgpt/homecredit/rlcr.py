"""EventGPT-D on Home Credit: EventGPT states its next event and its confidence that the claim is right.

From the history before the cut (the first day of the month before the decision month) the token decoder writes a
claim about the client's next event: its kind and the key fields of that kind,

    cb_pay      late, pdpd (n is generated first, as the decoder orders the fields, but not judged)
    app         status
    cb_open     st
    cb_close    cdpd
    tax         src
    dep         closed
    card        (the kind alone)

with a confidence q. The claim is right (c = 1) when the client's first event after the cut has the claimed kind and
every claimed key field.

EventGPT-D (the default, arm jev_ce with --fisher) takes q to be the model's own probability of its claim, as
Jev-style decision models read option probabilities: P(kind) times the key fields, summed over n for cb_pay. It
follows the open recipe of Geuter, Cheng and Alvarez-Melis (2026), "Releasing the Inner Jev: A Simple Recipe for
Open Decision Models" (arXiv:2610.17205):

    1. the residual head z = W0 y + A std(y) + b (their eq. 2): W0 is EventGPT's output layer, frozen, and A = 0,
       b = 0 at the start, so training starts at EventGPT zero-shot;
    2. cross-entropy of the true next event, with the token decoder (two blocks and the final norm) trained;
    3. Fisher scoring of the intercept b with the full expected information (fisher_intercept), which solves the
       maximum-likelihood score equations of b: for every token, the predicted probabilities sum to its observed count
       on the training rows. The paper fits a temperature instead; after the Fisher step the fitted temperature is 1.

The other arms:

    native  EventGPT zero-shot, no training: the greedy claim with its native confidence
    jev     the same residual head trained by the RLCR reward of Damani et al. (2025), "Beyond Binary Rewards:
            Training LMs to Reason About Their Uncertainty" (arXiv:2507.16806), r = c - (q - c)^2, with REINFORCE
            and group advantages over G sampled claims per row (as in the RLCR notebook of
            python-ml-projects/rlcr-calibration). q is differentiable, so the gradient of the expected reward is the
            REINFORCE term for the claim plus the direct gradient of -(q - c)^2. A temperature is fitted afterwards.
            On three seeds it matched the cross-entropy arm on accuracy with worse calibration (fitted temperature
            1.4); it is the variant for targets where correctness can be checked but no true answer can be imitated.
    text    confidence written as a token, as in Damani et al. (2025): after the claim the decoder writes one of
            K_CONF confidence tokens and q is the midpoint of its bin. The token is the last one of the sequence and
            is never read back, so its K_CONF output rows are a linear map of the decoder state at the claim's last
            token, initialized at zero. The claim and the confidence token are one sampled sequence trained by
            REINFORCE on r. Its q came out nearly constant.

After training, the jev arms get one temperature, fitted by NLL on the validation rows and kept at 1 or above, as in
Geuter, Cheng and Alvarez-Melis (2026). The history decoder is frozen; its state at the last event before the cut is
computed once per row and cached. Every arm is scored on the test fold with its greedy claim. The earlier version of
this module, with a separate Brier-trained confidence head, is kept with its results as
data/runs/hc-rlcr-s42/rlcr_v1.py.

Example, run from the repository root:

    MBD_OUT=data/home-credit/tokens uv run python -m eventgpt.homecredit.rlcr data/runs/hc-alq-s42/ckpt/hc-alq-s42.pt \
        --epochs 2 --out data/runs/hc-rlcr-s42/v3/s0
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
from torch import nn

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.homecredit import calibration
from eventgpt.homecredit.prepare import N_FOLDS
from eventgpt.homecredit.zero_shot import Events, auc, load_rows
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

GEN: dict[str, list[str]] = {
    "cb_pay": ["n", "late", "pdpd"],
    "app": ["status"],
    "cb_open": ["st"],
    "cb_close": ["cdpd"],
    "tax": ["src"],
    "dep": ["closed"],
    "card": [],
}
KEY: dict[str, list[str]] = GEN | {"cb_pay": ["late", "pdpd"]}
KINDS = list(GEN)
N_GEN = max(len(f) for f in GEN.values())
L = 3 + N_GEN  # [EVT] name dt and the generated fields
ARMS = ["native", "text", "jev", "jev_ce"]
K_CONF = 20  # confidence tokens of the text arm; bin k stands for q = (k + 0.5) / K_CONF
MID = (torch.arange(K_CONF, dtype=torch.float32) + 0.5) / K_CONF


# -------------------------------------------------------------------------------
# Claims
# -------------------------------------------------------------------------------
class Grammar:
    """Token ids of the claim grammar: the allowed tokens at every claim position for every kind."""

    def __init__(self, vocab: Vocab) -> None:
        v = vocab
        self.v = v
        self.evt = v.id("[EVT]")
        self.name_ids = torch.tensor([v.id(f"name={k}") for k in KINDS])
        self.kind_of = {v.id(f"name={k}"): i for i, k in enumerate(KINDS)}
        size = v.size
        self.allowed_name = torch.zeros(size, dtype=torch.bool)
        self.allowed_name[self.name_ids] = True
        # allowed[kind, j]: values of the kind's j-th generated field; an all-false row ends the claim
        self.allowed = torch.zeros(len(KINDS), N_GEN, size, dtype=torch.bool)
        for i, k in enumerate(KINDS):
            for j, f in enumerate(GEN[k]):
                ids = [t for s, t in v.structured.items() if s.startswith(f"{f}=")]
                self.allowed[i, j, ids] = True
        self.n_gen = torch.tensor([len(GEN[k]) for k in KINDS])
        # positions of the key fields among the generated ones, as a (kind, N_GEN) mask
        self.is_key = torch.zeros(len(KINDS), N_GEN, dtype=torch.bool)
        for i, k in enumerate(KINDS):
            for j, f in enumerate(GEN[k]):
                self.is_key[i, j] = f in KEY[k]

    def kind(self, name_tok: torch.Tensor) -> torch.Tensor:
        """Kind index of name tokens; -1 for a name outside the grammar."""
        lut = torch.full((self.v.size,), -1, dtype=torch.long)
        for t, i in self.kind_of.items():
            lut[t] = i
        return lut.to(name_tok.device)[name_tok]

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """(b, L, vocab) allowed next tokens at each position of claims x; position p predicts token p + 1."""
        b = x.size(0)
        out = torch.zeros(b, L, self.v.size, dtype=torch.bool, device=x.device)
        out[:, 0] = self.allowed_name.to(x.device)
        k = self.kind(x[:, 1]).clamp(min=0)
        out[:, 2 : 2 + N_GEN] = self.allowed.to(x.device)[k]
        # a position with nothing to predict (the dt, past the kind's last field) allows everything: an all-masked
        # log-softmax is NaN, and its gradient would be NaN even where the position is never read
        return out | ~out.any(-1, keepdim=True)


def decode(model: Any, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Token decoder states (b, t, d) of claim tokens x (b, t) given the history state ctx (b, d) of the last event.

    The same computation as EventGPT.decode_tokens for the tokens of one event: token embedding, position in the
    event, the previous event's history state, causal attention within the event, final norm.
    """
    t = x.size(1)
    pos = torch.arange(t, device=x.device)
    y = model.tok_emb(x) + model.pos_in_event(pos).unsqueeze(0) + ctx.unsqueeze(1)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device)).view(1, 1, t, t)
    for blk in model.tok_blocks:
        y = blk(y, causal)
    return model.ln_f(y)


def masked_logp(model: Any, y: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """(b, t, vocab) log-probabilities over the allowed tokens at each position."""
    return model.head(y).float().masked_fill(~allowed, float("-inf")).log_softmax(-1)


@torch.no_grad()
def sample_claims(model: Any, g: Grammar, ctx: torch.Tensor, dt: torch.Tensor, greedy: bool) -> torch.Tensor:
    """(b, L) claims, one token at a time; positions after a kind's last field stay 0."""
    b = ctx.size(0)
    x = torch.zeros(b, L, dtype=torch.long, device=ctx.device)
    x[:, 0] = g.evt

    def pick(lp: torch.Tensor) -> torch.Tensor:
        return lp.argmax(-1) if greedy else torch.multinomial(lp.exp().nan_to_num(0.0), 1).squeeze(1)

    lp = masked_logp(model, decode(model, ctx, x[:, :1])[:, 0], g.allowed_name.to(ctx.device).expand(b, -1))
    x[:, 1] = pick(lp)
    x[:, 2] = dt
    k = g.kind(x[:, 1])
    n_gen = g.n_gen.to(ctx.device)[k]
    for j in range(N_GEN):
        allowed = g.allowed.to(ctx.device)[k, j]
        live = n_gen > j
        if not live.any():
            break
        lp = masked_logp(model, decode(model, ctx, x[:, : 3 + j])[:, 2 + j], allowed | ~live.unsqueeze(1))
        x[:, 3 + j] = torch.where(live, pick(lp), torch.zeros_like(x[:, 0]))
    return x


def claim_logp_and_state(model: Any, g: Grammar, ctx: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of each claim's sampled tokens (dt excluded) and the decoder state at its last token."""
    y = decode(model, ctx, x)
    lp = masked_logp(model, y, g.mask(x))
    tok_lp = lp[:, :-1].gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)  # (b, L - 1): position p scores token p + 1
    k = g.kind(x[:, 1]).clamp(min=0)
    n_gen = g.n_gen.to(x.device)[k]
    ar = torch.arange(L - 1, device=x.device).unsqueeze(0)
    use = (ar == 0) | ((ar >= 2) & (ar < 2 + n_gen.unsqueeze(1)))
    total = torch.where(use, tok_lp, torch.zeros_like(tok_lp)).sum(1)
    last = 2 + n_gen
    return total, y[torch.arange(len(x), device=x.device), last]


def native_confidence(model: Any, g: Grammar, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The generator's own probability of the claim's key: P(kind) times the key fields, summed over n for cb_pay.

    Differentiable in the decoder and the output head, which the jev arm needs.
    """
    k = g.kind(x[:, 1]).clamp(min=0)
    pay = k == KINDS.index("cb_pay")
    lp = masked_logp(model, decode(model, ctx, x), g.mask(x))
    tok_p = lp[:, :-1].gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1).exp()
    n_gen = g.n_gen.to(x.device)[k]
    p = tok_p[:, 0] * torch.where(n_gen > 0, tok_p[:, 2], torch.ones_like(tok_p[:, 0]))  # kind, then the first field
    if pay.any():
        p = p.clone()  # cb_pay: sum over n of P(n) P(late | n) P(pdpd | n, late)
        xp = x[pay].clone()
        n_ids = g.allowed[KINDS.index("cb_pay"), 0].nonzero().squeeze(1).to(x.device)
        lp0 = masked_logp(model, decode(model, ctx[pay], xp[:, :3])[:, 2], g.allowed[KINDS.index("cb_pay"), 0].to(x.device))
        total = torch.zeros(len(xp), device=x.device)
        for nid in n_ids:
            xn = xp.clone()  # a fresh tensor per n: autograd keeps the index tensors of the gathers
            xn[:, 3] = nid
            q = masked_logp(model, decode(model, ctx[pay], xn), g.mask(xn))
            p_late = q[:, 3].gather(-1, xn[:, 4:5]).squeeze(-1).exp()
            p_pdpd = q[:, 4].gather(-1, xn[:, 5:6]).squeeze(-1).exp()
            total = total + lp0[:, nid].exp() * p_late * p_pdpd
        p[pay] = tok_p[pay, 0] * total
    return p


def correct(g: Grammar, x: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Correctness: 1 when the claim's kind and every key field match the true next event; truth is (b, L) like a claim."""
    k = g.kind(x[:, 1]).clamp(min=0)
    key = g.is_key.to(x.device)[k]
    fields_ok = ((x[:, 3:] == truth[:, 3:]) | ~key).all(1)
    return ((x[:, 1] == truth[:, 1]) & fields_ok).float()


# -------------------------------------------------------------------------------
# Rows: history state at the last event before the cut, and the true next event
# -------------------------------------------------------------------------------
def true_claims(fold: Fold, rows: pd.DataFrame, g: Grammar, dt: np.ndarray) -> np.ndarray:
    """(rows, L) the first event after the cut written as a claim; a missing field is -1."""
    v = fold.vocab
    inv = {i: s for s, i in v.structured.items()}
    out = np.zeros((len(rows), L), dtype=np.int64)
    e_all = fold.client_events[rows["client_idx"].to_numpy()] + rows["n_hist"].to_numpy()
    for r, e in enumerate(e_all):
        s = int(fold.event_start[e])
        t = int(fold.event_start[e + 1]) if e + 1 < len(fold.event_start) else len(fold.tokens)
        toks = [int(z) for z in fold.tokens[s:t]]
        out[r, 0], out[r, 1], out[r, 2] = g.evt, toks[1], dt[r]
        kind = inv[toks[1]][5:]
        by_field = {inv[z].split("=", 1)[0]: z for z in toks[3:] if z in inv and "=" in inv[z]}
        for j, f in enumerate(GEN.get(kind, [])):
            out[r, 3 + j] = by_field.get(f, -1)
    return out


@torch.no_grad()
def last_states(model: Any, fold: Fold, rows: pd.DataFrame, dev: torch.device, block: int, bsz: int) -> np.ndarray:
    """(rows, d) history state at the last event before the cut: the context the token decoder reads for the next event."""
    ci, n_hist = rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy()
    aligned = bool(getattr(model.cfg, "aligned", False))
    seqs = [history_tokens(fold, int(c), int(n), block, aligned=aligned) for c, n in zip(ci, n_hist, strict=True)]
    lengths = np.array([len(s) for s in seqs])
    order = np.argsort(lengths)
    out = np.zeros((len(seqs), model.cfg.d_model), dtype=np.float32)
    batches, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and j - i < bsz and (j + 1 - i) * lengths[order[j]] <= bsz * 256:
            j += 1
        batches.append(order[i:j])
        i = j
    t0 = time.time()
    for b_i, idx in enumerate(batches):
        x = np.zeros((len(idx), lengths[idx].max()), dtype=np.int64)
        for r, j in enumerate(idx):
            x[r, : lengths[j]] = seqs[j]
        xt = torch.from_numpy(x).to(dev)
        h, event, _, _, _ = model.stages(xt, xt != 0)
        rr = torch.arange(len(idx), device=dev)
        out[idx] = h[rr, event[rr, torch.from_numpy(lengths[idx] - 1).to(dev)]].float().cpu().numpy()
        del xt, h, event
        if dev.type == "mps":
            torch.mps.empty_cache()
        if b_i % 100 == 0:
            print(f"    {sum(len(z) for z in batches[: b_i + 1]):,}/{len(order):,} histories ({time.time() - t0:.0f}s)", flush=True)
    return out


def build_split(
    model: Any, vocab: Vocab, g: Grammar, folds: list[int], per_fold: int, seed: int, dev: torch.device, block: int, bsz: int
) -> dict[str, np.ndarray]:
    """Contexts, gap tokens and true next-event claims of rows from the given folds; rows need a next event."""
    ctx, dts, truth = [], [], []
    for k in folds:
        fold = Fold(k, vocab)
        rows = load_rows(fold, Events(fold), k, None, seed + k)
        ce = fold.client_events
        has_next = (ce[rows["client_idx"].to_numpy()] + rows["n_hist"].to_numpy()) < ce[rows["client_idx"].to_numpy() + 1]
        rows = rows[has_next].reset_index(drop=True)
        rows = rows.sample(min(per_fold, len(rows)), random_state=seed + k).reset_index(drop=True)
        ci, n_hist = rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy()
        last = fold.event_time[ce[ci] + n_hist - 1]
        dt = vocab.dt_tokens(np.maximum(rows["cut"].to_numpy() - last, 0).astype(np.float64))
        print(f"  fold {k}: {len(rows):,} rows", flush=True)
        truth.append(true_claims(fold, rows, g, dt))
        dts.append(dt)
        ctx.append(last_states(model, fold, rows, dev, block, bsz))
    return {"ctx": np.concatenate(ctx), "dt": np.concatenate(dts), "truth": np.concatenate(truth)}


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
class ConfidenceTokens(nn.Module):
    """The text arm's confidence tokens (Damani et al. 2025): K_CONF output rows on the decoder state at the claim's last token.

    The confidence token ends the sequence and is never fed back, so these rows act as a linear map; zero weights
    make the first confidence uniform over the bins.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.out = nn.Linear(d, K_CONF)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """(b, K_CONF) log-probabilities of the confidence tokens."""
        return self.out(y).float().log_softmax(-1)


class ResidualHead(nn.Module):
    """The jev arm's output head z = W0 y + A std(y) + b (Geuter, Cheng and Alvarez-Melis 2026, eq. 2).

    W0 is EventGPT's pretrained output head, frozen; std(y) standardizes the decoder state with means and standard
    deviations fixed from the pretrained model; A and b start at zero, so the head starts as the zero-shot readout.
    """

    def __init__(self, w0: nn.Linear, mean: torch.Tensor, sd: torch.Tensor) -> None:
        super().__init__()
        self.w0 = w0
        for p in self.w0.parameters():
            p.requires_grad_(False)
        d, v = mean.numel(), w0.weight.size(0)
        self.res = nn.Linear(d, v)
        nn.init.zeros_(self.res.weight)
        nn.init.zeros_(self.res.bias)
        self.mean: torch.Tensor
        self.sd: torch.Tensor
        self.register_buffer("mean", mean)
        self.register_buffer("sd", sd)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Logits over the vocabulary."""
        return self.w0(y) + self.res((y - self.mean) / self.sd)


@torch.no_grad()
def state_stats(model: Any, g: Grammar, ctx: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of the pretrained decoder state over the positions of greedy claims, for std(y)."""
    x = sample_claims(model, g, ctx, dt, greedy=True)
    y = decode(model, ctx, x)
    n_gen = g.n_gen[g.kind(x[:, 1]).clamp(min=0)]
    keep = torch.arange(L).unsqueeze(0) <= (2 + n_gen).unsqueeze(1)
    z = y[keep]
    return z.mean(0), z.std(0).clamp(min=1e-4)


def evaluate(
    model: Any, conf: ConfidenceTokens | None, g: Grammar, ctx: torch.Tensor, dt: torch.Tensor, truth: torch.Tensor, bsz: int = 2048
) -> dict[str, np.ndarray]:
    """Greedy claim, its correctness and its confidence per row.

    q is the most likely confidence token's value for the text arm (conf given) and the native confidence otherwise;
    q_mean is the expected value of the confidence token (text arm only).
    """
    xs, cs, qs, qm, ns = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ctx), bsz):
            c_, d_, t_ = ctx[i : i + bsz], dt[i : i + bsz], truth[i : i + bsz]
            x = sample_claims(model, g, c_, d_, greedy=True)
            native = native_confidence(model, g, c_, x)
            if conf is not None:
                _, y = claim_logp_and_state(model, g, c_, x)
                lpc = conf(y)
                qs.append(MID[lpc.argmax(-1)])
                qm.append((lpc.exp() * MID).sum(-1))
            else:
                qs.append(native)
                qm.append(torch.full((len(x),), float("nan")))
            xs.append(x)
            cs.append(correct(g, x, t_))
            ns.append(native)
    return {k: torch.cat(v).cpu().numpy() for k, v in (("claim", xs), ("c", cs), ("q", qs), ("q_mean", qm), ("native", ns))}


def true_ce(model: Any, g: Grammar, ctx: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the true next event's claim tokens, the jev_ce loss; rows with a missing field or an unknown kind skip."""
    ok = (truth >= 0).all(1) & (g.kind(truth[:, 1].clamp(min=0)) >= 0)
    if not ok.any():
        return torch.zeros((), device=ctx.device)
    lp, _ = claim_logp_and_state(model, g, ctx[ok], truth[ok])
    return -lp.mean()


def split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Training and validation indices (10% validation) and the generator, ready for the next draw."""
    rng = np.random.default_rng(seed)
    val = rng.random(n) < 0.1
    return np.flatnonzero(~val), np.flatnonzero(val), rng


def install_jev_head(model: Any, g: Grammar, data: dict[str, torch.Tensor], tr_idx: np.ndarray, rng: np.random.Generator) -> None:
    """Replace the output head by the residual head, with std(y) fixed from 4,096 training rows of the pretrained model."""
    sub = torch.from_numpy(rng.choice(tr_idx, min(4096, len(tr_idx)), replace=False))
    mean, sd = state_stats(model, g, data["ctx"][sub], data["dt"][sub])
    model.head = ResidualHead(model.head, mean, sd)


class Tempered(nn.Module):
    """Logits divided by one temperature t >= 1, fitted after training as in Geuter, Cheng and Alvarez-Melis (2026)."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head
        self.t = 1.0

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Tempered logits."""
        return self.head(y) / self.t


@torch.no_grad()
def truth_nll(model: Any, g: Grammar, ctx: torch.Tensor, truth: torch.Tensor, bsz: int = 2048) -> float:
    """Mean negative log probability of the true next event's key; rows with a missing field or an unknown kind skip."""
    ok = (truth >= 0).all(1) & (g.kind(truth[:, 1].clamp(min=0)) >= 0)
    c, t = ctx[ok], truth[ok]
    p = torch.cat([native_confidence(model, g, c[i : i + bsz], t[i : i + bsz]) for i in range(0, len(c), bsz)])
    return float(-p.clamp(min=1e-12).log().mean())


def fit_temperature(model: Any, g: Grammar, ctx: torch.Tensor, truth: torch.Tensor) -> tuple[float, dict[str, float]]:
    """Set the temperature that minimizes the validation NLL of the true next event, on a grid from 1 to 4.

    As in Geuter, Cheng and Alvarez-Melis (2026): one temperature, fitted by NLL on held-out rows and kept at 1 or
    above. Dividing every logit by t keeps each argmax, so the greedy claims stay the same and only q changes.
    """
    model.head = Tempered(model.head)
    nll = {}
    for t in np.round(np.arange(1.0, 4.01, 0.1), 2):
        model.head.t = float(t)
        nll[f"{t:.1f}"] = truth_nll(model, g, ctx, truth)
    best = float(min(nll, key=lambda k: nll[k]))
    model.head.t = best
    return best, nll


@torch.no_grad()
def fisher_intercept(model: Any, g: Grammar, ctx: torch.Tensor, truth: torch.Tensor, steps: int = 10, bsz: int = 4096) -> list[float]:
    """Fisher scoring of the residual head's intercept b on the cross-entropy of the true next event.

    Everything else is frozen. Each step is b += I^+ s, the Fisher-scoring step of the fisher-scoring package with the
    full expected information I = sum (diag(p) - p p^T) over the tokens a claim can contain (the pseudo-inverse, since
    a softmax is unchanged by a shift of all its logits), and s = sum (y - p). The diagonal of I alone zig-zags between
    tokens of the same field, whose probabilities sum to one. At the fixed point the maximum-likelihood score equations
    of b hold: for every token, its predicted probabilities summed over the rows and claim positions where it can appear
    equal its observed count. That is the property of an intercept fitted by maximum likelihood, so it holds for the
    cross-entropy arm and not for the reward. Returns the largest absolute score before each step and at the end.
    """
    head = model.head
    ok = (truth >= 0).all(1) & (g.kind(truth[:, 1].clamp(min=0)) >= 0)
    c, t = ctx[ok], truth[ok]
    zs, ms, ys = [], [], []
    ar = torch.arange(L - 1)
    for i in range(0, len(c), bsz):
        x = t[i : i + bsz]
        y = decode(model, c[i : i + bsz], x)
        z = head.w0(y) + head.res((y - head.mean) / head.sd) - head.res.bias  # the logits without b
        n_gen = g.n_gen[g.kind(x[:, 1]).clamp(min=0)]
        use = (ar == 0) | ((ar >= 2) & (ar < 2 + n_gen.unsqueeze(1)))
        zs.append(z[:, :-1][use].double())
        ms.append(g.mask(x)[:, :-1][use])
        ys.append(x[:, 1:][use])
    z, m, yt = torch.cat(zs), torch.cat(ms), torch.cat(ys)
    u = m.any(0).nonzero().squeeze(1)  # tokens some claim position allows
    z, m = z[:, u], m[:, u]
    onehot = torch.zeros_like(z).scatter_(1, torch.searchsorted(u, yt).unsqueeze(1), 1.0)
    b = head.res.bias.detach().clone().double()
    log = []
    for _ in range(steps + 1):
        p = (z + b[u]).masked_fill(~m, float("-inf")).softmax(-1)
        score = (onehot - p).sum(0)
        log.append(float(score.abs().max()))
        if len(log) > steps:
            break
        info = torch.diag(p.sum(0)) - p.T @ p
        b[u] += torch.linalg.pinv(info, hermitian=True) @ score
    head.res.bias.copy_(b.float())
    return log


def train(
    model: Any, g: Grammar, data: dict[str, torch.Tensor], arm: str, epochs: int, group: int, bsz: int, seed: int
) -> tuple[ConfidenceTokens | None, list[dict[str, float]]]:
    """Train an arm (text and jev by the RLCR reward, jev_ce by cross-entropy); the epoch with the best validation reward is kept."""
    torch.manual_seed(seed)
    tr_idx, va_idx, rng = split(len(data["ctx"]), seed)
    for p in model.parameters():
        p.requires_grad_(False)
    gen_params = [p for m in (*model.tok_blocks, model.ln_f) for p in m.parameters()]
    for p in gen_params:
        p.requires_grad_(True)
    groups: list[dict[str, Any]] = [{"params": gen_params, "lr": 3e-5}]
    conf: ConfidenceTokens | None = None
    if arm == "text":
        conf = ConfidenceTokens(model.cfg.d_model)
        groups.append({"params": list(conf.parameters()), "lr": 1e-3})
        extra = list(conf.parameters())
    else:
        install_jev_head(model, g, data, tr_idx, rng)
        extra = list(model.head.res.parameters())
        groups.append({"params": extra, "lr": 1e-4})  # the head's learning rate in Geuter et al. (2026)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    best, best_state, log = -float("inf"), None, []
    va = torch.from_numpy(va_idx)
    for ep in range(epochs):
        t0 = time.time()
        stats: dict[str, float] = {"loss": 0.0, "reward": 0.0, "c": 0.0, "q": 0.0}
        batches = np.array_split(rng.permutation(tr_idx), max(len(tr_idx) // bsz, 1))
        for b in batches:
            bt = torch.from_numpy(b)
            if arm == "jev_ce":
                loss = true_ce(model, g, data["ctx"][bt], data["truth"][bt])
                opt.zero_grad()
                loss.backward()
                opt.step()
                stats["loss"] += loss.item() / len(batches)
                continue
            ctx = data["ctx"][bt].repeat_interleave(group, 0)
            dt = data["dt"][bt].repeat_interleave(group, 0)
            truth = data["truth"][bt].repeat_interleave(group, 0)
            x = sample_claims(model, g, ctx, dt, greedy=False)
            c = correct(g, x, truth)
            logp, y = claim_logp_and_state(model, g, ctx, x)
            if conf is not None:  # text: sample the confidence token as the last token of the sequence
                lpc = conf(y)
                k = torch.multinomial(lpc.exp().detach(), 1).squeeze(1)
                q = MID[k]
                logp = logp + lpc.gather(1, k.unsqueeze(1)).squeeze(1)
                direct = torch.zeros((), device=c.device)
            else:  # jev: q is the model's own probability of the claim
                q_live = native_confidence(model, g, ctx, x)
                q = q_live.detach()
                direct = (q_live - c).square().mean()
            r = c - (q - c).square()
            adv = (r.view(-1, group) - r.view(-1, group).mean(1, keepdim=True)).view(-1)
            loss = 0.25 * -(adv * logp).mean() + direct
            opt.zero_grad()
            loss.backward()
            opt.step()
            for key, val_ in (("loss", loss.item()), ("reward", r.mean().item()), ("c", c.mean().item()), ("q", q.mean().item())):
                stats[key] += val_ / len(batches)
        ev = evaluate(model, conf, g, data["ctx"][va], data["dt"][va], data["truth"][va])
        v_reward = float(np.mean(ev["c"] - (ev["q"] - ev["c"]) ** 2))
        row = stats | {
            "epoch": ep + 1,
            "val_reward": v_reward,
            "val_c": float(ev["c"].mean()),
            "val_q": float(ev["q"].mean()),
            "s": time.time() - t0,
        }
        log.append(row)
        print(f"    {arm} epoch {ep + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        if v_reward > best:
            best = v_reward
            best_state = [p.detach().clone() for p in gen_params + extra]
    assert best_state is not None
    with torch.no_grad():
        for p, s_ in zip(gen_params + extra, best_state, strict=True):
            p.copy_(s_)
    return (conf.eval() if conf is not None else None), log


# -------------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------------
def metrics(c: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Accuracy, mean confidence, Brier, ECE, AUC of q for c, accuracy of the most confident 50% and 20%, mean reward."""
    bins = np.clip((q * 10).astype(int), 0, 9)
    ece = sum(abs(q[bins == b].mean() - c[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    order = np.argsort(-q)
    out = {
        "accuracy": float(c.mean()),
        "mean_q": float(q.mean()),
        "brier": float(np.mean((q - c) ** 2)),
        "ece": float(ece),
        "reward": float(np.mean(c - (q - c) ** 2)),
        "acc_top50": float(c[order[: len(c) // 2]].mean()),
        "acc_top20": float(c[order[: len(c) // 5]].mean()),
    }
    out["auc"], out["auc_se"] = auc(c.astype(np.int64), q) if 0 < c.sum() < len(c) else (float("nan"), float("nan"))
    return out | {k: v for k, v in calibration.summary(c, q).items() if k not in out}


def claim_shares(g: Grammar, x: np.ndarray) -> dict[str, float]:
    """Share of claims by kind."""
    inv = {i: s for s, i in g.v.structured.items()}
    names = pd.Series([inv.get(int(t), "?") for t in x[:, 1]])
    return {str(k): float(v) for k, v in names.value_counts(normalize=True).items()}


def main() -> None:
    """Build the rows, train the EventGPT-D arms, score them and EventGPT zero-shot on the test fold."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--per-fold", type=int, default=6000, help="training rows drawn per training fold")
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64, help="rows per batch when encoding histories")
    p.add_argument("--rows", type=int, default=64, help="rows per training step; each gets --group sampled claims")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["native", "jev_ce"], choices=ARMS, help="default: EventGPT and EventGPT-D")
    p.add_argument("--no-fisher", dest="fisher", action="store_false", help="skip the Fisher scoring of the intercept after jev_ce")
    p.add_argument("--reuse", action="store_true", help="load a trained arm from <out>/<arm>.pt instead of training it")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    t0 = time.time()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    g = Grammar(vocab)

    cache = a.out / "rows.npz"
    if cache.exists():
        z = np.load(cache)
        tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
        te = {k[3:]: z[k] for k in z.files if k.startswith("te_")}
    else:
        model = load_checkpoint(str(a.ckpt)).to(dev).eval()
        tr = build_split(model, vocab, g, [k for k in range(N_FOLDS) if k != test_fold], a.per_fold, a.seed, dev, a.block, a.batch)
        te = build_split(model, vocab, g, [test_fold], a.n_test, a.seed, dev, a.block, a.batch)
        np.savez(cache, **{f"tr_{k}": v for k, v in tr.items()}, **{f"te_{k}": v for k, v in te.items()})
        del model
    print(f"{len(tr['ctx']):,} training rows, {len(te['ctx']):,} test rows ({time.time() - t0:.0f}s)", flush=True)

    cpu = torch.device("cpu")
    to_t = lambda d: {k: torch.from_numpy(v) for k, v in d.items()}  # noqa: E731
    tr_t, te_t = to_t(tr), to_t(te)

    # the trivial answer: the most frequent true key in training, with its training frequency as confidence
    def key_of(x: np.ndarray) -> list[tuple[int, ...]]:
        k = g.kind(torch.from_numpy(x[:, 1]).clamp(min=0)).numpy()
        return [(int(r[1]), *[int(r[3 + j]) if g.is_key[kk, j] else 0 for j in range(N_GEN)]) for r, kk in zip(x, k, strict=True)]

    keys_tr = pd.Series(key_of(tr["truth"]))
    top = keys_tr.value_counts(normalize=True)
    c_const = np.array([k == top.index[0] for k in key_of(te["truth"])], dtype=np.float64)
    results: dict[str, Any] = {"constant": metrics(c_const, np.full(len(c_const), float(top.iloc[0])))}
    print(f"  most frequent next event: {top.index[0]} at {top.iloc[0]:.3f}", flush=True)

    claims: dict[str, np.ndarray] = {}
    logs: dict[str, Any] = {}
    for arm in a.arms:
        model = load_checkpoint(str(a.ckpt)).to(cpu).eval()  # each arm starts from the pretrained generator
        conf = None
        saved = a.out / f"{arm}.pt"
        if arm != "native" and a.reuse and saved.exists():  # the trained arm, rebuilt with the same std(y) rows
            state = torch.load(saved)
            if arm in ("jev", "jev_ce"):
                tr_idx, _, rng = split(len(tr_t["ctx"]), a.seed)
                install_jev_head(model, g, tr_t, tr_idx, rng)
            else:
                conf = ConfidenceTokens(model.cfg.d_model)
                conf.load_state_dict(state["conf"])
                conf.eval()
            missing = model.load_state_dict(state["tok"], strict=False)
            assert not [k for k in missing.missing_keys if k.startswith(("tok_blocks", "ln_f", "head.res"))]
            print(f"  {arm}: loaded {saved}", flush=True)
        elif arm != "native":
            conf, logs[arm] = train(model, g, tr_t, arm, a.epochs, a.group, a.rows, a.seed)
            keep = ("tok_blocks", "ln_f", "head.res")
            torch.save(
                {
                    "conf": conf.state_dict() if conf is not None else None,
                    "tok": {k: v for k, v in model.state_dict().items() if k.startswith(keep)},
                },
                saved,
            )
        if arm == "jev_ce" and a.fisher:  # Fisher scoring of the intercept on the training rows
            ev0 = evaluate(model, conf, g, te_t["ctx"], te_t["dt"], te_t["truth"])
            results["jev_ce_before_fisher"] = metrics(ev0["c"], ev0["q"])
            tr_idx, _, _ = split(len(tr_t["ctx"]), a.seed)
            ti = torch.from_numpy(tr_idx)
            flog = fisher_intercept(model, g, tr_t["ctx"][ti], tr_t["truth"][ti])
            results["jev_ce_fisher_scores"] = flog
            print(f"  jev_ce: Fisher scoring of b, largest |score| {flog[0]:.2f} after one step, {flog[-1]:.4f} at the end", flush=True)
        if arm in ("jev", "jev_ce"):  # the score at t = 1, then the fitted temperature
            ev1 = evaluate(model, conf, g, te_t["ctx"], te_t["dt"], te_t["truth"])
            results[f"{arm}_t1"] = metrics(ev1["c"], ev1["q"])
            _, va_idx, _ = split(len(tr_t["ctx"]), a.seed)
            va = torch.from_numpy(va_idx)
            t_fit, nll = fit_temperature(model, g, tr_t["ctx"][va], tr_t["truth"][va])
            results[f"{arm}_temperature"] = {"t": t_fit, "val_nll": nll}
            print(f"  {arm}: temperature {t_fit} (validation NLL {nll['1.0']:.4f} at 1.0, {min(nll.values()):.4f} fitted)", flush=True)
        ev = evaluate(model, conf, g, te_t["ctx"], te_t["dt"], te_t["truth"])
        results[arm] = metrics(ev["c"], ev["q"]) | {"claims": claim_shares(g, ev["claim"])}
        if arm == "text":
            results["text_q_mean"] = metrics(ev["c"], ev["q_mean"])
            results["text_native_q"] = metrics(ev["c"], ev["native"])
        claims[arm] = ev["claim"]
        claims[f"q_{arm}"], claims[f"c_{arm}"] = ev["q"], ev["c"]
        print(f"  {arm}: " + json.dumps({k: round(v, 4) for k, v in results[arm].items() if isinstance(v, float)}), flush=True)

    (a.out / "results.json").write_text(
        json.dumps({"ckpt": str(a.ckpt), "elapsed_s": round(time.time() - t0), "results": results, "logs": logs}, indent=1)
    )
    np.savez(a.out / "claims.npz", truth=te["truth"], **claims)
    show = ["accuracy", "mean_q", "brier", "ece", "reward", "auc", "acc_top50", "acc_top20"]
    print(pd.DataFrame({k: {m: v.get(m) for m in show} for k, v in results.items() if "accuracy" in v}).T.round(4).to_string())
    print(f"written to {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
