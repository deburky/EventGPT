"""Monthly rollouts: sample a month of events from EventGPT and score the event targets, churn and counts from them.

The zero-shot scorer (eventgpt.evaluation.zero_shot) reads one step of the model: the probability that the next
event is of kind (X, Y). The labels cover a calendar month, so here the model writes the month itself. For each
row it samples K continuations event by event, from the as-of date to the end of the label month (Moscow time),
and reads from every step the probability that the event at that step is of each kind and falls inside the month.

Where sampling starts. Pretraining never shows the probe's "now" event (`[EVT] dt=<gap>`), so a continuation
sampled after it would start from a state the model was not trained to continue. Sampling therefore starts right
after the last real event, from that event's hidden state, and the first sampled event is conditioned on the
silence the client has already kept: its gap from the last event is at least g, the time from the last event to
the as-of date. Name and gap are drawn jointly from P(name) P(gap | name) restricted to gaps >= g, which also
reweights the names by how likely each is to come that late; within a gap bucket the time is uniform in seconds.

Scores per row, averaged over the K paths:
- rollout: for each kind, the sum over steps of P(the event at this step is of the kind and inside the month),
  taken up to and including the path's first sampled event of the kind. Its mean over paths is an unbiased
  estimate of the probability of the kind in the month (the compensator of the first occurrence), and it varies
  smoothly with the history where the share of paths that hit a rare kind is mostly zero.
- rollout_prod: 1 - prod (1 - p_t) over the same steps, the form that never exceeds one.
- hit: the share of paths with a sampled event of the kind inside the month.
- churn: 1 - the same compensator for "a transaction inside the month", and churn_hit, the share of paths
  with none; expected_trx, the mean number of sampled transactions inside the month.
P(kind at a step) = P(name=trx_X) P(gap inside the month | name=trx_X) P(sub=Y | name=trx_X, gap), with the gap of
the sub term set to the step's sampled gap. The earlier-occurrence rule of zero_shot applies to the kind scores.

The history decoder keeps a key-value cache: the history is encoded once, and each sampled event costs the event
encoder on its own tokens and one history step. The token decoder reruns over the few tokens of the current event.
The history is the probe's window (as in zero_shot); the sampled events extend it, so a long month on a full
window runs past the 2,048 tokens the model was trained on. `--check` compares the cached computation with
`model.history` on real events and prints the largest difference.

`--sequences` also compares each uniformly sampled row's sampled months of transactions with the month that
followed, beside two baselines that write the month without the model (eventgpt.evaluation.sequences).

Example with R=data/runs/hier-alq-s42: `uv run python -m eventgpt.evaluation.rollout $R/hier-alq-s42.pt --probe-dir $R/probe-now-2k`
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import MAX_TIME_BUCKET, Vocab
from eventgpt.evaluation.probe import FIRST_KINDS, UNDEFINED, Fold, history_tokens, mark_undefined, score
from eventgpt.evaluation.sequences import METRICS, clustered_mean, mean_compare, poisson_months, repeat_month
from eventgpt.evaluation.zero_shot import FIRST, next_event_scores, seen_before
from eventgpt.models.flat import Block, load_checkpoint, rotary
from eventgpt.models.hierarchy import EventGPT

MAX_EVENT_TOKENS = 12  # [EVT] name dt and at most nine fields; a transaction has eight or nine
N_BUCKETS = MAX_TIME_BUCKET + 1


# -------------------------------------------------------------------------------
# Gap buckets in seconds
# -------------------------------------------------------------------------------
def bucket_edges() -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edge in seconds of every gap bucket, the inverse of time_bucket; the last bucket is open."""
    b = np.arange(N_BUCKETS, dtype=np.float64)
    lo = 8.0 * np.expm1(b / 8.0)
    hi = 8.0 * np.expm1((b + 1) / 8.0)
    hi[-1] = np.inf
    return lo, hi


def window_weight(lo: torch.Tensor, hi: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """(n, buckets) share of each bucket's seconds that lies in [start, end) per row, uniform within a bucket."""
    s, e = start.unsqueeze(1), end.unsqueeze(1)
    width = (hi - lo).unsqueeze(0)
    inside = (torch.minimum(hi.unsqueeze(0), e) - torch.maximum(lo.unsqueeze(0), s)).clamp(min=0.0)
    w = torch.where(torch.isinf(width), (inside > 0).float(), inside / width.clamp(min=1e-9))
    return w.clamp(0.0, 1.0)


# -------------------------------------------------------------------------------
# History decoder with a key-value cache
# -------------------------------------------------------------------------------
def _heads(blk: Block, x: torch.Tensor, pos: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queries, keys and values (b, h, t, hd) of a block, normed and rotated as in Block.forward."""
    b, t, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, dim=2)
    q, k, v = (z.view(b, t, blk.n_head, d // blk.n_head).transpose(1, 2) for z in (q, k, v))
    if blk.q_norm is not None and blk.k_norm is not None:
        q, k = blk.q_norm(q), blk.k_norm(k)
    if pos is not None:
        q, k = rotary(q, pos), rotary(k, pos)
    return q, k, v


def _finish(blk: Block, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Output projection, residual and feed-forward of a block, from the attention output y (b, h, t, hd)."""
    b, t, d = x.shape
    x = x + blk.proj(y.transpose(1, 2).reshape(b, t, d))
    return x + blk.experts[0](blk.ln2(x))


def _hist_blocks(model: EventGPT) -> list[Block]:
    """The history decoder's blocks, typed."""
    return [blk for blk in model.hist_blocks if isinstance(blk, Block)]


def _position(model: EventGPT, elapsed: torch.Tensor, ev_pos: torch.Tensor) -> torch.Tensor:
    """Rotary position of events from their elapsed seconds and index, as TimePositions computes it."""
    if model.cfg.time_pos == "ordinal":
        return ev_pos.float()
    e = elapsed.clamp(min=0.0)
    return e / 3600.0 if model.cfg.time_pos == "linear" else 64.0 * torch.log1p(e / 60.0)


@dataclass
class State:
    """Cached history of a batch of paths: per-layer keys and values, which slots hold an event, and the clocks."""

    keys: list[torch.Tensor]  # per history layer (n, heads, slots, head_dim)
    values: list[torch.Tensor]
    valid: torch.Tensor  # (n, slots) True where a slot holds an event of the path's client
    h: torch.Tensor  # (n, d) history state after the last event
    elapsed: torch.Tensor  # (n,) seconds from the client's first event in the window, as TimePositions counts
    ev_pos: torch.Tensor  # (n,) index of the last event from the client's first event
    profile: torch.Tensor | None  # (n, d) static path vector added to every later event, or None
    next_slot: int

    def repeat(self, k: int) -> State:
        """Each path k times in a row, for k samples per history."""
        rep = lambda z: z.repeat_interleave(k, dim=0)  # noqa: E731
        return State(
            [rep(z) for z in self.keys],
            [rep(z) for z in self.values],
            rep(self.valid),
            rep(self.h),
            rep(self.elapsed),
            rep(self.ev_pos),
            None if self.profile is None else rep(self.profile),
            self.next_slot,
        )


@torch.no_grad()
def prefill(model: EventGPT, idx: torch.Tensor, real: torch.Tensor, extra: int) -> tuple[State, torch.Tensor]:
    """Encode right-padded histories as EventGPT.stages does, keeping each history layer's keys and values.

    Returns the cache with room for `extra` more events, and the history states of every event (b, n_ev, d), which
    equal model.history(idx, real)[0].
    """
    cfg = model.cfg
    event, client, pos_in_event, n_ev = model.segmenter(idx, real)
    x = model.tok_emb(idx) if cfg.rope else model.tok_emb(idx) + model.pos_in_event(pos_in_event)
    route = model.router(idx, event, n_ev, model.kind_lut) if cfg.n_experts > 1 else None
    pk = None
    if cfg.packed:
        pk = model.packer(event, n_ev, real)
        enc_mask, _ = pk.masks()
        xp = pk.pack(x)
        route_p = pk.pack(route) if route is not None else None
        for blk in model.event_blocks:
            xp = blk(xp, enc_mask, route=route_p)
        ev = pk.pool(model.ln_event(xp))
    else:
        same_event, _ = model.masks(event)
        for blk in model.event_blocks:
            x = blk(x, same_event, route=route)
        ev = model.pool(model.ln_event(x), event, n_ev)
    first_ev, ev_pos, hist_mask = model.layout(event, client, n_ev)
    b = idx.size(0)
    rows = torch.arange(b, device=idx.device)
    last = event[rows, real.long().sum(1) - 1]  # last real event of each row
    profile = None
    if model.profile_proj is not None and pk is not None:
        sp = model.static_path(idx, pk, first_ev)
        ev = ev + sp
        profile = sp[rows, last]
    gap = torch.zeros(b, n_ev, device=idx.device)
    gap.scatter_add_(1, event, model.dt_seconds[idx])
    cum = torch.cumsum(gap, dim=1)
    elapsed = cum - torch.gather(cum, 1, first_ev) + torch.gather(gap, 1, first_ev)
    pos = model.time_positions(idx, event, n_ev, first_ev, model.dt_seconds, ev_pos) if cfg.rope else None
    h = ev if cfg.rope else ev + model.pos_event(ev_pos)
    slots = n_ev + extra
    keys, values = [], []
    for blk in _hist_blocks(model):
        q, k, v = _heads(blk, h, pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=hist_mask)
        h = _finish(blk, h, y)
        kc = k.new_zeros(b, k.size(1), slots, k.size(3))
        vc = v.new_zeros(b, v.size(1), slots, v.size(3))
        kc[:, :, :n_ev], vc[:, :, :n_ev] = k, v
        keys.append(kc)
        values.append(vc)
    h = model.ln_hist(h)
    occupied = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device).scatter_(1, event, True)
    h = h * occupied.unsqueeze(-1).to(h.dtype)
    valid = torch.zeros(b, slots, dtype=torch.bool, device=idx.device)
    valid[:, :n_ev] = hist_mask[rows, 0, last]
    state = State(keys, values, valid, h[rows, last], elapsed[rows, last], ev_pos[rows, last], profile, n_ev)
    return state, h


@torch.no_grad()
def encode_events(model: EventGPT, toks: torch.Tensor, filled: torch.Tensor) -> torch.Tensor:
    """Event vectors (n, d) of one event per row, tokens (n, L) with `filled` marking real slots, as the event encoder."""
    cfg = model.cfg
    n, width = toks.shape
    slot = torch.arange(width, device=toks.device).clamp(max=63)
    x = model.tok_emb(toks) if cfg.rope else model.tok_emb(toks) + model.pos_in_event(slot).unsqueeze(0)
    route = None
    if cfg.n_experts > 1:
        route = model.kind_lut[toks[:, 1]].unsqueeze(1).expand(-1, width)
    keys = filled.clone()
    keys[:, 0] = True
    mask = keys[:, None, None, :].expand(-1, 1, width, -1)
    for blk in model.event_blocks:
        x = blk(x, mask, route=route)
    w = filled.unsqueeze(-1).to(x.dtype)
    return (model.ln_event(x) * w).sum(1) / w.sum(1).clamp(min=1)


@torch.no_grad()
def history_step(model: EventGPT, st: State, toks: torch.Tensor, filled: torch.Tensor) -> None:
    """Append one event per path to the cache and move st.h to the state after it."""
    ev = encode_events(model, toks, filled)
    if st.profile is not None:
        ev = ev + st.profile
    st.elapsed = st.elapsed + (model.dt_seconds[toks] * filled.to(st.elapsed.dtype)).sum(1)
    st.ev_pos = (st.ev_pos + 1).clamp(max=model.cfg.max_events - 1)
    pos = _position(model, st.elapsed, st.ev_pos).unsqueeze(1) if model.cfg.rope else None
    h = ev.unsqueeze(1) if model.cfg.rope else (ev + model.pos_event(st.ev_pos)).unsqueeze(1)
    s = st.next_slot
    st.valid[:, s] = True
    mask = st.valid[:, None, None, : s + 1]
    for i, blk in enumerate(_hist_blocks(model)):
        q, k, v = _heads(blk, h, pos)
        st.keys[i][:, :, s] = k[:, :, 0]
        st.values[i][:, :, s] = v[:, :, 0]
        y = F.scaled_dot_product_attention(q, st.keys[i][:, :, : s + 1], st.values[i][:, :, : s + 1], attn_mask=mask)
        h = _finish(blk, h, y)
    st.h = model.ln_hist(h)[:, 0]
    st.next_slot = s + 1


@torch.no_grad()
def decode(model: EventGPT, toks: torch.Tensor, h: torch.Tensor, at: int, temperature: float = 1.0) -> torch.Tensor:
    """Token decoder log-probabilities (n, vocab) of the token after prefix position `at`, given h (n, d), logits divided by temperature."""
    t = toks.size(1)
    slot = torch.arange(t, device=toks.device).clamp(max=63)
    y = model.tok_emb(toks) + model.pos_in_event(slot).unsqueeze(0) + h.unsqueeze(1)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=toks.device))[None, None]
    for blk in model.tok_blocks:
        y = blk(y, mask)
    return torch.log_softmax(model.head(model.ln_f(y[:, at])).float() / temperature, -1)


# -------------------------------------------------------------------------------
# Sampling a month
# -------------------------------------------------------------------------------
@dataclass
class Tables:
    """Token ids the sampler needs, on the device."""

    evt: int
    names: torch.Tensor  # name tokens that may start a sampled event (every name but the profile)
    is_trx: torch.Tensor  # (len(names),) True for transaction names
    dt: torch.Tensor  # (buckets,) dt token per bucket
    stop: torch.Tensor  # tokens that end an event: [EVT], [USR] and padding
    kind_name: torch.Tensor  # (kinds,) index into names of each kind's type
    kind_sub: torch.Tensor  # (kinds,) sub token of each kind
    sub_of_tok: torch.Tensor  # (vocab,) subtype value of a sub token, -1 otherwise
    lo: torch.Tensor  # (buckets,) bucket edges in seconds
    hi: torch.Tensor


def tables(vocab: Vocab, dev: torch.device) -> Tables:
    """Token tables for the sampler from the vocabulary."""
    names = [t for t in vocab.structured if t.startswith("name=") and t != "name=profile"]
    sub_of_tok = torch.full((vocab.size,), -1, dtype=torch.long)
    for t, i in vocab.structured.items():
        if t.startswith("sub=") and t[4:].lstrip("-").isdigit():
            sub_of_tok[i] = int(t[4:])
    lo, hi = bucket_edges()
    return Tables(
        evt=vocab.id("[EVT]"),
        names=torch.tensor([vocab.id(t) for t in names], device=dev),
        is_trx=torch.tensor([t.startswith("name=trx_") for t in names], device=dev),
        dt=torch.tensor([vocab.id(f"dt={b}") for b in range(N_BUCKETS)], device=dev),
        stop=torch.tensor([vocab.id("[EVT]"), vocab.id("[USR]"), 0], device=dev),
        kind_name=torch.tensor([names.index(f"name=trx_{x}") for x, _ in FIRST_KINDS], device=dev),
        kind_sub=torch.tensor([vocab.id(f"sub={y}") for _, y in FIRST_KINDS], device=dev),
        sub_of_tok=sub_of_tok.to(dev),
        lo=torch.tensor(lo, dtype=torch.float32, device=dev),
        hi=torch.tensor(hi, dtype=torch.float32, device=dev),
    )


@torch.no_grad()
def sample_month(
    model: EventGPT,
    st: State,
    tb: Tables,
    silence: torch.Tensor,
    month: torch.Tensor,
    max_events: int,
    gen: torch.Generator,
    trace: list[dict[str, np.ndarray]] | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Sample every path's events to the end of its month and accumulate the per-path scores.

    silence: (n,) seconds from the last event to the as-of date; month: (n,) seconds from the as-of date to the end
    of the label month. Returns per-path tensors: comp, prod and hit (n, kinds), trx_comp, any_trx, n_trx and capped (n,).
    With trace, each step appends the sampled events: time from the as-of date, name token, subtype and whether
    the event falls inside the month, one entry per path. A temperature other than 1 divides every logit of the
    token decoder, so the paths and the scores both come from the tempered model.
    """
    n, dev = silence.size(0), silence.device
    k_kinds = len(FIRST_KINDS)
    n_names = len(tb.names)
    comp = torch.zeros(n, k_kinds, device=dev)
    prod = torch.ones(n, k_kinds, device=dev)
    hit = torch.zeros(n, k_kinds, dtype=torch.bool, device=dev)
    trx_comp = torch.zeros(n, device=dev)
    any_trx = torch.zeros(n, dtype=torch.bool, device=dev)
    n_trx = torch.zeros(n, device=dev)
    t_prev = -silence  # time of the last event, seconds from the as-of date
    active = torch.ones(n, dtype=torch.bool, device=dev)
    evt = torch.full((n, 1), tb.evt, dtype=torch.long, device=dev)
    # every path steps every time, so the cache is never gathered; a path that has left its month is masked out
    for step in range(max_events):
        if not active.any():
            break
        h = st.h
        # P(name) at [EVT], and P(gap bucket | name) at the name token for every name
        p_name = decode(model, evt, h, 0, temperature).exp()[:, tb.names]
        pair = torch.stack([evt.expand(-1, n_names), tb.names.unsqueeze(0).expand(n, -1)], -1).view(n * n_names, 2)
        p_gap = decode(model, pair, h.repeat_interleave(n_names, 0), 1, temperature).exp()[:, tb.dt].view(n, n_names, N_BUCKETS)
        p_gap = p_gap / p_gap.sum(-1, keepdim=True).clamp(min=1e-12)
        # the first event comes after the silence; every event's gap is counted from the previous event
        start = silence if step == 0 else torch.zeros(n, device=dev)
        w_after = window_weight(tb.lo, tb.hi, start, torch.full_like(start, math.inf))
        w_in = window_weight(tb.lo, tb.hi, start, month - t_prev)
        joint = p_name.unsqueeze(-1) * p_gap * w_after.unsqueeze(1)  # (n, names, buckets)
        z = joint.sum((1, 2)).clamp(min=1e-12)
        p_in = (p_name.unsqueeze(-1) * p_gap * w_in.unsqueeze(1)).sum(-1) / z.unsqueeze(1)  # (n, names): inside the month
        # sample (name, bucket) jointly, then the time uniformly within the bucket and after the silence
        flat = torch.multinomial((joint / z.view(n, 1, 1)).view(n, -1), 1, generator=gen).squeeze(1)
        ni, bi = flat // N_BUCKETS, flat % N_BUCKETS
        lo = torch.maximum(tb.lo[bi], start)
        hi = tb.hi[bi]
        u = torch.rand(n, device=dev, generator=gen)
        gap_s = torch.where(torch.isinf(hi), torch.full_like(lo, math.inf), lo + u * (hi - lo).clamp(min=0.0))
        t_new = t_prev + gap_s
        inside = active & (t_new < month)
        name_tok, dt_tok = tb.names[ni], tb.dt[bi]
        # P(sub=Y | name=trx_X, the sampled gap) for each kind's type
        kind_toks = torch.stack(
            [evt.expand(-1, k_kinds), tb.names[tb.kind_name].unsqueeze(0).expand(n, -1), dt_tok.unsqueeze(1).expand(-1, k_kinds)], -1
        ).view(n * k_kinds, 3)
        p_sub = decode(model, kind_toks, h.repeat_interleave(k_kinds, 0), 2, temperature).exp().view(n, k_kinds, -1)
        p_sub = torch.gather(p_sub, 2, tb.kind_sub.view(1, -1, 1).expand(n, -1, 1)).squeeze(2)
        p_kind = p_in[:, tb.kind_name] * p_sub
        q_trx = (p_in * tb.is_trx.unsqueeze(0)).sum(1)
        # the rest of the event's tokens
        toks = torch.zeros(n, MAX_EVENT_TOKENS, dtype=torch.long, device=dev)
        toks[:, 0], toks[:, 1], toks[:, 2] = tb.evt, name_tok, dt_tok
        filled = torch.zeros(n, MAX_EVENT_TOKENS, dtype=torch.bool, device=dev)
        filled[:, :3] = True
        open_ = inside.clone()
        for j in range(3, MAX_EVENT_TOKENS):
            if not open_.any():
                break
            logp = decode(model, toks[:, :j], h, j - 1, temperature)
            nxt = torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
            open_ = open_ & ~torch.isin(nxt, tb.stop)
            toks[:, j] = torch.where(open_, nxt, toks[:, j])
            filled[:, j] = open_
        sub = tb.sub_of_tok[toks[:, 3]]
        if trace is not None:
            step_events = {"t": t_new, "name": name_tok, "sub": sub, "inside": inside}
            trace.append({key: v.cpu().numpy() for key, v in step_events.items()})
        # accumulate: kind terms up to and including a path's first sampled event of the kind
        count = active.unsqueeze(1) & ~hit
        comp += p_kind * count
        prod *= torch.where(count, 1.0 - p_kind, torch.ones_like(p_kind))
        is_kind = torch.stack([(name_tok == tb.names[tb.kind_name[k]]) & (sub == y) for k, (_, y) in enumerate(FIRST_KINDS)], 1)
        hit |= is_kind & inside.unsqueeze(1)
        trx_comp += q_trx * (active & ~any_trx)
        new_trx = tb.is_trx[ni] & inside
        any_trx |= new_trx
        n_trx += new_trx.float()
        t_prev = torch.where(active, t_new, t_prev)
        active = inside
        if active.any():
            history_step(model, st, toks, filled)
    return {"comp": comp, "prod": prod, "hit": hit, "trx_comp": trx_comp, "any_trx": any_trx, "n_trx": n_trx, "capped": active}


# -------------------------------------------------------------------------------
# Checks and main
# -------------------------------------------------------------------------------
def path_mean(z: torch.Tensor, k: int) -> np.ndarray:
    """Mean over each row's k consecutive paths: (rows * k, ...) -> (rows, ...)."""
    return z.float().view(-1, k, *z.shape[1:]).mean(1).cpu().numpy()


def pad(seqs: list[np.ndarray], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded batch and its mask of real tokens."""
    x = np.zeros((len(seqs), max(len(s) for s in seqs)), dtype=np.int64)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = s
    xt = torch.from_numpy(x).to(dev)
    return xt, xt != 0


@torch.no_grad()
def check(model: EventGPT, fold: Fold, rows: pd.DataFrame, dev: torch.device, n_check: int = 4, m: int = 5) -> float:
    """Largest difference between cached incremental history states and model.history on the full sequence.

    For short histories (no truncation), prefill all but the last m events, feed the last m real events through
    history_step, and compare each step's state with the full forward pass; also compare the prefill itself.
    """
    worst = 0.0
    cand = rows[(rows["n_hist"] > m + 3) & (rows["n_hist"] < 120)].head(n_check)
    aligned = bool(getattr(model.cfg, "aligned", False))
    for _, r in cand.iterrows():
        ci, n = int(fold.index[r["client_id"]]), int(r["n_hist"])
        full = history_tokens(fold, ci, n, 100_000, aligned=aligned)
        part = history_tokens(fold, ci, n - m, 100_000, aligned=aligned)
        xt, real = pad([full], dev)
        h_full, event, _ = model.history(xt, real)
        xp, rp = pad([part], dev)
        st, h_pre = prefill(model, xp, rp, m)
        h_ref, _, _ = model.history(xp, rp)
        worst = max(worst, float((h_pre - h_ref).abs().max()))
        starts = np.flatnonzero(full == model.cfg.evt_id)
        tail = starts[-m:]
        ends = [*tail[1:], len(full)]
        for s, e in zip(tail, ends, strict=True):
            toks = torch.zeros(1, MAX_EVENT_TOKENS, dtype=torch.long, device=dev)
            toks[0, : e - s] = torch.from_numpy(full[s:e])
            filled = toks != 0
            history_step(model, st, toks, filled)
            ev_idx = int(event[0, s])
            worst = max(worst, float((st.h[0] - h_full[0, ev_idx]).abs().max()))
    return worst


def month_events(fold: Fold, start: int, stop: int, cut: float) -> tuple[np.ndarray, np.ndarray]:
    """Transactions start..stop of the fold as (days from the cut, kind = etype * 1000 + subtype)."""
    e = np.arange(start, stop)
    e = e[fold.is_trx[e]]
    return (fold.event_time[e] - cut) / 86_400.0, fold.etype[e].astype(np.int64) * 1000 + fold.esub[e]


def sequence_rows(
    trace: list[dict[str, np.ndarray]],
    sel: np.ndarray,
    k: int,
    rows: pd.DataFrame,
    fold: Fold,
    first: np.ndarray,
    n_hist: np.ndarray,
    month: np.ndarray,
    in_rand: np.ndarray,
    etype_of_tok: dict[int, int],
    rng: np.random.Generator,
    out: dict[str, list[dict[str, float]]],
    index: list[int],
) -> None:
    """Sequence metrics of the batch's uniformly sampled rows, for the model's months and the two baselines."""
    t = np.stack([s["t"] for s in trace], 1) / 86_400.0
    inside = np.stack([s["inside"] for s in trace], 1)
    sub = np.stack([s["sub"] for s in trace], 1)
    etype = np.vectorize(lambda tok: etype_of_tok.get(int(tok), -1))(np.stack([s["name"] for s in trace], 1))
    keep = inside & (etype >= 0)
    kinds = etype.astype(np.int64) * 1000 + sub
    cuts, ends = rows["cut"].to_numpy(), rows["n_month"].to_numpy()
    for b, j in enumerate(sel):
        if not in_rand[j]:
            continue
        a, h = int(first[j]), int(n_hist[j])
        real = month_events(fold, a + h, a + int(ends[j]), float(cuts[j]))
        past = month_events(fold, a, a + h, float(cuts[j]))
        paths = [(t[p][keep[p]], kinds[p][keep[p]]) for p in range(b * k, (b + 1) * k)]
        days = float(month[j]) / 86_400.0
        out["model"].append(mean_compare(paths, real))
        out["repeat"].append(mean_compare([repeat_month(*past, days)], real))
        out["poisson"].append(mean_compare(poisson_months(*past, days, k, rng), real))
        index.append(int(j))


def main() -> None:
    """Rollout scores, the single-step zero-shot score on the same rows, churn and count calibration."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--probe-dir", type=Path, required=True, help="probe output with test_rows")
    p.add_argument("--k", type=int, default=16, help="sampled months per row")
    p.add_argument("--n-random", type=int, default=20_000, help="test rows sampled uniformly beside the positives")
    p.add_argument("--max-events", type=int, default=120, help="cap on sampled events per month")
    p.add_argument("--batch", type=int, default=16, help="rows per batch; each batch samples batch * k paths")
    p.add_argument("--block", type=int, default=0, help="history window in tokens; 0 takes the probe run's window")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="score at most this many rows, for smoke tests")
    p.add_argument("--device", default="", help="cpu, cuda or mps; default cuda if available, else cpu")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast on cuda")
    p.add_argument("--out", type=Path, default=None, help="csv path; default <ckpt dir>/rollout.csv")
    p.add_argument("--save-scores", action="store_true", help="also write each row's scores to rollout_scores.parquet")
    p.add_argument("--check", action="store_true", help="compare cached history states with model.history and stop")
    p.add_argument("--temperature", type=float, default=1.0, help="divide the token decoder's logits by this when sampling and scoring")
    p.add_argument("--clients", choices=["all", "select", "report"], default="all", help="all test clients, or one half of them")
    p.add_argument("--uniform-only", action="store_true", help="skip the positives outside the uniform sample")
    p.add_argument("--sequences", action="store_true", help="also compare sampled months with the real month and two baselines")
    a = p.parse_args()

    t0 = time.time()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = Vocab.load(OUT / "vocab")
    model = load_checkpoint(str(a.ckpt)).to(dev).eval()
    if not isinstance(model, EventGPT):
        raise SystemExit("rollouts need an EventGPT checkpoint")
    block = a.block or int(json.loads((a.probe_dir / "results.json").read_text())["block"])
    te = mark_undefined(pd.read_parquet(a.probe_dir / "test_rows.parquet"))
    if a.clients != "all":
        # client ids are hex hashes: the last digit splits test clients into a half for choosing settings and a half for reporting
        odd = te["client_id"].str[-1].map(lambda c: int(c, 16) % 2 == 1)
        te = te[odd == (a.clients == "select")].reset_index(drop=True)
    fold = Fold(int(te["fold"].iloc[0]), vocab)
    if a.check:
        print(f"max |cached - full| history state: {check(model, fold, te, dev):.2e}")
        return
    rand = np.random.default_rng(a.seed).choice(len(te), a.n_random, replace=False)
    pick = rand if a.uniform_only else np.union1d(np.flatnonzero(te[FIRST].to_numpy().any(1)), rand)
    if a.limit:
        pick = np.sort(np.random.default_rng(a.seed).choice(pick, min(a.limit, len(pick)), replace=False))
    in_rand = np.isin(pick, rand)
    rows = te.iloc[pick].reset_index(drop=True)
    print(f"{a.ckpt.name}: {len(rows):,} rows, k={a.k}, on {dev}", flush=True)

    ci, n_hist = rows["client_id"].map(fold.index).to_numpy(), rows["n_hist"].to_numpy()
    first = fold.client_events[ci]
    last = fold.event_time[np.maximum(first + n_hist - 1, 0)]
    silence = np.where(n_hist > 0, np.maximum(rows["cut"].to_numpy() - last, 0), 0).astype(np.float64)
    month = (rows["cut_end"].to_numpy() - rows["cut"].to_numpy()).astype(np.float64)
    aligned = bool(getattr(model.cfg, "aligned", False))
    hist = [history_tokens(fold, int(c), int(n), block, aligned=aligned) for c, n in zip(ci, n_hist, strict=True)]
    head = np.array([vocab.id("[EVT]"), vocab.id(f"name=trx_{FIRST_KINDS[0][0]}")])
    gap_tok = vocab.dt_tokens(silence)
    zs_seqs = [history_tokens(fold, int(c), int(n), block, np.r_[head, g], aligned) for c, n, g in zip(ci, n_hist, gap_tok, strict=True)]
    tb = tables(vocab, dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    n_k = len(FIRST_KINDS)
    out = {key: np.zeros((len(rows), n_k)) for key in ("rollout", "rollout_prod", "hit", "zero_shot")}
    churn, churn_hit, expected, capped = (np.zeros(len(rows)) for _ in range(4))
    order = np.argsort([len(s) for s in hist])
    seq_rows: dict[str, list[dict[str, float]]] = {"model": [], "repeat": [], "poisson": []}
    seq_index: list[int] = []
    etype_of_tok = {i: int(t.removeprefix("name=trx_")) for t, i in vocab.structured.items() if t.startswith("name=trx_")}
    rng = np.random.default_rng(a.seed)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16 and dev.type == "cuda")
    t1 = time.time()
    for i in range(0, len(order), a.batch):
        sel = order[i : i + a.batch]
        with autocast:
            out["zero_shot"][sel] = next_event_scores(model, [zs_seqs[j] for j in sel], vocab, dev)
            xt, real = pad([hist[j] for j in sel], dev)
            st, _ = prefill(model, xt, real, a.max_events)
            st = st.repeat(a.k)
            s_t = torch.tensor(silence[sel], dtype=torch.float32, device=dev).repeat_interleave(a.k)
            m_t = torch.tensor(month[sel], dtype=torch.float32, device=dev).repeat_interleave(a.k)
            trace: list[dict[str, np.ndarray]] | None = [] if a.sequences else None
            r = sample_month(model, st, tb, s_t, m_t, a.max_events, gen, trace, a.temperature)
        if trace is not None:
            sequence_rows(trace, sel, a.k, rows, fold, first, n_hist, month, in_rand, etype_of_tok, rng, seq_rows, seq_index)
        per = functools.partial(path_mean, k=a.k)
        out["rollout"][sel], out["rollout_prod"][sel] = per(r["comp"]), 1.0 - per(r["prod"])
        out["hit"][sel] = per(r["hit"])
        churn[sel], churn_hit[sel] = 1.0 - per(r["trx_comp"]), 1.0 - per(r["any_trx"])
        expected[sel], capped[sel] = per(r["n_trx"]), per(r["capped"])
        done = i + len(sel)
        if (i // a.batch) % 20 == 0 or done == len(order):
            rate = done / (time.time() - t1)
            print(f"  {done:,}/{len(order):,} rows, {rate:.2f} rows/s ({rate * a.k:.1f} paths/s)", flush=True)

    unseen = ~seen_before(fold, ci, n_hist)
    res = []
    for k, lab in enumerate(FIRST):
        ev = in_rand | (rows[lab].to_numpy() == 1)
        y = rows[lab].to_numpy()[ev]
        res.append(
            (
                lab,
                int(y.sum()),
                *(score(y, (out[c][:, k] * unseen[:, k])[ev])["roc_auc"] for c in ("zero_shot", "rollout", "rollout_prod", "hit")),
            )
        )
    r_df = pd.DataFrame(res, columns=["label", "positives", "zero_shot", "rollout", "rollout_prod", "hit"]).set_index("label")
    y_c = rows["churn"].to_numpy()
    ev = in_rand & (y_c != UNDEFINED)
    r_df.loc["churn", ["positives", "rollout", "hit"]] = [
        int(y_c[ev].sum()),
        score(y_c[ev], churn[ev])["roc_auc"],
        score(y_c[ev], churn_hit[ev])["roc_auc"],
    ]
    realized = np.array([fold.is_trx[s + h : s + mth].sum() for s, h, mth in zip(first, n_hist, rows["n_month"].to_numpy(), strict=True)])
    u = in_rand
    calib = {
        "mean_expected_trx": float(expected[u].mean()),
        "mean_realized_trx": float(realized[u].mean()),
        "pearson": float(np.corrcoef(expected[u], realized[u])[0, 1]) if u.sum() > 2 else float("nan"),
        "spearman": float(pd.Series(expected[u]).corr(pd.Series(realized[u]), method="spearman")) if u.sum() > 2 else float("nan"),
        "capped_share": float(capped.mean()),
    }
    print(r_df.round(4).to_string())
    print(json.dumps({k: round(v, 4) for k, v in calib.items()}))
    dest = a.out or a.ckpt.parent / "rollout.csv"
    r_df.to_csv(dest)
    dest.with_name(dest.stem + "_calibration.json").write_text(json.dumps(calib, indent=1))
    if a.save_scores:
        saved = rows[["client_id", "mon", "cut", "n_hist", *FIRST, "churn"]].assign(in_rand=in_rand)
        for k, lab in enumerate(FIRST):
            for c in out:
                saved[f"{c}_{lab}"] = out[c][:, k]
            saved[f"unseen_{lab}"] = unseen[:, k]
        saved = saved.assign(rollout_churn=churn, hit_churn=churn_hit, expected_trx=expected, realized_trx=realized, capped=capped)
        saved.to_parquet(dest.with_name("rollout_scores.parquet"))
    if a.sequences:
        clients = rows["client_id"].to_numpy()[seq_index]
        per_row = pd.concat({src: pd.DataFrame(v) for src, v in seq_rows.items()}, axis=1)
        summary = []
        for m in METRICS:
            line: dict[str, object] = {"metric": m}
            for src in seq_rows:
                line[src], line[f"{src}_se"] = clustered_mean(per_row[(src, m)].to_numpy(), clients)
            for base in ("repeat", "poisson"):
                diff = (per_row[("model", m)] - per_row[(base, m)]).to_numpy()
                line[f"model-{base}"], line[f"model-{base}_se"] = clustered_mean(diff, clients)
            summary.append(line)
        s_df = pd.DataFrame(summary).set_index("metric")
        print(s_df.round(4).to_string())
        s_df.to_csv(dest.with_name(dest.stem + "_sequences.csv"))
        levels = zip(per_row.columns.get_level_values(0), per_row.columns.get_level_values(1), strict=True)
        per_row.columns = pd.Index([f"{src}_{m}" for src, m in levels])
        per_row.assign(client_id=clients, row=seq_index).to_parquet(dest.with_name(dest.stem + "_sequences.parquet"))
    print(f"written to {dest} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
