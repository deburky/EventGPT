"""EventGPT: bidirectional event encoder -> causal history decoder over events -> token decoder.

Tokens of one event (from [EVT] to the next [EVT] or [USR]) are encoded bidirectionally within the event and
mean-pooled to one vector. A causal decoder runs over the event vectors of each client (segment mask at [USR],
positions restart per client). The tokens of event i+1 are then predicted causally within the event, conditioned
on the history state at event i, with the same tied vocabulary head as the flat GPT. The probe reads the history
state of the last event in the prefix.

The bookkeeping (segmentation, pooling, layout of events per client, expert routing, time positions) lives in
parameter-free submodules so a module-level trace of the model reads as its stages rather than as tensor ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eventgpt.models.flat import Block, GPTConfig, dt_midpoint


@dataclass
class EventGPTConfig(GPTConfig):
    """GPTConfig plus the hybrid's layer split (n_layer_event, n_layer_tok), the [EVT] id and the event-position budget."""

    n_layer_event: int = 2
    n_layer_tok: int = 2
    evt_id: int = 2
    max_events: int = 512  # positions for the history decoder
    rope: bool = False  # no positions inside an event; time-aware rotary positions over events
    time_pos: str = "log"  # rotary position of an event with rope: log (64 ln(1 + minutes)), linear (hours) or ordinal (index)
    n_experts: int = 1  # feed-forward experts in the event encoder, hard-routed by event kind
    packed: bool = True  # token stages attend within packed per-event rows (linear in the window) rather than dense masks
    profile_path: bool = False  # the profile event's field embeddings added to every later event of the client (needs packed)
    profile_fields: int = 4  # field tokens of the profile event, after [EVT] name=profile dt=<b>


def _same(a: torch.Tensor) -> torch.Tensor:
    """(b, 1, t, t) mask, True where two positions share the value of a (b, t)."""
    return (a.unsqueeze(2) == a.unsqueeze(1)).unsqueeze(1)


def _tril(t: int, device: torch.device) -> torch.Tensor:
    """(1, 1, t, t) causal mask."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


# -------------------------------------------------------------------------------
# Parameter-free stages
# -------------------------------------------------------------------------------
class Segmenter(nn.Module):
    """Per token: event index within the window, client index within the window, position within the event.

    Padding (key_mask False) is moved to one extra trailing event so it neither attends into nor pools into
    the last real event; the history decoder is causal, so real events never see that trailing event.
    """

    def __init__(self, evt_id: int, usr_id: int) -> None:
        super().__init__()
        self.evt_id, self.usr_id = evt_id, usr_id

    def forward(self, idx: torch.Tensor, key_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Returns (event, client, pos_in_event), each (b, t), and the number of events in the window."""
        starts = (idx == self.evt_id) | (idx == self.usr_id)
        event = torch.cumsum(starts.long(), dim=1)  # 0 for a leading partial event
        if key_mask is not None:
            event = torch.where(key_mask, event, torch.full_like(event, int(event.max().item()) + 1))
        client = torch.cumsum((idx == self.usr_id).long(), dim=1)
        ar = torch.arange(idx.size(1), device=idx.device).unsqueeze(0).expand_as(idx)
        start_pos = torch.cummax(torch.where(starts, ar, torch.zeros_like(ar)), dim=1).values
        pos_in_event = (ar - start_pos).clamp(max=63)
        return event, client, pos_in_event, int(event.max().item()) + 1


class EventMasks(nn.Module):
    """Attention masks over tokens: same-event (bidirectional) and same-event causal, both (b, 1, t, t)."""

    def forward(self, event: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (same_event, same_event & causal)."""
        same = _same(event)
        return same, same & _tril(event.size(1), event.device)


class EventPool(nn.Module):
    """Mean of the token states of each event: (b, t, d) -> (b, n_ev, d)."""

    def forward(self, x: torch.Tensor, event: torch.Tensor, n_ev: int) -> torch.Tensor:
        """Scatter-mean by event index."""
        b = x.size(0)
        ev_sum = torch.zeros(b, n_ev, x.size(-1), device=x.device, dtype=x.dtype)
        ev_sum.scatter_add_(1, event.unsqueeze(-1).expand_as(x), x)
        counts = torch.zeros(b, n_ev, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, event, torch.ones_like(event, dtype=x.dtype))
        return ev_sum / counts.clamp(min=1).unsqueeze(-1)


class HistoryLayout(nn.Module):
    """Per event: client index, first event of its client, position within the client, and the causal per-client mask."""

    def __init__(self, max_events: int) -> None:
        super().__init__()
        self.max_events = max_events

    def forward(self, event: torch.Tensor, client: torch.Tensor, n_ev: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (first event of the client per event (b, n_ev), event position (b, n_ev), mask (b, 1, n_ev, n_ev))."""
        b = event.size(0)
        ev_client = torch.zeros(b, n_ev, dtype=torch.long, device=event.device)
        ev_client.scatter_(1, event, client)
        ar_ev = torch.arange(n_ev, device=event.device).unsqueeze(0).expand(b, -1)
        is_client_start = torch.zeros_like(ev_client, dtype=torch.bool)
        is_client_start[:, 1:] = ev_client[:, 1:] != ev_client[:, :-1]
        first_ev = torch.cummax(torch.where(is_client_start, ar_ev, torch.zeros_like(ar_ev)), dim=1).values
        ev_pos = (ar_ev - first_ev).clamp(max=self.max_events - 1)
        return first_ev, ev_pos, _same(ev_client) & _tril(n_ev, event.device)


class KindRouter(nn.Module):
    """Expert index per token: the kind of its event, read from the name token that follows [EVT]."""

    def __init__(self, evt_id: int) -> None:
        super().__init__()
        self.evt_id = evt_id

    def forward(self, idx: torch.Tensor, event: torch.Tensor, n_ev: int, kind_lut: torch.Tensor) -> torch.Tensor:
        """Route (b, t) with values in [0, n_experts)."""
        b, _ = idx.shape
        is_name = torch.zeros_like(idx, dtype=torch.bool)
        is_name[:, 1:] = idx[:, :-1] == self.evt_id
        kind_tok = torch.where(is_name, kind_lut[idx], torch.zeros_like(idx))
        ev_kind = torch.zeros(b, n_ev, dtype=torch.long, device=idx.device)
        ev_kind.scatter_add_(1, event, kind_tok)
        return torch.gather(ev_kind, 1, event)


class TimePositions(nn.Module):
    """Rotary position per event from the time since the client's first event in the window.

    log: 64 * log1p(minutes), the scheme of the paper; linear: hours; ordinal: the event's index from the client's
    first event, which keeps every other part of the model the same for an ablation of the time scale.
    """

    def __init__(self, scheme: str = "log") -> None:
        super().__init__()
        if scheme not in ("log", "linear", "ordinal"):
            raise ValueError(f"unknown time_pos {scheme}")
        self.scheme = scheme

    def forward(
        self, idx: torch.Tensor, event: torch.Tensor, n_ev: int, first_ev: torch.Tensor, dt_seconds: torch.Tensor, ev_pos: torch.Tensor
    ) -> torch.Tensor:
        """Positions (b, n_ev)."""
        if self.scheme == "ordinal":
            return ev_pos.float()
        b, _ = idx.shape
        gap = torch.zeros(b, n_ev, device=idx.device)
        gap.scatter_add_(1, event, dt_seconds[idx])
        elapsed = torch.cumsum(gap, dim=1)
        elapsed = elapsed - torch.gather(elapsed, 1, first_ev) + torch.gather(gap, 1, first_ev)
        # float32 sums of 1e10+ seconds lose whole minutes; on MPS the difference can dip below zero and log1p would give NaN
        elapsed = elapsed.clamp(min=0.0)
        return elapsed / 3600.0 if self.scheme == "linear" else 64.0 * torch.log1p(elapsed / 60.0)


class EventPacker(nn.Module):
    """Lay each event's tokens out as one short row, so attention within an event is cheap.

    The event encoder and the token decoder never attend across an event boundary, yet a dense (t, t) mask
    makes them pay for every pair of tokens in the window: memory grows with the square of the window, and at
    8k tokens that is gigabytes per layer. Packed, a window of t tokens is (b * n_ev) rows of L slots, L the
    longest event (about ten on MBD), and attention costs n_ev * L^2. Same computation, same masks, per row.

    Only real tokens are packed: padding (key_mask False) stays out and comes back as zeros.
    """

    def forward(self, event: torch.Tensor, n_ev: int, real: torch.Tensor) -> Packing:
        """Row and slot of every real token, and which slots of each row hold one."""
        b, t = event.shape
        ar = torch.arange(t, device=event.device).unsqueeze(0).expand(b, -1)
        starts = torch.ones_like(event, dtype=torch.bool)
        starts[:, 1:] = event[:, 1:] != event[:, :-1]
        slot = ar - torch.cummax(torch.where(starts, ar, torch.zeros_like(ar)), dim=1).values
        bi, ti = real.nonzero(as_tuple=True)
        row = bi * n_ev + event[bi, ti]
        col = slot[bi, ti]
        width = int(col.max().item()) + 1 if len(col) else 1
        filled = torch.zeros(b * n_ev, width, dtype=torch.bool, device=event.device)
        filled[row, col] = True
        return Packing(bi, ti, row, col, b, t, n_ev, width, filled)


@dataclass
class Packing:
    """Where each real token sits in the packed (b * n_ev, width) layout, from EventPacker."""

    bi: torch.Tensor
    ti: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    b: int
    t: int
    n_ev: int
    width: int
    filled: torch.Tensor  # (b * n_ev, width), True where a slot holds a token

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        """(b, t, ...) -> (b * n_ev, width, ...), empty slots zero."""
        out = x.new_zeros((self.b * self.n_ev, self.width, *x.shape[2:]))
        out[self.row, self.col] = x[self.bi, self.ti]
        return out

    def unpack(self, xp: torch.Tensor) -> torch.Tensor:
        """(b * n_ev, width, d) -> (b, t, d), padding positions zero."""
        out = xp.new_zeros((self.b, self.t, xp.size(-1)))
        out[self.bi, self.ti] = xp[self.row, self.col]
        return out

    def masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(bidirectional, causal) masks over the slots of each row, both (b * n_ev, 1, width, width).

        A row with no tokens may attend to its first slot rather than to nothing, which SDPA turns into
        NaN that the backward pass would spread; its outputs are never read.
        """
        keys = self.filled.clone()
        keys[:, 0] = True
        bi = keys[:, None, None, :].expand(-1, 1, self.width, -1)
        causal = torch.tril(torch.ones(self.width, self.width, dtype=torch.bool, device=keys.device))
        return bi, bi & causal

    def pool(self, xp: torch.Tensor) -> torch.Tensor:
        """Mean over the filled slots of each row: (b * n_ev, width, d) -> (b, n_ev, d); zero for an empty event."""
        w = self.filled.unsqueeze(-1).to(xp.dtype)
        mean = (xp * w).sum(1) / w.sum(1).clamp(min=1)
        return mean.view(self.b, self.n_ev, -1)


class PreviousEventContext(nn.Module):
    """History state of the previous event, broadcast to every token of the current one (zero for the first event)."""

    def forward(self, h: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """(b, n_ev, d) history states and (b, t) event index -> (b, t, d)."""
        prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return torch.gather(prev, 1, event.unsqueeze(-1).expand(-1, -1, h.size(-1)))


# -------------------------------------------------------------------------------
# The model
# -------------------------------------------------------------------------------
class EventGPT(nn.Module):
    """Bidirectional event encoder, causal history decoder over event vectors, token decoder of the next event."""

    def __init__(self, cfg: EventGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_in_event = nn.Embedding(64, d)
        self.pos_event = nn.Embedding(cfg.max_events, d)
        self.drop = nn.Dropout(cfg.dropout)
        bi = GPTConfig(
            cfg.vocab_size,
            cfg.block_size,
            cfg.n_layer_event,
            cfg.n_head,
            d,
            cfg.dropout,
            causal=False,
            usr_id=cfg.usr_id,
            qk_norm=cfg.qk_norm,
        )
        ca = GPTConfig(
            cfg.vocab_size,
            cfg.block_size,
            cfg.n_layer,
            cfg.n_head,
            d,
            cfg.dropout,
            causal=True,
            usr_id=cfg.usr_id,
            qk_norm=cfg.qk_norm,
        )
        self.segmenter = Segmenter(cfg.evt_id, cfg.usr_id)
        self.masks = EventMasks()
        self.router = KindRouter(cfg.evt_id)
        self.event_blocks = nn.ModuleList([Block(bi, cfg.n_experts) for _ in range(cfg.n_layer_event)])
        self.ln_event = nn.LayerNorm(d)
        self.pool = EventPool()
        self.packer = EventPacker()
        self.layout = HistoryLayout(cfg.max_events)
        self.time_positions = TimePositions(cfg.time_pos)
        self.hist_blocks = nn.ModuleList([Block(ca) for _ in range(cfg.n_layer)])
        self.ln_hist = nn.LayerNorm(d)
        self.context = PreviousEventContext()
        self.tok_blocks = nn.ModuleList([Block(ca) for _ in range(cfg.n_layer_tok)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        # vocabulary tables filled by set_vocab_tables: event kind per name token, seconds per dt token
        self.kind_lut: torch.Tensor
        self.dt_seconds: torch.Tensor
        self.register_buffer("kind_lut", torch.zeros(cfg.vocab_size, dtype=torch.long))
        self.register_buffer("dt_seconds", torch.zeros(cfg.vocab_size))
        # static path: the profile event's field embeddings, projected and added to the client's later events;
        # zero-initialized, so a model starts out as one without the path
        self.profile_proj: nn.Linear | None = None
        self.profile_name: torch.Tensor
        if cfg.profile_path:
            self.profile_proj = nn.Linear(cfg.profile_fields * d, d, bias=False)
            self.register_buffer("profile_name", torch.tensor(-1))
        self.apply(self._init)
        if self.profile_proj is not None:
            nn.init.zeros_(self.profile_proj.weight)

    def set_vocab_tables(self, structured: dict[str, int]) -> None:
        """Fill kind_lut (name token -> expert) and dt_seconds (dt token -> bucket midpoint) from the vocabulary."""
        kinds = {"trx_": 1, "geo": 2}
        for tok, i in structured.items():
            if tok == "name=profile" and self.cfg.profile_path:
                self.profile_name.fill_(i)
            if tok.startswith("name="):
                name = tok[5:]
                self.kind_lut[i] = next((k for pre, k in kinds.items() if name.startswith(pre)), 0)
            elif tok.startswith("dt="):
                b = int(tok[3:])
                self.dt_seconds[i] = dt_midpoint(b)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------------
    def stages(
        self, idx: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | Packing]:
        """Event encoder and history decoder; returns (h, event, ev, pos_in_event, token mask or packing)."""
        event, client, pos_in_event, n_ev = self.segmenter(idx, key_mask)
        # 1. event encoder: bidirectional within an event (no positions when rope: key order carries no meaning)
        x = self.tok_emb(idx) if self.cfg.rope else self.tok_emb(idx) + self.pos_in_event(pos_in_event)
        x = self.drop(x)
        route = self.router(idx, event, n_ev, self.kind_lut) if self.cfg.n_experts > 1 else None
        tok: torch.Tensor | Packing
        if self.cfg.packed:
            pk = self.packer(event, n_ev, key_mask if key_mask is not None else torch.ones_like(event, dtype=torch.bool))
            enc_mask, _ = pk.masks()
            xp = pk.pack(x)
            route_p = pk.pack(route) if route is not None else None
            for blk in self.event_blocks:
                xp = blk(xp, enc_mask, route=route_p)
            ev = pk.pool(self.ln_event(xp))
            tok = pk
        else:
            same_event, tok_mask = self.masks(event)
            for blk in self.event_blocks:
                x = blk(x, same_event, route=route)
            ev = self.pool(self.ln_event(x), event, n_ev)
            tok = tok_mask
        # 2. history decoder: causal over events, masked to the same client, positions restart per client
        first_ev, ev_pos, hist_mask = self.layout(event, client, n_ev)
        if self.profile_proj is not None and isinstance(tok, Packing):
            ev = ev + self.static_path(idx, tok, first_ev)
        rope_pos = None
        if self.cfg.rope:
            h = ev
            rope_pos = self.time_positions(idx, event, n_ev, first_ev, self.dt_seconds, ev_pos)
        else:
            h = ev + self.pos_event(ev_pos)
        for blk in self.hist_blocks:
            h = blk(h, hist_mask, pos=rope_pos)
        h = self.ln_hist(h)
        # an event with no tokens (event 0 when a window opens on [USR], as every aligned window does) pools to
        # an exact zero vector that stays zero through every layer at initialization, and the layer norms' factor
        # of 1/sqrt(eps) on it compounds into gradients near 1e18 at width 384 and past float32 range at 512; its
        # state feeds only the context of the next event, so it is zeroed, which also cuts that gradient path
        occupied = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device).scatter_(1, event, True)
        h = h * occupied.unsqueeze(-1).to(h.dtype)
        return h, event, ev, pos_in_event, tok

    def static_path(self, idx: torch.Tensor, pk: Packing, first_ev: torch.Tensor) -> torch.Tensor:
        """(b, n_ev, d) projected profile field embeddings for every event from the client's profile event on.

        A client's events are [USR] (event first_ev), its profile event (first_ev + 1) and the rest. The [USR]
        event gets nothing: its state conditions the prediction of the profile event's own tokens.
        """
        assert self.profile_proj is not None
        b, n_ev = first_ev.shape
        k = self.cfg.profile_fields
        if pk.width < 3 + k:
            return torch.zeros(b, n_ev, self.cfg.d_model, device=idx.device, dtype=self.tok_emb.weight.dtype)
        ids = pk.pack(idx).view(b, n_ev, pk.width)
        is_profile = ids[:, :, 1] == self.profile_name
        p_ev = self.profile_proj(self.tok_emb(ids[:, :, 3 : 3 + k]).flatten(-2)) * is_profile.unsqueeze(-1)
        src = (first_ev + 1).clamp(max=n_ev - 1)
        ar = torch.arange(n_ev, device=idx.device).unsqueeze(0)
        after = (ar >= first_ev + 1).unsqueeze(-1)
        return torch.gather(p_ev, 1, src.unsqueeze(-1).expand(-1, -1, p_ev.size(-1))) * after

    def history(self, idx: torch.Tensor, key_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (history states per event (b, n_ev, d), event index per token, event vectors)."""
        h, event, ev, _, _ = self.stages(idx, key_mask)
        return h, event, ev

    def hidden(self, idx: torch.Tensor, doc_mask: bool = True, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Per-token state = history state of the token's event, so the last token of a prefix carries the summary."""
        h, event, _ = self.history(idx, key_mask)
        return torch.gather(h, 1, event.unsqueeze(-1).expand(-1, -1, h.size(-1)))

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, usr_idx: torch.Tensor | None = None, doc_mask: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Next-token logits from the token decoder conditioned on the previous event's history state.

        Args:
            idx: token ids (b, t)
            targets: next-token ids with -100 to ignore
            usr_idx: unused, kept for the GPT call signature
            doc_mask: unused, segmentation is always per client
        """
        h, event, _, pos_in_event, tok = self.stages(idx)
        logits = self.head(self.decode_tokens(idx, h, event, pos_in_event, tok))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def decode_tokens(
        self, idx: torch.Tensor, h: torch.Tensor, event: torch.Tensor, pos_in_event: torch.Tensor, tok: torch.Tensor | Packing
    ) -> torch.Tensor:
        """Token decoder states (b, t, d), before the vocabulary head, from the outputs of `stages`.

        Tokens of event i see the history state of event i-1 (zero for the first event in the window), so idx may
        change inside the last event without running `stages` again: the zero-shot scorer swaps the event name there.
        """
        y = self.drop(self.tok_emb(idx) + self.pos_in_event(pos_in_event) + self.context(h, event))
        if isinstance(tok, Packing):
            _, dec_mask = tok.masks()
            yp = tok.pack(y)
            for blk in self.tok_blocks:
                yp = blk(yp, dec_mask)
            y = tok.unpack(yp)
        else:
            for blk in self.tok_blocks:
                y = blk(y, tok)
        return self.ln_f(y)

    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
