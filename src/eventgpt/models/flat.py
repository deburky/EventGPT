"""Flat GPT decoder over event token streams, and the transformer block EventGPT is built from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters of the flat transformer; causal=False makes it the PRAGMA-style encoder."""

    vocab_size: int
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.1
    causal: bool = True  # False = bidirectional encoder trained with masked modelling (PRAGMA style)
    usr_id: int = 1  # [USR] token id; segment boundaries for the block-diagonal mask
    qk_norm: bool = False  # RMSNorm on queries and keys inside each head (Qwen3), bounds attention logits
    aligned: bool = False  # trained on client-aligned windows (eventgpt.data.windows); the probe builds histories the same way


class RMSNorm(nn.Module):
    """Root-mean-square norm over the last dimension, scale only, computed in float32."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise (..., d) and rescale."""
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (h * self.scale).to(x.dtype)


def _mlp(d: int) -> nn.Sequential:
    """GELU feed-forward, 4x expansion."""
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


def rotary(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding on (b, h, t, hd) with a real-valued position per token (b, t).

    Positions need not be integers: the hybrid passes scaled log elapsed time, so the relative rotation between
    two events depends on the time between them (PRAGMA's time-aware RoPE) rather than on their index.
    """
    hd = x.size(-1)
    freqs = base ** (-torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd)
    ang = pos.float().unsqueeze(-1) * freqs  # (b, t, hd/2)
    cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)  # broadcast over heads
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)


class Block(nn.Module):
    """Pre-norm transformer block.

    Self-attention with an optional explicit mask, GELU MLP, residual adds.
    """

    def __init__(self, cfg: GPTConfig, n_experts: int = 1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # n_experts > 1: one feed-forward per event kind, hard-routed by the caller (no learned gate)
        self.experts = nn.ModuleList([_mlp(cfg.d_model) for _ in range(n_experts)])
        self.drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.causal = cfg.causal
        hd = cfg.d_model // cfg.n_head
        self.q_norm = RMSNorm(hd) if cfg.qk_norm else None
        self.k_norm = RMSNorm(hd) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        route: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one block.

        Args:
            x: states (b, t, d)
            attn_mask: bool (b, 1, t, t), True = attend; None uses is_causal
            pos: rotary positions (b, t) applied to q and k; None = no rotation
            route: expert index per token (b, t); None = expert 0 for every token
        """
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if pos is not None:
            q, k = rotary(q, pos), rotary(k, pos)
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.proj(y))
        h = self.ln2(x)
        if route is None or len(self.experts) == 1:
            return x + self.drop(self.experts[0](h))
        out = torch.zeros_like(h)
        for e, mlp in enumerate(self.experts):
            sel = route == e
            if sel.any():
                out[sel] = mlp(h[sel]).to(out.dtype)  # under autocast the expert returns bfloat16 while out is float32
        return x + self.drop(out)


class GPT(nn.Module):
    """Flat GPT: tied embeddings, learned positions, n_layer blocks, final norm; the encoder head also sees the [USR] state."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # encoder: the masked-position state is concatenated with the sequence's [USR] state before the head,
        # so [USR] is trained as the client summary (PRAGMA's history-encoder readout)
        self.usr_proj = nn.Linear(2 * cfg.d_model, cfg.d_model) if not cfg.causal else None
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def segment_mask(self, idx: torch.Tensor) -> torch.Tensor:
        """Block-diagonal attention mask over client segments (a segment starts at each [USR]), causal if the model is.

        Used for packed pretraining windows so no token attends across a client boundary. Shape (b, 1, t, t), True = attend.
        """
        b, t = idx.shape
        seg = torch.cumsum((idx == self.cfg.usr_id).long(), dim=1)  # segment index per position
        same = seg.unsqueeze(2) == seg.unsqueeze(1)
        if self.cfg.causal:
            same &= torch.tril(torch.ones(t, t, dtype=torch.bool, device=idx.device)).unsqueeze(0)
        return same.unsqueeze(1)

    def hidden(self, idx: torch.Tensor, doc_mask: bool = False, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Final-layer states.

        With doc_mask, attention is block-diagonal over client segments and positions restart
        at every [USR], so a client packed mid-window is conditioned exactly as it would be alone.

        key_mask (b, t), True = real token: right-padding in probe batches is never attended to.
        Without it a bidirectional model reads a padded tail as a run of [PAD], which is also its
        [MASK], so the [USR] state would depend on the batch composition.
        """
        b, t = idx.shape
        ar = torch.arange(t, device=idx.device)
        if doc_mask:
            is_usr = idx == self.cfg.usr_id
            seg_start = torch.cummax(torch.where(is_usr, ar.unsqueeze(0), torch.zeros_like(idx)), dim=1).values
            pos = (ar.unsqueeze(0) - seg_start).clamp(max=self.cfg.block_size - 1)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        else:
            x = self.drop(self.tok_emb(idx) + self.pos_emb(ar))
        mask = self.segment_mask(idx) if doc_mask else None
        if key_mask is not None:
            keys = key_mask.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, t) broadcast over queries
            if mask is None:  # an explicit mask replaces is_causal in the block, so the triangle must be rebuilt here
                mask = (
                    torch.tril(torch.ones(t, t, dtype=torch.bool, device=idx.device)).view(1, 1, t, t)
                    if self.cfg.causal
                    else torch.ones(1, 1, t, t, dtype=torch.bool, device=idx.device)
                )
            mask = mask & keys
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        usr_idx: torch.Tensor | None = None,
        doc_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Logits per position, plus the loss when targets are given.

        Args:
            idx: token ids (b, t)
            targets: next-token ids, or masked targets with -100 elsewhere
            usr_idx: per-token index of the governing [USR], encoder head only
            doc_mask: block-diagonal attention over client segments
        """
        h = self.hidden(idx, doc_mask)
        if self.usr_proj is not None and usr_idx is not None:
            usr = torch.gather(h, 1, usr_idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            h = F.gelu(self.usr_proj(torch.cat([h, usr], dim=-1)))
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    """Rebuild a saved model from its checkpoint, flat GPT or EventGPT."""
    ck = torch.load(path, map_location=map_location)
    if ck.get("arch") == "eventgpt":
        from eventgpt.models.hierarchy import EventGPT, EventGPTConfig

        model: nn.Module = EventGPT(EventGPTConfig(**ck["cfg"]))
    else:
        model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model
