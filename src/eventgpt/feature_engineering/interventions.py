"""Interventions on attention heads, and attributions to keys, to test what a model's output depends on.

AttentionRecorder says where each head looks. The tools here change what the heads do and measure the output:

    with AttentionIntervention(model.hist_blocks) as iv:
        iv.collect_means = True                      # 1. population mean of every head's output
        model.stages(x_ref, x_ref != 0)
        iv.collect_means = False

        iv.ablate({3: [11]}, mode="mean")            # 2. module 3, head 11 replaced by its mean output
        out = model.stages(x, x != 0)

        iv.ablate({})                                # 3. attention to chosen keys removed in every head, renormalized
        iv.knockout(keys)                            # keys (b, t) bool, per batch
        out = model.stages(x, x != 0)

    with AttentionIntervention(blocks, gradients=True) as iv:   # 4. first-order effect of mean-ablating each head
        score(model(x)).sum().backward()
        est = iv.mean_ablation_estimate()            # (b, modules, heads)

    with AttentionIntervention(blocks, weight_gradients=True) as iv:   # 5. gradient-weighted attention rollout
        score(model(x)).sum().backward()
        w, g = iv.weight_gradients()                 # (modules, b, heads, t, t) each
    importance = grad_rollout(w, g, readout)          # (b, t)

    occlusion(score_rows, seqs, {"late": late_id}, starts={evt_id}) # 6. drop events carrying a token, rescore

Like AttentionRecorder.sdpa, the intervention wraps F.scaled_dot_product_attention while the context is open, so it
works for any block that calls it through the module attribute; outside the chosen modules calls pass through.
Do not nest it with AttentionRecorder.sdpa on the same modules (the recorder would see nothing, or the weights before
the intervention); record=True keeps the weights the intervened heads actually used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from eventgpt.feature_engineering.attention_heads import sdpa_weights


# -------------------------------------------------------------------------------
# Head interventions
# -------------------------------------------------------------------------------
class AttentionIntervention:
    """Change the per-head attention of chosen modules that call F.scaled_dot_product_attention.

    Modules are addressed by their index in `modules`, heads by their index in the module. Per forward pass:

        ablate({i: [h, ...]}, mode)   head outputs replaced by zero or by their mean over positions (collect_means)
        knockout(keys, heads)         attention to the keys where `keys` (b, t_k) is True removed in the given heads
                                      ({i: [h, ...]}, or None for every head of every module), weights renormalized;
                                      a query left with no key gets a zero output
        collect_means = True          accumulate each head's mean output over rows and positions (weighted by
                                      `position_weight` (b, t) when set) for mode="mean"
        gradients=True                a zero offset on each module's output as a gradient handle, for
                                      mean_ablation_estimate()
        record=True                   attention computed explicitly and its weights kept, knockouts included, for
                                      weights() (ablation acts on head outputs, so it does not change them)
        weight_gradients=True         attention computed explicitly with a zero offset on the weights, for
                                      weight_gradients() and grad_rollout()

    Interventions apply on every forward inside the context until changed. Dropout is not applied by the explicit
    paths (knockout, weight_gradients); use them in evaluation mode. Selected modules may be nested (a block and its
    attention submodule): each call is handled for the innermost selected module running. One attention call per
    module and forward is expected: with several, every call is intervened, but gradients() and weight_gradients()
    keep the last one's. The patch is process-wide and not thread-safe: do not run models in other threads while the
    context is open.
    """

    def __init__(self, modules: Iterable[nn.Module], gradients: bool = False, weight_gradients: bool = False, record: bool = False) -> None:
        self.modules = list(modules)
        assert self.modules, "no attention modules to intervene on"
        self._index = {id(m): i for i, m in enumerate(self.modules)}
        self.gradients, self.explicit, self.record = gradients, weight_gradients, record
        self._used: dict[int, torch.Tensor] = {}
        self._ablate: dict[int, list[int]] = {}
        self._mode = "zero"
        self._keys: torch.Tensor | None = None
        self._knock_heads: dict[int, list[int]] | None = None
        self.collect_means = False
        self.position_weight: torch.Tensor | None = None
        self._sum: dict[int, torch.Tensor] = {}
        self._count: dict[int, float] = {}
        self._means: dict[int, torch.Tensor] = {}
        self._out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._w: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._active: int | None = None  # the innermost selected module whose forward is running
        self._stack: list[int] = []  # the running selected modules, outermost first, so leaving one restores its caller
        self._sdpa: Callable[..., torch.Tensor] | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> AttentionIntervention:
        """Wrap the function and install the hooks that mark the running module."""
        self._sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._wrapper
        for m in self.modules:
            self._handles.append(m.register_forward_pre_hook(self._enter_module))
            # always_call: the mark is cleared even when the forward raises and the caller catches the error
            self._handles.append(m.register_forward_hook(self._leave_module, always_call=True))
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Restore the function and remove the hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._sdpa is not None:
            F.scaled_dot_product_attention = self._sdpa
            self._sdpa = None
        self._active = None
        self._stack = []

    def _enter_module(self, m: nn.Module, args: tuple[Any, ...]) -> None:
        """Mark module m as running, on top of any selected module that called it."""
        self._stack.append(self._index[id(m)])
        self._active = self._stack[-1]

    def _leave_module(self, m: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        """Clear the mark; the calling selected module, if any, is running again."""
        if self._stack:
            self._stack.pop()
        self._active = self._stack[-1] if self._stack else None

    def ablate(self, heads: Mapping[int, Sequence[int]], mode: str = "zero") -> None:
        """Ablate these heads ({module index: [head, ...]}) on the following passes; {} ablates none."""
        assert mode in ("zero", "mean"), f"mode is 'zero' or 'mean', not {mode!r}"
        if mode == "mean" and heads:
            missing = [i for i in heads if i not in self.head_means()]
            assert not missing, f"no means for modules {missing}; run passes with collect_means = True first"
        self._ablate, self._mode = {int(i): list(h) for i, h in heads.items()}, mode

    def knockout(self, keys: torch.Tensor | None, heads: Mapping[int, Sequence[int]] | None = None) -> None:
        """Remove attention to keys (b, t_k) bool in these heads (None: all) on the following passes; keys None stops it."""
        self._keys = keys
        self._knock_heads = None if heads is None else {int(i): list(h) for i, h in heads.items()}

    def head_means(self) -> dict[int, torch.Tensor]:
        """{module index: (heads, head_dim)} mean head outputs collected so far."""
        for i, s in self._sum.items():
            self._means[i] = s / max(self._count[i], 1e-12)
        return self._means

    def _wrapper(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """The original function, or the intervened attention when a chosen module is running."""
        original = self._sdpa
        assert original is not None
        i = self._active
        if i is None:
            return original(query, key, value, *args, **kwargs)
        names = ("attn_mask", "dropout_p", "is_causal", "scale")
        given = dict(zip(names, args, strict=False)) | kwargs
        mask, causal, scale, gqa = (
            given.get("attn_mask"),
            bool(given.get("is_causal", False)),
            given.get("scale"),
            bool(given.get("enable_gqa", False)),
        )
        knock = self._keys is not None and (self._knock_heads is None or i in self._knock_heads)
        if self.explicit or knock or self.record:
            w = sdpa_weights(query, key, mask, causal, scale, gqa)
            v = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3) if gqa and value.size(-3) != query.size(-3) else value
            if knock:
                assert self._keys is not None
                hs = list(range(query.size(-3))) if self._knock_heads is None else self._knock_heads[i]
                masked = w[:, hs] * (~self._keys.to(w.device).bool())[:, None, None, :]
                w = w.clone()
                w[:, hs] = masked / masked.sum(-1, keepdim=True).clamp(min=1e-12)
            if self.explicit:
                eps = torch.zeros_like(w, requires_grad=True)
                self._w[i] = (w.detach(), eps)
                w = w + eps
            if self.record:
                self._used[i] = w.detach().float().cpu()
            y = (w @ v.to(w.dtype)).to(query.dtype)  # in the weights' precision: float32, or float64 inputs' own
        else:
            y = original(query, key, value, *args, **kwargs)
        if self.collect_means:
            pw = (
                torch.ones(y.size(0), y.size(-2), device=y.device)
                if self.position_weight is None
                else self.position_weight.to(y.device).float()
            )
            s = torch.einsum("bhtd,bt->hd", y.detach().float(), pw)
            self._sum[i] = s if i not in self._sum else self._sum[i] + s
            self._count[i] = self._count.get(i, 0.0) + float(pw.sum())
        hs = self._ablate.get(i, [])
        if hs:
            y = y.clone()
            for h in hs:
                y[:, h] = 0 if self._mode == "zero" else self._means[i][h].to(y.dtype)
        if self.gradients:
            eps = torch.zeros_like(y, requires_grad=True)
            self._out[i] = (y.detach(), eps)
            y = y + eps
        return y

    def mean_ablation_estimate(self, position_weight: torch.Tensor | None = None) -> np.ndarray:
        """(b, modules, heads) first-order change of the backpropagated output when each head is mean-ablated.

        grad . (mean - output) summed over positions (weighted by position_weight (b, t), e.g. 0 on padding) and
        head dimensions, per row. Needs gradients=True, collected means, and a backward pass inside the context.
        """
        assert self.gradients, "construct with gradients=True"
        means = self.head_means()
        out = []
        for i in range(len(self.modules)):
            assert i in self._out and self._out[i][1].grad is not None, f"module {i} has no gradient; run backward() first"
            y, eps = self._out[i]
            assert eps.grad is not None
            delta = means[i][None, :, None, :].to(y.device) - y.float()
            if position_weight is not None:
                delta = delta * position_weight.to(y.device).float()[:, None, :, None]
            out.append((eps.grad.float() * delta).sum((2, 3)).cpu().numpy())
        return np.stack(out, 1)

    def weights(self) -> np.ndarray:
        """(modules, b, heads, t, t) attention weights the modules used on the last pass, knockouts included (record=True)."""
        assert self.record, "construct with record=True"
        missing = [i for i in range(len(self.modules)) if i not in self._used]
        assert not missing, f"modules {missing} recorded no weights; run a forward pass inside the context"
        out = np.stack([self._used[i].numpy() for i in range(len(self.modules))])
        self._used = {}
        return out

    def weight_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        """(modules, b, heads, t, t) attention weights and the gradient of the backpropagated output with respect to them."""
        assert self.explicit, "construct with weight_gradients=True"
        ws, gs = [], []
        for i in range(len(self.modules)):
            assert i in self._w and self._w[i][1].grad is not None, f"module {i} has no gradient; run backward() first"
            w, eps = self._w[i]
            assert eps.grad is not None
            ws.append(w.float().cpu().numpy())
            gs.append(eps.grad.float().cpu().numpy())
        return np.stack(ws), np.stack(gs)


# -------------------------------------------------------------------------------
# Attributions to keys
# -------------------------------------------------------------------------------
def rollout(weights: np.ndarray, readout: np.ndarray, residual: float = 0.5) -> np.ndarray:
    """(b, t) attention rollout: head-mean weights mixed with the identity, multiplied through the layers in order.

    weights (layers, b, heads, t, t); readout (b, t) how the output reads the last layer's positions (one-hot at the
    query position, or pooling weights).
    """
    a = np.asarray(weights).mean(2)
    t = a.shape[-1]
    r = np.broadcast_to(np.eye(t), a.shape[1:]).copy()
    for layer in a:
        r = ((1 - residual) * layer + residual * np.eye(t)) @ r
    return np.einsum("bt,bts->bs", np.asarray(readout), r)


def grad_rollout(weights: np.ndarray, grads: np.ndarray, readout: np.ndarray) -> np.ndarray:
    """(b, t) gradient-weighted rollout: per layer A = mean over heads of relu(grad * weights), R <- R + A R.

    weights and grads (layers, b, heads, t, t), e.g. from AttentionIntervention.weight_gradients(); readout (b, t).
    """
    a = np.clip(np.asarray(grads) * np.asarray(weights), 0, None).mean(2)
    t = a.shape[-1]
    r = np.broadcast_to(np.eye(t), a.shape[1:]).copy()
    for layer in a:
        r = r + layer @ r
    return np.einsum("bt,bts->bs", np.asarray(readout), r)


def drop_events(tokens: np.ndarray, starts: Iterable[int], drop: int, keep: Iterable[int] = ()) -> np.ndarray:
    """One sequence without the events that contain token `drop`.

    An event runs from a token in `starts` to the next one (tokens before the first start form event 0). Events holding
    a token in `keep` (a client boundary, say) stay. If every event would go, the sequence is returned unchanged.
    """
    s = np.asarray(tokens)
    ev = np.cumsum(np.isin(s, list(starts)))
    bad = np.setdiff1d(np.unique(ev[s == drop]), np.unique(ev[np.isin(s, list(keep))]))
    out = s[~np.isin(ev, bad)]
    return out if len(out) else s


def occlusion(
    score: Callable[[list[np.ndarray]], np.ndarray],
    seqs: Sequence[np.ndarray],
    tokens: Mapping[str, int],
    starts: Iterable[int],
    keep: Iterable[int] = (),
) -> pd.DataFrame:
    """Per row and named token: change of score when the events containing the token are dropped from the input.

    `score` maps a list of token sequences to one number per sequence (a logit, say). Rows without the token get 0,
    and only rows with it are rescored. Columns are the names in `tokens`.
    """
    starts, keep = list(starts), list(keep)
    base = np.asarray(score(list(seqs)), dtype=np.float64)
    out = {}
    for name, tok in tokens.items():
        rows = [r for r, s in enumerate(seqs) if (np.asarray(s) == tok).any()]
        d = np.zeros(len(seqs))
        if rows:
            d[rows] = np.asarray(score([drop_events(seqs[r], starts, tok, keep) for r in rows]), dtype=np.float64) - base[rows]
        out[name] = d
    return pd.DataFrame(out)
