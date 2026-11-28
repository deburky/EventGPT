"""Features from attention heads: what each head of a transformer reads, as columns of a table.

For a query position q of a row (a client's state at the cut, a test applicant, the last token) and the attention
weights a_j of one head over the keys j, the head's summary of a key attribute f is the attention-weighted mean

    s_f = sum_j a_j f(key j)

so a head that attends to late payments gives a large s_late. Every (layer, head, attribute) becomes one feature, and
each head also gets its focus, the entropy of a in nats.

The module depends only on numpy, pandas and torch, so other projects can use it with the EventGPT package installed:

    from eventgpt.feature_engineering import AttentionRecorder, HeadFeatures

    with AttentionRecorder.torch_mha(model) as rec:         # every nn.MultiheadAttention, e.g. nn.TransformerEncoder
        model(x)
    attn = rec.weights()                                    # (layers, b, heads, t, t)
    table = HeadFeatures(["late", "tax"]).transform(attn, query, attrs)

    out = hf_model(**batch, output_attentions=True)          # Hugging Face: no recorder needed
    table = HeadFeatures(names).transform(out.attentions, query, attrs)

    AttentionRecorder.flat_blocks(model.hist_blocks)          # EventGPT and its flat GPT
    AttentionRecorder.sdpa(blocks)                            # any block calling F.scaled_dot_product_attention
    AttentionRecorder(modules, read=..., prepare=...)        # any other attention module

Counts and ordinal fields: HeadFeatures(names, counts=True) gives attention-weighted counts (share times the keys the
head saw) instead of shares, and HeadFeatures(names, fields=["amt_bin"]) with field_attributes adds the
attention-weighted mean of an ordinal field. HeadFeatureStore saves the tables of a long extraction in chunks by row
id, so a stopped run resumes, and records in its meta what produced them (model_fingerprint).

Key attributes can come from the input tokens alone: token_attributes gives one 0/1 attribute per vocabulary token
("this key contains late=2"), named from the vocabulary, with group_tokens collecting an event's tokens into its key
for models that attend over events. HeadFeatures.match(table, reference, on="case_id") joins a reference table by id.

Interpretation needs care (Jain and Wallace, 2019): a summary that correlates with an engineered feature can follow
from attention spread evenly over the keys, so enrichment, which compares a head's mass on an attribute with the
attribute's share among the keys, says more about what a head looks for than a correlation does.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn

Reader = Callable[[nn.Module, tuple[Any, ...], dict[str, Any], Any], torch.Tensor | None]
Preparer = Callable[[nn.Module, tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]


# -------------------------------------------------------------------------------
# Recording per-head weights
# -------------------------------------------------------------------------------
class AttentionRecorder:
    """Record the per-head attention weights (b, heads, t, t) of chosen modules on every forward pass.

    `read(module, args, kwargs, output)` returns a module's weights after its forward pass; `prepare(module, args,
    kwargs)`, if given, rewrites its inputs before it (for instance to ask for the weights); `setup` and `teardown` run
    on each module when the context opens and closes. Use as a context manager; weights() returns the last pass.

    Set `query` to a (b,) tensor of query positions before a forward pass to keep only those rows of the weights,
    (b, heads, 1, t) per module, which saves memory over long sequences. A (b, Q) tensor keeps Q rows per sequence,
    (b, heads, Q, t): with a causal model one pass over a sequence then gives the attention at Q cut-offs at once
    (every row only sees keys up to its query), and flatten_queries makes one row per query for HeadFeatures.
    """

    def __init__(
        self,
        modules: Iterable[nn.Module],
        read: Reader,
        prepare: Preparer | None = None,
        setup: Callable[[nn.Module], None] | None = None,
        teardown: Callable[[nn.Module], None] | None = None,
    ) -> None:
        self.modules = list(modules)
        assert self.modules, "no attention modules to record"
        self.read, self.prepare, self.setup, self.teardown = read, prepare, setup, teardown
        self._store: list[torch.Tensor | None] = [None] * len(self.modules)
        self._handles: list[Any] = []
        self._fastpath: bool | None = None
        self._wrap_sdpa = False  # set by AttentionRecorder.sdpa
        self._sdpa: Callable[..., torch.Tensor] | None = None  # the F.scaled_dot_product_attention found on entering
        self._active: int | None = None  # index of the innermost recorded module whose forward is running (sdpa)
        self._stack: list[int] = []  # the running recorded modules, outermost first, so nesting restores the parent
        self._index = {id(m): i for i, m in enumerate(self.modules)}
        self._captured: dict[int, torch.Tensor] = {}
        self._keep_outputs = False  # set by AttentionRecorder.sdpa(outputs=True)
        self._captured_out: dict[int, torch.Tensor] = {}
        self._out_store: list[torch.Tensor | None] = [None] * len(self.modules)
        self.query: torch.Tensor | None = None

    def __enter__(self) -> AttentionRecorder:
        """Install the hooks."""
        if self._wrap_sdpa:  # taken now, not at construction, so nested or reused recorders restore what they found
            self._sdpa = F.scaled_dot_product_attention
            self._patch_sdpa()
        for i, m in enumerate(self.modules):
            if self.setup is not None:
                self.setup(m)
            if self.prepare is not None:
                self._handles.append(m.register_forward_pre_hook(self.prepare, with_kwargs=True))
            # always_call: the hook runs even when the forward raises, so a caught error leaves no module marked running
            self._handles.append(m.register_forward_hook(self._hook(i), with_kwargs=True, always_call=True))
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Remove the hooks and undo the setup."""
        for h in self._handles:
            h.remove()
        self._handles = []
        for m in self.modules:
            if self.teardown is not None:
                self.teardown(m)
        if self._fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(self._fastpath)
        if self._sdpa is not None:
            F.scaled_dot_product_attention = self._sdpa
            self._sdpa = None
        self._active = None
        self._stack = []

    def _hook(self, i: int) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any], Any], None]:
        """The forward hook that stores module i's weights."""

        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            """Store the weights the reader returns."""
            w = self.read(module, args, kwargs, output)
            if w is not None:
                rows_done = self._wrap_sdpa and self.query is not None  # the sdpa wrapper computed only the query rows
                self._store[i] = (w if rows_done else self._rows(w)).detach().float().cpu()
            y = self._captured_out.pop(i, None)
            if y is not None:
                self._out_store[i] = self._rows(y).detach().float().cpu()

        return hook

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        """The query rows of (b, heads, t, ...) when `query` is set: (b, heads, Q, ...), Q = 1 for a (b,) query."""
        if self.query is None:
            return x
        q = self.query.to(x.device).long()
        q2 = q[:, None] if q.dim() == 1 else q
        ar = torch.arange(x.size(0), device=x.device)[:, None]
        return x[ar, :, q2].movedim(1, 2)  # indexing gives (b, Q, heads, ...)

    def weights(self) -> np.ndarray:
        """(layers, b, heads, t, t) weights of the last forward pass, in module order; the store is cleared."""
        missing = [i for i, w in enumerate(self._store) if w is None]
        assert not missing, f"modules {missing} recorded no weights; run a forward pass inside the context"
        out = np.stack([w.numpy() for w in self._store if w is not None])
        self._store = [None] * len(self.modules)
        assert out.ndim == 5, f"expected per-head weights (layers, b, heads, t, t), got shape {out.shape}"
        return out

    def outputs(self) -> np.ndarray:
        """(layers, b, heads, t, head_dim) per-head attention outputs of the last pass (AttentionRecorder.sdpa(outputs=True)).

        Each is the function's own result before the heads are merged and projected: weights @ values, with dropout
        applied when the module ran with dropout_p > 0, while weights() are kept without dropout. The store is cleared.
        """
        assert self._keep_outputs, "outputs are recorded by AttentionRecorder.sdpa(modules, outputs=True)"
        missing = [i for i, y in enumerate(self._out_store) if y is None]
        assert not missing, f"modules {missing} recorded no outputs; run a forward pass inside the context"
        out = np.stack([y.numpy() for y in self._out_store if y is not None])
        self._out_store = [None] * len(self.modules)
        return out

    @classmethod
    def torch_mha(cls, model: nn.Module) -> AttentionRecorder:
        """Every nn.MultiheadAttention in a model, asked for unaveraged weights.

        The fast path of nn.TransformerEncoderLayer bypasses the attention module in evaluation mode, so it is switched
        off while recording and restored afterwards.
        """

        def prepare(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            """Ask the module for per-head weights."""
            return args, kwargs | {"need_weights": True, "average_attn_weights": False}

        def read(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> torch.Tensor | None:
            """The weights are the second output (none when the forward raised)."""
            return output[1] if output is not None else None

        rec = cls((m for m in model.modules() if isinstance(m, nn.MultiheadAttention)), read=read, prepare=prepare)
        rec._fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        return rec

    @classmethod
    def flat_blocks(cls, blocks: Iterable[nn.Module]) -> AttentionRecorder:
        """The attention blocks of EventGPT and the flat GPT (eventgpt.models.flat.Block), through their own storage."""

        def setup(m: Any) -> None:
            """Turn on per-head storage (flat.Block's keep_attn and keep_heads)."""
            m.keep_attn, m.keep_heads = True, True

        def teardown(m: Any) -> None:
            """Turn it off and clear it."""
            m.keep_attn, m.keep_heads, m.attn = False, False, None

        def read(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> torch.Tensor | None:
            """The weights the block stored."""
            w = getattr(m, "attn", None)
            return w if isinstance(w, torch.Tensor) else None

        return cls(blocks, read=read, setup=setup, teardown=teardown)

    @classmethod
    def sdpa(cls, modules: Iterable[nn.Module], outputs: bool = False) -> AttentionRecorder:
        """Modules that compute attention with F.scaled_dot_product_attention and keep no weights of their own.

        While the context is open, F.scaled_dot_product_attention is wrapped: when it is called inside the forward of
        one of `modules`, the wrapper also computes softmax(q k^T * scale + mask) from the same q, k, attn_mask,
        is_causal, scale and enable_gqa it was given, and stores it for that module; the attention itself is still
        computed by the original function, so outputs do not change. The query and key are taken after whatever the
        block does to them (norms, rotary positions), which a recomputation from the block's input could get wrong.

        The block must call it through the module attribute (F.scaled_dot_product_attention or
        torch.nn.functional.scaled_dot_product_attention), not a name imported before the context opened. One call per
        module and forward is expected; with several, the last is kept. Dropout is not applied to the stored weights.
        The wrapper is process-wide while the context is open: other models calling the function pass through it, but
        only calls inside one of `modules` are recorded, and outputs never change. Do not open it inside or around an
        AttentionIntervention on the same modules: the two wrap the same function, and depending on the order the
        recorder sees nothing or the weights before the intervention; use AttentionIntervention(record=True) instead.

        With `outputs`, the function's own result for each module, per head (b, heads, t, head_dim), is kept as well;
        read it with outputs(). With `query` set, only those rows of the weights are computed, so the memory is
        linear in the sequence length. The patch is not thread-safe: do not run models in other threads while it is
        open.
        """

        def prepare(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            """Mark this module as the one running, on top of any recorded module that called it."""
            rec._stack.append(rec._index[id(m)])
            rec._active = rec._stack[-1]
            return args, kwargs

        def read(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> torch.Tensor | None:
            """The weights the wrapper captured during this forward; the calling module, if any, is running again."""
            if rec._stack:
                rec._stack.pop()
            rec._active = rec._stack[-1] if rec._stack else None
            return rec._captured.pop(rec._index[id(m)], None)

        rec = cls(modules, read=read, prepare=prepare)
        rec._wrap_sdpa = True
        rec._keep_outputs = outputs
        return rec

    def _patch_sdpa(self) -> None:
        """Replace F.scaled_dot_product_attention by a wrapper that also records the weights of the active module."""
        original = self._sdpa
        assert original is not None

        def wrapper(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            """Record, then attend with the original function."""
            active = self._active
            if active is not None:
                names = ("attn_mask", "dropout_p", "is_causal", "scale")  # the positional ones; enable_gqa is keyword-only
                given = dict(zip(names, args, strict=False)) | kwargs
                self._captured[active] = sdpa_weights(
                    query,
                    key,
                    given.get("attn_mask"),
                    bool(given.get("is_causal", False)),
                    given.get("scale"),
                    bool(given.get("enable_gqa", False)),
                    rows=self.query,
                )
            y = original(query, key, value, *args, **kwargs)
            if active is not None and self._keep_outputs:
                self._captured_out[active] = y
            return y

        F.scaled_dot_product_attention = wrapper


def sdpa_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """The (b, heads, t_q, t_k) weights F.scaled_dot_product_attention uses for the same arguments.

    A bool mask keeps keys where True, a float mask is added to the scores; rows with no allowed key are zero (where
    the fused kernel would give NaN or an arbitrary average). The result is float32, or the inputs' precision when it
    is higher. With `rows` (b,) or (b, Q), only those query rows are computed, giving (b, heads, Q, t_k) without
    building the full matrix.
    """
    dt = torch.promote_types(query.dtype, torch.float32)
    b, lq, lk = query.size(0), query.size(-2), key.size(-2)
    if enable_gqa and key.size(-3) != query.size(-3):
        key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
    r = None if rows is None else rows.to(query.device).long()
    r = r[:, None] if r is not None and r.dim() == 1 else r  # (b, Q)
    ar = torch.arange(b, device=query.device)[:, None]
    q = query if r is None else query[ar, :, r].movedim(1, 2)  # (b, heads, Q, d)
    s = (q.to(dt) @ key.to(dt).transpose(-1, -2)) * (query.size(-1) ** -0.5 if scale is None else scale)
    if is_causal:  # key j is allowed for query i when j <= i (top-left alignment, as in PyTorch)
        keys = torch.arange(lk, device=s.device)
        if r is None:
            allowed = keys[None, :] <= torch.arange(lq, device=s.device)[:, None]
        else:
            allowed = (keys[None, None, :] <= r[:, :, None])[:, None]  # (b, 1, Q, t_k)
        s = s.masked_fill(~allowed, float("-inf"))
    if attn_mask is not None:
        m = attn_mask.to(s.device)
        if r is not None:
            while m.dim() < 4:
                m = m.unsqueeze(0)
            m = m.expand(b, m.size(1), lq, lk)[ar, :, r].movedim(1, 2)
        s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m.to(dt)
    return torch.nan_to_num(s.softmax(-1))


def flatten_queries(weights: Any, n_queries: np.ndarray | Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per query from weights recorded with a (b, Q) query: (layers, N, heads, t), and each row's sequence and query.

    `n_queries` (b,) keeps only the first n_queries[i] queries of sequence i, for sequences padded to Q queries. The
    rows come sequence by sequence, queries in order; index per-sequence arrays (keys, attributes) with the returned
    sequence index to match them.
    """
    w = stack_layers(weights)
    assert w.ndim == 5, f"expected (layers, b, heads, Q, t), got shape {w.shape}"
    b, q = w.shape[1], w.shape[3]
    n = np.full(b, q) if n_queries is None else np.asarray(n_queries, dtype=int)
    assert n.shape == (b,), f"n_queries has {n.size} counts for {b} sequences"
    assert (n <= q).all(), "n_queries above the recorded queries"
    seq = np.repeat(np.arange(b), n)
    qi = np.concatenate([np.arange(k) for k in n]) if len(n) else np.zeros(0, dtype=int)
    return w[:, seq, :, qi, :].transpose(1, 0, 2, 3), seq, qi  # indexing gives (N, layers, heads, t)


def stack_layers(attentions: Sequence[Any] | np.ndarray | torch.Tensor) -> np.ndarray:
    """(layers, b, heads, t, t) from stacked weights or a sequence of per-layer (b, heads, t, t), e.g. Hugging Face's."""
    if isinstance(attentions, np.ndarray):
        return attentions
    if isinstance(attentions, torch.Tensor):
        return attentions.detach().float().cpu().numpy()
    return np.stack([a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a) for a in attentions])


# -------------------------------------------------------------------------------
# Key attributes from tokens
# -------------------------------------------------------------------------------
def group_tokens(tokens: np.ndarray, slot: np.ndarray, n_slots: int, pad: int = 0) -> np.ndarray:
    """(b, n_slots, m) the token ids of each key slot, padded with `pad`, from tokens (b, T) and the slot of each (b, T).

    For a model that attends over events (EventGPT's history), slot is each token's event; for one that attends over
    tokens, keys are the tokens themselves and no grouping is needed.
    """
    tokens, slot = np.asarray(tokens), np.asarray(slot)
    valid = tokens != pad
    counts = np.zeros((len(tokens), n_slots), dtype=np.int64)
    for r in range(len(tokens)):
        np.add.at(counts[r], slot[r][valid[r]], 1)
    out = np.full((len(tokens), n_slots, max(int(counts.max()), 1)), pad, dtype=tokens.dtype)
    fill = np.zeros_like(counts)
    for r in range(len(tokens)):
        for tok, k in zip(tokens[r][valid[r]], slot[r][valid[r]], strict=True):
            out[r, k, fill[r, k]] = tok
            fill[r, k] += 1
    return out


def token_attributes(
    keys: np.ndarray,
    names: Mapping[int, str] | Sequence[str],
    top: int | None = 50,
    include: Iterable[str] | None = None,
    query: np.ndarray | None = None,
    pad: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """0/1 key attributes "the key contains token X", one per token, named from the vocabulary.

    Args:
        keys: (b, t) one token id per key, or (b, t, m) the token ids of each key padded with `pad`.
        names: token id to its name (a mapping, or a sequence indexed by id).
        top: keep the `top` tokens contained in the most keys (None keeps every token); ignored with `include`.
        include: names of the tokens to keep instead.
        query: (b,) each row's query position; the base shares then count keys up to it, the ones a causal head sees.
        pad: the padding id, never an attribute.

    Returns:
        attrs (b, t, A), the attribute names, and base (b, A): each row's share of its non-empty keys (up to the query
        position when given) that contain the token, for HeadFeatures.enrichment.
    """
    k = np.asarray(keys)
    k = k[..., None] if k.ndim == 2 else k
    flat = np.sort(k.reshape(-1, k.shape[-1]), 1)
    first = np.ones_like(flat, dtype=bool)
    first[:, 1:] = flat[:, 1:] != flat[:, :-1]
    ids, cnt = np.unique(flat[first & (flat != pad)], return_counts=True)
    name_of = {int(i): str(names[int(i)]) for i in ids}
    if include is not None:
        wanted = set(include)
        chosen = [int(i) for i in ids if name_of[int(i)] in wanted]
    else:
        order = np.argsort(-cnt, kind="stable")
        chosen = [int(ids[j]) for j in (order if top is None else order[:top])]
    attrs = np.stack([(k == i).any(-1) for i in chosen], -1).astype(np.float64)
    seen = (k != pad).any(-1)
    if query is not None:
        seen = seen & (np.arange(k.shape[1])[None, :] <= np.asarray(query)[:, None])
    base = (attrs * seen[..., None]).sum(1) / np.clip(seen.sum(1, keepdims=True), 1, None)
    return attrs, [name_of[i] for i in chosen], base


@dataclass
class FieldValues:
    """Numeric values of ordinal fields at every key: values and presence (b, t, F), and the field names."""

    values: np.ndarray
    present: np.ndarray
    names: list[str]


def field_attributes(
    keys: np.ndarray,
    names: Mapping[int, str] | Sequence[str],
    fields: Sequence[str],
    values: Mapping[str, Mapping[str, float]] | None = None,
    pad: int = 0,
) -> FieldValues:
    """The value of each ordinal field at every key, read from tokens named field=value.

    Args:
        keys: (b, t) one token id per key, or (b, t, m) the token ids of each key padded with `pad`.
        names: token id to its name (a mapping, or a sequence indexed by id).
        fields: the fields to read, such as ["amt_bin", "dt"].
        values: per field, values for non-numeric tokens, e.g. {"amt_bin": {"zero": -1}}; a value that is neither a
            number nor listed leaves the field absent at that key.
        pad: the padding id.

    Returns:
        FieldValues: a key's value is the mean of its tokens' values for the field (most keys have one), and present
        says whether the key has the field at all. HeadFeatures(fields=...) turns them into attention-weighted means.
    """
    k = np.asarray(keys)
    k = k[..., None] if k.ndim == 2 else k
    ids = np.unique(k[k != pad])
    lookup = {int(i): str(names[int(i)]) for i in ids}
    size = int(ids.max()) + 1 if len(ids) else 1
    vals = np.full((len(fields), size), np.nan)
    for i, name in lookup.items():
        f, _, v = name.partition("=")
        if f in fields:
            mapped = (values or {}).get(f, {})
            try:
                vals[list(fields).index(f), i] = float(mapped[v]) if v in mapped else float(v)
            except ValueError:
                pass
    safe = np.where(k == pad, 0, k)
    per_token = vals[:, safe]  # (F, b, t, m)
    per_token[:, k == pad] = np.nan
    finite = ~np.isnan(per_token)
    n = finite.sum(-1)
    present = n > 0
    mean = np.where(finite, per_token, 0.0).sum(-1) / np.maximum(n, 1)  # 0 where absent; present says which
    return FieldValues(mean.transpose(1, 2, 0), present.transpose(1, 2, 0), list(fields))


# -------------------------------------------------------------------------------
# Features
# -------------------------------------------------------------------------------
def query_rows(attn: Any, query: np.ndarray | None) -> np.ndarray:
    """(b, layers, heads, t) the weights of each row's query position, from any of the shapes HeadFeatures takes."""
    a = stack_layers(attn)
    if a.ndim == 5 and a.shape[3] == 1:  # query rows from a recorder with `query` set, or one-key sequences
        a, query = a[:, :, :, 0, :], None
    if query is None:
        assert a.ndim == 4, f"(layers, b, heads, Q, t) with Q > 1: use flatten_queries first, got shape {a.shape}"
        return np.asarray(a).transpose(1, 0, 2, 3)
    rows = np.arange(a.shape[1])
    return a[:, rows, :, np.asarray(query), :]  # numpy moves the advanced-index axis to the front


def head_summaries(attn: np.ndarray, query: np.ndarray | None, attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attention-weighted key attributes and focus of every head at one query position per row.

    Args:
        attn: (layers, b, heads, t, t) weights, row q of the last two axes being query position q; or, with query
            None, only the query rows, (layers, b, heads, t), to save memory over many rows.
        query: (b,) the query position of each row, or None.
        attrs: (b, t, A) the attributes of every key position; give 0 to keys that should count as none of them.

    Returns:
        summaries (b, layers, heads, A) and focus (b, layers, heads), the entropy of each head's weights in nats.
    """
    if query is None:
        w = np.asarray(attn).transpose(1, 0, 2, 3)  # (b, layers, heads, t)
    else:
        rows = np.arange(attn.shape[1])
        w = attn[:, rows, :, query, :]  # (b, layers, heads, t): numpy moves the advanced-index axis to the front
    summ = np.einsum("blht,bta->blha", w, attrs)
    focus = -(w * np.log(np.clip(w, 1e-12, 1.0))).sum(-1)
    return summ, focus


class HeadFeatures:
    """Attention-weighted summaries of named key attributes, for every layer and head, as a table.

    `counts` turns the summaries of the named attributes (all with True) into attention-weighted counts: the share
    among the keys the head saw times their number, so a head reports how many late payments it reads and not only
    their share; the columns are named _{name}_count. A key is seen when its weight is nonzero (masked, padded and,
    for a causal head, later keys have none) and, when transform gets `valid`, when it is marked real; both the share
    and the number then use the seen keys only.
    `fields` adds, for each ordinal field of a FieldValues given to transform, the attention-weighted mean value over
    the keys that have the field (_{field}_mean, NaN when the head puts no weight on such keys).
    """

    def __init__(self, names: Sequence[str], counts: bool | Sequence[str] = False, fields: Sequence[str] = ()) -> None:
        self.names = list(names)
        self.counts = set(self.names) if counts is True else set(counts or [])
        unknown = self.counts - set(self.names)
        assert not unknown, f"counts for attributes not in names: {sorted(unknown)}"
        self.fields = list(fields)

    def summaries(self, attn: Any, query: np.ndarray | None, attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """head_summaries on weights given stacked or per layer (query None: only the query rows, see head_summaries).

        Weights recorded with AttentionRecorder.query set, (layers, b, heads, 1, t), are the query rows already; they
        are used as such and `query` is not needed.
        """
        assert attrs.shape[-1] == len(self.names), f"{attrs.shape[-1]} attributes for {len(self.names)} names"
        w = query_rows(attn, query)
        return np.einsum("blht,bta->blha", w, attrs), -(w * np.log(np.clip(w, 1e-12, 1.0))).sum(-1)

    def transform(
        self,
        attn: Any,
        query: np.ndarray | None,
        attrs: np.ndarray,
        fields: FieldValues | None = None,
        valid: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """The feature table: per layer and head, its attributes, fields and focus.

        Columns are L{layer}H{head}_{name} (or _{name}_count), _{field}_mean and _focus. `valid` (b, t) bool marks
        the real keys, for counts; without it a key counts as seen when its weight is nonzero, which excludes padding
        and masked keys.
        """
        w = query_rows(attn, query)
        assert attrs.shape[-1] == len(self.names), f"{attrs.shape[-1]} attributes for {len(self.names)} names"
        summ = np.einsum("blht,bta->blha", w, attrs)
        focus = -(w * np.log(np.clip(w, 1e-12, 1.0))).sum(-1)
        # counts: the share among the keys seen, times their number; a key is seen when the head gives it weight (a
        # zero weight is a masked, padded or, for a causal head, later key) and, with `valid`, when it is real
        seen = (w > 0) if valid is None else ((w > 0) & np.asarray(valid, dtype=bool)[:, None, None, :])
        n_seen = seen.sum(-1)  # (b, layers, heads)
        ws = w * seen
        with np.errstate(invalid="ignore", divide="ignore"):
            share_seen = np.einsum("blht,bta->blha", ws, attrs) / ws.sum(-1, keepdims=True)
        share_seen = np.nan_to_num(share_seen)
        means = None
        if self.fields:
            assert fields is not None, "HeadFeatures(fields=...) needs transform(..., fields=field_attributes(...))"
            idx = [fields.names.index(f) for f in self.fields]
            p = fields.present[..., idx].astype(np.float64)
            mass = np.einsum("blht,btf->blhf", w, p)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(mass > 0, np.einsum("blht,btf->blhf", w, fields.values[..., idx] * p) / mass, np.nan)
        b, n_layers, n_heads, _ = summ.shape
        scaled = summ.copy()
        is_count = np.array([name in self.counts for name in self.names], dtype=bool)
        if is_count.any():
            scaled[..., is_count] = share_seen[..., is_count] * n_seen[..., None]
        block = np.concatenate([scaled, *([means] if means is not None else []), focus[..., None]], -1)
        return pd.DataFrame(block.reshape(b, -1), columns=self._columns(n_layers, n_heads, means is not None))

    def _columns(self, n_layers: int, n_heads: int, with_fields: bool) -> list[str]:
        """Column names in the order transform lays the values out: per layer and head, attributes, fields, focus."""
        per_head = [f"{n}_count" if n in self.counts else n for n in self.names]
        per_head += [f"{f}_mean" for f in self.fields] if with_fields else []
        per_head += ["focus"]
        return [f"L{ly + 1}H{h + 1}_{c}" for ly in range(n_layers) for h in range(n_heads) for c in per_head]

    def enrichment(self, attn: Any, query: np.ndarray | None, attrs: np.ndarray, base: np.ndarray) -> pd.DataFrame:
        """Per head (rows L{layer}H{head}) and 0/1 attribute: mean attention mass over its mean share among the keys.

        `base` (b, A) is each row's share of keys with the attribute, over the keys the heads can attend to; 1 means no
        preference, 8 means the head puts eight times the attribute's share of its weight on it.
        """
        summ, _ = self.summaries(attn, query, attrs)
        ratio = summ.mean(0) / np.clip(np.asarray(base).mean(0), 1e-12, None)
        n_layers, n_heads, _ = ratio.shape
        index = [f"L{ly + 1}H{h + 1}" for ly in range(n_layers) for h in range(n_heads)]
        return pd.DataFrame(ratio.reshape(n_layers * n_heads, -1), index=index, columns=self.names)

    def coverage(self, attn: Any, query: np.ndarray | None, attrs: np.ndarray) -> pd.DataFrame:
        """Per head (rows L{layer}H{head}): mean attention mass on keys carrying at least one of the attributes.

        The rest of a head's weight falls on keys the feature columns do not describe.
        """
        summ, _ = self.summaries(attn, query, (np.asarray(attrs) > 0).any(-1, keepdims=True).repeat(len(self.names), -1).astype(np.float64))
        mass = summ[..., 0].mean(0)
        n_layers, n_heads = mass.shape
        index = [f"L{ly + 1}H{h + 1}" for ly in range(n_layers) for h in range(n_heads)]
        return pd.DataFrame({"coverage": mass.reshape(-1)}, index=index)

    @staticmethod
    def match(features: pd.DataFrame, reference: pd.DataFrame, top: int = 3, on: str | None = None) -> dict[str, list[tuple[str, float]]]:
        """For each feature column, the reference columns with the largest absolute Spearman correlation.

        Rows are taken as aligned, or, with `on`, joined on that id column of both tables. The reference may hold
        more rows than the features (a whole dataset against a scored sample); a warning is given only when feature
        rows find no partner in it. Only numeric reference columns are compared.
        """
        if on is not None:
            joined = features.merge(reference, on=on, how="inner", suffixes=("", "__ref"))
            lost = len(features) - len(joined)
            if lost:
                warnings.warn(f"match on {on!r}: {lost} of {len(features)} feature rows have no partner in the reference", stacklevel=2)
            feat_cols = [c for c in features.columns if c != on]
            ref_cols = [c if c not in features.columns else f"{c}__ref" for c in reference.columns if c != on]
            features, reference = joined[feat_cols], joined[ref_cols]
        ref = reference.select_dtypes("number").rank(pct=True)
        out = {}
        for c in features.columns:
            r = ref.corrwith(features[c].rank(pct=True)).dropna()
            best = r.abs().sort_values(ascending=False).head(top).index
            out[c] = [(str(k), float(r[k])) for k in best]
        return out


# -------------------------------------------------------------------------------
# Storing features over a large dataset
# -------------------------------------------------------------------------------
def to_sql_names(columns: Sequence[str]) -> list[str]:
    """Column names as plain SQL identifiers: lower case, other characters as _, a leading digit escaped with _.

    Redshift folds identifiers to lower case, quoted ones too, and Feast writes feature names into its Redshift SQL
    unquoted, so a name such as L1H1_s11=22 must become l1h1_s11_22 there. Two names that would become one, and names
    above Redshift's 127 characters, are refused. The names keep their order, so zip them with the originals to map
    back.
    """
    out = []
    for c in columns:
        n = re.sub(r"[^a-z0-9_]", "_", c.lower())
        out.append(f"_{n}" if n[:1].isdigit() else n)
    seen: dict[str, str] = {}
    for c, n in zip(columns, out, strict=True):
        if seen.setdefault(n, c) != c:
            raise ValueError(f"columns {seen[n]!r} and {c!r} both become {n!r}")
    long = [n for n in out if len(n.encode()) > 127]
    if long:
        raise ValueError(f"{len(long)} names above 127 characters, e.g. {long[0]!r}")
    return out


def model_fingerprint(model: nn.Module, config: Mapping[str, Any] | None = None) -> str:
    """A short hash of a model's state dict (names, dtypes, shapes, values) and of `config`, for a store's meta.

    Two models give the same fingerprint exactly when their state dicts are equal, so it tells a retrained or
    fine-tuned checkpoint from the one a store was built with. Settings that change the outputs without changing the
    weights (a time encoding switched off, a context length) are not in the state dict: give them as config. Non-tensor
    extra state is hashed by its repr. Fingerprint the model as it is used for extraction, not while it trains.
    """
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(f"{name}|extra|{v!r}|".encode())
    if config:
        h.update(json.dumps(dict(config), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _key(x: Any) -> str:
    """A stable text key for a row id: numbers, strings and tuples of them, numpy scalars as their Python values."""
    if isinstance(x, tuple):
        return "(" + ", ".join(_key(v) for v in x) + ("," if len(x) == 1 else "") + ")"
    if isinstance(x, np.generic):
        x = x.item()
    return repr(x)


class HeadFeatureStore:
    """Feature tables saved to a directory in chunks, by row id, so an extraction over a large dataset can resume.

        store = HeadFeatureStore("features/", chunk_rows=20_000)
        for ids, batch in batches:
            if store.has_all(ids):              # already extracted in an earlier run
                continue
            ...                                 # record attention, build the table
            store.add(ids, table)
        store.flush()
        features = store.read(all_ids)          # in the order of all_ids

    Row ids may be any numbers, strings or tuples of them, such as (client, month); numpy scalars count as their
    Python values. Each chunk is a parquet file of at most chunk_rows rows, written to a temporary file and renamed when
    complete, so a run stopped during a write leaves no partial chunk, and named uniquely, so two stores writing to the
    same directory do not overwrite each other. A run stopped part-way loses at most the rows not yet flushed; rows
    added again, within a batch or across runs, are ignored, so an interrupted batch can simply be redone.

    `meta` records what produced the features, saved as store.json with the first store that gives it:

        meta = {"model": model_fingerprint(model), "checkpoint": path, "attributes": names}
        store = HeadFeatureStore("features/", meta=meta)

    Reopening the directory with a different meta raises ValueError, so rows from a retrained model or another
    attribute list never mix with the stored ones. Opening a store that has a meta without giving one reads the
    saved meta as store.meta and makes the store read-only, since the new rows' origin is unknown. Values that are not
    JSON (paths) are saved as their text.
    """

    def __init__(self, directory: str | Path, chunk_rows: int = 20_000, meta: Mapping[str, Any] | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        assert chunk_rows > 0, "chunk_rows must be positive"
        self.chunk_rows = chunk_rows
        self.read_only = False
        self.meta = self._meta(meta)
        self._schema: list[tuple[str, str]] | None = None
        self._buffer: list[pd.DataFrame] = []
        self._buffered = 0
        self._written = 0
        self._done: set[str] = set()
        for part in self._chunks():
            self._done.update(self._keys(part))

    def _meta(self, meta: Mapping[str, Any] | None) -> dict[str, Any]:
        """The saved meta, after checking it against the given one; the given one is saved when none is."""
        path = self.dir / "store.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        given = None if meta is None else json.loads(json.dumps(dict(meta), default=str))  # as it reads back
        if saved is None:
            if given is None:
                return {}
            if self._chunks():
                raise ValueError(f"{self.dir} has chunks without meta; give it no meta or use a new directory")
            tmp = self.dir / f".store-{uuid.uuid4().hex[:8]}.json.tmp"
            tmp.write_text(json.dumps(given, indent=1, sort_keys=True))
            try:
                os.link(tmp, path)  # created only if absent: of two stores starting at once, one meta wins
                saved = given
            except FileExistsError:
                saved = json.loads(path.read_text())
            finally:
                tmp.unlink()
        if given is None:
            self.read_only = True
        elif given != saved:
            diff = sorted(k for k in saved.keys() | given.keys() if saved.get(k) != given.get(k))
            raise ValueError(f"{self.dir} holds features from another setup: meta differs in {diff}")
        return dict(saved)

    def _chunks(self) -> list[Path]:
        """The complete chunk files, oldest first."""
        return sorted(self.dir.glob("chunk-*.parquet"))

    @staticmethod
    def _keys(part: Path, table: pd.DataFrame | None = None) -> list[str]:
        """The row keys of a chunk (its row_key column, or the row_id column of chunks from earlier versions)."""
        if table is None:
            col = "row_key" if "row_key" in pq.read_schema(part).names else "row_id"
            table = pd.read_parquet(part, columns=[col])
        if "row_key" in table.columns:
            return table["row_key"].tolist()
        return [_key(v) for v in table["row_id"].tolist()]

    def __enter__(self) -> HeadFeatureStore:
        """The store itself; leaving the context flushes it."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Write what is buffered, also when the extraction stopped with an error."""
        self.flush()

    def has(self, row_id: Any) -> bool:
        """Whether the row is saved or buffered."""
        return _key(row_id) in self._done

    def has_all(self, ids: Iterable[Any]) -> bool:
        """Whether every one of these rows is saved or buffered."""
        return all(_key(i) in self._done for i in ids)

    def add(self, ids: Sequence[Any], table: pd.DataFrame) -> None:
        """Buffer a batch's rows (in the table's order), skipping rows already stored or repeated; writes full chunks."""
        assert len(ids) == len(table), f"{len(ids)} ids for {len(table)} rows"
        if self.read_only:
            raise ValueError(f"{self.dir} has a meta: open it with that meta to add rows")
        self._check_schema(table)
        keys = [_key(i) for i in ids]
        keep, seen = [], set()
        for k in keys:
            keep.append(k not in self._done and k not in seen)
            seen.add(k)
        mask = np.array(keep, dtype=bool)
        if not mask.any():
            return
        part = table.reset_index(drop=True)[mask].copy()
        new = [k for k, ok in zip(keys, keep, strict=True) if ok]
        part.insert(0, "row_key", new)
        self._buffer.append(part)
        self._buffered += len(part)
        self._done.update(new)
        while self._buffered >= self.chunk_rows:
            self._write(self.chunk_rows)

    def _check_schema(self, table: pd.DataFrame) -> None:
        """Refuse a batch whose feature columns or types differ from the stored ones (or from the first batch)."""
        got = [(f.name, str(f.type)) for f in pa.Schema.from_pandas(table, preserve_index=False)]
        if self._schema is None:
            parts = self._chunks()
            if parts:
                stored = pq.read_schema(parts[0])
                self._schema = [(f.name, str(f.type)) for f in stored if f.name not in ("row_key", "row_id")]
            else:
                self._schema = got
        if got != self._schema:
            want, have = dict(self._schema), dict(got)
            diff = sorted(c for c in want.keys() | have.keys() if want.get(c) != have.get(c))
            raise ValueError(f"batch columns differ from the stored ones in {diff[:5]}")

    def _write(self, n: int) -> None:
        """Write the first n buffered rows as a new chunk: to a temporary file first, then renamed."""
        buffered = pd.concat(self._buffer, ignore_index=True)
        head, rest = buffered.iloc[:n], buffered.iloc[n:]
        self._written += 1  # orders this store's chunks when the clock does not (microseconds on macOS)
        name = f"chunk-{time.time_ns():020d}-{os.getpid()}-{self._written:06d}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = self.dir / f".{name}.tmp"
        head.to_parquet(tmp, index=False)
        os.replace(tmp, self.dir / name)
        self._buffer = [rest] if len(rest) else []
        self._buffered = len(rest)

    def flush(self) -> None:
        """Write every buffered row, in chunks of at most chunk_rows."""
        while self._buffered:
            self._write(min(self.chunk_rows, self._buffered))

    def export_feast(
        self,
        directory: str | Path,
        ids: Sequence[Any],
        entities: pd.DataFrame,
        timestamp_field: str = "event_timestamp",
        columns: Sequence[str] | None = None,
        sql_names: bool = False,
    ) -> Path:
        """Write the rows `ids` as a Feast offline source: entity columns, timestamps and features, chunk by chunk.

        `entities` gives, for each id in order, the join keys of its Feast entity (one column each, such as client_id)
        and the time the features were known in `timestamp_field`, so that get_historical_features joins them point in
        time. A created_timestamp column records the export time. Each stored chunk gives one parquet
        file, written whole and renamed, so the memory follows one chunk; read the directory with a FileSource, or
        load it into the offline store (Redshift, BigQuery) as one table. feature_view builds the matching view, with
        the same columns. The export is written to a directory of its own and renamed into place when complete, so it
        replaces an earlier export whole, and a stopped or concurrent export never leaves a mix of parts.

        sql_names=True writes the feature names as plain SQL identifiers (see sql_names), as a Redshift source needs:
        Redshift folds names to lower case and Feast's Redshift SQL leaves them unquoted. Give the same flag to
        redshift_table and feature_view.
        """
        assert len(entities) == len(ids), f"{len(entities)} entity rows for {len(ids)} ids"
        assert timestamp_field in entities.columns, f"entities has no {timestamp_field!r} column"
        clash = {"row_key", "row_id", "created_timestamp"} & set(entities.columns)
        assert not clash, f"entity columns {sorted(clash)} are reserved"
        self.flush()
        feature_names = self.columns() if columns is None else list(columns)
        if sql_names:
            feature_names = to_sql_names(feature_names)
        taken_names = set(feature_names) & ({"created_timestamp"} | set(entities.columns))
        assert not taken_names, f"feature columns {sorted(taken_names)} clash with the entity or timestamp columns"
        out = Path(directory)
        if out.exists() and any(not p.name.startswith("part-") for p in out.iterdir()):
            raise ValueError(f"{out} holds files other than an export's parts; give an empty or new directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:8]}"  # this export's own, renamed into place at the end
        stage.mkdir()
        ent = entities.reset_index(drop=True).copy()
        ent[timestamp_field] = pd.to_datetime(ent[timestamp_field], utc=True)
        where: dict[str, int] = {}
        for i, k in enumerate(_key(r) for r in ids):
            assert k not in where, f"id {ids[i]!r} given twice"
            where[k] = i
        created = pd.Timestamp.now(tz="UTC")
        written: set[str] = set()
        for part in self._chunks():
            id_col = "row_key" if "row_key" in pq.read_schema(part).names else "row_id"
            t = pd.read_parquet(part, columns=None if columns is None else [id_col, *columns])
            keys = self._keys(part, t)
            take = [(j, where[k]) for j, k in enumerate(keys) if k in where and k not in written]
            if not take:
                continue
            rows, pos = [j for j, _ in take], [i for _, i in take]
            feats = t.drop(columns=[id_col]).iloc[rows].reset_index(drop=True)
            if sql_names:
                feats.columns = to_sql_names(list(feats.columns))
            table = pd.concat([ent.iloc[pos].reset_index(drop=True), feats], axis=1)
            table["created_timestamp"] = created
            table.to_parquet(stage / f"part-{part.stem.removeprefix('chunk-')}.parquet", index=False)
            written.update(keys[j] for j in rows)
        missing = [r for r in ids if _key(r) not in written]
        if missing:
            shutil.rmtree(stage)
            raise AssertionError(f"{len(missing)} ids not stored, e.g. {missing[:3]}")
        old = out.parent / f".{out.name}.old-{uuid.uuid4().hex[:8]}"
        if out.exists():
            os.replace(out, old)  # an earlier export of this directory goes whole, none of its parts stay
        os.replace(stage, out)
        shutil.rmtree(old, ignore_errors=True)
        return out

    def columns(self) -> list[str]:
        """The stored feature columns, without the row id."""
        parts = self._chunks()
        assert parts, f"no chunks in {self.dir}"
        return [c for c in pq.read_schema(parts[0]).names if c not in ("row_key", "row_id")]

    def read(self, ids: Sequence[Any] | None = None, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Every stored row, or these rows in this order (all must be stored), without the row id.

        With `ids`, each chunk is filtered as it is read, so the memory follows the rows asked for and not the store;
        `columns` reads only those feature columns. A row stored twice is read from its oldest chunk.
        """
        self.flush()
        parts = self._chunks()
        assert parts, f"no chunks in {self.dir}"
        wanted = None if ids is None else [_key(i) for i in ids]
        want = None if wanted is None else set(wanted)
        got: list[pd.DataFrame] = []
        taken: set[str] = set()
        for part in parts:
            id_col = "row_key" if "row_key" in pq.read_schema(part).names else "row_id"
            t = pd.read_parquet(part, columns=None if columns is None else [id_col, *columns])
            keys = self._keys(part, t)
            mask = np.array([k not in taken and (want is None or k in want) for k in keys], dtype=bool)
            if mask.any():
                t = t.drop(columns=[id_col])[mask]
                t.index = pd.Index([k for k, m in zip(keys, mask, strict=True) if m], name="row_key")
                taken.update(t.index)
                got.append(t)
        table = pd.concat(got) if got else pd.DataFrame(columns=list(columns or []))
        if wanted is None:
            return table.reset_index(drop=True)
        missing = [i for i, k in zip(ids or [], wanted, strict=True) if k not in table.index]
        assert not missing, f"{len(missing)} rows not stored, e.g. {missing[:3]}"
        return table.loc[wanted].reset_index(drop=True)
