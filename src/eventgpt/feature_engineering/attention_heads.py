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
    AttentionRecorder(modules, read=..., prepare=...)        # any other attention module

Key attributes can come from the input tokens alone: token_attributes gives one 0/1 attribute per vocabulary token
("this key contains late=2"), named from the vocabulary, with group_tokens collecting an event's tokens into its key
for models that attend over events. HeadFeatures.match(table, reference, on="case_id") joins a reference table by id.

Interpretation needs care (Jain and Wallace, 2019): a summary that correlates with an engineered feature can follow
from attention spread evenly over the keys, so enrichment, which compares a head's mass on an attribute with the
attribute's share among the keys, says more about what a head looks for than a correlation does.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd
import torch
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

    def __enter__(self) -> AttentionRecorder:
        """Install the hooks."""
        for i, m in enumerate(self.modules):
            if self.setup is not None:
                self.setup(m)
            if self.prepare is not None:
                self._handles.append(m.register_forward_pre_hook(self.prepare, with_kwargs=True))
            self._handles.append(m.register_forward_hook(self._hook(i), with_kwargs=True))
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

    def _hook(self, i: int) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any], Any], None]:
        """The forward hook that stores module i's weights."""

        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            """Store the weights the reader returns."""
            w = self.read(module, args, kwargs, output)
            if w is not None:
                self._store[i] = w.detach().float().cpu()

        return hook

    def weights(self) -> np.ndarray:
        """(layers, b, heads, t, t) weights of the last forward pass, in module order; the store is cleared."""
        missing = [i for i, w in enumerate(self._store) if w is None]
        assert not missing, f"modules {missing} recorded no weights; run a forward pass inside the context"
        out = np.stack([w.numpy() for w in self._store if w is not None])
        self._store = [None] * len(self.modules)
        assert out.ndim == 5, f"expected per-head weights (layers, b, heads, t, t), got shape {out.shape}"
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
            """The weights are the second output."""
            return output[1]

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


# -------------------------------------------------------------------------------
# Features
# -------------------------------------------------------------------------------
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
    """Attention-weighted summaries of named key attributes, for every layer and head, as a table."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)

    def summaries(self, attn: Any, query: np.ndarray | None, attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """head_summaries on weights given stacked or per layer (query None: only the query rows, see head_summaries)."""
        a = stack_layers(attn)
        assert attrs.shape[-1] == len(self.names), f"{attrs.shape[-1]} attributes for {len(self.names)} names"
        return head_summaries(a, None if query is None else np.asarray(query), attrs)

    def transform(self, attn: Any, query: np.ndarray | None, attrs: np.ndarray) -> pd.DataFrame:
        """One column per layer, head and attribute (L{layer}H{head}_{name}) and per head its focus (_focus)."""
        summ, focus = self.summaries(attn, query, attrs)
        _, n_layers, n_heads, _ = summ.shape
        cols, data = [], []
        for ly in range(n_layers):
            for h in range(n_heads):
                for a, name in enumerate(self.names):
                    cols.append(f"L{ly + 1}H{h + 1}_{name}")
                    data.append(summ[:, ly, h, a])
                cols.append(f"L{ly + 1}H{h + 1}_focus")
                data.append(focus[:, ly, h])
        return pd.DataFrame(np.stack(data, 1), columns=cols)

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

    @staticmethod
    def match(features: pd.DataFrame, reference: pd.DataFrame, top: int = 3, on: str | None = None) -> dict[str, list[tuple[str, float]]]:
        """For each feature column, the reference columns with the largest absolute Spearman correlation.

        Rows are taken as aligned, or, with `on`, joined on that id column of both tables (an inner join; a warning
        says how many rows of either table found no partner). Only numeric reference columns are compared.
        """
        if on is not None:
            joined = features.merge(reference, on=on, how="inner", suffixes=("", "__ref"))
            lost = (len(features) - len(joined), len(reference) - len(joined))
            if any(lost):
                warnings.warn(f"match on {on!r}: {lost[0]} feature rows and {lost[1]} reference rows have no partner", stacklevel=2)
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
