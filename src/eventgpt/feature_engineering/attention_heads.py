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

Interpretation needs care (Jain and Wallace, 2019): a summary that correlates with an engineered feature can follow
from attention spread evenly over the keys, so enrichment, which compares a head's mass on an attribute with the
attribute's share among the keys, says more about what a head looks for than a correlation does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
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
# Features
# -------------------------------------------------------------------------------
def head_summaries(attn: np.ndarray, query: np.ndarray, attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attention-weighted key attributes and focus of every head at one query position per row.

    Args:
        attn: (layers, b, heads, t, t) weights; row q of the last two axes is query position q.
        query: (b,) the query position of each row.
        attrs: (b, t, A) the attributes of every key position; give 0 to keys that should count as none of them.

    Returns:
        summaries (b, layers, heads, A) and focus (b, layers, heads), the entropy of each head's weights in nats.
    """
    rows = np.arange(attn.shape[1])
    w = attn[:, rows, :, query, :]  # (b, layers, heads, t): numpy moves the advanced-index axis to the front
    summ = np.einsum("blht,bta->blha", w, attrs)
    focus = -(w * np.log(np.clip(w, 1e-12, 1.0))).sum(-1)
    return summ, focus


class HeadFeatures:
    """Attention-weighted summaries of named key attributes, for every layer and head, as a table."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)

    def summaries(self, attn: Any, query: np.ndarray, attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """head_summaries on weights given stacked or per layer."""
        a = stack_layers(attn)
        assert attrs.shape[-1] == len(self.names), f"{attrs.shape[-1]} attributes for {len(self.names)} names"
        return head_summaries(a, np.asarray(query), attrs)

    def transform(self, attn: Any, query: np.ndarray, attrs: np.ndarray) -> pd.DataFrame:
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

    def enrichment(self, attn: Any, query: np.ndarray, attrs: np.ndarray, base: np.ndarray) -> pd.DataFrame:
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
    def match(features: pd.DataFrame, reference: pd.DataFrame, top: int = 3) -> dict[str, list[tuple[str, float]]]:
        """For each feature column, the reference columns with the largest absolute Spearman correlation, rows aligned."""
        ref = reference.rank(pct=True)
        out = {}
        for c in features.columns:
            r = ref.corrwith(features[c].rank(pct=True)).dropna()
            best = r.abs().sort_values(ascending=False).head(top).index
            out[c] = [(str(k), float(r[k])) for k in best]
        return out
