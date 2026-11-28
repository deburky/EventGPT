"""Features from hidden states: a transformer's embedding of each row, as columns of a table.

Attention features say what a head reads; the hidden state at the query says what the model concluded, and is usually
the stronger input to a downstream model. The two share the query conventions and HeadFeatureStore:

    from eventgpt.feature_engineering import EmbeddingFeatures, HeadFeatureStore, OutputRecorder

    with OutputRecorder(model.final_norm) as rec:            # any module whose output is (b, t, d)
        model(x)
    emb = EmbeddingFeatures(pool="query")
    table = emb.transform(rec.output(), query)                # columns emb_0 ... emb_{d-1}

    out = hf_model(**batch, output_hidden_states=True)        # Hugging Face: no recorder needed
    table = emb.transform(out.hidden_states[-1], valid=batch["attention_mask"], pool="mean")

Embeddings are wide (hundreds of dimensions). EmbeddingFeatures(components=32) projects them on their first principal
components, fitted with partial_fit batch by batch over the training rows, so the projection never needs every
embedding in memory and is fitted without the test rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

Pool = Literal["query", "mean", "last"]


# -------------------------------------------------------------------------------
# Recording
# -------------------------------------------------------------------------------
class OutputRecorder:
    """The outputs of one or more modules in the last forward pass, by forward hooks.

    A module returning a tuple (as many transformer layers do) is read at `index`, its hidden states by default.
    Outputs are copied to the CPU, so recording keeps neither the graph nor device memory, and a later in-place change
    to the output does not change the recording. With `root` (the model called), every forward pass of root starts by
    forgetting the last pass, so a module the pass skips gives no output rather than an earlier batch's. Hooks see
    every call of a module, so use one recorder for one forward pass at a time.
    """

    def __init__(self, modules: nn.Module | Sequence[nn.Module], index: int = 0, root: nn.Module | None = None) -> None:
        self.modules = [modules] if isinstance(modules, nn.Module) else list(modules)
        assert self.modules, "no modules to record"
        self.index = index
        self.root = root
        self._out: list[torch.Tensor | None] = [None] * len(self.modules)
        self._hooks: list[Any] = []

    def __enter__(self) -> OutputRecorder:
        """Register a forward hook on every module."""
        self.clear()
        if self.root is not None:
            self._hooks.append(self.root.register_forward_pre_hook(lambda m, args: self.clear()))
        for i, m in enumerate(self.modules):
            self._hooks.append(m.register_forward_hook(self._hook(i)))
        return self

    def clear(self) -> None:
        """Forget the recorded outputs."""
        self._out = [None] * len(self.modules)

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Remove the hooks, also when the forward pass failed."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _hook(self, i: int) -> Any:
        """A hook keeping module i's output."""

        def keep(m: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            """Keep the output (or its element at index) of the latest call."""
            y = output[self.index] if isinstance(output, (tuple, list)) else output
            assert isinstance(y, torch.Tensor), f"module {type(m).__name__} gave {type(y).__name__}, not a tensor"
            self._out[i] = y.detach().to("cpu", copy=True)

        return keep

    def output(self, i: int = 0) -> torch.Tensor:
        """Module i's output in the last forward pass."""
        y = self._out[i]
        assert y is not None, f"no output of module {i} in the last forward pass"
        return y

    def outputs(self) -> list[torch.Tensor]:
        """Every module's output in the last forward pass, in the order the modules were given."""
        return [self.output(i) for i in range(len(self.modules))]


# -------------------------------------------------------------------------------
# Features
# -------------------------------------------------------------------------------
def _array(x: Any) -> np.ndarray:
    """A float64 numpy array from a tensor (any dtype or device) or an array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().to(torch.float64).numpy()  # to the CPU first: MPS has no float64
    return np.asarray(x, dtype=np.float64)


class EmbeddingFeatures:
    """One row per query (or per sequence) of a model's hidden states, raw or projected on principal components.

    pool says which positions make a row's embedding:

        "query"  the hidden state at query[i] (b,), or one row per query for query (b, Q), as flatten_queries orders
                 them; n_queries keeps the first n_queries[i] queries of sequence i
        "mean"   the mean over the positions where valid (b, t) is true, or over every position
        "last"   the last position where valid is true, or the last position

    With components=k, fit or partial_fit learns the mean and the first k principal components of the pooled
    embeddings, and transform gives their scores, columns {prefix}_pc1 ... {prefix}_pc{k}; explained_variance_ratio
    says how much each keeps. Each component's sign is fixed so its largest loading is positive, so refits agree.
    """

    def __init__(self, prefix: str = "emb", pool: Pool = "query", components: int | None = None) -> None:
        assert pool in ("query", "mean", "last"), f"unknown pool {pool!r}"
        assert components is None or components > 0, "components must be positive"
        self.prefix = prefix
        self.pool_by = pool
        self.components = components
        self.reset()

    def reset(self) -> None:
        """Forget the fitted projection."""
        self._n = 0
        self._sum: np.ndarray | None = None
        self._outer: np.ndarray | None = None
        self._fitted: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def pool(
        self,
        hidden: Any,
        query: Any = None,
        valid: Any = None,
        n_queries: np.ndarray | Sequence[int] | None = None,
        pool: Pool | None = None,
    ) -> np.ndarray:
        """The pooled embeddings (rows, d) of hidden states (b, t, d)."""
        h = _array(hidden)
        assert h.ndim == 3, f"expected hidden states (b, t, d), got shape {h.shape}"
        b, t, _ = h.shape
        how = pool or self.pool_by
        assert how in ("query", "mean", "last"), f"unknown pool {how!r}"
        v = np.ones((b, t), dtype=bool) if valid is None else _array(valid).astype(bool)
        assert v.shape == (b, t), f"valid has shape {v.shape}, expected {(b, t)}"
        if how == "mean":
            n = v.sum(1, keepdims=True)
            assert (n > 0).all(), "a sequence with no valid position"
            return np.einsum("bt,btd->bd", v.astype(np.float64), h) / n
        if how == "last":
            assert v.any(1).all(), "a sequence with no valid position"
            last = t - 1 - np.argmax(v[:, ::-1], axis=1)
            return h[np.arange(b), last]
        assert query is not None, 'pool="query" needs query'
        q = np.asarray(_array(query), dtype=int)
        if q.ndim == 1:
            assert n_queries is None, "n_queries is for query (b, Q)"
            return h[np.arange(b), q]
        assert q.ndim == 2 and len(q) == b, f"query has shape {q.shape}, expected (b,) or (b, Q)"
        n = np.full(b, q.shape[1]) if n_queries is None else np.asarray(n_queries, dtype=int)
        assert n.shape == (b,), f"n_queries has {n.size} counts for {b} sequences"
        assert (n <= q.shape[1]).all(), "n_queries above the given queries"
        seq = np.repeat(np.arange(b), n)
        qi = np.concatenate([np.arange(k) for k in n]) if len(n) else np.zeros(0, dtype=int)
        return h[seq, q[seq, qi]]

    def partial_fit(self, pooled: Any) -> EmbeddingFeatures:
        """Add pooled embeddings (rows, d) of training rows to the principal components."""
        x = _array(pooled)
        assert x.ndim == 2, f"expected pooled embeddings (rows, d), got shape {x.shape}"
        if self._sum is None or self._outer is None:
            self._sum, self._outer = np.zeros(x.shape[1]), np.zeros((x.shape[1], x.shape[1]))
        assert x.shape[1] == len(self._sum), f"{x.shape[1]} dimensions, fitted on {len(self._sum)}"
        self._n += len(x)
        self._sum += x.sum(0)
        self._outer += x.T @ x
        self._fitted = None  # solved again when next needed
        return self

    def fit(self, pooled: Any) -> EmbeddingFeatures:
        """Principal components of pooled embeddings (rows, d), forgetting any earlier fit."""
        self.reset()
        return self.partial_fit(pooled)

    def _solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean, components and explained variance ratios from the running sums, solved once per fit."""
        assert self.components is not None, "no components to fit: EmbeddingFeatures(components=k)"
        assert self._sum is not None and self._outer is not None, "fit or partial_fit first"
        if self._fitted is not None:
            return self._fitted
        d = len(self._sum)
        assert self.components <= d, f"{self.components} components of {d} dimensions"
        mean = self._sum / self._n
        cov = self._outer / self._n - np.outer(mean, mean)
        val, vec = np.linalg.eigh((cov + cov.T) / 2)
        order = np.argsort(val)[::-1][: self.components]
        comp = vec[:, order].T
        comp *= np.sign(comp[np.arange(len(comp)), np.abs(comp).argmax(1)])[:, None]
        total = max(float(np.clip(val, 0, None).sum()), 1e-300)
        self._fitted = (mean, comp, np.clip(val[order], 0, None) / total)
        return self._fitted

    @property
    def mean_(self) -> np.ndarray:
        """The mean of the fitted embeddings (d,)."""
        return self._solve()[0]

    @property
    def components_(self) -> np.ndarray:
        """The principal components (components, d), unit length."""
        return self._solve()[1]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """The share of the total variance each component keeps."""
        return self._solve()[2]

    def names(self, d: int | None = None) -> list[str]:
        """The column names: {prefix}_0 ... for raw embeddings of d dimensions, {prefix}_pc1 ... when projected."""
        if self.components is not None:
            return [f"{self.prefix}_pc{i + 1}" for i in range(self.components)]
        assert d is not None, "raw embeddings need their dimension"
        return [f"{self.prefix}_{i}" for i in range(d)]

    def transform(
        self,
        hidden: Any,
        query: Any = None,
        valid: Any = None,
        n_queries: np.ndarray | Sequence[int] | None = None,
        pool: Pool | None = None,
    ) -> pd.DataFrame:
        """The feature table: pooled embeddings, projected on the fitted components when there are any."""
        x = self.pool(hidden, query, valid, n_queries, pool)
        if self.components is not None:
            mean, comp, _ = self._solve()
            x = (x - mean) @ comp.T
        return pd.DataFrame(x, columns=self.names(x.shape[1]))
