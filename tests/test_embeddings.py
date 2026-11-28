"""Tests of embedding features, the output recorder, store meta and model fingerprints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from eventgpt.feature_engineering import EmbeddingFeatures, HeadFeatureStore, OutputRecorder, model_fingerprint


def hidden(b: int = 3, t: int = 5, d: int = 4) -> np.ndarray:
    """Hidden states whose value at (i, j, k) is 100 i + 10 j + k, so a pooled row says where it came from."""
    i, j, k = np.meshgrid(np.arange(b), np.arange(t), np.arange(d), indexing="ij")
    return (100 * i + 10 * j + k).astype(np.float64)


# -------------------------------------------------------------------------------
# Pooling
# -------------------------------------------------------------------------------
def test_query_pooling_takes_each_rows_query_position() -> None:
    """A (b,) query gives the hidden state at each sequence's query, named emb_0 ... emb_{d-1}."""
    table = EmbeddingFeatures().transform(hidden(), np.array([4, 0, 2]))
    assert list(table.columns) == ["emb_0", "emb_1", "emb_2", "emb_3"]
    assert table["emb_0"].tolist() == [40.0, 100.0, 220.0]


def test_several_queries_per_sequence_follow_flatten_queries() -> None:
    """A (b, Q) query gives one row per query, sequence by sequence; n_queries drops the padded queries."""
    q = np.array([[1, 3, 3], [0, 2, 4], [4, 4, 4]])
    table = EmbeddingFeatures().transform(hidden(), q, n_queries=[2, 3, 1])
    assert table["emb_0"].tolist() == [10.0, 30.0, 100.0, 120.0, 140.0, 240.0]


def test_mean_and_last_pooling_use_only_valid_positions() -> None:
    """Mean and last pooling skip the positions marked invalid (padding)."""
    valid = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 0, 0]], dtype=bool)
    emb = EmbeddingFeatures(pool="mean")
    assert emb.transform(hidden(), valid=valid)["emb_1"].tolist() == [6.0, 121.0, 216.0]
    assert emb.transform(hidden(), valid=valid, pool="last")["emb_1"].tolist() == [11.0, 141.0, 221.0]


# -------------------------------------------------------------------------------
# Principal components
# -------------------------------------------------------------------------------
def test_components_match_a_direct_pca_and_fit_in_batches() -> None:
    """Fitting batch by batch gives the components of the whole set; scores match a direct SVD up to sign."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6)) + 3.0
    whole = EmbeddingFeatures(components=3).fit(x)
    parts = EmbeddingFeatures(components=3)
    for chunk in np.array_split(x, 7):
        parts.partial_fit(chunk)
    assert np.allclose(whole.components_, parts.components_) and np.allclose(whole.mean_, parts.mean_)
    xc = x - x.mean(0)
    _, sv, vt = np.linalg.svd(xc, full_matrices=False)
    h = x[:, None, :]  # (rows, 1, d): one position per row
    scores = parts.transform(h, np.zeros(len(x), dtype=int))
    assert list(scores.columns) == ["emb_pc1", "emb_pc2", "emb_pc3"]
    assert np.allclose(np.abs(scores.to_numpy()), np.abs(xc @ vt[:3].T))
    assert np.allclose(parts.explained_variance_ratio, sv[:3] ** 2 / (sv**2).sum())


def test_projection_needs_a_fit() -> None:
    """Asking for components before any fit fails with a clear message."""
    with pytest.raises(AssertionError, match="fit or partial_fit"):
        EmbeddingFeatures(components=2).transform(hidden(), np.zeros(3, dtype=int))


# -------------------------------------------------------------------------------
# Recording
# -------------------------------------------------------------------------------
def test_output_recorder_reads_layer_outputs_and_tuples() -> None:
    """The recorder keeps each module's output of the last pass, reading tuple outputs at index, and unhooks on exit."""
    torch.manual_seed(0)
    layers = nn.ModuleList(nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True) for _ in range(2))
    x = torch.randn(2, 5, 8)
    with OutputRecorder(list(layers)) as rec:
        h1 = layers[0](x)
        h2 = layers[1](h1)
    assert torch.equal(rec.output(0), h1.detach()) and torch.equal(rec.outputs()[1], h2.detach())
    assert all(not m._forward_hooks for m in layers)
    mha = nn.MultiheadAttention(8, 2, batch_first=True)
    with OutputRecorder(mha, index=0) as rec:
        out, _ = mha(x, x, x)
    assert torch.equal(rec.output(), out.detach())


# -------------------------------------------------------------------------------
# Store meta and fingerprints
# -------------------------------------------------------------------------------
def test_fingerprint_tells_models_apart() -> None:
    """Equal state dicts give one fingerprint; changing one weight changes it."""
    torch.manual_seed(0)
    a = nn.Linear(4, 2)
    b = nn.Linear(4, 2)
    b.load_state_dict(a.state_dict())
    assert model_fingerprint(a) == model_fingerprint(b)
    with torch.no_grad():
        b.weight[0, 0] += 1e-6
    assert model_fingerprint(a) != model_fingerprint(b)


def test_store_meta_is_saved_checked_and_read_back(tmp_path: Path) -> None:
    """The first meta is saved; the same meta or none reopens the store, a different one raises."""
    meta = {"model": "abc", "attributes": ("late", "tax")}
    store = HeadFeatureStore(tmp_path, meta=meta)
    store.add([1], pd.DataFrame({"x": [1.0]}))
    store.flush()
    assert HeadFeatureStore(tmp_path, meta=meta).meta == {"model": "abc", "attributes": ["late", "tax"]}
    assert HeadFeatureStore(tmp_path).meta["model"] == "abc"
    with pytest.raises(ValueError, match=r"meta differs in \['model'\]"):
        HeadFeatureStore(tmp_path, meta={**meta, "model": "def"})


def test_store_refuses_meta_for_chunks_written_without_it(tmp_path: Path) -> None:
    """Meta cannot be claimed for chunks already written without it; a store without meta has an empty one."""
    store = HeadFeatureStore(tmp_path)
    store.add([1], pd.DataFrame({"x": [1.0]}))
    store.flush()
    assert HeadFeatureStore(tmp_path).meta == {}
    with pytest.raises(ValueError, match="chunks without meta"):
        HeadFeatureStore(tmp_path, meta={"model": "abc"})


def test_embeddings_go_through_the_store(tmp_path: Path) -> None:
    """An embedding table is stored and read back by id like attention features."""
    emb = EmbeddingFeatures(prefix="h")
    with HeadFeatureStore(tmp_path, meta={"pool": "query"}) as store:
        store.add(["a", "b", "c"], emb.transform(hidden(), np.array([1, 2, 3])))
    back = HeadFeatureStore(tmp_path).read(["c", "a"], columns=["h_0", "h_3"])
    assert back.to_numpy().tolist() == [[230.0, 233.0], [10.0, 13.0]]


# -------------------------------------------------------------------------------
# Review: recorder, pooling, fingerprints, meta
# -------------------------------------------------------------------------------
def test_recorder_forgets_skipped_modules_and_keeps_its_copy() -> None:
    """With root, a module skipped in a pass gives no output; a later in-place change leaves the recording as it was."""

    class Gate(nn.Module):
        """Runs its inner layer only when asked."""

        def __init__(self) -> None:
            """One linear layer behind the gate."""
            super().__init__()
            self.inner = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor, run: bool) -> torch.Tensor:
            """The layer's output, or x unchanged."""
            return self.inner(x) if run else x

    g = Gate()
    with OutputRecorder(g.inner, root=g) as rec:
        y = g(torch.ones(2, 3), run=True)
        kept = rec.output().clone()
        y.data.add_(5.0)  # an in-place change after the hook ran
        assert torch.equal(rec.output(), kept)
        g(torch.ones(2, 3), run=False)
        with pytest.raises(AssertionError, match="no output of module 0"):
            rec.output()


def test_pool_checks_its_mode_and_query_counts() -> None:
    """An unknown pool override and n_queries of the wrong length are refused instead of giving other rows."""
    emb = EmbeddingFeatures()
    with pytest.raises(AssertionError, match="unknown pool"):
        emb.transform(hidden(), np.zeros(3, dtype=int), pool=cast(Any, "max"))
    with pytest.raises(AssertionError, match="2 counts for 3 sequences"):
        emb.transform(hidden(), np.zeros((3, 2), dtype=int), n_queries=[1, 2])


def test_fingerprint_sees_shapes_extra_state_and_config() -> None:
    """Equal values in another shape, non-tensor extra state and a config each change the fingerprint."""

    class Shaped(nn.Module):
        """One buffer of a given shape, with optional extra state."""

        def __init__(self, shape: tuple[int, ...], extra: object = None) -> None:
            """A buffer of the values 0 to 5 in this shape."""
            super().__init__()
            self.register_buffer("w", torch.arange(6.0).reshape(shape))
            self.extra = extra

        def get_extra_state(self) -> object:
            """The extra state, kept in the state dict."""
            return self.extra

        def set_extra_state(self, state: object) -> None:
            """Restore the extra state."""
            self.extra = state

    a, b = Shaped((2, 3)), Shaped((3, 2))
    assert model_fingerprint(a) != model_fingerprint(b)
    assert model_fingerprint(Shaped((2, 3), {"time_pos": True})) != model_fingerprint(Shaped((2, 3), {"time_pos": False}))
    assert model_fingerprint(a, {"context": 2048}) != model_fingerprint(a, {"context": 1024}) != model_fingerprint(a)


def test_meta_with_paths_and_a_store_opened_without_meta_is_read_only(tmp_path: Path) -> None:
    """Paths in meta are saved as text; a store with a meta opened without one reads but does not add."""
    meta = {"checkpoint": Path("runs/x.pt")}
    with HeadFeatureStore(tmp_path, meta=meta) as store:
        store.add([1], pd.DataFrame({"x": [1.0]}))
    assert HeadFeatureStore(tmp_path, meta=meta).meta == {"checkpoint": "runs/x.pt"}
    plain = HeadFeatureStore(tmp_path)
    assert plain.read_only and plain.read([1])["x"].tolist() == [1.0]
    with pytest.raises(ValueError, match="open it with that meta"):
        plain.add([2], pd.DataFrame({"x": [2.0]}))


def test_two_stores_starting_at_once_keep_one_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When another store saves its meta between the check and the write, this store compares with it and refuses."""
    real_link = __import__("os").link

    def other_first(src: str, dst: str) -> None:
        """Another store's meta lands first, then this link is tried."""
        Path(dst).write_text('{"model": "other"}')
        real_link(src, dst)

    monkeypatch.setattr("eventgpt.feature_engineering.attention_heads.os.link", other_first)
    with pytest.raises(ValueError, match=r"meta differs in \['model'\]"):
        HeadFeatureStore(tmp_path, meta={"model": "mine"})
    assert HeadFeatureStore(tmp_path).meta == {"model": "other"} and not list(tmp_path.glob(".*.tmp"))


def test_batches_with_other_columns_or_types_are_refused(tmp_path: Path) -> None:
    """A batch with a missing column or another type than the stored ones is refused, before or after a flush."""
    store = HeadFeatureStore(tmp_path)
    store.add([1], pd.DataFrame({"x": np.float32([1.0]), "y": np.float32([1.0])}))
    with pytest.raises(ValueError, match=r"\['y'\]"):
        store.add([2], pd.DataFrame({"x": np.float32([2.0])}))
    store.flush()
    with pytest.raises(ValueError, match=r"\['x'\]"):
        HeadFeatureStore(tmp_path).add([3], pd.DataFrame({"x": np.float64([3.0]), "y": np.float32([3.0])}))
