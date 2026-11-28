"""Tests of embedding features, the output recorder, store meta and model fingerprints."""

from __future__ import annotations

from pathlib import Path

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
