"""Tests of eventgpt.feature_engineering: recording per-head attention and turning it into features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from eventgpt.feature_engineering import AttentionRecorder, HeadFeatures, group_tokens, head_summaries, token_attributes
from eventgpt.models.flat import Block, GPTConfig


# -------------------------------------------------------------------------------
# Features
# -------------------------------------------------------------------------------
def test_uniform_head_reads_the_mean_attribute() -> None:
    """A head spreading its weight evenly over the keys summarizes each attribute by its mean, with focus log(t)."""
    t = 5
    attn = np.full((1, 1, 1, t, t), 1.0 / t)
    attrs = np.arange(t, dtype=float).reshape(1, t, 1)
    summ, focus = head_summaries(attn, np.array([t - 1]), attrs)
    assert np.isclose(summ[0, 0, 0, 0], 2.0)
    assert np.isclose(focus[0, 0, 0], np.log(t))


def test_focused_head_reads_its_key() -> None:
    """A head with all its weight on key 2 returns key 2's attributes and focus 0, per row and per head."""
    attn = np.zeros((2, 3, 4, 6, 6))
    attn[..., 2] = 1.0
    attrs = np.random.default_rng(0).normal(size=(3, 6, 2))
    summ, focus = head_summaries(attn, np.array([5, 3, 4]), attrs)
    assert summ.shape == (3, 2, 4, 2)
    assert np.allclose(summ, attrs[:, 2][:, None, None, :])
    assert np.allclose(focus, 0.0)


def test_query_position_is_per_row() -> None:
    """Each row is read at its own query position."""
    attn = np.zeros((1, 2, 1, 3, 3))
    attn[0, 0, 0, 1, 0] = 1.0  # row 0, query 1 looks at key 0
    attn[0, 1, 0, 2, 1] = 1.0  # row 1, query 2 looks at key 1
    attrs = np.array([[[10.0], [20.0], [30.0]], [[1.0], [2.0], [3.0]]])
    summ, _ = head_summaries(attn, np.array([1, 2]), attrs)
    assert summ[:, 0, 0, 0].tolist() == [10.0, 2.0]


def test_table_enrichment_and_matches() -> None:
    """Columns are named by layer and head; per-layer input equals stacked input; enrichment is mass over share."""
    rng = np.random.default_rng(1)
    attn = rng.dirichlet(np.ones(4), size=(2, 30, 3, 4))  # (layers, b, heads, t, t)
    attrs = rng.integers(0, 2, size=(30, 4, 2)).astype(float)
    hf = HeadFeatures(["late", "tax"])
    query = np.full(30, 3)
    table = hf.transform(attn, query, attrs)
    assert table.shape == (30, 2 * 3 * 3)
    assert list(table.columns[:3]) == ["L1H1_late", "L1H1_tax", "L1H1_focus"]
    assert np.allclose(table.to_numpy(), hf.transform([torch.from_numpy(a) for a in attn], query, attrs).to_numpy())
    base = attrs.mean(1)
    enr = hf.enrichment(attn, query, attrs, base)
    summ, _ = head_summaries(attn, query, attrs)
    assert enr.shape == (6, 2) and enr.index[0] == "L1H1"
    assert np.isclose(enr.loc["L2H3", "tax"], summ[:, 1, 2, 1].mean() / base[:, 1].mean())
    ref = pd.DataFrame({"copy": table["L2H3_tax"] * 3 + 1, "noise": rng.random(30)})
    assert HeadFeatures.match(table[["L2H3_tax"]], ref)["L2H3_tax"][0] == ("copy", 1.0)


# -------------------------------------------------------------------------------
# Recording
# -------------------------------------------------------------------------------
def test_flat_blocks_record_heads_and_keep_the_averaged_mode() -> None:
    """EventGPT's blocks give causal per-head rows summing to 1, clean up on exit, and keep their averaged mode."""
    torch.manual_seed(0)
    blocks = nn.ModuleList([Block(GPTConfig(vocab_size=10, n_head=4, d_model=16, dropout=0.0)) for _ in range(2)]).eval()
    x = torch.randn(2, 7, 16)
    with AttentionRecorder.flat_blocks(blocks) as rec:
        h = x
        for blk in blocks:
            h = blk(h)
        attn = rec.weights()
    assert attn.shape == (2, 2, 4, 7, 7)
    assert np.allclose(attn.sum(-1), 1.0, atol=1e-5)
    assert np.allclose(np.triu(attn[0, 0, 0], 1), 0.0)
    assert all(b.attn is None and not b.keep_attn and not b.keep_heads for b in blocks)
    blocks[0].keep_attn = True
    with torch.no_grad():
        blocks[0](x)
    assert blocks[0].attn is not None
    assert np.allclose(blocks[0].attn.numpy(), attn[0].mean(1), atol=1e-6)


def test_torch_encoder_records_through_its_fast_path() -> None:
    """nn.TransformerEncoder in evaluation mode: one layer per entry, per-head weights equal to a direct call."""
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)
    enc = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False).eval()
    x = torch.randn(2, 6, 16)
    before = torch.backends.mha.get_fastpath_enabled()
    with torch.no_grad(), AttentionRecorder.torch_mha(enc) as rec:
        enc(x)
        attn = rec.weights()
    assert torch.backends.mha.get_fastpath_enabled() == before
    assert attn.shape == (3, 2, 4, 6, 6)
    assert np.allclose(attn.sum(-1), 1.0, atol=1e-5)
    mha = enc.layers[0].self_attn
    with torch.no_grad():
        _, direct = mha(x, x, x, need_weights=True, average_attn_weights=False)
    assert np.allclose(attn[0], direct.numpy(), atol=1e-6)


def test_custom_module_with_a_reader() -> None:
    """Any module works when a reader says where its weights are."""

    class Toy(nn.Module):
        """Uniform attention over the keys, returned next to its output."""

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Uniform weights (b, 1, t, t)."""
            b, t, _ = x.shape
            return x, torch.full((b, 1, t, t), 1.0 / t)

    def read(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> torch.Tensor:
        """The weights are the second output."""
        return output[1]

    toy = Toy()
    with AttentionRecorder([toy], read=read) as rec:
        toy(torch.zeros(3, 4, 2))
        attn = rec.weights()
    assert attn.shape == (1, 3, 1, 4, 4)
    table = HeadFeatures(["x"]).transform(attn, np.array([3, 3, 3]), np.ones((3, 4, 1)))
    assert np.allclose(table["L1H1_x"], 1.0) and np.allclose(table["L1H1_focus"], np.log(4))


# -------------------------------------------------------------------------------
# Linking to the data
# -------------------------------------------------------------------------------
def test_group_tokens_collects_each_events_tokens() -> None:
    """Tokens are grouped by their slot, in order, padded with 0."""
    tokens = np.array([[5, 6, 7, 8, 0]])
    slot = np.array([[1, 1, 2, 2, 0]])
    out = group_tokens(tokens, slot, 3)
    assert out.shape == (1, 3, 2)
    assert out[0].tolist() == [[0, 0], [5, 6], [7, 8]]


def test_token_attributes_name_count_and_base() -> None:
    """One attribute per token, the most frequent first, base shares over the keys up to the query."""
    names = ["[PAD]", "late=0", "late=1", "tax"]
    keys = np.array([[[1, 0], [2, 3], [2, 0], [0, 0]]])  # keys: late=0 / late=1 + tax / late=1 / empty
    attrs, got, base = token_attributes(keys, names, top=2)
    assert got == ["late=1", "late=0"]
    assert attrs[0, :, 0].tolist() == [0, 1, 1, 0] and attrs[0, :, 1].tolist() == [1, 0, 0, 0]
    assert np.allclose(base[0], [2 / 3, 1 / 3])  # three non-empty keys
    _, _, base_q = token_attributes(keys, names, include=["tax"], query=np.array([1]))
    assert np.allclose(base_q[0], [1 / 2])  # keys 0 and 1 only
    attrs1, got1, _ = token_attributes(np.array([[1, 2, 2]]), names, top=None)  # one token per key
    assert got1 == ["late=1", "late=0"] and attrs1.shape == (1, 3, 2)


def test_tokens_to_enrichment_end_to_end() -> None:
    """A head that always attends to the late=1 key shows enrichment above 1 on late=1 and below 1 elsewhere."""
    names = ["[PAD]", "late=0", "late=1"]
    rng = np.random.default_rng(2)
    keys = rng.integers(1, 3, size=(40, 6))
    attn = np.full((1, 40, 1, 6, 6), 1e-9)
    for r in range(40):
        hits = np.flatnonzero(keys[r] == 2)
        attn[0, r, 0, 5, hits if len(hits) else [5]] = 1.0
    attn /= attn.sum(-1, keepdims=True)
    query = np.full(40, 5)
    attrs, got, base = token_attributes(keys, names, query=query)
    enr = HeadFeatures(got).enrichment(attn, query, attrs, base)
    assert enr.loc["L1H1", "late=1"] > 1.5 and enr.loc["L1H1", "late=0"] < 0.5


def test_match_joins_on_an_id() -> None:
    """match(on=...) aligns rows by id whatever their order, and warns about rows without a partner."""
    rng = np.random.default_rng(3)
    feats = pd.DataFrame({"case_id": np.arange(20), "L1H1_late": rng.random(20)})
    ref = pd.DataFrame({"case_id": np.arange(25)[::-1], "n_late": 0.0, "label": "x"})
    known = ref["case_id"] < 20
    ref.loc[known, "n_late"] = 2 * feats.set_index("case_id").loc[ref.loc[known, "case_id"], "L1H1_late"].to_numpy()
    with pytest.warns(UserWarning, match="5 reference rows"):
        got = HeadFeatures.match(feats, ref, on="case_id")
    assert got["L1H1_late"][0] == ("n_late", 1.0)


def test_query_rows_alone_give_the_same_features() -> None:
    """Passing only the query rows (query None) gives the same table as the full weights."""
    rng = np.random.default_rng(4)
    attn = rng.dirichlet(np.ones(5), size=(2, 8, 3, 5))
    query = rng.integers(0, 5, size=8)
    attrs = rng.random((8, 5, 2))
    hf = HeadFeatures(["a", "b"])
    rows = attn[:, np.arange(8), :, query, :].transpose(1, 0, 2, 3)
    assert np.allclose(hf.transform(attn, query, attrs).to_numpy(), hf.transform(rows, None, attrs).to_numpy())
