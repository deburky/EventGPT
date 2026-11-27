"""Tests of eventgpt.feature_engineering interventions and attributions, and the recorder's query rows and outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from eventgpt.feature_engineering import (
    AttentionIntervention,
    AttentionRecorder,
    HeadFeatures,
    drop_events,
    grad_rollout,
    occlusion,
    rollout,
    sdpa_weights,
)
from eventgpt.models.flat import Block, GPTConfig


class Attend(nn.Module):
    """Plain multi-head attention through F.scaled_dot_product_attention, returning per-head outputs (b, heads, t, d)."""

    def __init__(self, heads: int = 2, d: int = 4) -> None:
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(heads * d, heads * d) for _ in range(3))
        self.heads, self.d = heads, d

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attention outputs per head."""
        b, t, _ = x.shape
        q, k, v = (f(x).view(b, t, self.heads, self.d).transpose(1, 2) for f in (self.q, self.k, self.v))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def blocks_and_input() -> tuple[nn.ModuleList, torch.Tensor, torch.Tensor]:
    """Two flat blocks without dropout, an input and a mask that always keeps the diagonal."""
    torch.manual_seed(0)
    blocks = nn.ModuleList([Block(GPTConfig(vocab_size=10, n_head=4, d_model=16, dropout=0.0)) for _ in range(2)]).eval()
    mask = (torch.rand(3, 1, 6, 6) > 0.3) | torch.eye(6, dtype=torch.bool)
    return blocks, torch.randn(3, 6, 16), mask


def run(blocks: nn.ModuleList, x: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Blocks in sequence."""
    for blk in blocks:
        x = blk(x, **kw)
    return x


# -------------------------------------------------------------------------------
# Recorder: query rows and per-head outputs
# -------------------------------------------------------------------------------
def test_recorder_query_rows_and_outputs() -> None:
    """With query set, each module keeps only the query rows; outputs equal weights times values."""
    torch.manual_seed(1)
    att = Attend()
    x, q = torch.randn(2, 5, 8), torch.tensor([4, 2])
    with torch.no_grad(), AttentionRecorder.sdpa([att], outputs=True) as rec:
        y = att(x)
        full, out = rec.weights(), rec.outputs()
        rec.query = q
        att(x)
        rows, out_rows = rec.weights(), rec.outputs()
    assert np.allclose(out[0], y.numpy(), atol=1e-6)
    assert rows.shape == (1, 2, 2, 1, 5) and out_rows.shape == (1, 2, 2, 1, 4)
    assert np.allclose(rows[0, :, :, 0], full[0, np.arange(2), :, q.numpy()], atol=1e-7)
    assert np.allclose(out_rows[0, :, :, 0], out[0, np.arange(2), :, q.numpy()], atol=1e-7)


def test_coverage_is_mass_on_described_keys() -> None:
    """A head with half its weight on keys that carry an attribute has coverage 0.5."""
    attn = np.zeros((1, 1, 1, 4))
    attn[..., [0, 3]] = 0.5
    attrs = np.zeros((1, 4, 2))
    attrs[0, 0, 1] = 1.0
    cov = HeadFeatures(["a", "b"]).coverage(attn, None, attrs)
    assert cov.loc["L1H1", "coverage"] == 0.5


# -------------------------------------------------------------------------------
# Interventions
# -------------------------------------------------------------------------------
def test_no_intervention_leaves_outputs_unchanged() -> None:
    """Inside the context with nothing set, outputs are bit-identical and the function is restored after."""
    blocks, x, mask = blocks_and_input()
    original = torch.nn.functional.scaled_dot_product_attention
    with torch.no_grad():
        plain = run(blocks, x, attn_mask=mask)
        with AttentionIntervention(blocks):
            same = run(blocks, x, attn_mask=mask)
    assert torch.equal(plain, same)
    assert torch.nn.functional.scaled_dot_product_attention is original


def test_zero_and_mean_ablation_set_the_head_output() -> None:
    """Zero ablation gives zeros, mean ablation the collected mean, only in the chosen head; others are untouched."""
    torch.manual_seed(2)
    att = Attend()
    x = torch.randn(3, 5, 8)
    with torch.no_grad(), AttentionIntervention([att]) as iv:
        iv.collect_means = True
        y = att(x)
        iv.collect_means = False
        iv.ablate({0: [1]}, mode="zero")
        z = att(x)
        iv.ablate({0: [1]}, mode="mean")
        m = att(x)
    assert torch.equal(z[:, 0], y[:, 0]) and torch.all(z[:, 1] == 0)
    assert torch.allclose(m[:, 1], y[:, 1].mean((0, 1)).expand_as(m[:, 1]), atol=1e-6)
    assert torch.equal(m[:, 0], y[:, 0])


def test_knockout_renormalizes_over_the_remaining_keys() -> None:
    """Knocking out keys in one head equals attention with those keys masked; other heads keep the original output."""
    torch.manual_seed(3)
    att = Attend()
    x = torch.randn(2, 5, 8)
    keys = torch.zeros(2, 5, dtype=torch.bool)
    keys[:, 1] = True
    with torch.no_grad():
        y = att(x)
        b, t, _ = x.shape
        q, k, v = (f(x).view(b, t, 2, 4).transpose(1, 2) for f in (att.q, att.k, att.v))
        ref = sdpa_weights(q, k, ~keys[:, None, None, :]) @ v
        with AttentionIntervention([att]) as iv:
            iv.knockout(keys, heads={0: [0]})
            got = att(x)
            iv.knockout(None)
            back = att(x)
    assert torch.allclose(got[:, 0], ref[:, 0], atol=1e-5)
    assert torch.allclose(got[:, 1], y[:, 1], atol=1e-5)
    assert torch.allclose(back, y, atol=1e-6)


def test_knockout_in_blocks_with_a_mask() -> None:
    """In flat blocks with an explicit mask, removing nothing is a no-op and removing a key changes later outputs."""
    blocks, x, mask = blocks_and_input()
    keys = torch.zeros(3, 6, dtype=torch.bool)
    with torch.no_grad():
        plain = run(blocks, x, attn_mask=mask)
        with AttentionIntervention(blocks) as iv:
            iv.knockout(keys)
            same = run(blocks, x, attn_mask=mask)
            keys[:, 0] = True
            iv.knockout(keys)
            changed = run(blocks, x, attn_mask=mask)
    assert torch.allclose(plain, same, atol=1e-5)
    assert not torch.allclose(plain[:, 1:], changed[:, 1:], atol=1e-3)


def test_mean_ablation_estimate_is_exact_for_a_linear_readout() -> None:
    """When the output is linear in the head outputs, the first-order estimate equals the real change."""
    torch.manual_seed(4)
    att = Attend()
    x = torch.randn(3, 5, 8)
    w = torch.randn(2, 5, 4)

    def readout(y: torch.Tensor) -> torch.Tensor:
        """Linear in the per-head outputs, one number per row."""
        return torch.einsum("bhtd,htd->b", y, w)

    with AttentionIntervention([att], gradients=True) as iv:
        iv.collect_means = True
        with torch.no_grad():
            base = readout(att(x))
        iv.collect_means = False
        readout(att(x)).sum().backward()
        est = iv.mean_ablation_estimate()
        real = []
        for h in range(2):
            iv.ablate({0: [h]}, mode="mean")
            with torch.no_grad():
                real.append((readout(att(x)) - base).numpy())
    assert est.shape == (3, 1, 2)
    assert np.allclose(est[:, 0], np.stack(real, 1), atol=1e-4)


def test_weight_gradients_and_grad_rollout() -> None:
    """Explicit weights equal the kernel's; their gradients feed grad_rollout, which gives the identity without signal."""
    blocks, x, mask = blocks_and_input()
    with torch.no_grad():
        plain = run(blocks, x, attn_mask=mask)
    with AttentionIntervention(blocks, weight_gradients=True) as iv:
        out = run(blocks, x, attn_mask=mask)
        out[:, -1].sum().backward()
        w, g = iv.weight_gradients()
    assert torch.allclose(out, plain, atol=1e-5)
    assert w.shape == g.shape == (2, 3, 4, 6, 6)
    readout = np.zeros((3, 6))
    readout[:, -1] = 1.0
    imp = grad_rollout(w, g, readout)
    assert imp.shape == (3, 6) and np.all(imp >= 0)
    assert np.allclose(grad_rollout(w, np.zeros_like(g), readout), readout)


# -------------------------------------------------------------------------------
# Attributions to keys
# -------------------------------------------------------------------------------
def test_rollout_of_uniform_causal_attention() -> None:
    """One layer, residual 0: rollout read at the last position is that position's attention row."""
    t = 4
    w = np.tril(np.ones((t, t)))
    w = (w / w.sum(1, keepdims=True))[None, None, None]
    readout = np.zeros((1, t))
    readout[0, -1] = 1.0
    assert np.allclose(rollout(w, readout, residual=0.0), [[0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(rollout(w, readout, residual=1.0), readout)


def test_drop_events_and_occlusion() -> None:
    """Events holding the token go, except ones with a kept token; occlusion rescored only the rows that had it."""
    evt, usr = 1, 2
    s = np.array([usr, 5, evt, 7, 8, evt, 7, evt, 9])
    assert drop_events(s, [evt, usr], 7).tolist() == [usr, 5, evt, 9]
    assert drop_events(np.array([usr, 7]), [evt, usr], 7, keep=[usr]).tolist() == [usr, 7]
    assert drop_events(np.array([evt, 7]), [evt, usr], 7).tolist() == [evt, 7]
    seqs = [s, np.array([usr, 5, evt, 9])]
    calls: list[int] = []

    def score(batch: list[np.ndarray]) -> np.ndarray:
        """Sequence length as the score."""
        calls.append(len(batch))
        return np.array([len(b) for b in batch], dtype=float)

    d = occlusion(score, seqs, {"seven": 7, "nine": 9}, starts=[evt, usr])
    assert d["seven"].tolist() == [-5.0, 0.0]
    assert d["nine"].tolist() == [-2.0, -2.0]
    assert calls == [2, 1, 2]
