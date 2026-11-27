"""Tests of counts, ordinal fields and the chunked store in eventgpt.feature_engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eventgpt.feature_engineering import HeadFeatures, HeadFeatureStore, field_attributes


def uniform_over(n_keys: list[int], t: int) -> np.ndarray:
    """(1, b, 1, t) query rows spreading the weight evenly over the first n keys of each row, 0 on the rest."""
    w = np.zeros((1, len(n_keys), 1, t))
    for r, n in enumerate(n_keys):
        w[0, r, 0, :n] = 1.0 / n
    return w


# -------------------------------------------------------------------------------
# Counts
# -------------------------------------------------------------------------------
def test_counts_are_shares_times_the_keys_seen() -> None:
    """A uniform head over n keys gives the number of keys with the attribute; padding (weight 0) is not counted."""
    attrs = np.zeros((2, 6, 1))
    attrs[0, [0, 1], 0] = 1  # row 0: 2 of its 4 keys
    attrs[1, [0], 0] = 1  # row 1: 1 of its 6 keys
    w = uniform_over([4, 6], 6)
    shares = HeadFeatures(["late"]).transform(w, None, attrs)
    counts = HeadFeatures(["late"], counts=True).transform(w, None, attrs)
    assert np.allclose(shares["L1H1_late"], [0.5, 1 / 6])
    assert np.allclose(counts["L1H1_late_count"], [2.0, 1.0])
    assert "L1H1_late" not in counts.columns


def test_counts_for_some_attributes_and_with_a_valid_mask() -> None:
    """Only the named attributes become counts; a valid mask overrides the nonzero-weight rule."""
    attrs = np.ones((1, 4, 2))
    w = uniform_over([4], 4)
    table = HeadFeatures(["a", "b"], counts=["b"]).transform(w, None, attrs, valid=np.array([[True, True, False, False]]))
    assert np.isclose(table["L1H1_a"].iat[0], 1.0) and np.isclose(table["L1H1_b_count"].iat[0], 2.0)
    with pytest.raises(AssertionError):
        HeadFeatures(["a"], counts=["missing"])


# -------------------------------------------------------------------------------
# Ordinal fields
# -------------------------------------------------------------------------------
NAMES = ["[PAD]", "amt_bin=3", "amt_bin=7", "amt_bin=zero", "kind=pay", "dt=2", "amt_bin=na"]


def test_field_attributes_read_values_and_presence() -> None:
    """Numeric tokens are read, listed ones are mapped, others and keys without the field are absent."""
    keys = np.array([[[1, 4], [2, 5], [3, 4], [4, 0], [6, 4], [0, 0]]])
    fv = field_attributes(keys, NAMES, ["amt_bin", "dt"], values={"amt_bin": {"zero": -1}})
    assert fv.names == ["amt_bin", "dt"]
    assert fv.present[0, :, 0].tolist() == [True, True, True, False, False, False]
    assert fv.values[0, :3, 0].tolist() == [3.0, 7.0, -1.0]
    assert fv.present[0, :, 1].tolist() == [False, True, False, False, False, False] and fv.values[0, 1, 1] == 2.0


def test_field_means_over_the_keys_that_have_the_field() -> None:
    """A uniform head averages the field over keys having it; a head on keys without it gives NaN."""
    keys = np.array([[[1], [2], [4], [4]], [[4], [4], [4], [2]]])  # row 1: only key 3 has an amount
    fv = field_attributes(keys, NAMES, ["amt_bin"])
    w = uniform_over([4, 3], 4)  # row 1 looks only at its first 3 keys
    table = HeadFeatures([], fields=["amt_bin"]).transform(w, None, np.zeros((2, 4, 0)), fields=fv)
    assert np.isclose(table["L1H1_amt_bin_mean"].iat[0], 5.0)  # (3 + 7) / 2
    assert np.isnan(table["L1H1_amt_bin_mean"].iat[1])
    with pytest.raises(AssertionError):
        HeadFeatures([], fields=["amt_bin"]).transform(w, None, np.zeros((2, 4, 0)))


# -------------------------------------------------------------------------------
# Store
# -------------------------------------------------------------------------------
def test_store_chunks_resumes_and_reads_in_order(tmp_path: Path) -> None:
    """Rows are written in chunks, a reopened store skips saved rows, and read returns any requested order."""
    table = pd.DataFrame({"x": np.arange(10.0)})
    ids = [f"r{i}" for i in range(10)]
    store = HeadFeatureStore(tmp_path, chunk_rows=4)
    store.add(ids[:3], table.iloc[:3])
    store.add(ids[3:6], table.iloc[3:6])  # 6 buffered: one chunk written
    assert len(list(tmp_path.glob("chunk-*.parquet"))) == 1
    store.flush()
    again = HeadFeatureStore(tmp_path, chunk_rows=4)
    assert again.has_all(ids[:6]) and not again.has(ids[6])
    again.add(ids[4:], table.iloc[4:])  # r4 and r5 are already stored and are skipped
    out = again.read(ids[::-1])
    assert out["x"].tolist() == list(range(9, -1, -1))
    assert len(pd.concat([pd.read_parquet(p) for p in tmp_path.glob("chunk-*.parquet")])) == 10
    with pytest.raises(AssertionError):
        again.read(["missing"])
    assert again.columns() == ["x"]
    both = HeadFeatureStore(tmp_path / "two")
    both.add(["a", "b"], pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
    assert both.read(["b"], columns=["y"]).to_dict("list") == {"y": [4.0]}


def test_store_flushes_when_the_extraction_fails(tmp_path: Path) -> None:
    """Leaving the context with an error still writes the buffered rows."""
    with pytest.raises(RuntimeError), HeadFeatureStore(tmp_path, chunk_rows=100) as store:
        store.add(["a", "b"], pd.DataFrame({"x": [1.0, 2.0]}))
        raise RuntimeError("stopped")
    assert HeadFeatureStore(tmp_path).has_all(["a", "b"])


def test_table_layout_matches_a_column_by_column_build() -> None:
    """Names, order and values equal a reference that builds every column one at a time."""
    rng = np.random.default_rng(5)
    w = rng.dirichlet(np.ones(6), size=(3, 7, 4))  # (layers, b, heads, t) query rows
    attrs = rng.integers(0, 2, size=(7, 6, 3)).astype(float)
    keys = rng.integers(1, 3, size=(7, 6))
    fv = field_attributes(keys, NAMES, ["amt_bin"])
    hf = HeadFeatures(["a", "b", "c"], counts=["b"], fields=["amt_bin"])
    table = hf.transform(w, None, attrs, fields=fv)
    rows = w.transpose(1, 0, 2, 3)
    cols, data = [], []
    for ly in range(3):
        for h in range(4):
            wt = rows[:, ly, h]
            for a, name in enumerate(["a", "b", "c"]):
                share = (wt * attrs[..., a]).sum(-1)
                cols.append(f"L{ly + 1}H{h + 1}_{name}_count" if name == "b" else f"L{ly + 1}H{h + 1}_{name}")
                data.append(share * (wt > 0).sum(-1) if name == "b" else share)
            p = fv.present[..., 0]
            cols.append(f"L{ly + 1}H{h + 1}_amt_bin_mean")
            data.append((wt * fv.values[..., 0] * p).sum(-1) / (wt * p).sum(-1))
            cols.append(f"L{ly + 1}H{h + 1}_focus")
            data.append(-(wt * np.log(wt)).sum(-1))
    assert list(table.columns) == cols
    assert np.allclose(table.to_numpy(), np.stack(data, 1))
