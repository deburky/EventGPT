"""Tests of counts, ordinal fields and the chunked store in eventgpt.feature_engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


# -------------------------------------------------------------------------------
# Review: weighting, valid keys, shapes
# -------------------------------------------------------------------------------
def test_counts_and_field_means_follow_uneven_weights() -> None:
    """With uneven weights the count is the weighted share times the keys seen, and the field mean is weighted."""
    w = np.array([[[[0.7, 0.2, 0.1, 0.0]]]])  # (layers, b, heads, t): 3 keys seen
    attrs = np.array([[[1.0], [0.0], [1.0], [1.0]]])  # attribute on keys 0, 2 and the unseen key 3
    counts = HeadFeatures(["late"], counts=True).transform(w, None, attrs)
    assert np.isclose(counts["L1H1_late_count"].iat[0], (0.7 + 0.1) * 3)  # 2.4, not the raw 2
    keys = np.array([[[1], [2], [4], [2]]])  # amt_bin 3, 7, none, 7
    fv = field_attributes(keys, NAMES, ["amt_bin"])
    table = HeadFeatures([], fields=["amt_bin"]).transform(w, None, np.zeros((1, 4, 0)), fields=fv)
    assert np.isclose(table["L1H1_amt_bin_mean"].iat[0], (0.7 * 3 + 0.2 * 7) / 0.9)  # 3.89, not the plain mean 5


def test_counts_use_only_valid_keys_for_share_and_number() -> None:
    """A key marked invalid adds to neither the share nor the number of keys, even with weight on it."""
    w = np.array([[[[0.5, 0.25, 0.25]]]])
    attrs = np.array([[[0.0], [0.0], [1.0]]])  # only the invalid key 2 has the attribute
    valid = np.array([[True, True, False]])
    table = HeadFeatures(["late"], counts=True).transform(w, None, attrs, valid=valid)
    assert table["L1H1_late_count"].iat[0] == 0.0


def test_one_key_sequences() -> None:
    """Weights of one-key sequences, (layers, b, heads, 1, 1), give one row each, with or without a query."""
    w = np.ones((2, 3, 4, 1, 1))
    attrs = np.ones((3, 1, 1))
    for q in (None, np.zeros(3, dtype=int)):
        table = HeadFeatures(["a"]).transform(w, q, attrs)
        assert table.shape == (3, 2 * 4 * 2) and np.allclose(table.filter(like="_a"), 1.0)


# -------------------------------------------------------------------------------
# Review: the store
# -------------------------------------------------------------------------------
def test_a_write_stopped_part_way_leaves_no_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When writing a chunk fails part-way, no chunk file is left behind and the store reopens and resumes."""
    store = HeadFeatureStore(tmp_path, chunk_rows=10)
    store.add([1], pd.DataFrame({"x": [1.0]}))
    store.flush()
    real = pd.DataFrame.to_parquet

    def stopped(self: pd.DataFrame, path: Any, *args: Any, **kwargs: Any) -> None:
        """Write a truncated file where asked, then fail, like a process stopped during the write."""
        Path(path).write_bytes(b"PAR1 truncated")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", stopped)
    store.add([2], pd.DataFrame({"x": [2.0]}))
    with pytest.raises(KeyboardInterrupt):
        store.flush()
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real)
    again = HeadFeatureStore(tmp_path)  # a partial chunk-*.parquet would fail to open here
    assert again.has(1) and not again.has(2)
    again.add([2], pd.DataFrame({"x": [2.0]}))
    assert again.read([1, 2])["x"].tolist() == [1.0, 2.0]


def test_two_stores_counting_the_same_chunks_keep_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two stores that see the same existing chunks when they write still write separate chunks."""
    a, b = HeadFeatureStore(tmp_path), HeadFeatureStore(tmp_path)
    a.add(["a1", "a2"], pd.DataFrame({"x": [1.0, 2.0]}))
    b.add(["b1"], pd.DataFrame({"x": [3.0]}))
    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([]))  # both see no chunk yet, as when writing at once
    a.flush()
    b.flush()
    monkeypatch.setattr(Path, "glob", real_glob)
    assert len(list(tmp_path.glob("chunk-*.parquet"))) == 2
    assert HeadFeatureStore(tmp_path).read(["b1", "a2", "a1"])["x"].tolist() == [3.0, 2.0, 1.0]


def test_store_ids_repeated_tuples_and_numpy_ints(tmp_path: Path) -> None:
    """A repeated id in one batch keeps its first row; tuple ids round-trip; numpy and Python ints are one id."""
    store = HeadFeatureStore(tmp_path)
    store.add([("c1", 1), ("c1", 1), ("c2", 3)], pd.DataFrame({"x": [1.0, 9.0, 2.0]}))
    store.add([np.int64(7)], pd.DataFrame({"x": [5.0]}))
    store.flush()
    again = HeadFeatureStore(tmp_path)
    assert again.has(("c1", 1)) and again.has(7) and not again.has(("c1", 2))
    assert again.read([("c2", 3), ("c1", 1), 7])["x"].tolist() == [2.0, 1.0, 5.0]
    assert len(again.read()) == 3


def test_store_chunks_never_exceed_chunk_rows(tmp_path: Path) -> None:
    """One batch larger than chunk_rows is split into chunks of at most chunk_rows."""
    store = HeadFeatureStore(tmp_path, chunk_rows=3)
    store.add(list(range(8)), pd.DataFrame({"x": np.arange(8.0)}))
    store.flush()
    sizes = sorted(len(pd.read_parquet(p)) for p in tmp_path.glob("chunk-*.parquet"))
    assert sizes == [2, 3, 3]
    assert HeadFeatureStore(tmp_path).read(list(range(8)))["x"].tolist() == list(range(8))


def test_store_reads_chunks_from_the_earlier_format(tmp_path: Path) -> None:
    """Chunks written with a row_id column (the first version) are read with the new ones."""
    pd.DataFrame({"row_id": [0, 1], "x": [1.0, 2.0]}).to_parquet(tmp_path / "chunk-00000.parquet", index=False)
    store = HeadFeatureStore(tmp_path)
    assert store.has(1) and store.has(np.int64(0))
    store.add([1, 2], pd.DataFrame({"x": [9.0, 3.0]}))  # 1 is stored already
    assert store.read([2, 1, 0])["x"].tolist() == [3.0, 2.0, 1.0]
    assert store.columns() == ["x"]


def test_chunks_of_one_store_sort_in_writing_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks written within one clock tick still sort oldest first, so a row stored twice is read from its first chunk."""
    monkeypatch.setattr("eventgpt.feature_engineering.attention_heads.time.time_ns", lambda: 1_000)
    store = HeadFeatureStore(tmp_path, chunk_rows=1)
    for i in range(12):
        store.add([i], pd.DataFrame({"x": [float(i)]}))
    store.flush()
    order = [int(pd.read_parquet(p)["x"].iat[0]) for p in store._chunks()]
    assert order == list(range(12))
