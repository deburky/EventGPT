"""Tests of the Feast export: the offline source a store writes, and a feature view served end to end by Feast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eventgpt.feature_engineering import HeadFeatureStore, meta_tags


def filled_store(directory: Path) -> HeadFeatureStore:
    """A store of 5 rows keyed (client, month), in chunks of 2, with float32 features and a meta."""
    store = HeadFeatureStore(directory, chunk_rows=2, meta={"model": "abc123", "attributes": ["late"]})
    ids = [("c1", 1), ("c1", 2), ("c2", 1), ("c2", 2), ("c3", 1)]
    table = pd.DataFrame({"L1H1_late": np.arange(5, dtype=np.float32), "L1H1_focus": np.ones(5, dtype=np.float32)})
    store.add(ids, table)
    store.flush()
    return store


def entities_for(ids: list[tuple[str, int]]) -> pd.DataFrame:
    """Entity rows for (client, month) ids: the join keys and the month's end as the time the features were known."""
    return pd.DataFrame(
        {
            "client_id": [c for c, _ in ids],
            "mon": [m for _, m in ids],
            "event_timestamp": [pd.Timestamp(2024, m, 28) for _, m in ids],
        }
    )


# -------------------------------------------------------------------------------
# The offline source
# -------------------------------------------------------------------------------
def test_export_writes_entities_timestamps_and_features(tmp_path: Path) -> None:
    """Each requested id appears once with its entity keys, a UTC timestamp, created_timestamp and its features."""
    store = filled_store(tmp_path / "store")
    ids = [("c2", 2), ("c1", 1), ("c3", 1)]
    out = store.export_feast(tmp_path / "src", ids, entities_for(ids))
    assert len(list(out.glob("part-*.parquet"))) == 3 and not list(out.glob(".*"))
    t = pd.read_parquet(out).sort_values(["client_id", "mon"]).reset_index(drop=True)
    assert t[["client_id", "mon"]].values.tolist() == [["c1", 1], ["c2", 2], ["c3", 1]]
    assert t["L1H1_late"].tolist() == [0.0, 3.0, 4.0] and t["L1H1_late"].dtype == np.float32
    assert str(t["event_timestamp"].dt.tz) == "UTC" and "created_timestamp" in t


def test_export_checks_its_inputs(tmp_path: Path) -> None:
    """Unstored ids, repeated ids and entity rows without a timestamp are refused."""
    store = filled_store(tmp_path / "store")
    with pytest.raises(AssertionError, match="not stored"):
        store.export_feast(tmp_path / "a", [("c9", 1)], entities_for([("c9", 1)]))
    with pytest.raises(AssertionError, match="given twice"):
        store.export_feast(tmp_path / "b", [("c1", 1)] * 2, entities_for([("c1", 1)] * 2))
    with pytest.raises(AssertionError, match="event_timestamp"):
        store.export_feast(tmp_path / "c", [("c1", 1)], entities_for([("c1", 1)]).drop(columns="event_timestamp"))


def test_meta_becomes_string_tags() -> None:
    """Feast tags are strings: strings stay, other values become JSON."""
    assert meta_tags({"model": "abc", "attributes": ["late", "tax"], "layers": 6}) == {
        "model": "abc",
        "attributes": '["late", "tax"]',
        "layers": "6",
    }


# -------------------------------------------------------------------------------
# Feast end to end
# -------------------------------------------------------------------------------
def test_feast_serves_the_export_point_in_time_and_online(tmp_path: Path) -> None:
    """Feast applies the view, joins the features point in time for training, and serves them after materializing."""
    feast = pytest.importorskip("feast")
    from feast import Entity, FileSource, ValueType

    from eventgpt.feature_engineering import feature_view

    store = filled_store(tmp_path / "store")
    ids = [("c1", 1), ("c1", 2), ("c2", 1), ("c2", 2), ("c3", 1)]
    src = store.export_feast(tmp_path / "src", ids, entities_for(ids))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "feature_store.yaml").write_text(
        "project: heads\nprovider: local\nregistry: registry.db\n"
        "online_store:\n  type: sqlite\n  path: online.db\noffline_store:\n  type: dask\n"
        "entity_key_serialization_version: 3\n"
    )
    fs = feast.FeatureStore(repo_path=str(repo))
    client = Entity(name="client", join_keys=["client_id"], value_type=ValueType.STRING)
    source = FileSource(name="heads_source", path=str(src), timestamp_field="event_timestamp", created_timestamp_column="created_timestamp")
    view = feature_view(store, "heads", [client], source, online=True)
    assert view.tags["model"] == "abc123" and [f.name for f in view.schema] == ["L1H1_late", "L1H1_focus"]
    fs.apply([client, view])
    # training: a label on 15 February sees c1's January state (0.0), not February's, which is known on the 28th
    labels = pd.DataFrame({"client_id": ["c1", "c2"], "event_timestamp": pd.to_datetime(["2024-02-15", "2024-03-01"], utc=True)})
    hist = fs.get_historical_features(labels, ["heads:L1H1_late"]).to_df().sort_values("client_id")
    assert hist["L1H1_late"].tolist() == [0.0, 3.0]
    fs.materialize_incremental(end_date=pd.Timestamp("2024-12-31", tz="UTC").to_pydatetime())
    online = fs.get_online_features(["heads:L1H1_late"], [{"client_id": "c1"}, {"client_id": "c3"}]).to_dict()
    assert online["L1H1_late"] == [1.0, 4.0]  # each client's latest state
