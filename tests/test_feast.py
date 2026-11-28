"""Tests of the Feast export: the offline source a store writes, and a feature view served end to end by Feast."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eventgpt.feature_engineering import HeadFeatureStore, column_groups, meta_tags, redshift_table, to_sql_names


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


def test_column_groups_keep_heads_whole_under_the_limit() -> None:
    """Groups stay under the limit, never split a head, keep every column once, and split only a too-wide prefix."""
    cols = [f"L{lay}H{h}_{a}" for lay in (1, 2) for h in (1, 2, 3) for a in ("x", "y", "z")]  # 6 heads of 3
    groups = column_groups(cols, max_columns=7)
    assert [len(g) for g in groups] == [6, 6, 6]
    assert all(len({c.split("_", 1)[0] for c in g}) == 2 for g in groups) and sum(groups, []) == cols
    wide = column_groups([f"emb_pc{i}" for i in range(10)] + ["L1H1_x"], max_columns=4)
    assert [len(g) for g in wide] == [4, 4, 3]  # the rest of the split prefix shares a table with the next head


def test_redshift_table_types_and_limits(tmp_path: Path) -> None:
    """The DDL types float32 features as real, adds the entity and timestamps, and refuses more than 1,600 columns."""
    store = filled_store(tmp_path / "store")
    ddl = redshift_table(store, "heads_0", {"client_id": "varchar(64)", "mon": "integer"})
    assert '"client_id" varchar(64)' in ddl and '"event_timestamp" timestamp' in ddl and '"L1H1_late" real' in ddl
    assert ddl.endswith('DISTKEY("client_id") SORTKEY("event_timestamp")')
    with pytest.raises(AssertionError, match="column_groups"):
        redshift_table(store, "t", {f"k{i}": "integer" for i in range(1597)})


def test_sql_names_are_plain_identifiers() -> None:
    """Names become lower-case identifiers of letters, digits and _, in order; clashes and long names are refused."""
    assert to_sql_names(["L1H1_s11=22", "L6H2_name=trx_40_count", "3d", "emb_pc1"]) == [
        "l1h1_s11_22",
        "l6h2_name_trx_40_count",
        "_3d",
        "emb_pc1",
    ]
    with pytest.raises(ValueError, match="both become 'a_b'"):
        to_sql_names(["a=b", "A_b"])
    with pytest.raises(ValueError, match="127"):
        to_sql_names(["x" * 128])


def test_sql_names_for_redshift(tmp_path: Path) -> None:
    """With sql_names=True the export, the DDL and the view's features use the same plain identifiers."""
    feast = pytest.importorskip("feast")
    from eventgpt.feature_engineering import feature_view

    store = HeadFeatureStore(tmp_path / "store")
    store.add([1], pd.DataFrame({"L1H1_s11=22": np.float32([1.0]), "L1H1_focus": np.float32([2.0])}))
    store.flush()
    ent = pd.DataFrame({"client_id": ["c1"], "event_timestamp": [pd.Timestamp(2024, 1, 31)]})
    out = store.export_feast(tmp_path / "src", [1], ent, sql_names=True)
    assert {"l1h1_s11_22", "l1h1_focus"} <= set(pd.read_parquet(out).columns)
    assert '"l1h1_s11_22" real' in redshift_table(store, "t", {"client_id": "varchar(64)"}, sql_names=True)
    source = feast.FileSource(name="s", path=str(out), timestamp_field="event_timestamp")
    view = feature_view(store, "v", [feast.Entity(name="client", join_keys=["client_id"])], source, timedelta(days=62), sql_names=True)
    assert [f.name for f in view.features] == ["l1h1_s11_22", "l1h1_focus"]


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
    view = feature_view(store, "heads", [client], source, ttl=timedelta(days=62), online=True)
    assert view.tags["model"] == "abc123" and [f.name for f in view.features] == ["L1H1_late", "L1H1_focus"]
    fs.apply([client, view])
    # training: a label on 15 February sees c1's January state (0.0), not February's, which is known on the 28th
    labels = pd.DataFrame({"client_id": ["c1", "c2"], "event_timestamp": pd.to_datetime(["2024-02-15", "2024-03-01"], utc=True)})
    hist = fs.get_historical_features(labels, ["heads:L1H1_late"]).to_df().sort_values("client_id")
    assert hist["L1H1_late"].tolist() == [0.0, 3.0]
    fs.materialize(
        start_date=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(), end_date=pd.Timestamp("2024-12-31", tz="UTC").to_pydatetime()
    )
    online = fs.get_online_features(["heads:L1H1_late"], [{"client_id": "c1"}, {"client_id": "c3"}]).to_dict()
    assert online["L1H1_late"] == [1.0, 4.0]  # each client's latest state


# -------------------------------------------------------------------------------
# Review: exports and tags
# -------------------------------------------------------------------------------
def test_a_new_export_replaces_the_old_one_whole(tmp_path: Path) -> None:
    """Exporting other ids to the same directory leaves none of the earlier parts, and no staging directory."""
    store = filled_store(tmp_path / "store")
    every = [("c1", 1), ("c1", 2), ("c2", 1), ("c2", 2), ("c3", 1)]
    store.export_feast(tmp_path / "src", every, entities_for(every))
    out = store.export_feast(tmp_path / "src", [("c1", 1)], entities_for([("c1", 1)]))
    assert len(pd.read_parquet(out)) == 1 and len(list(out.iterdir())) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "store"]


def test_a_failed_export_leaves_the_earlier_one(tmp_path: Path) -> None:
    """An export that fails (an id not stored) keeps the earlier export as it was and cleans up its staging."""
    store = filled_store(tmp_path / "store")
    store.export_feast(tmp_path / "src", [("c1", 1)], entities_for([("c1", 1)]))
    with pytest.raises(AssertionError, match="not stored"):
        store.export_feast(tmp_path / "src", [("c1", 1), ("c9", 9)], entities_for([("c1", 1), ("c9", 9)]))
    assert len(pd.read_parquet(tmp_path / "src")) == 1 and sorted(p.name for p in tmp_path.iterdir()) == ["src", "store"]


def test_export_refuses_other_files_and_name_clashes(tmp_path: Path) -> None:
    """A directory with other files is not replaced; a feature named like a timestamp or entity column is refused."""
    store = filled_store(tmp_path / "store")
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="other than an export's parts"):
        store.export_feast(tmp_path / "mine", [("c1", 1)], entities_for([("c1", 1)]))
    odd = HeadFeatureStore(tmp_path / "odd")
    odd.add([1], pd.DataFrame({"created_timestamp": [1.0]}))
    with pytest.raises(AssertionError, match="clash"):
        odd.export_feast(tmp_path / "x", [1], pd.DataFrame({"event_timestamp": [pd.Timestamp(2024, 1, 1)]}))


def test_caller_tags_cannot_replace_the_meta(tmp_path: Path) -> None:
    """A tag that would change the store's meta in the view is refused; other tags are added."""
    feast = pytest.importorskip("feast")
    from eventgpt.feature_engineering import feature_view

    store = filled_store(tmp_path / "store")
    source = feast.FileSource(name="s", path=str(tmp_path), timestamp_field="event_timestamp")
    client = [feast.Entity(name="client", join_keys=["client_id"])]
    assert feature_view(store, "v", client, source, timedelta(days=62), tags={"team": "risk"}).tags["team"] == "risk"
    with pytest.raises(ValueError, match="replace the store's meta"):
        feature_view(store, "v", client, source, timedelta(days=62), tags={"model": "other"})
