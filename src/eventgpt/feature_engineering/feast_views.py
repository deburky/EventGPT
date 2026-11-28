"""Feast feature views for the tables of a HeadFeatureStore, typed from the stored columns.

A store of attention or embedding features can hold thousands of columns, too many to declare as Fields by hand, so
feature_view reads their types from the store, and carries the store's meta (the model fingerprint, the checkpoint,
the attribute list) into the view's tags, where the Feast registry shows which model produced them:

    from feast import Entity, FileSource, ValueType
    from eventgpt.feature_engineering import HeadFeatureStore, feature_view

    store = HeadFeatureStore("features/")
    store.export_feast("feast_source/", ids, entities)       # entities: client_id and event_timestamp per id
    client = Entity(name="client", join_keys=["client_id"], value_type=ValueType.STRING)
    source = FileSource(name="attention_source", path="feast_source/", timestamp_field="event_timestamp",
                        created_timestamp_column="created_timestamp")
    attention = feature_view(store, "attention_heads", [client], source, ttl=timedelta(days=62))

The entity is the client and the cut is the timestamp: get_historical_features then gives each label the client's
latest state known before it, and materialize puts each client's latest state in the online store. On AWS the same
directory is staged in S3, loaded into Redshift with COPY and read with a RedshiftSource. Redshift folds names to lower
case and Feast's Redshift SQL leaves them unquoted, so the export, the table and the view all take sql_names=True,
which writes L1H1_s11=22 as l1h1_s11_22. A Redshift table holds at most
1,600 columns, so a wide store is split with column_groups, which keeps each head's columns together, into one table
and one view per group; redshift_table writes each table's CREATE TABLE from the stored types:

    for k, cols in enumerate(column_groups(store.columns())):
        store.export_feast(f"export/heads_{k}/", ids, entities, columns=cols, sql_names=True)   # then upload to S3
        run(redshift_table(store, f"heads_{k}", {"client_id": "varchar(64)"}, cols, sql_names=True))
        run(f"COPY heads_{k} FROM 's3://bucket/heads_{k}/' IAM_ROLE '...' FORMAT AS PARQUET")
        source = RedshiftSource(name=f"heads_{k}_source", table=f"heads_{k}", timestamp_field="event_timestamp",
                                created_timestamp_column="created_timestamp")
        views.append(feature_view(store, f"heads_{k}", [client], source, timedelta(days=62), cols, sql_names=True))

Feast is imported only by feature_view, so the rest of feature_engineering does not need it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from eventgpt.feature_engineering.attention_heads import to_sql_names

if TYPE_CHECKING:
    from feast import Entity, FeatureView
    from feast.data_source import DataSource

    from eventgpt.feature_engineering.attention_heads import HeadFeatureStore


def _feast_type(t: pa.DataType) -> Any:
    """The Feast type of a parquet column type."""
    from feast import types

    if pa.types.is_float32(t) or pa.types.is_float16(t):
        return types.Float32
    if pa.types.is_floating(t):
        return types.Float64
    if pa.types.is_boolean(t):
        return types.Bool
    if pa.types.is_integer(t):
        return types.Int64 if t.bit_width > 32 or (pa.types.is_unsigned_integer(t) and t.bit_width == 32) else types.Int32
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return types.String
    raise TypeError(f"no Feast type for column type {t}")


REDSHIFT_MAX_COLUMNS = 1600


def column_groups(columns: Sequence[str], max_columns: int = REDSHIFT_MAX_COLUMNS - 8) -> list[list[str]]:
    """Feature columns in groups of at most max_columns, each head's columns (one L{layer}H{head} prefix) kept together.

    The default leaves room in a 1,600-column Redshift table for up to 6 entity columns and the two timestamps. Columns
    without a head prefix (embeddings: emb_pc1, ...) are grouped by their own prefix in the same way.
    """
    by_prefix: dict[str, list[str]] = {}
    for c in columns:
        by_prefix.setdefault(c.split("_", 1)[0], []).append(c)
    groups: list[list[str]] = [[]]
    for cols in by_prefix.values():
        if len(cols) > max_columns:  # one prefix wider than a table: split it on its own
            groups += [cols[i : i + max_columns] for i in range(0, len(cols), max_columns)]
            continue
        if len(groups[-1]) + len(cols) > max_columns:
            groups.append([])
        groups[-1] += cols
    return [g for g in groups if g]


def _redshift_type(t: pa.DataType) -> str:
    """The Redshift column type of a parquet column type."""
    if pa.types.is_float32(t) or pa.types.is_float16(t):
        return "real"
    if pa.types.is_floating(t):
        return "double precision"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_integer(t):
        return "bigint" if t.bit_width > 32 or (pa.types.is_unsigned_integer(t) and t.bit_width == 32) else "integer"
    raise TypeError(f"no Redshift type for column type {t}")


def redshift_table(
    store: HeadFeatureStore,
    table: str,
    entities: dict[str, str],
    columns: Sequence[str] | None = None,
    timestamp_field: str = "event_timestamp",
    sql_names: bool = False,
) -> str:
    """CREATE TABLE for an export_feast of these columns, distributed on the first entity column and sorted by time.

    The table has the entity columns with the Redshift types given, the two timestamps, and the features typed from
    the store (float32 as real, float64 as double precision). sql_names=True names the features as
    export_feast(sql_names=True) writes them.
    """
    parts = store._chunks()
    assert parts, f"no chunks in {store.dir}"
    schema = pq.read_schema(parts[0])
    names = store.columns() if columns is None else list(columns)
    n = len(entities) + 2 + len(names)
    assert n <= REDSHIFT_MAX_COLUMNS, f"{n} columns, above Redshift's {REDSHIFT_MAX_COLUMNS}: split with column_groups"
    cols = [f'"{c}" {t}' for c, t in entities.items()]
    cols += [f'"{timestamp_field}" timestamp', '"created_timestamp" timestamp']
    shown = to_sql_names(names) if sql_names else names
    cols += [f'"{c}" {_redshift_type(schema.field(o).type)}' for c, o in zip(shown, names, strict=True)]
    body = ",\n    ".join(cols)
    return f'CREATE TABLE "{table}" (\n    {body})\nDISTKEY("{next(iter(entities))}") SORTKEY("{timestamp_field}")'


def meta_tags(meta: dict[str, Any]) -> dict[str, str]:
    """A store's meta as Feast tags: strings as they are, every other value as JSON."""
    return {k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in meta.items()}


def _merge_tags(meta: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """The store's meta tags and the caller's, refusing a caller tag that would replace a meta value."""
    clash = sorted(k for k in extra if k in meta and extra[k] != meta[k])
    if clash:
        raise ValueError(f"tags {clash} would replace the store's meta")
    return {**meta, **extra}


def feature_view(
    store: HeadFeatureStore,
    name: str,
    entities: Sequence[Entity],
    source: DataSource,
    ttl: timedelta,
    columns: Sequence[str] | None = None,
    online: bool = False,
    description: str = "",
    tags: dict[str, str] | None = None,
    sql_names: bool = False,
) -> FeatureView:
    """A Feast FeatureView of the store's feature columns (or `columns`), typed from the stored parquet schema.

    The store's meta becomes the view's tags (with `tags` added, which may not change it), so the model fingerprint
    sits in the registry next to the features.

    The ttl is how old a state may be: a training example more than ttl after the client's latest cut gets missing
    values, so for monthly cuts a ttl of about two months keeps one missed month. Feast does not apply the ttl to
    online reads (up to 0.68), so materialize more often than the ttl. The first materialize_incremental looks back
    only one ttl from its end date, so load older cuts with materialize(start_date, end_date).

    online=False keeps a wide view out of the online store until it is wanted there. sql_names=True names the features
    as export_feast(sql_names=True) writes them, as a Redshift source must be.
    """
    from feast import FeatureView, Field

    parts = store._chunks()
    assert parts, f"no chunks in {store.dir}"
    schema = pq.read_schema(parts[0])
    names = store.columns() if columns is None else list(columns)
    unknown = [c for c in names if c not in schema.names]
    assert not unknown, f"columns not in the store: {unknown[:5]}"
    shown = to_sql_names(names) if sql_names else names
    fields = [Field(name=c, dtype=_feast_type(schema.field(o).type)) for c, o in zip(shown, names, strict=True)]
    return FeatureView(
        name=name,
        entities=list(entities),
        ttl=ttl,
        schema=fields,
        source=source,
        online=online,
        description=description,
        tags=_merge_tags(meta_tags(store.meta), tags or {}),
    )
