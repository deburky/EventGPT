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
    attention = feature_view(store, "attention_heads", [client], source)

The entity is the client and the cut is the timestamp: get_historical_features then gives each label the client's
latest state known before it, and materialize puts each client's latest state in the online store. On AWS the same
directory is loaded into Redshift (or read from S3) and passed as a RedshiftSource instead. Feast is imported only
here, so the rest of feature_engineering does not need it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

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


def meta_tags(meta: dict[str, Any]) -> dict[str, str]:
    """A store's meta as Feast tags: strings as they are, every other value as JSON."""
    return {k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in meta.items()}


def feature_view(
    store: HeadFeatureStore,
    name: str,
    entities: Sequence[Entity],
    source: DataSource,
    columns: Sequence[str] | None = None,
    ttl: timedelta = timedelta(days=3650),
    online: bool = False,
    description: str = "",
    tags: dict[str, str] | None = None,
) -> FeatureView:
    """A Feast FeatureView of the store's feature columns (or `columns`), typed from the stored parquet schema.

    The store's meta becomes the view's tags (with `tags` added), so the model fingerprint sits in the registry next
    to the features. The ttl defaults to ten years because each row is a state at a cut, a point event, and a
    training cohort may join cuts from anywhere in history; online=False keeps a wide view out of the online store
    until it is wanted there.
    """
    from feast import FeatureView, Field

    parts = store._chunks()
    assert parts, f"no chunks in {store.dir}"
    schema = pq.read_schema(parts[0])
    names = store.columns() if columns is None else list(columns)
    unknown = [c for c in names if c not in schema.names]
    assert not unknown, f"columns not in the store: {unknown[:5]}"
    fields = [Field(name=c, dtype=_feast_type(schema.field(c).type)) for c in names]
    return FeatureView(
        name=name,
        entities=list(entities),
        ttl=ttl,
        schema=fields,
        source=source,
        online=online,
        description=description,
        tags={**meta_tags(store.meta), **(tags or {})},
    )
