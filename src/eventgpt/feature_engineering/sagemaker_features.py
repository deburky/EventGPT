"""SageMaker Feature Store feature groups for the tables of a HeadFeatureStore.

A feature group holds a record identifier, an event time and up to 2,500 features in all; each record is put with
PutRecord, its values as strings. The online store keeps each identifier's record with the latest event time and the
offline store keeps every record, so a client's cuts give its history offline and its latest state online:

    import boto3
    from eventgpt.feature_engineering import HeadFeatureStore, column_groups, feature_definitions, put_records

    store = HeadFeatureStore("features/")
    sm, runtime = boto3.client("sagemaker"), boto3.client("sagemaker-featurestore-runtime")
    for k, cols in enumerate(column_groups(store.columns(), max_columns=SAGEMAKER_MAX_FEATURES - 2)):
        sm.create_feature_group(FeatureGroupName=f"heads-{k}", RecordIdentifierFeatureName="client_id",
                                EventTimeFeatureName="event_time",
                                FeatureDefinitions=feature_definitions(store, "client_id", "event_time", cols),
                                OnlineStoreConfig={"EnableOnlineStore": True}, OfflineStoreConfig=..., RoleArn=...)
        table = pd.concat([entities, store.read(ids, columns=cols)], axis=1)   # or an export_feast part at a time
        put_records(runtime, f"heads-{k}", table, "client_id", "event_timestamp", columns=cols, workers=8)

Feature names follow SageMaker's rules (sagemaker_names): letters, digits, _ and -, at most 64 characters, starting and
ending with a letter or digit, none of is_deleted, write_time and api_invocation_time. They are lower case as well,
since the offline store is queried through Athena, which folds names to lower case. The store's meta goes in the
group's Description or tags; boto3 is not imported here, so any client with put_record works.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from eventgpt.feature_engineering.attention_heads import to_sql_names

if TYPE_CHECKING:
    from eventgpt.feature_engineering.attention_heads import HeadFeatureStore

SAGEMAKER_MAX_FEATURES = 2500
SAGEMAKER_RESERVED = ("is_deleted", "write_time", "api_invocation_time")
_NAME = re.compile(r"[a-zA-Z0-9]([-_]*[a-zA-Z0-9]){0,63}")


def sagemaker_names(columns: Sequence[str]) -> list[str]:
    """Column names as SageMaker feature names: the plain identifiers of to_sql_names, without leading or trailing _.

    Names that would clash, break SageMaker's pattern (at most 64 characters) or take a reserved name are refused.
    The names keep their order, so zip them with the originals to map back.
    """
    out = [n.strip("_") for n in to_sql_names(columns)]
    seen: dict[str, str] = {}
    for c, n in zip(columns, out, strict=True):
        if not _NAME.fullmatch(n):
            raise ValueError(f"column {c!r} gives {n!r}, not a SageMaker feature name (at most 64 characters)")
        if n in SAGEMAKER_RESERVED:
            raise ValueError(f"column {c!r} gives the reserved feature name {n!r}")
        if seen.setdefault(n, c) != c:
            raise ValueError(f"columns {seen[n]!r} and {c!r} both become {n!r}")
    return out


def _sagemaker_type(t: pa.DataType) -> str:
    """The SageMaker feature type of a parquet column type."""
    if pa.types.is_floating(t):
        return "Fractional"
    if pa.types.is_integer(t) or pa.types.is_boolean(t):
        return "Integral"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "String"
    raise TypeError(f"no SageMaker feature type for column type {t}")


def feature_definitions(
    store: HeadFeatureStore,
    record_id: str,
    event_time: str,
    columns: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """The FeatureDefinitions of a group of these feature columns, named by sagemaker_names.

    The record identifier and event time come first, as String, then the features typed from the store: floats
    Fractional, integers and booleans Integral.
    """
    parts = store._chunks()
    assert parts, f"no chunks in {store.dir}"
    schema = pq.read_schema(parts[0])
    names = store.columns() if columns is None else list(columns)
    n = len(names) + 2
    assert n <= SAGEMAKER_MAX_FEATURES, f"{n} features, above {SAGEMAKER_MAX_FEATURES}: split with column_groups"
    keys = sagemaker_names([record_id, event_time])
    assert keys == [record_id, event_time], f"record_id and event_time must be SageMaker names, got {keys}"
    shown = sagemaker_names(names)
    assert not {record_id, event_time} & set(shown), "a feature is named like the record identifier or event time"
    defs = [{"FeatureName": record_id, "FeatureType": "String"}, {"FeatureName": event_time, "FeatureType": "String"}]
    return defs + [{"FeatureName": s, "FeatureType": _sagemaker_type(schema.field(c).type)} for s, c in zip(shown, names, strict=True)]


def _text(v: Any) -> str | None:
    """A value as SageMaker's ValueAsString, or None for a missing value (the feature is left out of the record)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return repr(float(v))  # round-trips: float32 values come back exactly when read as float32
    return str(v)


def records(
    table: pd.DataFrame,
    record_id: str,
    timestamp: str,
    columns: Sequence[str] | None = None,
    event_time: str = "event_time",
) -> Iterator[list[dict[str, str]]]:
    """The PutRecord records of a table of a record id, a timestamp and features (store names), in time order.

    Rows come oldest first. The online store keeps each record id's latest event time, so the order does not change
    what it holds, but a store that keeps the last write instead (oblako before 0.7.4) holds the same state. The
    timestamp is written as event_time in UTC (2022-02-28T00:00:00Z); missing values are left out of the record, as
    SageMaker allows.
    """
    feats = [c for c in table.columns if c not in (record_id, timestamp)] if columns is None else list(columns)
    names = sagemaker_names(feats)
    when = pd.to_datetime(table[timestamp], utc=True)
    order = np.argsort(when.to_numpy(), kind="stable")
    ids = table[record_id].astype(str).to_numpy()
    stamps = when.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    values = table[feats].to_numpy(dtype=object)
    for i in order:
        rec = [{"FeatureName": record_id, "ValueAsString": ids[i]}, {"FeatureName": event_time, "ValueAsString": stamps[i]}]
        for name, v in zip(names, values[i], strict=True):
            s = _text(v)
            if s is not None:
                rec.append({"FeatureName": name, "ValueAsString": s})
        yield rec


def put_records(
    runtime: Any,
    group: str,
    table: pd.DataFrame,
    record_id: str,
    timestamp: str,
    columns: Sequence[str] | None = None,
    event_time: str = "event_time",
    workers: int = 1,
) -> int:
    """Put every row of the table in a feature group with runtime.put_record; the number of records put.

    With workers > 1 the records are put from that many threads, each record id always from the same thread and in
    time order, so its latest state is still written last. boto3 clients are safe to share between threads.
    """
    recs = list(records(table, record_id, timestamp, columns, event_time))
    if workers <= 1:
        for rec in recs:
            runtime.put_record(FeatureGroupName=group, Record=rec)
        return len(recs)
    lanes: list[list[list[dict[str, str]]]] = [[] for _ in range(workers)]
    lane_of: dict[str, int] = {}
    for rec in recs:  # each id to one lane, the lanes filled evenly
        rid = rec[0]["ValueAsString"]
        lanes[lane_of.setdefault(rid, len(lane_of) % workers)].append(rec)

    def put(lane: list[list[dict[str, str]]]) -> None:
        """Put one lane's records in order."""
        for rec in lane:
            runtime.put_record(FeatureGroupName=group, Record=rec)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, lanes))
    return len(recs)
