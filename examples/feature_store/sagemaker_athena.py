"""A HeadFeatureStore in SageMaker Feature Store: feature groups, PutRecord, online reads and Athena point in time.

The store's columns are split under SageMaker's 2,500 features per group, and the rows are put in shuffled order with
put_records. The online store (GetRecord, BatchGetRecord) is checked against each client's latest state, including a
late write of an older cut, and the offline store against every row: the Parquet under the group's resolved S3
location, and a point-in-time query in Athena on the group's Glue table. Everything created is removed.

The defaults point at oblako 0.7.4 or later (https://github.com/almostly/oblako), a local AWS, installed with the
sagemaker (or parquet) extra and started with `oblako up sagemaker`, `oblako up iceberg` and `oblako up trino` (or
`oblako up athena` from 0.7.5). On AWS pass your own role and bucket. The entities parquet has one row per stored id,
in the store's id order: client_id and event_timestamp.

    AWS_PROFILE=oblako uv run --with boto3 python examples/feature_store/sagemaker_athena.py \
        --store features/ --entities entities.parquet --rows 2000
"""

from __future__ import annotations

import argparse
import io
import time
from typing import Any

import boto3
import numpy as np
import pandas as pd

from eventgpt.feature_engineering import (
    SAGEMAKER_MAX_FEATURES,
    HeadFeatureStore,
    column_groups,
    feature_definitions,
    put_records,
    sagemaker_names,
)


# -------------------------------------------------------------------------------
# Athena
# -------------------------------------------------------------------------------
def athena_rows(athena: Any, sql: str, database: str, output: str) -> list[list[str | None]]:
    """The rows of an Athena query, after waiting for it to finish."""
    qid = athena.start_query_execution(
        QueryString=sql, QueryExecutionContext={"Database": database}, ResultConfiguration={"OutputLocation": output}
    )["QueryExecutionId"]
    while (state := athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"])["State"] in ("QUEUED", "RUNNING"):
        time.sleep(0.5)
    assert state["State"] == "SUCCEEDED", state
    out = []
    for page in athena.get_paginator("get_query_results").paginate(QueryExecutionId=qid):
        out += [[d.get("VarCharValue") for d in r["Data"]] for r in page["ResultSet"]["Rows"]]
    return out[1:]


# -------------------------------------------------------------------------------
# Groups, records and reads
# -------------------------------------------------------------------------------
def main() -> None:
    """Create the groups, put the records, read them back online and offline, compare with the store, clean up."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="a HeadFeatureStore directory")
    ap.add_argument("--entities", required=True, help="parquet of client_id and event_timestamp, one row per id")
    ap.add_argument("--rows", type=int, default=2000, help="the first ids to put")
    ap.add_argument("--bucket", default="eventgpt-sm-features")
    ap.add_argument("--role", default="arn:aws:iam::000000000000:role/oblako")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    entities = pd.read_parquet(a.entities).iloc[: a.rows].reset_index(drop=True)
    entities["client_id"] = entities["client_id"].astype(str)
    entities["event_timestamp"] = pd.to_datetime(entities["event_timestamp"], utc=True)
    n_rows = len(entities)
    store = HeadFeatureStore(a.store)
    groups = column_groups(store.columns(), max_columns=SAGEMAKER_MAX_FEATURES - 2)
    print(
        f"{len(store.columns()):,} columns in {len(groups)} groups of {[len(g) for g in groups]}; {entities['client_id'].nunique()} clients"
    )
    sm, rt, s3, athena = (boto3.client(s) for s in ("sagemaker", "sagemaker-featurestore-runtime", "s3", "athena"))
    s3.create_bucket(Bucket=a.bucket)
    names = [f"heads-{k}" for k in range(len(groups))]
    shuffled = np.random.default_rng(0).permutation(n_rows)  # out of time order: put_records orders each id's writes
    try:
        for name, cols in zip(names, groups, strict=True):
            sm.create_feature_group(
                FeatureGroupName=name,
                RecordIdentifierFeatureName="client_id",
                EventTimeFeatureName="event_time",
                FeatureDefinitions=feature_definitions(store, "client_id", "event_time", cols),
                OnlineStoreConfig={"EnableOnlineStore": True},
                OfflineStoreConfig={"S3StorageConfig": {"S3Uri": f"s3://{a.bucket}/offline"}},
                RoleArn=a.role,
                Description=f"{len(cols)} features of {a.store}",
            )
            table = pd.concat([entities.iloc[shuffled].reset_index(drop=True), store.read(list(shuffled), columns=cols)], axis=1)
            n = put_records(rt, name, table, "client_id", "event_timestamp", columns=cols, workers=a.workers)
            print(f"  {name}: {len(cols):,} features, {n:,} records put ({time.time() - t0:.0f}s)")

        # online: each client's latest cut, by GetRecord and BatchGetRecord
        by_time = entities.sort_values("event_timestamp").reset_index()
        clients = list(entities["client_id"].unique()[:10])
        last = by_time.groupby("client_id").tail(1).set_index("client_id").loc[clients, "index"]
        ok = True
        for name, cols in zip(names, groups, strict=True):
            pick = cols[:: max(1, len(cols) // 5)][:5]
            want = store.read(list(last), columns=pick).to_numpy(np.float32)
            got = []
            for c in clients:
                rec = {
                    f["FeatureName"]: f["ValueAsString"]
                    for f in rt.get_record(FeatureGroupName=name, RecordIdentifierValueAsString=c)["Record"]
                }
                got.append([float(rec[s]) for s in sagemaker_names(pick)])
            ok &= bool(np.array_equal(np.float32(got), want))
        batch = rt.batch_get_record(Identifiers=[{"FeatureGroupName": names[0], "RecordIdentifiersValueAsString": clients}])
        n_batch = len(batch["Records"])
        print(f"online: {len(clients)} clients x {len(names)} groups equal to each latest state: {ok}; BatchGetRecord {n_batch} records")

        # an older cut written after the latest one: the online store keeps the latest event time
        first = by_time.groupby("client_id").head(1).set_index("client_id").loc[clients[0], "index"]
        old = pd.concat([entities.iloc[[first]].reset_index(drop=True), store.read([first], columns=groups[0])], axis=1)
        put_records(rt, names[0], old, "client_id", "event_timestamp", columns=groups[0])
        rec = {
            f["FeatureName"]: f["ValueAsString"]
            for f in rt.get_record(FeatureGroupName=names[0], RecordIdentifierValueAsString=clients[0])["Record"]
        }
        written, latest = entities["event_timestamp"].iloc[first], entities["event_timestamp"].iloc[last.iloc[0]]
        print(f"late write of {written:%Y-%m-%d}: online at {rec['event_time'][:10]}, latest {latest:%Y-%m-%d}")

        # offline: every row under the group's resolved S3 location, and point in time through Athena
        for name, cols in zip(names, groups, strict=True):
            desc = sm.describe_feature_group(FeatureGroupName=name)["OfflineStoreConfig"]
            prefix = desc["S3StorageConfig"]["ResolvedOutputS3Uri"].removeprefix(f"s3://{a.bucket}/")
            pages = s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=prefix)
            keys = [o["Key"] for p in pages for o in p.get("Contents", []) if o["Key"].endswith(".parquet")]
            off = pd.concat(
                [pd.read_parquet(io.BytesIO(s3.get_object(Bucket=a.bucket, Key=k)["Body"].read())) for k in keys], ignore_index=True
            )
            off["event_time"] = pd.to_datetime(off["event_time"], utc=True)
            pick = sagemaker_names(cols[:3])
            mine = pd.concat([entities[["client_id"]], store.read(list(range(n_rows)), columns=cols[:3])], axis=1)
            mine.columns = ["client_id", *pick]
            mine["event_time"] = entities["event_timestamp"]
            both = off.merge(mine, on=["client_id", "event_time"], suffixes=("", "_store"))
            same = all(np.array_equal(both[c].to_numpy(np.float32), both[f"{c}_store"].to_numpy(np.float32)) for c in pick)
            print(f"offline {name}: {len(both):,} rows equal to the store: {same and len(both) == n_rows + (name == names[0])}")
            cat = desc["DataCatalogConfig"]
            for cutoff in sorted(mine["event_time"].unique())[1::4][:3]:
                stamp = pd.Timestamp(cutoff).strftime("%Y-%m-%dT%H:%M:%SZ")
                sql = f"""SELECT client_id, {pick[0]} FROM (
                    SELECT *, row_number() OVER (PARTITION BY client_id
                        ORDER BY event_time DESC, api_invocation_time DESC, write_time DESC) AS rn
                    FROM "{cat["Database"]}"."{cat["TableName"]}" WHERE event_time <= '{stamp}'
                ) WHERE rn = 1 AND NOT is_deleted"""
                back = pd.Series(
                    {c: float(v) for c, v in athena_rows(athena, sql, cat["Database"], f"s3://{a.bucket}/athena/")}, dtype=np.float64
                )
                seen = (
                    mine[mine["event_time"] <= cutoff]
                    .sort_values("event_time")
                    .groupby("client_id")
                    .tail(1)
                    .set_index("client_id")[pick[0]]
                )
                pit_ok = len(back) == len(seen) and np.array_equal(back.loc[seen.index].to_numpy(np.float32), seen.to_numpy(np.float32))
                print(f"  Athena as of {stamp[:10]}: {len(back)} clients, equal to the store: {pit_ok} ({time.time() - t0:.0f}s)")
    finally:
        glue = boto3.client("glue")
        for name in names:
            try:
                cat = sm.describe_feature_group(FeatureGroupName=name)["OfflineStoreConfig"]["DataCatalogConfig"]
                glue.delete_table(DatabaseName=cat["Database"], Name=cat["TableName"])  # SageMaker leaves it behind
            except Exception as e:  # noqa: BLE001  a group or table not created
                print(f"  {name} table: {type(e).__name__}")
            try:
                sm.delete_feature_group(FeatureGroupName=name)
            except Exception as e:  # noqa: BLE001  a group not created
                print(f"  {name}: {type(e).__name__}")
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=a.bucket, Key=obj["Key"])
        s3.delete_bucket(Bucket=a.bucket)
        print(f"cleaned up ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
