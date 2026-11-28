"""A HeadFeatureStore through Feast on AWS: export, S3, COPY into Redshift, a point-in-time join and DynamoDB.

Every column of the store is exported, split by column_groups into Redshift-sized tables, one feature view each. The
training set and the online reads are checked against the store itself, and everything created is removed. The
defaults point at oblako (https://github.com/almostly/oblako), a local AWS; on AWS pass your own cluster, role and
bucket. The entities parquet has one row per stored id, in the store's id order: client_id and event_timestamp.

    AWS_PROFILE=oblako uv run --with 'feast[aws]==0.68.0' python examples/feature_store/feast_redshift.py \
        --store features/ --entities entities.parquet --rows 2000
"""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from feast import Entity, FeatureStore, RepoConfig, ValueType
from feast.infra.offline_stores.redshift import RedshiftOfflineStoreConfig
from feast.infra.offline_stores.redshift_source import RedshiftSource
from feast.infra.online_stores.dynamodb import DynamoDBOnlineStoreConfig

from eventgpt.feature_engineering import HeadFeatureStore, column_groups, feature_view, redshift_table, to_sql_names


# -------------------------------------------------------------------------------
# Redshift
# -------------------------------------------------------------------------------
class Redshift:
    """SQL through the Redshift Data API, one statement at a time."""

    def __init__(self, cluster: str, database: str, user: str) -> None:
        """A Data API client for this cluster, database and user."""
        self.client = boto3.client("redshift-data")
        self.target = {"ClusterIdentifier": cluster, "Database": database, "DbUser": user}

    def run(self, sql: str) -> None:
        """Run one statement and wait for it, raising its error if it fails."""
        sid = self.client.execute_statement(Sql=sql, **self.target)["Id"]
        while (state := self.client.describe_statement(Id=sid))["Status"] not in ("FINISHED", "FAILED", "ABORTED"):
            time.sleep(0.5)
        if state["Status"] != "FINISHED":
            raise RuntimeError(f"{state['Status']}: {state.get('Error')}")


# -------------------------------------------------------------------------------
# Export, load, apply, join and serve
# -------------------------------------------------------------------------------
def main() -> None:
    """Export, load, apply, join and serve; compare with the store; clean up."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="a HeadFeatureStore directory")
    ap.add_argument("--entities", required=True, help="parquet of client_id and event_timestamp, one row per id")
    ap.add_argument("--rows", type=int, default=2000, help="the first ids to export")
    ap.add_argument("--out", default="feast_export", help="local directory for the export")
    ap.add_argument("--bucket", default="eventgpt-heads")
    ap.add_argument("--role", default="arn:aws:iam::123456789012:role/redshift-copy")
    ap.add_argument("--cluster", default="analytics")
    ap.add_argument("--database", default="oblako")
    ap.add_argument("--user", default="oblako")
    ap.add_argument("--region", default="us-east-1")
    a = ap.parse_args()
    t0 = time.time()
    entities = pd.read_parquet(a.entities).iloc[: a.rows].reset_index(drop=True)
    entities["event_timestamp"] = pd.to_datetime(entities["event_timestamp"], utc=True)
    ids = list(range(len(entities)))
    store = HeadFeatureStore(a.store)
    groups = column_groups(store.columns())
    print(f"{len(store.columns()):,} columns in {len(groups)} groups of {[len(g) for g in groups]}; {len(ids):,} rows")

    s3 = boto3.client("s3")
    s3.create_bucket(Bucket=a.bucket)
    sql = Redshift(a.cluster, a.database, a.user)
    client = Entity(name="client", join_keys=["client_id"], value_type=ValueType.STRING)
    config = RepoConfig(
        project="eventgpt_heads",
        provider="aws",
        registry=f"s3://{a.bucket}/feast/registry.db",
        offline_store=RedshiftOfflineStoreConfig(
            cluster_id=a.cluster,
            region=a.region,
            database=a.database,
            user=a.user,
            s3_staging_location=f"s3://{a.bucket}/feast",
            iam_role=a.role,
        ),
        online_store=DynamoDBOnlineStoreConfig(region=a.region),
        entity_key_serialization_version=3,
    )
    fs = FeatureStore(config=config)
    try:
        views = []
        for k, cols in enumerate(groups):
            out = store.export_feast(Path(a.out) / f"heads_{k}", ids, entities, columns=cols, sql_names=True)
            for part in sorted(out.glob("part-*.parquet")):
                s3.upload_file(str(part), a.bucket, f"heads_{k}/{part.name}")
            sql.run(f'DROP TABLE IF EXISTS "heads_{k}"')
            sql.run(redshift_table(store, f"heads_{k}", {"client_id": "varchar(64)"}, cols, sql_names=True))
            sql.run(f"COPY \"heads_{k}\" FROM 's3://{a.bucket}/heads_{k}/' IAM_ROLE '{a.role}' FORMAT AS PARQUET")
            source = RedshiftSource(
                name=f"heads_{k}_source",
                table=f"heads_{k}",
                timestamp_field="event_timestamp",
                created_timestamp_column="created_timestamp",
            )
            views.append(
                feature_view(store, f"heads_{k}", [client], source, ttl=timedelta(days=62), columns=cols, online=k == 0, sql_names=True)
            )
            print(f"  heads_{k}: {len(cols):,} columns exported, staged and loaded ({time.time() - t0:.0f}s)")
        fs.apply([client, *views])
        print(f"applied {len(fs.list_feature_views())} views ({time.time() - t0:.0f}s)")

        # training: a label 15 days after a cut sees that cut; one column from each view
        picks = [g[0] for g in groups]
        sample = entities.iloc[:: max(1, len(entities) // 40)].head(40)
        labels = sample.assign(event_timestamp=sample["event_timestamp"] + timedelta(days=15))
        refs = [f"heads_{k}:{to_sql_names([c])[0]}" for k, c in enumerate(picks)]
        hist = fs.get_historical_features(labels, refs).to_df()
        want = store.read(list(sample.index), columns=picks).set_index(
            sample["client_id"].astype(str) + "|" + sample["event_timestamp"].astype(str)
        )
        key = hist["client_id"].astype(str) + "|" + (pd.to_datetime(hist["event_timestamp"], utc=True) - timedelta(days=15)).astype(str)
        got = hist.set_index(key)[to_sql_names(picks)].loc[want.index]
        ok = np.allclose(got.to_numpy(float), want.to_numpy(float), rtol=1e-6)
        print(f"point-in-time join: {len(hist)} labels x {len(refs)} views, equal to the store: {ok} ({time.time() - t0:.0f}s)")

        # serving: the latest state of each client, from DynamoDB; materialize loads every cut, not only the last ttl
        start, end = entities["event_timestamp"].min(), entities["event_timestamp"].max() + timedelta(days=1)
        fs.materialize(start_date=start.to_pydatetime(), end_date=end.to_pydatetime())
        some = list(entities["client_id"].unique()[:5])
        online = fs.get_online_features(
            [f"heads_0:{to_sql_names([c])[0]}" for c in groups[0][:3]], [{"client_id": c} for c in some]
        ).to_dict()
        last = entities.sort_values("event_timestamp").groupby("client_id").tail(1).reset_index().set_index("client_id").loc[some, "index"]
        latest = store.read(list(last), columns=groups[0][:3])
        ok = np.allclose(np.array([online[to_sql_names([c])[0]] for c in groups[0][:3]], dtype=float).T, latest.to_numpy(float), rtol=1e-6)
        print(f"online: {len(some)} clients x 3 features, equal to each client's latest state: {ok} ({time.time() - t0:.0f}s)")
    finally:
        fs.teardown()
        for k in range(len(groups)):
            sql.run(f'DROP TABLE IF EXISTS "heads_{k}"')
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=a.bucket, Key=obj["Key"])
        s3.delete_bucket(Bucket=a.bucket)
        print(f"cleaned up ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
