# Feature stores

Two end-to-end runs of a `HeadFeatureStore` (attention or embedding features from `eventgpt.feature_engineering`)
through a feature store on AWS. Each script checks what the feature store returns against the store itself and
removes everything it created.

| Script | Path | Checks |
|---|---|---|
| `feast_redshift.py` | `export_feast`, S3, `COPY` into Redshift, Feast with a `RedshiftSource`, DynamoDB online | point-in-time join, online reads |
| `sagemaker_athena.py` | SageMaker Feature Store groups, `put_records`, the offline store's Glue table | online reads and a late write, offline rows, point-in-time query in Athena |

Both take the store directory and an entities parquet with one row per stored id, in the store's id order:
`client_id` and `event_timestamp`. A wide store is split with `column_groups`, under Redshift's 1,600 columns per
table and SageMaker's 2,500 features per group.

The defaults point at [oblako](https://github.com/almostly/oblako), a local AWS:

```bash
uv tool install 'oblako[sagemaker]>=0.7.5'
oblako up && oblako up redshift-data && oblako up sagemaker && oblako up athena
oblako configure

AWS_PROFILE=oblako uv run --with 'feast[aws]==0.68.0' python examples/feature_store/feast_redshift.py \
    --store features/ --entities entities.parquet
AWS_PROFILE=oblako uv run --with boto3 python examples/feature_store/sagemaker_athena.py \
    --store features/ --entities entities.parquet
```

On AWS, pass your own bucket and IAM role, and for Redshift the cluster, database and user.

With 4,536 head features of MBD-mini (2,000 client-months of 172 clients), the Feast run takes under a minute and the
SageMaker run about ten minutes on oblako 0.7.5. Every comparison with the store matched.
