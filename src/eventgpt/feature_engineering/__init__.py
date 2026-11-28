"""Features engineered from trained models, reusable outside EventGPT.

They depend only on numpy, pandas and torch; HeadFeatureStore also needs pyarrow to write parquet,
feature_view needs Feast, and put_records
takes any SageMaker Feature Store runtime client (boto3).
"""

from eventgpt.feature_engineering.attention_heads import (
    AttentionRecorder,
    FieldValues,
    HeadFeatures,
    HeadFeatureStore,
    field_attributes,
    flatten_queries,
    group_tokens,
    head_summaries,
    model_fingerprint,
    query_rows,
    sdpa_weights,
    stack_layers,
    to_sql_names,
    token_attributes,
)
from eventgpt.feature_engineering.embeddings import EmbeddingFeatures, OutputRecorder
from eventgpt.feature_engineering.feast_views import column_groups, feature_view, meta_tags, redshift_table
from eventgpt.feature_engineering.interventions import AttentionIntervention, drop_events, grad_rollout, occlusion, rollout
from eventgpt.feature_engineering.sagemaker_features import (
    SAGEMAKER_MAX_FEATURES,
    feature_definitions,
    put_records,
    records,
    sagemaker_names,
)

__all__ = [
    "SAGEMAKER_MAX_FEATURES",
    "AttentionIntervention",
    "AttentionRecorder",
    "EmbeddingFeatures",
    "FieldValues",
    "HeadFeatureStore",
    "HeadFeatures",
    "column_groups",
    "drop_events",
    "feature_definitions",
    "feature_view",
    "field_attributes",
    "flatten_queries",
    "grad_rollout",
    "group_tokens",
    "head_summaries",
    "meta_tags",
    "model_fingerprint",
    "OutputRecorder",
    "occlusion",
    "put_records",
    "query_rows",
    "records",
    "redshift_table",
    "rollout",
    "sagemaker_names",
    "sdpa_weights",
    "stack_layers",
    "to_sql_names",
    "token_attributes",
]
