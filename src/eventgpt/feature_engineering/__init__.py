"""Features engineered from trained models, reusable outside EventGPT.

They depend only on numpy, pandas and torch; HeadFeatureStore also needs pyarrow to write parquet,
and feature_view needs Feast.
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
    token_attributes,
)
from eventgpt.feature_engineering.embeddings import EmbeddingFeatures, OutputRecorder
from eventgpt.feature_engineering.feast_views import feature_view, meta_tags
from eventgpt.feature_engineering.interventions import AttentionIntervention, drop_events, grad_rollout, occlusion, rollout

__all__ = [
    "AttentionIntervention",
    "AttentionRecorder",
    "EmbeddingFeatures",
    "FieldValues",
    "HeadFeatureStore",
    "HeadFeatures",
    "drop_events",
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
    "query_rows",
    "rollout",
    "sdpa_weights",
    "stack_layers",
    "token_attributes",
]
