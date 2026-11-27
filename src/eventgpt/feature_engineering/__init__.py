"""Features engineered from trained models, reusable outside EventGPT.

They depend only on numpy, pandas and torch; HeadFeatureStore also needs pyarrow to write parquet.
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
    query_rows,
    sdpa_weights,
    stack_layers,
    token_attributes,
)
from eventgpt.feature_engineering.interventions import AttentionIntervention, drop_events, grad_rollout, occlusion, rollout

__all__ = [
    "AttentionIntervention",
    "AttentionRecorder",
    "FieldValues",
    "HeadFeatureStore",
    "HeadFeatures",
    "drop_events",
    "field_attributes",
    "flatten_queries",
    "grad_rollout",
    "group_tokens",
    "head_summaries",
    "occlusion",
    "query_rows",
    "rollout",
    "sdpa_weights",
    "stack_layers",
    "token_attributes",
]
