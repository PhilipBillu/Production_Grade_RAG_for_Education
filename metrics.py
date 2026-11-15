"""
Module: metrics.py
==================
Prometheus Instrumentation for Production-Grade RAG for Education.

Tracks:
1. End-to-end Pipeline Latency (P50, P95, P99) via high-resolution Histogram buckets.
2. Request counts and Throughput by endpoint, method, and HTTP status code.
3. Pipeline Failure Rates & Error classifications.
4. Token Consumption metrics (Prompt, Completion, Total) via tiktoken.
5. Refusal triggers (Pre-generation gate vs Post-generation refusal).
"""

from typing import Tuple
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Summary,
    generate_latest,
)

# ==============================================================================
# 1. Latency Metrics (Histogram for P50, P95, P99 Percentiles)
# ==============================================================================

# Latency histogram with fine-grained buckets for accurate quantile calculation:
# PromQL P50: histogram_quantile(0.50, sum(rate(rag_pipeline_latency_seconds_bucket[5m])) by (le))
# PromQL P95: histogram_quantile(0.95, sum(rate(rag_pipeline_latency_seconds_bucket[5m])) by (le))
# PromQL P99: histogram_quantile(0.99, sum(rate(rag_pipeline_latency_seconds_bucket[5m])) by (le))
RAG_PIPELINE_LATENCY_SECONDS = Histogram(
    "rag_pipeline_latency_seconds",
    "End-to-end latency of the educational RAG pipeline in seconds",
    ["endpoint", "status"],
    buckets=(
        0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0
    ),
)

# Optional client-side Summary quantiles for immediate inspection
RAG_PIPELINE_LATENCY_SUMMARY = Summary(
    "rag_pipeline_latency_summary_seconds",
    "Summary quantiles of pipeline latency in seconds",
    ["endpoint"],
)


# ==============================================================================
# 2. Throughput & Request Counters
# ==============================================================================

RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of HTTP requests processed by the RAG application",
    ["endpoint", "method", "status_code"],
)

RAG_REQUEST_FAILURES_TOTAL = Counter(
    "rag_request_failures_total",
    "Total number of failed requests in the RAG pipeline",
    ["endpoint", "error_type"],
)


# ==============================================================================
# 3. Token Consumption & Refusal Metrics
# ==============================================================================

RAG_TOKENS_CONSUMED_TOTAL = Counter(
    "rag_tokens_consumed_total",
    "Total tokens consumed by LLM generation (tracked using tiktoken cl100k_base)",
    ["token_type"],  # 'prompt', 'completion', 'total'
)

RAG_REFUSALS_TOTAL = Counter(
    "rag_refusals_total",
    "Total count of educational refusal triggers",
    ["stage"],  # 'pre_gate' (CrossEncoder score < threshold) or 'post_gate' ('I don't know')
)


# ==============================================================================
# Exporter Helper
# ==============================================================================

def get_metrics_payload() -> Tuple[bytes, str]:
    """Generates the latest Prometheus metrics in the official text exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST
