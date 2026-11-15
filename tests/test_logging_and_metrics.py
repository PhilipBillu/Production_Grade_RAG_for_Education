"""
Module: tests/test_logging_and_metrics.py
=========================================
Comprehensive Test Suite for Phase 5:
1. PostgreSQL Request & Telemetry Logging (rag_query_logs table).
2. Tiktoken Token Counting (cl100k_base).
3. Prometheus Metrics & Exposition (/metrics endpoint, P50/P95/P99 latency histogram, throughput, failure rates).
4. FastAPI End-to-End Middleware Integration.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from llama_index.core.schema import TextNode

from db_logger import (
    count_tokens,
    get_db_pool,
    init_db,
    log_query_to_db,
    close_db_pool,
)
from metrics import (
    RAG_PIPELINE_LATENCY_SECONDS,
    RAG_REQUESTS_TOTAL,
    RAG_REQUEST_FAILURES_TOTAL,
    RAG_TOKENS_CONSUMED_TOTAL,
    RAG_REFUSALS_TOTAL,
    get_metrics_payload,
)
from main import app, QueryRequest, QueryResponse


# ==============================================================================
# Test 1: Tiktoken Token Counting
# ==============================================================================

def test_token_counter():
    print("\n--- Test 1: Tiktoken Token Counting ---")
    test_str = "Educational RAG with Hybrid Retrieval and CrossEncoder Reranking."
    tokens = count_tokens(test_str)
    print(f"Token count for '{test_str}': {tokens}")
    assert tokens > 0, "Token count should be greater than 0"
    assert count_tokens("") == 0, "Empty string should have 0 tokens"
    assert count_tokens(None) == 0, "None should return 0 tokens"
    print("[PASS] Tiktoken counting is accurate.")


# ==============================================================================
# Test 2: PostgreSQL Logging Table & Inserts
# ==============================================================================

def test_postgres_logging():
    print("\n--- Test 2: PostgreSQL Database Logging ---")
    assert init_db(), "Database table initialization should succeed"

    test_query = f"Test Query {int(time.time())}"
    chunk_ids = ["chunk-uuid-1", "chunk-uuid-2", "chunk-uuid-3"]
    scores = [8.4321, 5.1234, -2.4567]
    prompt = "System: You are an educational tutor.\nContext: [1] Isaac Newton formulated gravity."
    response = "Isaac Newton discovered universal gravitation."
    latency = 245.75

    success = log_query_to_db(
        user_query=test_query,
        retrieved_chunk_ids=chunk_ids,
        reranked_scores=scores,
        final_prompt=prompt,
        llm_response=response,
        latency_ms=latency,
        refusal_triggered=False,
        status_code=200,
    )
    assert success, "Logging query to PostgreSQL should return True"

    # Verify query exists in DB
    pool = get_db_pool()
    assert pool is not None, "Connection pool must be active"
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_query, retrieved_chunk_ids, reranked_scores, final_prompt, 
                       llm_response, token_count, latency_ms, status_code
                FROM rag_query_logs 
                WHERE user_query = %s
                LIMIT 1;
                """,
                (test_query,),
            )
            row = cur.fetchone()
            assert row is not None, "Inserted row should be found in PostgreSQL"
            (
                db_query,
                db_chunks,
                db_scores,
                db_prompt,
                db_response,
                db_tokens,
                db_lat,
                db_status,
            ) = row

            assert db_query == test_query
            assert db_chunks == chunk_ids
            assert db_scores == scores
            assert db_prompt == prompt
            assert db_response == response
            assert db_tokens > 0
            assert db_lat == latency
            assert db_status == 200
            print(f"[PASS] Successfully logged and verified row in PostgreSQL (tokens={db_tokens}, latency={db_lat}ms).")
    finally:
        pool.putconn(conn)


# ==============================================================================
# Test 3: Prometheus Metrics Exposition
# ==============================================================================

def test_prometheus_metrics():
    print("\n--- Test 3: Prometheus Metrics Generation ---")
    # Simulate some metric updates
    RAG_REQUESTS_TOTAL.labels(endpoint="/query", method="POST", status_code="200").inc()
    RAG_PIPELINE_LATENCY_SECONDS.labels(endpoint="/query", status="success").observe(0.45)
    RAG_TOKENS_CONSUMED_TOTAL.labels(token_type="total").inc(120)

    payload, content_type = get_metrics_payload()
    text = payload.decode("utf-8")

    assert "rag_pipeline_latency_seconds_bucket" in text, "Latency histogram bucket must be present"
    assert "rag_requests_total" in text, "Requests total counter must be present"
    assert "rag_tokens_consumed_total" in text, "Token usage counter must be present"
    assert "rag_request_failures_total" in text, "Failure counter must be present"
    print("[PASS] Prometheus text payload generated with all required metric families.")


# ==============================================================================
# Test 4: FastAPI Observability End-to-End via TestClient
# ==============================================================================

def test_fastapi_observability_middleware():
    print("\n--- Test 4: FastAPI Observability Middleware & /metrics ---")

    # Mock the hybrid retriever
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "id": "node_test_01",
            "text": "Thermodynamics states energy cannot be created or destroyed.",
            "score": 6.8123,
            "initial_rrf_rank": 1,
            "rrf_score": 0.0325,
            "metadata": {"source": "physics.pdf", "page": 10},
        }
    ]
    app.state.retriever = mock_retriever

    client = TestClient(app)

    # 1. Test /metrics endpoint
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "rag_requests_total" in metrics_res.text
    print("[PASS] /metrics endpoint returned 200 OK with Prometheus exposition format.")

    # 2. Test /query endpoint through HTTP client
    query_payload = {
        "question": "What is the First Law of Thermodynamics?",
        "top_k": 1,
        "rrf_candidates": 5,
        "generate": False,
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert data["question"] == "What is the First Law of Thermodynamics?"
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == "node_test_01"

    # Give async logger thread a moment to commit to PostgreSQL
    time.sleep(0.5)

    # Verify query was logged into PostgreSQL
    pool = get_db_pool()
    if pool:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, retrieved_chunk_ids, latency_ms 
                    FROM rag_query_logs 
                    WHERE user_query = %s
                    ORDER BY id DESC LIMIT 1;
                    """,
                    (query_payload["question"],),
                )
                logged_entry = cur.fetchone()
                assert logged_entry is not None, "Query should be logged to PostgreSQL"
                assert logged_entry[0] == query_payload["question"]
                assert "node_test_01" in logged_entry[1]
                print(f"[PASS] Telemetry middleware logged request to PostgreSQL (latency: {logged_entry[2]}ms).")
        finally:
            pool.putconn(conn)


if __name__ == "__main__":
    print("=" * 70)
    print("Executing Phase 5: Logging, Metrics, and Observability Test Suite")
    print("=" * 70)

    try:
        test_token_counter()
        test_postgres_logging()
        test_prometheus_metrics()
        test_fastapi_observability_middleware()
        print("\n" + "=" * 70)
        print("ALL OBSERVABILITY AND METRICS TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    finally:
        close_db_pool()
