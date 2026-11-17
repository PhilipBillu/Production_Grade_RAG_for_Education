"""
Module: main.py
===============
FastAPI Application for Production-Grade RAG for Education.

Exposes:
- POST /query   : Accepts user questions, executes Hybrid Retrieval (BM25 + Weaviate Vector Store),
                  fuses candidates via Reciprocal Rank Fusion (top 20), reranks with CrossEncoder,
                  generates structured responses with Groq API (openai/gpt-oss-20b),
                  logs telemetry to PostgreSQL, and tracks Prometheus metrics.
- GET  /health  : Health check, database readiness, and model status.
- GET  /metrics : Prometheus telemetry endpoint tracking P50, P95, P99 latencies, token consumption, and failure rates.
- GET  /        : API index and documentation overview.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from retriever import HybridRetriever
from generator import RAGGenerator, RAGResponse
from db_logger import init_db, close_db_pool, log_query_async, count_tokens
from metrics import (
    RAG_PIPELINE_LATENCY_SECONDS,
    RAG_PIPELINE_LATENCY_SUMMARY,
    RAG_REQUESTS_TOTAL,
    RAG_REQUEST_FAILURES_TOTAL,
    RAG_TOKENS_CONSUMED_TOTAL,
    RAG_REFUSALS_TOTAL,
    get_metrics_payload,
)

# Load environment configuration
PROJECT_ROOT: Path = Path(__file__).resolve().parent
ENV_PATH: Path = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Global retriever instance
hybrid_retriever: Optional[HybridRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: initializes models, databases, and connection pools at startup,

    and closes them cleanly at shutdown.
    """
    global hybrid_retriever
    print("\n" + "=" * 70)
    print("  Initializing Production-Grade RAG Engine (FastAPI + Weaviate + Postgres)")
    print("=" * 70)

    # 1. Initialize PostgreSQL table for query telemetry
    db_initialized = init_db()
    if db_initialized:
        print("[Lifespan] PostgreSQL 'rag_query_logs' table ready.")
    else:
        print("[Lifespan Warning] PostgreSQL table initialization skipped or failed.")

    # 2. Initialize HybridRetriever (Weaviate + BM25 + CrossEncoder)
    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_http_port = int(os.getenv("WEAVIATE_PORT", 8080))
    weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
    index_name = os.getenv("INDEX_NAME", "DocumentChunk")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    reranker_model = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    try:
        hybrid_retriever = HybridRetriever(
            weaviate_host=weaviate_host,
            weaviate_http_port=weaviate_http_port,
            weaviate_grpc_port=weaviate_grpc_port,
            index_name=index_name,
            embedding_model_name=embedding_model,
            reranker_model_name=reranker_model,
            data_dir=PROJECT_ROOT / "data",
        )
        app.state.retriever = hybrid_retriever
        print("[Lifespan] HybridRetriever successfully initialized.")
    except Exception as exc:
        print(f"[Lifespan Error] Failed to initialize HybridRetriever: {exc}")
        app.state.retriever = None

    # 3. Initialize RAGGenerator (Dual Provider: Groq or Ollama)
    try:
        rag_generator = RAGGenerator()
        app.state.generator = rag_generator
        print(f"[Lifespan] RAGGenerator successfully initialized with provider '{rag_generator.provider}' (model: '{rag_generator.model}').")
    except Exception as exc:
        print(f"[Lifespan Error] Failed to initialize RAGGenerator: {exc}")
        app.state.generator = None

    yield

    print("[Lifespan] Shutting down application, closing Weaviate and PostgreSQL connections...")
    if hybrid_retriever:
        hybrid_retriever.close()
    close_db_pool()
    print("[Lifespan] Shutdown complete.")


# Initialize FastAPI App
app = FastAPI(
    title="Production-Grade RAG for Education API",
    version="1.0.0",
    description=(
        "Educational RAG API combining Weaviate dense vector search (BAAI/bge-small-en-v1.5), "
        "BM25 lexical search (rank_bm25), Reciprocal Rank Fusion (RRF), Cross-Encoder neural "
        "reranking (cross-encoder/ms-marco-MiniLM-L-6-v2), Groq generation (openai/gpt-oss-20b), "
        "PostgreSQL query telemetry logging, and Prometheus metrics."
    ),
    lifespan=lifespan,
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Observability & Logging HTTP Middleware
# ==============================================================================

@app.middleware("http")
async def rag_telemetry_middleware(request: Request, call_next):
    """Intercepts every incoming HTTP request to:

    1. Measure exact pipeline latency in milliseconds and seconds.
    2. Record Prometheus metrics (requests total, latency histograms for P50/P95/P99, failure counters).
    3. Asynchronously write detailed query telemetry to PostgreSQL (rag_query_logs).
    """
    start_time = time.perf_counter()
    endpoint = request.url.path
    method = request.method
    status_code = 500
    error_message: Optional[str] = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        status_code = 500
        error_message = str(exc)
        # Record pipeline failure in Prometheus
        RAG_REQUEST_FAILURES_TOTAL.labels(
            endpoint=endpoint,
            error_type=type(exc).__name__,
        ).inc()
        raise exc
    finally:
        duration_seconds = time.perf_counter() - start_time
        latency_ms = round(duration_seconds * 1000.0, 2)

        # 1. Update Prometheus Throughput & Latency Metrics
        RAG_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code),
        ).inc()

        status_label = "success" if status_code < 400 else "error"
        RAG_PIPELINE_LATENCY_SECONDS.labels(
            endpoint=endpoint,
            status=status_label,
        ).observe(duration_seconds)
        RAG_PIPELINE_LATENCY_SUMMARY.labels(endpoint=endpoint).observe(duration_seconds)

        # 2. Process Telemetry for Query Endpoints
        telemetry = getattr(request.state, "rag_telemetry", None)
        if telemetry:
            prompt_tokens = telemetry.get("prompt_tokens", 0)
            completion_tokens = telemetry.get("completion_tokens", 0)
            total_tokens = telemetry.get("token_count", prompt_tokens + completion_tokens)

            # Record token usage in Prometheus
            if total_tokens > 0:
                RAG_TOKENS_CONSUMED_TOTAL.labels(token_type="prompt").inc(prompt_tokens)
                RAG_TOKENS_CONSUMED_TOTAL.labels(token_type="completion").inc(completion_tokens)
                RAG_TOKENS_CONSUMED_TOTAL.labels(token_type="total").inc(total_tokens)

            refusal_triggered = telemetry.get("refusal_triggered", False)
            if refusal_triggered:
                stage = telemetry.get("refusal_stage", "pre_gate")
                RAG_REFUSALS_TOTAL.labels(stage=stage).inc()

            # Async log to PostgreSQL
            log_kwargs = {
                "user_query": telemetry.get("user_query", ""),
                "retrieved_chunk_ids": telemetry.get("retrieved_chunk_ids", []),
                "reranked_scores": telemetry.get("reranked_scores", []),
                "final_prompt": telemetry.get("final_prompt", ""),
                "llm_response": telemetry.get("llm_response", ""),
                "token_count": total_tokens,
                "latency_ms": latency_ms,
                "refusal_triggered": refusal_triggered,
                "status_code": status_code,
                "error_message": error_message or telemetry.get("error_message"),
            }
            log_query_async(log_kwargs)
        elif endpoint == "/query" and error_message:
            # Query endpoint encountered an error before telemetry populated
            log_kwargs = {
                "user_query": getattr(request.state, "user_query", "Unknown"),
                "retrieved_chunk_ids": [],
                "reranked_scores": [],
                "final_prompt": "",
                "llm_response": "",
                "token_count": 0,
                "latency_ms": latency_ms,
                "refusal_triggered": False,
                "status_code": status_code,
                "error_message": error_message,
            }
            log_query_async(log_kwargs)


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class QueryRequest(BaseModel):
    """Incoming query payload."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is Weaviate and how does it store vectors?",
                "top_k": 5,
                "rrf_candidates": 20,
            }
        }
    )

    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="The educational question or query to search for.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Final number of cross-encoder reranked chunks to return.",
    )
    rrf_candidates: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of candidate chunks from RRF to pass into the CrossEncoder.",
    )
    generate: bool = Field(
        default=True,
        description="Whether to execute Groq LLM generation (openai/gpt-oss-20b) or return retrieval only.",
    )
    refusal_threshold: float = Field(
        default=0.4,
        description="Cross-Encoder relevance score threshold. If all chunks score below this, refusal is triggered.",
    )


class RetrievedChunk(BaseModel):
    """Structured representation of an individual retrieved and reranked chunk."""
    id: Optional[str] = Field(default=None, description="Unique chunk/node identifier.")
    text: str = Field(..., description="Chunk text content.")
    score: float = Field(..., description="Cross-encoder relevance score (higher is more relevant).")
    initial_rrf_rank: int = Field(..., description="Rank position (1-based) from Reciprocal Rank Fusion.")
    rrf_score: float = Field(..., description="Combined RRF score from BM25 and Weaviate dense search.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserved from ingestion (source, page, file_type).")


class QueryResponse(BaseModel):
    """API response for /query endpoint."""
    question: str = Field(..., description="Echo of the input question.")
    rag_response: Optional[RAGResponse] = Field(default=None, description="Structured generation output (answer, citations, confidence).")
    refusal_triggered: bool = Field(default=False, description="Whether the educational refusal mechanism was triggered.")
    refusal_reason: Optional[str] = Field(default=None, description="Explanation for refusal if triggered.")
    total_retrieved: int = Field(..., description="Number of reranked results returned.")
    results: List[RetrievedChunk] = Field(..., description="Top reranked educational chunks.")


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service status and API navigation."""
    return {
        "service": "Production-Grade RAG for Education API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "metrics": "GET /metrics",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint verifying database connectivity and model status."""
    retriever: Optional[HybridRetriever] = getattr(app.state, "retriever", None)
    generator: Optional[RAGGenerator] = getattr(app.state, "generator", None)

    weaviate_status = "offline"
    corpus_size = 0

    if retriever and hasattr(retriever, "weaviate_client"):
        try:
            if retriever.weaviate_client.is_ready():
                weaviate_status = "connected"
        except Exception:
            weaviate_status = "disconnected"

    if retriever:
        corpus_size = len(retriever.corpus_nodes)

    # LLM Provider status (Groq vs Ollama)
    llm_provider = generator.provider if generator else os.getenv("LLM_PROVIDER", "groq").lower().strip()
    active_model = generator.model if generator else (os.getenv("OLLAMA_MODEL", "educational-llama3") if llm_provider == "ollama" else os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))

    if llm_provider == "ollama":
        provider_status = "connected" if (generator and generator.is_ollama_ready()) else "configured"
    else:
        provider_status = "connected" if (generator and generator.client) else ("configured" if os.getenv("GROQ_API_KEY") else "unconfigured")

    # PostgreSQL check
    pg_status = "connected" if init_db() else "disconnected"

    return {
        "status": "healthy" if weaviate_status == "connected" and pg_status == "connected" else "degraded",
        "weaviate": weaviate_status,
        "postgresql": pg_status,
        "bm25_corpus_size": corpus_size,
        "llm": {
            "provider": llm_provider,
            "status": provider_status,
            "model": active_model,
            "refusal_threshold": float(os.getenv("REFUSAL_THRESHOLD", 0.4)),
        },
        "groq": {
            "status": provider_status if llm_provider == "groq" else "unconfigured",
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "refusal_threshold": float(os.getenv("REFUSAL_THRESHOLD", 0.4)),
        },
        "models": {
            "embedding": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            "reranker": os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        },
    }


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics():
    """Exposes Prometheus metrics for scraping (P50/P95/P99 latency, failure rates, tokens)."""
    payload, content_type = get_metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
async def query_rag(payload: QueryRequest, request: Request = None):
    """Executes the full Production RAG Pipeline:

    1. Semantic search: Retrieves candidate chunks from Weaviate vector store.
    2. Lexical search: Retrieves candidate chunks using BM25.
    3. Hybrid Fusion: Merges rankings using Reciprocal Rank Fusion (RRF, k=60) -> top 20 candidates.
    4. Neural Reranking: Scores (question, chunk) pairs with CrossEncoder -> returns top 5 chunks.
    5. Generation & Refusal: Invokes Groq (openai/gpt-oss-20b) with structured Pydantic output,
       or refuses if CrossEncoder scores < refusal_threshold.
    6. Observability & Logging: Telemetry recorded via middleware to PostgreSQL and Prometheus.
    """
    if request is not None:
        request.state.user_query = payload.question

    retriever: Optional[HybridRetriever] = getattr(app.state, "retriever", None)

    if not retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HybridRetriever is not initialized. Please verify Docker/Weaviate is running.",
        )

    try:
        results = retriever.search(
            query=payload.question,
            top_k=payload.top_k,
            rrf_candidates=payload.rrf_candidates,
        )

        rag_response: Optional[RAGResponse] = None
        refusal_triggered = False
        refusal_reason: Optional[str] = None
        final_prompt = ""
        llm_response_str = ""
        refusal_stage = "pre_gate"

        if payload.generate:
            generator: Optional[RAGGenerator] = getattr(app.state, "generator", None)
            if generator:
                gen_result = generator.generate(
                    question=payload.question,
                    chunks=results,
                    threshold=payload.refusal_threshold,
                )
                rag_response = gen_result.get("rag_response")
                refusal_triggered = gen_result.get("refusal_triggered", False)
                refusal_reason = gen_result.get("refusal_reason")
                final_prompt = gen_result.get("prompt_used", "")
                llm_response_str = (
                    rag_response.answer if rag_response else gen_result.get("raw_output", "")
                )
                if refusal_triggered:
                    refusal_stage = (
                        "pre_gate" if "CrossEncoder" in (refusal_reason or "") else "post_gate"
                    )

        # Calculate exact token usage with tiktoken
        prompt_tokens = count_tokens(final_prompt)
        completion_tokens = count_tokens(llm_response_str)
        token_count = prompt_tokens + completion_tokens

        retrieved_chunk_ids = [
            str(item.get("id") or f"chunk_{i+1}") for i, item in enumerate(results)
        ]
        reranked_scores = [float(item["score"]) for item in results]

        # Attach telemetry to request.state for middleware logging to PostgreSQL
        if request is not None:
            request.state.rag_telemetry = {
                "user_query": payload.question,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "reranked_scores": reranked_scores,
                "final_prompt": final_prompt,
                "llm_response": llm_response_str,
                "token_count": token_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "refusal_triggered": refusal_triggered,
                "refusal_stage": refusal_stage,
                "refusal_reason": refusal_reason,
            }

        return QueryResponse(
            question=payload.question,
            rag_response=rag_response,
            refusal_triggered=refusal_triggered,
            refusal_reason=refusal_reason,
            total_retrieved=len(results),
            results=[
                RetrievedChunk(
                    id=item.get("id"),
                    text=item["text"],
                    score=item["score"],
                    initial_rrf_rank=item["initial_rrf_rank"],
                    rrf_score=item["rrf_score"],
                    metadata=item["metadata"],
                )
                for item in results
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval & Generation error: {str(exc)}",
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[Main] Starting Uvicorn server on http://{host}:{port}...")
    uvicorn.run("main:app", host=host, port=port, reload=True)
