"""
Module: main.py
===============
FastAPI Application for Production-Grade RAG for Education.

Exposes:
- POST /query : Accepts user questions, executes Hybrid Retrieval (BM25 + Weaviate Vector Store),
                fuses candidates via Reciprocal Rank Fusion (top 20), and reranks with
                sentence_transformers.CrossEncoder to return the top 5 chunks.
- GET  /health: Health check, database readiness, and model status.
- GET  /      : API index and documentation overview.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from retriever import HybridRetriever
from generator import RAGGenerator, RAGResponse

# Load environment configuration
PROJECT_ROOT: Path = Path(__file__).resolve().parent
ENV_PATH: Path = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Global retriever instance
hybrid_retriever: Optional[HybridRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: initializes models and connections at startup,

    and closes database sessions cleanly at shutdown.
    """
    global hybrid_retriever
    print("\n" + "=" * 70)
    print("  Initializing Production-Grade RAG Engine (FastAPI + Weaviate)")
    print("=" * 70)

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

    # Initialize RAGGenerator (Groq + LangChain PydanticOutputParser)
    try:
        rag_generator = RAGGenerator()
        app.state.generator = rag_generator
        print("[Lifespan] RAGGenerator successfully initialized.")
    except Exception as exc:
        print(f"[Lifespan Error] Failed to initialize RAGGenerator: {exc}")
        app.state.generator = None

    yield

    print("[Lifespan] Shutting down application and closing Weaviate connection...")
    if hybrid_retriever:
        hybrid_retriever.close()
    print("[Lifespan] Shutdown complete.")


# Initialize FastAPI App
app = FastAPI(
    title="Production-Grade RAG for Education API",
    version="1.0.0",
    description=(
        "Educational RAG API combining Weaviate dense vector search (BAAI/bge-small-en-v1.5), "
        "BM25 lexical search (rank_bm25), Reciprocal Rank Fusion (RRF), and "
        "Cross-Encoder neural reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)."
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

    groq_status = "unconfigured"
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if generator and generator.client:
        groq_status = "connected"
    elif os.getenv("GROQ_API_KEY"):
        groq_status = "configured"

    return {
        "status": "healthy" if weaviate_status == "connected" else "degraded",
        "weaviate": weaviate_status,
        "bm25_corpus_size": corpus_size,
        "groq": {
            "status": groq_status,
            "model": groq_model,
            "refusal_threshold": float(os.getenv("REFUSAL_THRESHOLD", 0.4)),
        },
        "models": {
            "embedding": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            "reranker": os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        },
    }


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
async def query_rag(request: QueryRequest):
    """Executes the full Production RAG Pipeline:

    1. Semantic search: Retrieves candidate chunks from Weaviate vector store.
    2. Lexical search: Retrieves candidate chunks using BM25.
    3. Hybrid Fusion: Merges rankings using Reciprocal Rank Fusion (RRF, k=60) -> top 20 candidates.
    4. Neural Reranking: Scores (question, chunk) pairs with CrossEncoder -> returns top 5 chunks.
    5. Generation & Refusal: Invokes Groq (openai/gpt-oss-20b) with structured Pydantic output,
       or refuses if CrossEncoder scores < refusal_threshold.
    """
    retriever: Optional[HybridRetriever] = getattr(app.state, "retriever", None)

    if not retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HybridRetriever is not initialized. Please verify Docker/Weaviate is running.",
        )

    try:
        results = retriever.search(
            query=request.question,
            top_k=request.top_k,
            rrf_candidates=request.rrf_candidates,
        )

        rag_response: Optional[RAGResponse] = None
        refusal_triggered = False
        refusal_reason: Optional[str] = None

        if request.generate:
            generator: Optional[RAGGenerator] = getattr(app.state, "generator", None)
            if generator:
                gen_result = generator.generate(
                    question=request.question,
                    chunks=results,
                    threshold=request.refusal_threshold,
                )
                rag_response = gen_result.get("rag_response")
                refusal_triggered = gen_result.get("refusal_triggered", False)
                refusal_reason = gen_result.get("refusal_reason")

        return QueryResponse(
            question=request.question,
            rag_response=rag_response,
            refusal_triggered=refusal_triggered,
            refusal_reason=refusal_reason,
            total_retrieved=len(results),
            results=[
                RetrievedChunk(
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
