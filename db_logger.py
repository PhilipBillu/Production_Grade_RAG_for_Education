"""
Module: db_logger.py
====================
PostgreSQL Logging & Token Telemetry for Production-Grade RAG for Education.

Features:
1. Threaded Connection Pooling for PostgreSQL using psycopg2.
2. Auto-initialization of 'rag_query_logs' table with proper indexing and JSONB columns:
   - timestamp: TIMESTAMPTZ DEFAULT NOW()
   - user_query: TEXT
   - retrieved_chunk_ids: JSONB
   - reranked_scores: JSONB
   - final_prompt: TEXT
   - llm_response: TEXT
   - token_count: INT (calculated using tiktoken)
   - latency_ms: DOUBLE PRECISION
   - refusal_triggered: BOOLEAN
   - status_code: INT
   - error_message: TEXT
3. Accurate offline-capable token counting with tiktoken ('cl100k_base').
4. Asynchronous / Background logging to ensure database writes never block HTTP responses.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json

# Configure logging
logger = logging.getLogger("rag.db_logger")
logger.setLevel(logging.INFO)

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Ensure tiktoken uses local offline cache directory
TIKTOKEN_CACHE_DIR = PROJECT_ROOT / "tiktoken_cache"
if TIKTOKEN_CACHE_DIR.exists():
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)

# Initialize tiktoken encoder
try:
    import tiktoken
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            _ENCODER = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            _ENCODER = None
except ImportError:
    _ENCODER = None

# Global PostgreSQL connection pool
_DB_POOL: Optional[pool.ThreadedConnectionPool] = None
_POOL_LOCK = threading.Lock()


def count_tokens(text: Optional[str]) -> int:
    """Calculates the exact token count of text using tiktoken (cl100k_base),

    with a safe fallback heuristic (~4 chars/token) if tiktoken is unavailable.
    """
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Standard rule-of-thumb fallback: ~4 characters per token in English
    return max(1, len(text) // 4)


def get_db_pool() -> Optional[pool.ThreadedConnectionPool]:
    """Returns or initializes the thread-safe connection pool for PostgreSQL."""
    global _DB_POOL
    if _DB_POOL is not None and not _DB_POOL.closed:
        return _DB_POOL

    with _POOL_LOCK:
        if _DB_POOL is not None and not _DB_POOL.closed:
            return _DB_POOL

        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", 5432))
        dbname = os.getenv("POSTGRES_DB", "rag_db")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")

        try:
            _DB_POOL = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                connect_timeout=3,
            )
            logger.info(f"[DB Logger] Connected to PostgreSQL pool at {host}:{port}/{dbname}.")
        except Exception as exc:
            logger.warning(f"[DB Logger] Could not connect to PostgreSQL ({exc}). Logging to DB will be disabled.")
            _DB_POOL = None

    return _DB_POOL


def close_db_pool() -> None:
    """Closes all pooled database connections cleanly."""
    global _DB_POOL
    with _POOL_LOCK:
        if _DB_POOL is not None and not _DB_POOL.closed:
            _DB_POOL.closeall()
            logger.info("[DB Logger] Closed PostgreSQL connection pool.")
            _DB_POOL = None


def init_db() -> bool:
    """Initializes the 'rag_query_logs' table in PostgreSQL if it does not already exist."""
    db_pool = get_db_pool()
    if not db_pool:
        logger.warning("[DB Logger] Database pool unavailable; skipping table initialization.")
        return False

    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rag_query_logs (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_query TEXT NOT NULL,
                    retrieved_chunk_ids JSONB,
                    reranked_scores JSONB,
                    final_prompt TEXT,
                    llm_response TEXT,
                    token_count INTEGER DEFAULT 0,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    refusal_triggered BOOLEAN DEFAULT FALSE,
                    status_code INTEGER DEFAULT 200,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_rag_query_logs_timestamp 
                ON rag_query_logs(timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_rag_query_logs_status 
                ON rag_query_logs(status_code);
            """)
            conn.commit()
            logger.info("[DB Logger] 'rag_query_logs' table verified/initialized in PostgreSQL.")
            return True
    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error(f"[DB Logger] Error initializing database table: {exc}")
        return False
    finally:
        if conn and db_pool:
            db_pool.putconn(conn)


def log_query_to_db(
    user_query: str,
    retrieved_chunk_ids: Optional[List[Any]] = None,
    reranked_scores: Optional[List[float]] = None,
    final_prompt: Optional[str] = None,
    llm_response: Optional[str] = None,
    token_count: Optional[int] = None,
    latency_ms: float = 0.0,
    refusal_triggered: bool = False,
    status_code: int = 200,
    error_message: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> bool:
    """Inserts a single query telemetry log into 'rag_query_logs'.

    If token_count is not provided, it is automatically computed using tiktoken
    from the final_prompt and llm_response.
    """
    db_pool = get_db_pool()
    if not db_pool:
        return False

    if token_count is None:
        prompt_tokens = count_tokens(final_prompt)
        response_tokens = count_tokens(llm_response)
        token_count = prompt_tokens + response_tokens

    ts = timestamp or datetime.now(timezone.utc)
    chunk_ids_json = Json(retrieved_chunk_ids if retrieved_chunk_ids is not None else [])
    scores_json = Json(reranked_scores if reranked_scores is not None else [])

    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_query_logs (
                    timestamp,
                    user_query,
                    retrieved_chunk_ids,
                    reranked_scores,
                    final_prompt,
                    llm_response,
                    token_count,
                    latency_ms,
                    refusal_triggered,
                    status_code,
                    error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    ts,
                    user_query,
                    chunk_ids_json,
                    scores_json,
                    final_prompt or "",
                    llm_response or "",
                    int(token_count),
                    float(latency_ms),
                    bool(refusal_triggered),
                    int(status_code),
                    error_message,
                ),
            )
            conn.commit()
            return True
    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error(f"[DB Logger] Failed to insert query log: {exc}")
        return False
    finally:
        if conn and db_pool:
            db_pool.putconn(conn)


def log_query_async(log_kwargs: Dict[str, Any]) -> None:
    """Fires a background worker thread to log without blocking the HTTP response."""
    threading.Thread(
        target=log_query_to_db,
        kwargs=log_kwargs,
        daemon=True,
    ).start()
