"""
Module: tests/test_hybrid_retriever_and_api.py
==============================================
Unit and integration test suite for:
1. BM25LexicalRetriever
2. Reciprocal Rank Fusion (RRF)
3. CrossEncoderReranker
4. FastAPI app endpoints (/health, /query, /)
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from retriever import (
    BM25LexicalRetriever,
    CrossEncoderReranker,
    reciprocal_rank_fusion,
    simple_tokenizer,
)
from main import app


# ==============================================================================
# Test 1: BM25 Lexical Retriever
# ==============================================================================

def test_bm25_lexical_retriever():
    print("\n--- Test: BM25 Lexical Retriever ---")

    node1 = TextNode(
        text="The quick brown fox jumps over the lazy dog.",
        id_="node_fox",
        metadata={"source": "animals.txt", "topic": "nature"}
    )
    node2 = TextNode(
        text="Calculus is the study of continuous change and differentiation.",
        id_="node_math",
        metadata={"source": "calculus.pdf", "topic": "math"}
    )
    node3 = TextNode(
        text="Deep learning neural networks rely heavily on gradient descent optimization.",
        id_="node_ml",
        metadata={"source": "ml.html", "topic": "ai"}
    )

    retriever = BM25LexicalRetriever(nodes=[node1, node2, node3], top_k=2)

    # Query for calculus
    results_math = retriever._retrieve(QueryBundle(query_str="differentiation in calculus"))
    assert len(results_math) > 0, "BM25 should return results"
    assert results_math[0].node.node_id == "node_math", f"Expected node_math, got {results_math[0].node.node_id}"
    assert results_math[0].score > 0, "Score should be positive"

    # Query for neural networks
    results_ml = retriever._retrieve(QueryBundle(query_str="neural networks gradient descent"))
    assert results_ml[0].node.node_id == "node_ml"

    print("[PASS] BM25 Lexical Retriever operates correctly.")


# ==============================================================================
# Test 2: Reciprocal Rank Fusion (RRF)
# ==============================================================================

def test_reciprocal_rank_fusion():
    print("\n--- Test: Reciprocal Rank Fusion (RRF) ---")

    node_a = TextNode(text="Document A on Physics", id_="A")
    node_b = TextNode(text="Document B on Chemistry", id_="B")
    node_c = TextNode(text="Document C on Biology", id_="C")
    node_d = TextNode(text="Document D on Math", id_="D")

    # Dense retriever ranking: [A (rank 1), B (rank 2), C (rank 3)]
    dense_list = [
        NodeWithScore(node=node_a, score=0.95),
        NodeWithScore(node=node_b, score=0.85),
        NodeWithScore(node=node_c, score=0.75),
    ]

    # Lexical retriever ranking: [B (rank 1), D (rank 2), A (rank 3)]
    lexical_list = [
        NodeWithScore(node=node_b, score=5.2),
        NodeWithScore(node=node_d, score=3.1),
        NodeWithScore(node=node_a, score=1.8),
    ]

    # Expected RRF scores with k=60:
    # Node B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # Node A: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # Node D: 1/(60+2) = 1/62 = 0.016129
    # Node C: 1/(60+3) = 1/63 = 0.015873

    fused = reciprocal_rank_fusion([dense_list, lexical_list], k=60, top_k=3)

    assert len(fused) == 3, f"Expected 3 fused results, got {len(fused)}"
    assert fused[0].node.node_id == "B", f"Expected B as top rank, got {fused[0].node.node_id}"
    assert fused[1].node.node_id == "A", f"Expected A as second rank, got {fused[1].node.node_id}"
    assert fused[2].node.node_id == "D", f"Expected D as third rank, got {fused[2].node.node_id}"

    # Verify score accuracy
    assert abs(fused[0].score - (1/61 + 1/62)) < 1e-5
    assert abs(fused[1].score - (1/61 + 1/63)) < 1e-5

    print("[PASS] Reciprocal Rank Fusion computes correct scores and ranking.")


# ==============================================================================
# Test 3: Cross-Encoder Reranker
# ==============================================================================

def test_cross_encoder_reranker():
    print("\n--- Test: Cross-Encoder Reranker ---")

    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    node1 = TextNode(text="Paris is the capital of France.", id_="paris")
    node2 = TextNode(text="The mitochondria is the powerhouse of the biological cell.", id_="cell")
    node3 = TextNode(text="Python was created by Guido van Rossum and released in 1991.", id_="python")

    candidates = [
        NodeWithScore(node=node1, score=0.03),
        NodeWithScore(node=node2, score=0.02),
        NodeWithScore(node=node3, score=0.01),
    ]

    # Query about biology
    reranked = reranker.rerank(
        query="What produces energy inside a biological cell?",
        nodes_with_score=candidates,
        top_k=2
    )

    assert len(reranked) == 2, f"Expected top 2, got {len(reranked)}"
    top_node_with_score, top_score, initial_rank = reranked[0]
    assert top_node_with_score.node.node_id == "cell", f"Expected 'cell' to rank #1, got {top_node_with_score.node.node_id}"
    assert initial_rank == 2, f"Expected initial RRF rank 2 for 'cell', got {initial_rank}"

    print("[PASS] Cross-Encoder Reranker re-orders semantic candidates with high precision.")


# ==============================================================================
# Test 4: FastAPI Application Endpoints
# ==============================================================================

def test_fastapi_endpoints():
    print("\n--- Test: FastAPI App Endpoints ---")

    mock_retriever = MagicMock()
    mock_retriever.corpus_nodes = [TextNode(text="dummy node")]
    mock_retriever.weaviate_client.is_ready.return_value = True
    mock_retriever.search.return_value = [
        {
            "text": "Retrieval-Augmented Generation improves domain accuracy.",
            "score": 3.8412,
            "initial_rrf_rank": 1,
            "rrf_score": 0.0325,
            "metadata": {"source": "data/sample.pdf", "page_number": 1, "total_pages": 2},
        },
        {
            "text": "Weaviate is a production vector database with gRPC.",
            "score": 2.9123,
            "initial_rrf_rank": 2,
            "rrf_score": 0.0318,
            "metadata": {"source": "data/sample.html", "file_type": "web_page"},
        },
    ]

    from main import QueryRequest, root, health_check, query_rag
    from pydantic import ValidationError

    def run_coro(coro):
        """Runs a simple async endpoint coroutine synchronously without event loop sockets."""
        try:
            coro.send(None)
        except StopIteration as exc:
            return exc.value

    # Assign mock retriever to app.state
    app.state.retriever = mock_retriever

    # 1. Test root endpoint
    res_root = run_coro(root())
    assert res_root["status"] == "online"
    assert "endpoints" in res_root

    # 2. Test health endpoint
    res_health = run_coro(health_check())
    assert res_health["status"] == "healthy"
    assert res_health["models"]["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # 3. Test validation error on empty question
    try:
        QueryRequest(question="", top_k=2)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # 4. Test query endpoint execution
    req = QueryRequest(question="What is RAG in education?", top_k=2, rrf_candidates=20)
    res_query = run_coro(query_rag(req))

    assert res_query.question == "What is RAG in education?"
    assert res_query.total_retrieved == 2
    assert len(res_query.results) == 2

    # Verify score and metadata preservation
    first = res_query.results[0]
    assert first.score == 3.8412
    assert first.initial_rrf_rank == 1
    assert first.metadata["source"] == "data/sample.pdf"
    assert first.metadata["page_number"] == 1

    print("[PASS] FastAPI endpoints (/health, /query, /) tested and verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Hybrid Retriever & FastAPI Test Suite")
    print("=" * 60)

    test_bm25_lexical_retriever()
    test_reciprocal_rank_fusion()
    test_cross_encoder_reranker()
    test_fastapi_endpoints()

    print("\n" + "=" * 60)
    print("ALL HYBRID RETRIEVER & API TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
