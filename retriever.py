"""
Module: retriever.py
====================
Hybrid Retrieval & Cross-Encoder Reranking Engine for Production-Grade RAG.

Architecture:
1. Semantic Retrieval: Weaviate Vector Store Index (dense vectors via BAAI/bge-small-en-v1.5).
2. Lexical Retrieval: BM25 (rank_bm25.BM25Okapi) over tokenized educational corpus chunks.
3. Hybrid Fusion: Reciprocal Rank Fusion (RRF) combining dense & lexical candidate rankings.
   Supports both native manual RRF and LlamaIndex QueryFusionRetriever.
4. Cross-Encoder Reranker: sentence_transformers.CrossEncoder ('cross-encoder/ms-marco-MiniLM-L-6-v2').
   Scores top 20 candidates from RRF and returns the refined top 5 chunks.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# LlamaIndex Imports
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.llms.mock import MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate

# Load environment configuration
PROJECT_ROOT: Path = Path(__file__).resolve().parent
ENV_PATH: Path = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


def simple_tokenizer(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens for BM25 indexing."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ==============================================================================
# 1. BM25 Lexical Retriever
# ==============================================================================

class BM25LexicalRetriever(BaseRetriever):
    """Lexical keyword retriever wrapping rank_bm25.BM25Okapi as a LlamaIndex BaseRetriever."""

    def __init__(self, nodes: List[BaseNode], top_k: int = 20) -> None:
        super().__init__()
        self.nodes: List[BaseNode] = list(nodes)
        self.top_k: int = top_k
        self.tokenized_corpus: List[List[str]] = [
            simple_tokenizer(node.get_content()) for node in self.nodes
        ]
        self.bm25: Optional[BM25Okapi] = None
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def update_corpus(self, nodes: List[BaseNode]) -> None:
        """Updates the corpus with new nodes and rebuilds the BM25 index."""
        self.nodes = list(nodes)
        self.tokenized_corpus = [
            simple_tokenizer(node.get_content()) for node in self.nodes
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Executes BM25 scoring against the corpus and returns top_k NodeWithScore results."""
        if not self.bm25 or not self.nodes:
            return []

        query_tokens = simple_tokenizer(query_bundle.query_str)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Sort indices by BM25 score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )[:self.top_k]

        results: List[NodeWithScore] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            results.append(NodeWithScore(node=self.nodes[idx], score=score))

        return results


# ==============================================================================
# 2. Reciprocal Rank Fusion (RRF)
# ==============================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[NodeWithScore]],
    k: int = 60,
    top_k: int = 20
) -> List[NodeWithScore]:
    """Combines multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in M} 1 / (k + rank_m(d))

    Parameters
    ----------
    ranked_lists : List[List[NodeWithScore]]
        Lists of ranked NodeWithScore objects from different retrievers (e.g. dense, lexical).
    k : int, default=60
        Smoothing constant preventing high rank bias (standard Cormack et al. constant).
    top_k : int, default=20
        Number of top fused candidates to return.

    Returns
    -------
    List[NodeWithScore]
        Merged and re-ranked list of top_k NodeWithScore candidates with updated RRF scores.
    """
    rrf_scores: Dict[str, float] = {}
    node_map: Dict[str, BaseNode] = {}

    for candidate_list in ranked_lists:
        for rank_0, node_with_score in enumerate(candidate_list):
            node = node_with_score.node
            # Use node_id as unique identifier, or text hash fallback
            unique_id = node.node_id or str(hash(node.get_content()))
            rank_1 = rank_0 + 1  # 1-based rank

            # Compute RRF contribution
            score_delta = 1.0 / (k + rank_1)
            rrf_scores[unique_id] = rrf_scores.get(unique_id, 0.0) + score_delta
            node_map[unique_id] = node

    # Sort candidates by accumulated RRF score descending
    sorted_unique_ids = sorted(
        rrf_scores.keys(),
        key=lambda uid: rrf_scores[uid],
        reverse=True
    )[:top_k]

    fused_results: List[NodeWithScore] = [
        NodeWithScore(node=node_map[uid], score=rrf_scores[uid])
        for uid in sorted_unique_ids
    ]

    return fused_results


# ==============================================================================
# 3. Cross-Encoder Reranker
# ==============================================================================

class CrossEncoderReranker:
    """Reranks candidate chunks using a trained transformer Cross-Encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        print(f"[Reranker] Loading CrossEncoder model '{model_name}'...")
        try:
            # Fast path: load cached weights directly without network probing
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name)
        print(f"[Reranker] CrossEncoder model successfully loaded.")

    def rerank(
        self,
        query: str,
        nodes_with_score: List[NodeWithScore],
        top_k: int = 5
    ) -> List[Tuple[NodeWithScore, float, int]]:
        """Reranks input nodes against the query using cross-attention.

        Parameters
        ----------
        query : str
            The user question.
        nodes_with_score : List[NodeWithScore]
            Candidate nodes from prior retrieval/fusion stage (e.g. top 20 from RRF).
        top_k : int, default=5
            Final number of highest-scoring chunks to return.

        Returns
        -------
        List[Tuple[NodeWithScore, float, int]]
            List of tuples: (NodeWithScore, cross_encoder_score, original_rrf_rank).
        """
        if not nodes_with_score:
            return []

        # Prepare sentence pairs for cross-encoder inference: (query, passage)
        pairs: List[Tuple[str, str]] = [
            (query, item.node.get_content()) for item in nodes_with_score
        ]

        # Predict cross-attention relevance scores
        raw_scores = self.model.predict(pairs)
        scores: List[float] = [float(s) for s in raw_scores]

        # Combine with original RRF rank (1-indexed)
        candidates_with_scores: List[Tuple[NodeWithScore, float, int]] = []
        for rank_1, (node_with_score, ce_score) in enumerate(zip(nodes_with_score, scores), start=1):
            candidates_with_scores.append((node_with_score, ce_score, rank_1))

        # Sort by cross-encoder score descending
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

        return candidates_with_scores[:top_k]


# ==============================================================================
# 4. HybridRetriever Orchestrator
# ==============================================================================

class HybridRetriever:
    """Orchestrates Hybrid Retrieval combining BM25 lexical search, Weaviate vector search,

    Reciprocal Rank Fusion (RRF), and Cross-Encoder neural reranking.
    """

    def __init__(
        self,
        weaviate_host: str = "localhost",
        weaviate_http_port: int = 8080,
        weaviate_grpc_port: int = 50051,
        index_name: str = "DocumentChunk",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        data_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.weaviate_host = weaviate_host
        self.weaviate_http_port = weaviate_http_port
        self.weaviate_grpc_port = weaviate_grpc_port
        self.index_name = index_name
        self.data_dir = Path(data_dir or (PROJECT_ROOT / "data")).resolve()

        # 1. Initialize HuggingFace dense embedding model
        print(f"[HybridRetriever] Initializing Dense Embeddings with '{embedding_model_name}'...")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = self.embed_model
        Settings.llm = None

        # 2. Connect to local Weaviate instance
        print(f"[HybridRetriever] Connecting to Weaviate at {weaviate_host}:{weaviate_http_port} (gRPC: {weaviate_grpc_port})...")
        self.weaviate_client = weaviate.connect_to_local(
            host=weaviate_host,
            port=weaviate_http_port,
            grpc_port=weaviate_grpc_port,
        )

        # Wrap vector store in LlamaIndex WeaviateVectorStore
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.weaviate_client,
            index_name=self.index_name,
        )

        # Create VectorStoreIndex referencing existing Weaviate collection
        self.vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

        # 3. Initialize BM25 Lexical Retriever
        self.corpus_nodes: List[BaseNode] = self._load_corpus()
        print(f"[HybridRetriever] Initializing BM25 index over {len(self.corpus_nodes)} corpus nodes...")
        self.bm25_retriever = BM25LexicalRetriever(nodes=self.corpus_nodes, top_k=20)

        # 4. Initialize Cross-Encoder Reranker
        self.reranker = CrossEncoderReranker(model_name=reranker_model_name)

        # 5. Initialize optional LlamaIndex QueryFusionRetriever
        self._init_query_fusion_retriever()

        print("[HybridRetriever] Ready for hybrid queries!")

    def _load_corpus(self) -> List[BaseNode]:
        """Loads corpus nodes from Weaviate objects or falls back to local data directory."""
        nodes: List[BaseNode] = []

        # Attempt to read existing objects from Weaviate
        try:
            collection = self.weaviate_client.collections.get(self.index_name)
            response = collection.query.fetch_objects(limit=10000)
            for item in response.objects:
                text = item.properties.get("text") or item.properties.get("content") or ""
                if not text:
                    continue
                # Extract metadata properties
                meta = {k: v for k, v in item.properties.items() if k not in ("text", "content", "_node_content")}
                node = TextNode(text=str(text), id_=str(item.uuid), metadata=meta)
                nodes.append(node)
            if nodes:
                print(f"[HybridRetriever] Loaded {len(nodes)} chunks directly from Weaviate '{self.index_name}'.")
                return nodes
        except Exception as exc:
            print(f"[HybridRetriever] Notice: Could not read collection from Weaviate ({exc}). Checking data directory...")

        # Fallback: Parse and chunk files in data directory if Weaviate is empty
        if self.data_dir.exists():
            from ingest import load_documents_from_directory, chunk_documents
            docs = load_documents_from_directory(self.data_dir)
            if docs:
                nodes = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
                print(f"[HybridRetriever] Loaded and chunked {len(nodes)} nodes from local '{self.data_dir.name}'.")

        return nodes

    def _init_query_fusion_retriever(self) -> None:
        """Sets up LlamaIndex QueryFusionRetriever with reciprocal_rerank mode."""
        try:
            dense_retriever = self.vector_index.as_retriever(similarity_top_k=20)
            self.query_fusion_retriever = QueryFusionRetriever(
                retrievers=[dense_retriever, self.bm25_retriever],
                llm=MockLLM(),
                mode=FUSION_MODES.RECIPROCAL_RERANK,
                similarity_top_k=20,
                num_queries=1,
                use_async=False,
            )
        except Exception as exc:
            print(f"[HybridRetriever] QueryFusionRetriever initialization note: {exc}. Using robust native RRF.")
            self.query_fusion_retriever = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_candidates: int = 20,
        use_llamaindex_fusion: bool = False
    ) -> List[Dict[str, Any]]:
        """Executes full hybrid search pipeline:

        1. Semantic Search (Weaviate dense vectors) -> top 20
        2. Lexical Search (BM25 Okapi) -> top 20
        3. Reciprocal Rank Fusion (RRF) -> top 20 fused
        4. Cross-Encoder Reranker -> top 5 refined

        Parameters
        ----------
        query : str
            The search question.
        top_k : int, default=5
            Number of final reranked chunks to return.
        rrf_candidates : int, default=20
            Number of candidate chunks from RRF to pass to the reranker.
        use_llamaindex_fusion : bool, default=False
            Whether to use LlamaIndex's QueryFusionRetriever or native RRF algorithm.

        Returns
        -------
        List[Dict[str, Any]]
            Top results with text, cross-encoder score, initial RRF rank, and metadata.
        """
        print(f"\n[Retrieval Pipeline] Query: '{query}'")
        # 1. Semantic Search (Weaviate)
        dense_retriever = self.vector_index.as_retriever(similarity_top_k=rrf_candidates)
        dense_nodes = dense_retriever.retrieve(query)
        print(f"  [1. Semantic / Weaviate] Retrieved {len(dense_nodes)} vector candidates.")

        # 2. Lexical Search (BM25)
        self.bm25_retriever.top_k = rrf_candidates
        lexical_nodes = self.bm25_retriever.retrieve(query)
        print(f"  [2. Lexical / BM25]    Retrieved {len(lexical_nodes)} keyword candidates.")

        # 3. Hybrid Fusion via Reciprocal Rank Fusion
        if use_llamaindex_fusion and self.query_fusion_retriever:
            try:
                fused_candidates = self.query_fusion_retriever.retrieve(query)[:rrf_candidates]
            except Exception:
                fused_candidates = reciprocal_rank_fusion(
                    [dense_nodes, lexical_nodes], k=60, top_k=rrf_candidates
                )
        else:
            fused_candidates = reciprocal_rank_fusion(
                [dense_nodes, lexical_nodes], k=60, top_k=rrf_candidates
            )
        print(f"  [3. RRF Hybrid Fusion]  Merged into top {len(fused_candidates)} fused candidates.")

        # 4. Cross-Encoder Reranking (Pass top 20 from RRF, return top 5)
        reranked_results = self.reranker.rerank(
            query=query,
            nodes_with_score=fused_candidates,
            top_k=top_k,
        )
        print(f"  [4. CrossEncoder Rerank] Reordered and selected top {len(reranked_results)} chunks.")
        for r_idx, (_, c_score, orig_rank) in enumerate(reranked_results, start=1):
            print(f"     Rank {r_idx}: CrossEncoder Score = {c_score:.4f} (Original RRF Rank: {orig_rank})")

        # Format output payload
        output: List[Dict[str, Any]] = []
        for node_with_score, ce_score, rrf_rank in reranked_results:
            node = node_with_score.node
            meta = dict(node.metadata) if hasattr(node, "metadata") and node.metadata else {}
            node_id = getattr(node, "node_id", getattr(node, "id_", f"chunk_{rrf_rank}"))
            output.append({
                "id": str(node_id),
                "text": node.get_content(),
                "score": round(float(ce_score), 4),
                "initial_rrf_rank": rrf_rank,
                "rrf_score": round(float(node_with_score.score), 6),
                "metadata": meta,
            })

        return output

    def close(self) -> None:
        """Closes Weaviate connection cleanly."""
        try:
            if hasattr(self, "weaviate_client") and self.weaviate_client:
                self.weaviate_client.close()
        except Exception:
            pass
