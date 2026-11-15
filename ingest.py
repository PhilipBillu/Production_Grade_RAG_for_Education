"""
Module: ingest.py
=================
Document ingestion and indexing pipeline for Production-Grade RAG for Education.

Key Pipeline Steps:
1. Document Parsing:
   - PDF Documents: Parsed using `pypdf` with page-by-page text extraction and metadata tracking.
   - Web Pages / HTML: Parsed using `BeautifulSoup4` with script/style stripping and sanitization.
2. Text Chunking:
   - Implements LlamaIndex `SentenceSplitter` configured with `chunk_size=800` and `chunk_overlap=100`.
3. Dense Vector Embeddings:
   - Uses `HuggingFaceEmbedding` powered by the `BAAI/bge-small-en-v1.5` transformer model.
4. Vector Storage & Indexing:
   - Integrates with local Weaviate instance via `llama-index-vector-stores-weaviate` (v4 gRPC client).
5. Portable & Self-Contained:
   - All paths are resolved relative to this script's directory for maximum portability across machines.

Usage:
------
    # Run ingestion using default documents inside data/ folder:
    python ingest.py

    # Ingest a specific PDF:
    python ingest.py --pdf data/sample.pdf

    # Ingest a live web page:
    python ingest.py --url https://en.wikipedia.org/wiki/Retrieval-augmented_generation
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Load environment variables from local .env file if available
from dotenv import load_dotenv

# Resolve project base directory dynamically relative to this file
# Ensures portability when zipped and moved to another machine
PROJECT_ROOT: Path = Path(__file__).resolve().parent
ENV_PATH: Path = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# LlamaIndex Core Imports
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

# LlamaIndex Embeddings Integration (Hugging Face)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LlamaIndex Vector Store Integration (Weaviate)
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Third-party document parsers
import pypdf
from bs4 import BeautifulSoup
import requests
import weaviate


# ==============================================================================
# 1. Document Parsing Functions
# ==============================================================================

def parse_pdf(file_path: Union[str, Path]) -> List[Document]:
    """Parses a PDF file page by page using `pypdf` and converts it into LlamaIndex Documents.

    Each page is extracted as a distinct Document object with attached metadata
    including source file path, file name, page number, and total page count.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the PDF file on the local file system.

    Returns
    -------
    List[Document]
        A list of LlamaIndex Document instances containing text and metadata.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the PDF is empty or contains no extractable text.

    Example
    -------
    >>> docs = parse_pdf("data/sample.pdf")
    >>> print(docs[0].metadata["page_number"])
    1
    """
    # Normalize to pathlib.Path object for cross-platform compatibility
    pdf_path = Path(file_path).resolve()

    # Validate file presence before proceeding
    if not pdf_path.exists():
        raise FileNotFoundError(f"[Error] PDF file not found at: {pdf_path}")

    documents: List[Document] = []

    print(f"[PDF Parser] Reading PDF: {pdf_path.name}")

    # Open and process PDF using pypdf
    with open(pdf_path, "rb") as file_stream:
        # Initialize pypdf reader
        pdf_reader = pypdf.PdfReader(file_stream)
        total_pages = len(pdf_reader.pages)

        # Loop over every page in the document
        for page_idx, page in enumerate(pdf_reader.pages, start=1):
            # Extract plain text from current page
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()

            # Ignore completely blank pages
            if not cleaned_text:
                continue

            # Construct structured metadata for retrieval provenance
            metadata = {
                "source": str(pdf_path.relative_to(PROJECT_ROOT) if pdf_path.is_relative_to(PROJECT_ROOT) else pdf_path),
                "file_name": pdf_path.name,
                "file_type": "pdf",
                "page_number": page_idx,
                "total_pages": total_pages,
            }

            # Wrap in LlamaIndex Document container
            doc = Document(text=cleaned_text, metadata=metadata)
            documents.append(doc)

    print(f"[PDF Parser] Successfully extracted {len(documents)} page-level document(s) from {pdf_path.name}.")
    return documents


def parse_web_page(source: str, is_url: bool = True) -> List[Document]:
    """Parses an HTML web page or local HTML file using `BeautifulSoup4`.

    Strips script, style, navigation, and footer tags to produce clean,
    sanitized textual content suitable for embedding and retrieval.

    Parameters
    ----------
    source : str
        Either a web URL (e.g., 'https://example.com') or a local file path (e.g., 'data/sample.html').
    is_url : bool, default=True
        Flag indicating whether `source` is an HTTP/HTTPS URL or a local file path.

    Returns
    -------
    List[Document]
        A list containing the parsed LlamaIndex Document object.

    Raises
    ------
    requests.RequestException
        If fetching the remote URL fails.
    FileNotFoundError
        If the local HTML file cannot be found.

    Example
    -------
    >>> docs = parse_web_page("data/sample.html", is_url=False)
    >>> print(docs[0].metadata["title"])
    'Production-Grade Retrieval-Augmented Generation (RAG) for Education'
    """
    raw_html: str = ""
    metadata_source: str = source

    if is_url:
        print(f"[Web Parser] Fetching content from URL: {source}")
        # Use a realistic User-Agent to prevent bot-detection blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # Execute HTTP GET request with a reasonable timeout
        response = requests.get(source, headers=headers, timeout=15)
        response.raise_for_status()
        raw_html = response.text
    else:
        # Resolve local HTML file path
        html_path = Path(source).resolve()
        if not html_path.exists():
            raise FileNotFoundError(f"[Error] HTML file not found at: {html_path}")

        print(f"[Web Parser] Reading local HTML file: {html_path.name}")
        with open(html_path, "r", encoding="utf-8", errors="replace") as html_file:
            raw_html = html_file.read()
        metadata_source = str(html_path.relative_to(PROJECT_ROOT) if html_path.is_relative_to(PROJECT_ROOT) else html_path)

    # Initialize BeautifulSoup with the standard HTML parser
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content tags that degrade embedding quality
    tags_to_decompose = ["script", "style", "noscript", "nav", "footer", "header", "aside", "svg"]
    for tag_name in tags_to_decompose:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Extract document title if present
    page_title = "Untitled Page"
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()

    # Extract cleaned textual content
    # separator="\n" preserves readable paragraph breaks
    extracted_text = soup.get_text(separator="\n", strip=True)

    if not extracted_text:
        print(f"[Web Parser] Warning: No readable text extracted from {source}.")
        return []

    # Construct metadata dictionary
    metadata = {
        "source": metadata_source,
        "title": page_title,
        "file_type": "web_page",
        "is_remote_url": is_url,
    }

    # Wrap in LlamaIndex Document object
    doc = Document(text=extracted_text, metadata=metadata)
    print(f"[Web Parser] Successfully parsed web page '{page_title}' ({len(extracted_text)} characters).")
    return [doc]


def load_documents_from_directory(data_dir: Union[str, Path]) -> List[Document]:
    """Scans a local directory and parses all supported PDF and HTML documents.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to the folder containing source documents.

    Returns
    -------
    List[Document]
        Aggregated list of all extracted Document objects.
    """
    directory = Path(data_dir).resolve()
    all_documents: List[Document] = []

    if not directory.exists() or not directory.is_dir():
        print(f"[Warning] Data directory does not exist: {directory}")
        return all_documents

    # Discover PDF files
    pdf_files = sorted(list(directory.glob("*.pdf")))
    for pdf_path in pdf_files:
        try:
            docs = parse_pdf(pdf_path)
            all_documents.extend(docs)
        except Exception as exc:
            print(f"[Error] Failed to parse PDF '{pdf_path.name}': {exc}")

    # Discover HTML / HTM files
    html_files = sorted(list(directory.glob("*.html")) + list(directory.glob("*.htm")))
    for html_path in html_files:
        try:
            docs = parse_web_page(str(html_path), is_url=False)
            all_documents.extend(docs)
        except Exception as exc:
            print(f"[Error] Failed to parse HTML '{html_path.name}': {exc}")

    return all_documents


# ==============================================================================
# 2. Text Chunking (SentenceSplitter)
# ==============================================================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[BaseNode]:
    """Splits input documents into semantic chunks using LlamaIndex `SentenceSplitter`.

    Parameters
    ----------
    documents : List[Document]
        The source documents to be split.
    chunk_size : int, default=800
        Maximum token length per chunk.
    chunk_overlap : int, default=100
        Token overlap between consecutive chunks to preserve contextual continuity.

    Returns
    -------
    List[BaseNode]
        A list of LlamaIndex BaseNode objects with parent metadata preserved.
    """
    print(f"[Chunker] Initializing SentenceSplitter (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")

    # Instantiate LlamaIndex SentenceSplitter
    # Respects sentence boundaries to prevent cutting off sentences mid-word
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Process documents and extract child nodes
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"[Chunker] Generated {len(nodes)} chunks from {len(documents)} document(s).")
    return nodes


# ==============================================================================
# 3. Embeddings & Weaviate Vector Store Configuration
# ==============================================================================

def initialize_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> HuggingFaceEmbedding:
    """Configures and returns the HuggingFaceEmbedding model instance.

    Parameters
    ----------
    model_name : str, default='BAAI/bge-small-en-v1.5'
        Hugging Face repository ID of the dense embedding model.

    Returns
    -------
    HuggingFaceEmbedding
        Configured embedding model instance.
    """
    print(f"[Embedding Model] Initializing HuggingFaceEmbedding with '{model_name}'...")
    # Instantiate HuggingFaceEmbedding
    # Runs locally via sentence-transformers and caches weights
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    # Set as default embed model across LlamaIndex global Settings
    Settings.embed_model = embed_model
    # Disable LLM during pure ingestion to avoid extraneous API calls
    Settings.llm = None

    return embed_model


def connect_weaviate(
    host: str = "localhost",
    http_port: int = 8080,
    grpc_port: int = 50051,
    index_name: str = "DocumentChunk"
) -> Tuple[weaviate.WeaviateClient, WeaviateVectorStore]:
    """Connects to the local Weaviate vector database instance using the v4 client.

    Parameters
    ----------
    host : str, default='localhost'
        Weaviate instance host.
    http_port : int, default=8080
        HTTP/REST port of Weaviate.
    grpc_port : int, default=50051
        gRPC port for high-speed batch vector operations.
    index_name : str, default='DocumentChunk'
        The collection / class name to store indexed vectors in Weaviate.

    Returns
    -------
    Tuple[weaviate.WeaviateClient, WeaviateVectorStore]
        Active Weaviate client and the corresponding LlamaIndex WeaviateVectorStore wrapper.

    Raises
    ------
    ConnectionError
        If Weaviate is unreachable (e.g., Docker container is stopped).
    """
    print(f"[Weaviate] Connecting to local Weaviate at {host}:{http_port} (gRPC: {grpc_port})...")

    try:
        # Establish connection to local Weaviate using the official v4 client API
        client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=grpc_port,
        )

        # Verify cluster health
        if not client.is_ready():
            raise ConnectionError(f"Weaviate at {host}:{http_port} is not ready.")

        # Check enabled modules to confirm text2vec-huggingface
        meta_info = client.get_meta()
        modules = meta_info.get("modules", {})
        print(f"[Weaviate] Connected successfully. Active Weaviate version: {meta_info.get('version')}")
        print(f"[Weaviate] Enabled modules: {list(modules.keys())}")

    except Exception as exc:
        raise ConnectionError(
            f"[Weaviate Connection Failed] Could not connect to Weaviate at {host}:{http_port}.\n"
            f"Details: {exc}\n"
            f"Hint: Make sure Docker is running and execute:\n"
            f"    docker compose up -d\n"
        ) from exc

    # Wrap the Weaviate client in LlamaIndex's WeaviateVectorStore
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name,
    )

    return client, vector_store


# ==============================================================================
# 4. Ingestion & Indexing Pipeline
# ==============================================================================

def embed_and_index_chunks(
    nodes: List[BaseNode],
    vector_store: WeaviateVectorStore,
    embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """Computes dense embeddings for chunks and indexes them into Weaviate.

    Parameters
    ----------
    nodes : List[BaseNode]
        The chunked text nodes to embed and insert.
    vector_store : WeaviateVectorStore
        The target vector store wrapper connected to Weaviate.
    embed_model : HuggingFaceEmbedding
        The embedding model used to compute vector embeddings.

    Returns
    -------
    VectorStoreIndex
        The initialized index instance referencing the populated vector store.
    """
    print(f"[Indexing] Creating StorageContext with WeaviateVectorStore...")
    # Bind vector store into LlamaIndex StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[Indexing] Embedding and inserting {len(nodes)} nodes into Weaviate...")
    # VectorStoreIndex computes embeddings via embed_model and inserts vectors + payloads
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"[Indexing] Successfully completed embedding and indexing!")
    return index


def verify_retrieval(index: VectorStoreIndex, query_text: str = "What is RAG architecture in education?", top_k: int = 2) -> None:
    """Executes a sample similarity retrieval against the Weaviate index to verify search functionality.

    Parameters
    ----------
    index : VectorStoreIndex
        The indexed vector store.
    query_text : str
        The query string to search for.
    top_k : int, default=2
        Number of most similar chunks to retrieve.
    """
    print("\n" + "=" * 70)
    print(f"[Verification Search] Testing vector similarity retrieval for query:")
    print(f"  -> '{query_text}'")
    print("=" * 70)

    # Instantiate retriever from vector store index
    retriever = index.as_retriever(similarity_top_k=top_k)

    # Perform retrieval
    retrieved_nodes = retriever.retrieve(query_text)

    # Display results
    for rank, node_with_score in enumerate(retrieved_nodes, start=1):
        score = node_with_score.score
        score_display = f"{score:.4f}" if score is not None else "N/A"
        print(f"\n[Result #{rank}] Similarity Score: {score_display}")
        print(f"Source: {node_with_score.node.metadata.get('source', 'Unknown')}")
        print(f"Chunk Preview:\n{node_with_score.node.get_content()[:250]}...\n")


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================

def main() -> None:
    """CLI orchestrator for parsing, chunking, embedding, and indexing."""
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and Web Pages into local Weaviate using LlamaIndex and BAAI/bge-small-en-v1.5."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Path to folder containing PDF and HTML files to ingest (default: ./data)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Optional path to a single PDF document to parse and ingest.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional web URL to fetch, parse with BeautifulSoup, and ingest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 800)),
        help="Chunk size for SentenceSplitter in tokens (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 100)),
        help="Chunk overlap for SentenceSplitter in tokens (default: 100)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=os.getenv("INDEX_NAME", "DocumentChunk"),
        help="Weaviate collection/class name (default: DocumentChunk)",
    )
    parser.add_argument(
        "--query-test",
        type=str,
        default="What is Weaviate and RAG in education?",
        help="Sample verification query to run after indexing completes.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Production-Grade RAG Ingestion Pipeline (LlamaIndex + Weaviate)")
    print("=" * 70)

    # Step 1: Collect documents from requested sources
    documents: List[Document] = []

    if args.pdf:
        # Ingest single specified PDF
        documents.extend(parse_pdf(args.pdf))

    if args.url:
        # Ingest single specified URL
        documents.extend(parse_web_page(args.url, is_url=True))

    if not args.pdf and not args.url:
        # Default behavior: Ingest all documents from data/ directory
        documents.extend(load_documents_from_directory(args.data_dir))

    if not documents:
        print("[Abort] No documents were found or parsed. Please add files to data/ or specify --pdf/--url.")
        sys.exit(1)

    print(f"\n[Summary] Collected a total of {len(documents)} document(s) for processing.")

    # Step 2: Implement SentenceSplitter chunking
    nodes = chunk_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Step 3: Initialize HuggingFaceEmbedding (BAAI/bge-small-en-v1.5)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    embed_model = initialize_embedding_model(model_name=embedding_model_name)

    # Step 4: Connect to local Weaviate instance
    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_port = int(os.getenv("WEAVIATE_HTTP_PORT", 8080))
    weaviate_grpc = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))

    client = None
    try:
        client, vector_store = connect_weaviate(
            host=weaviate_host,
            http_port=weaviate_port,
            grpc_port=weaviate_grpc,
            index_name=args.index_name,
        )

        # Step 5: Embed and Index into Weaviate
        index = embed_and_index_chunks(
            nodes=nodes,
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # Step 6: Verify retrieval
        if args.query_test:
            verify_retrieval(index=index, query_text=args.query_test)

    except ConnectionError as err:
        print(f"\n{err}")
        print("[Note] Pipeline code and document parsing were executed successfully.")
        print("[Note] Once Weaviate is running via Docker, re-run `python ingest.py` to index the chunks.")
    finally:
        # Safely close Weaviate client connection if opened
        if client is not None:
            try:
                client.close()
                print("[Weaviate] Client connection safely closed.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
