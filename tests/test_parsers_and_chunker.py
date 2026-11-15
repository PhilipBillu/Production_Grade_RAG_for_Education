"""
Module: tests.test_parsers_and_chunker
======================================
Standalone unit test suite for verifying PDF parsing, HTML web parsing,
and LlamaIndex SentenceSplitter text chunking without requiring an active
Weaviate Docker container.
"""

import sys
from pathlib import Path

# Add project root to sys.path so ingest.py functions can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import parsing and chunking functions from ingest.py
from ingest import parse_pdf, parse_web_page, chunk_documents


def test_pdf_parsing() -> None:
    """Verifies that `pypdf` correctly extracts text and page metadata from sample.pdf."""
    print("\n--- Running Test: PDF Parsing (pypdf) ---")
    sample_pdf_path = PROJECT_ROOT / "data" / "sample.pdf"

    assert sample_pdf_path.exists(), f"Sample PDF not found at {sample_pdf_path}"

    documents = parse_pdf(sample_pdf_path)

    # Validate document page count
    assert len(documents) == 2, f"Expected 2 pages, got {len(documents)}"

    # Validate metadata on Page 1
    doc_p1 = documents[0]
    assert doc_p1.metadata["page_number"] == 1, "Page number mismatch on Page 1"
    assert doc_p1.metadata["total_pages"] == 2, "Total page count mismatch"
    assert "Retrieval-Augmented Generation" in doc_p1.text, "Expected content missing from Page 1"

    # Validate metadata on Page 2
    doc_p2 = documents[1]
    assert doc_p2.metadata["page_number"] == 2, "Page number mismatch on Page 2"
    assert "BAAI/bge-small-en-v1.5" in doc_p2.text, "Expected content missing from Page 2"

    print("[PASS] PDF parsing verified successfully.")


def test_html_parsing() -> None:
    """Verifies that `BeautifulSoup4` correctly cleans HTML, removes scripts, and extracts metadata."""
    print("\n--- Running Test: HTML Web Parsing (BeautifulSoup) ---")
    sample_html_path = PROJECT_ROOT / "data" / "sample.html"

    assert sample_html_path.exists(), f"Sample HTML not found at {sample_html_path}"

    documents = parse_web_page(str(sample_html_path), is_url=False)

    assert len(documents) == 1, f"Expected 1 document, got {len(documents)}"

    doc = documents[0]

    # Validate title extraction
    assert "Retrieval-Augmented Generation" in doc.metadata["title"], "Title mismatch in HTML metadata"

    # Validate script tag sanitization
    assert "console.log" not in doc.text, "Script tag was not stripped by BeautifulSoup!"
    assert "Telemetry tracker active" not in doc.text, "Script content leaked into extracted text!"

    # Validate text content
    assert "Weaviate is a cloud-native" in doc.text, "Expected text content missing from HTML extraction"

    print("[PASS] HTML web parsing verified successfully.")


def test_sentence_splitter_chunking() -> None:
    """Verifies that LlamaIndex SentenceSplitter respects chunk_size=800 and chunk_overlap=100."""
    print("\n--- Running Test: Chunking (SentenceSplitter) ---")
    sample_pdf_path = PROJECT_ROOT / "data" / "sample.pdf"
    sample_html_path = PROJECT_ROOT / "data" / "sample.html"

    docs = parse_pdf(sample_pdf_path) + parse_web_page(str(sample_html_path), is_url=False)
    assert len(docs) > 0, "No documents available for chunking"

    nodes = chunk_documents(docs, chunk_size=800, chunk_overlap=100)

    assert len(nodes) >= len(docs), "Expected at least as many nodes as source documents"

    # Verify each node maintains parent metadata
    for idx, node in enumerate(nodes):
        assert "source" in node.metadata, f"Node {idx} is missing 'source' metadata"
        assert len(node.get_content().strip()) > 0, f"Node {idx} is empty"

    print(f"[PASS] SentenceSplitter chunking verified successfully ({len(nodes)} chunks generated).")


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Parser and Chunker Test Suite")
    print("=" * 60)
    test_pdf_parsing()
    test_html_parsing()
    test_sentence_splitter_chunking()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
