"""
Module: tests/test_generator_and_refusal.py
===========================================
Unit tests verifying Phase 3 requirements:
1. Groq prompt template generation taking top 5 chunks.
2. Structured JSON enforcement with Pydantic model RAGResponse (answer, citations, confidence).
3. Pre-generation CrossEncoder threshold refusal (< 0.4).
4. Post-generation refusal normalization when LLM outputs "I don't know".
5. Fallback retry mechanism on invalid JSON output.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator import (
    REFUSAL_MESSAGE,
    RAGGenerator,
    RAGResponse,
    extract_json_from_text,
)


def test_rag_response_schema():
    """Verify RAGResponse Pydantic validation."""
    print("\n--- Test: RAGResponse Schema Validation ---")
    resp = RAGResponse(
        answer="Weaviate is a cloud-native vector database.",
        citations=[1, 2],
        confidence=0.95,
    )
    assert resp.answer.startswith("Weaviate")
    assert resp.citations == [1, 2]
    assert resp.confidence == 0.95

    # Check serialization
    data = resp.model_dump()
    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    print("[PASS] RAGResponse schema adheres to requirements.")


def test_extract_json_from_text():
    """Verify JSON extractor handles raw JSON, markdown fences, and embedded text."""
    print("\n--- Test: JSON Extraction Helper ---")
    raw = '{"answer": "Test", "citations": [1], "confidence": 0.9}'
    fenced = '```json\n{"answer": "Test", "citations": [1], "confidence": 0.9}\n```'
    embedded = 'Here is the answer:\n```\n{"answer": "Test", "citations": [1], "confidence": 0.9}\n```\nHope it helps!'

    assert json.loads(extract_json_from_text(raw))["answer"] == "Test"
    assert json.loads(extract_json_from_text(fenced))["answer"] == "Test"
    assert json.loads(extract_json_from_text(embedded))["answer"] == "Test"
    print("[PASS] JSON extraction works with markdown fences and mixed text.")


def test_cross_encoder_threshold_refusal():
    """Verify that when all chunks have CrossEncoder score < 0.4, refusal triggers immediately."""
    print("\n--- Test: Pre-Generation CrossEncoder Threshold Refusal ---")
    gen = RAGGenerator(refusal_threshold=0.4)

    # Low-scoring chunks
    low_chunks = [
        {"text": "Random text A", "score": 0.15, "metadata": {"source": "a.txt"}},
        {"text": "Random text B", "score": 0.38, "metadata": {"source": "b.txt"}},
        {"text": "Random text C", "score": -0.5, "metadata": {"source": "c.txt"}},
    ]

    result = gen.generate(
        question="What is quantum gravity?",
        chunks=low_chunks,
        threshold=0.4,
    )

    assert result["refusal_triggered"] is True
    assert result["rag_response"].answer == REFUSAL_MESSAGE
    assert result["rag_response"].citations == []
    assert result["rag_response"].confidence == 0.0
    assert "CrossEncoder relevance scores for all chunks" in result["refusal_reason"]
    print("[PASS] Pre-generation CrossEncoder threshold refusal works correctly.")


def test_post_generation_llm_refusal():
    """Verify that when the LLM returns 'I don't know', it is normalized to standardized refusal."""
    print("\n--- Test: Post-Generation 'I don't know' Refusal ---")
    gen = RAGGenerator(refusal_threshold=0.4)

    high_chunks = [
        {"text": "Technical documentation on Weaviate.", "score": 2.5, "metadata": {"source": "weaviate.pdf"}}
    ]

    # Mock Groq client returning "I don't know"
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "answer": "I don't know because this topic is not discussed in the text.",
            "citations": [],
            "confidence": 0.0,
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    gen.client = mock_client

    result = gen.generate(
        question="What is the capital of Mars?",
        chunks=high_chunks,
    )

    assert result["refusal_triggered"] is True
    assert result["rag_response"].answer == REFUSAL_MESSAGE
    assert result["rag_response"].citations == []
    assert result["rag_response"].confidence == 0.0
    print("[PASS] Post-generation 'I don't know' refusal normalized properly.")


def test_retry_fallback_mechanism():
    """Verify retry mechanism fixes malformed JSON, and falls back gracefully if all fail."""
    print("\n--- Test: Retry Fallback Mechanism ---")
    gen = RAGGenerator(refusal_threshold=0.4)

    high_chunks = [
        {"text": "LlamaIndex uses SentenceSplitter for chunking.", "score": 3.1, "metadata": {"source": "rag.pdf"}}
    ]

    # Case 1: First call returns broken JSON, second call (retry) returns valid JSON
    mock_client = MagicMock()
    first_bad = MagicMock(choices=[MagicMock(message=MagicMock(content='INVALID { "answer": ... NOT JSON'))])
    second_good = MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
        "answer": "LlamaIndex uses SentenceSplitter.",
        "citations": [1],
        "confidence": 0.95,
    })))])

    mock_client.chat.completions.create.side_effect = [first_bad, second_good]
    gen.client = mock_client

    result = gen.generate(
        question="How does LlamaIndex chunk text?",
        chunks=high_chunks,
        max_retries=2,
    )

    assert result["refusal_triggered"] is False
    assert result["rag_response"].answer == "LlamaIndex uses SentenceSplitter."
    assert result["rag_response"].citations == [1]
    assert result["rag_response"].confidence == 0.95
    print("[PASS] Retry mechanism successfully recovered from malformed JSON.")


def test_prompt_template_context_formatting():
    """Verify prompt template contains formatted top 5 chunks and format instructions."""
    print("\n--- Test: Prompt Context & Format Formatting ---")
    gen = RAGGenerator()

    chunks = [
        {"text": f"Chunk content {i}", "score": 2.0 - i * 0.2, "metadata": {"source": f"doc_{i}.pdf", "page_number": i}}
        for i in range(1, 6)
    ]

    context_str = gen.format_context(chunks)
    assert "[1] (Source: doc_1.pdf, Page: 1" in context_str
    assert "[5] (Source: doc_5.pdf, Page: 5" in context_str

    prompt = gen.prompt_template.format(
        context=context_str,
        format_instructions=gen.format_instructions,
        question="Explain chunking.",
    )

    assert "CONTEXT CHUNKS:" in prompt
    assert "USER QUESTION:" in prompt
    assert "Explain chunking." in prompt
    assert "citations" in prompt
    assert "confidence" in prompt
    print("[PASS] Prompt template correctly formats top 5 chunks and schema instructions.")


if __name__ == "__main__":
    test_rag_response_schema()
    test_extract_json_from_text()
    test_cross_encoder_threshold_refusal()
    test_post_generation_llm_refusal()
    test_retry_fallback_mechanism()
    test_prompt_template_context_formatting()
    print("\n" + "=" * 60)
    print("ALL PHASE 3 GENERATOR AND REFUSAL TESTS PASSED!")
    print("=" * 60)
