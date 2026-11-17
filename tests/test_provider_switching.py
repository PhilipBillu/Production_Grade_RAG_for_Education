"""
Module: tests/test_provider_switching.py
========================================
Unit and Integration Tests for Dual LLM Provider Switching (Phase 7):
1. Verifies Groq baseline provider behavior when LLM_PROVIDER=groq.
2. Verifies Local Ollama fine-tuned provider behavior when LLM_PROVIDER=ollama.
3. Verifies /health endpoint accurately reflects provider switching.
4. Verifies mock Ollama generation with structured Pydantic output.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator import RAGGenerator, RAGResponse


def test_groq_provider_initialization():
    print("\n--- Test 1: Groq Provider Initialization ---")
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "dummy_key"}):
        gen = RAGGenerator()
        assert gen.provider == "groq"
        assert gen.model == "openai/gpt-oss-20b"
        print("[PASS] Groq baseline provider initialized correctly.")


def test_ollama_provider_initialization():
    print("\n--- Test 2: Ollama Provider Initialization ---")
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "educational-llama3",
    }):
        gen = RAGGenerator()
        assert gen.provider == "ollama"
        assert gen.model == "educational-llama3"
        assert gen.ollama_base_url == "http://localhost:11434"
        print("[PASS] Local Ollama provider initialized correctly.")


def test_ollama_mock_generation():
    print("\n--- Test 3: Ollama Generation with Structured JSON Output ---")
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "educational-llama3",
    }):
        gen = RAGGenerator()

        mock_chunks = [
            {
                "id": "chunk_1",
                "text": "Newton's Second Law is F = ma, describing force as the rate of change of momentum.",
                "score": 6.85,
                "metadata": {"source": "physics.pdf", "page": 1},
            }
        ]

        mock_ollama_response = {
            "message": {
                "role": "assistant",
                "content": '{"answer": "Newton\'s Second Law is formulated as F = ma [1].", "citations": [1], "confidence": 0.98}'
            }
        }

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_ollama_response
            mock_post.return_value = mock_resp

            result = gen.generate(
                question="What is Newton's Second Law?",
                chunks=mock_chunks,
                threshold=0.4,
            )

            assert mock_post.called, "requests.post should be called to Ollama endpoint"
            rag_res = result["rag_response"]
            assert isinstance(rag_res, RAGResponse)
            assert "F = ma" in rag_res.answer
            assert rag_res.citations == [1]
            assert rag_res.confidence == 0.98
            assert result["refusal_triggered"] is False
            print("[PASS] Ollama generation, JSON parsing, and citation extraction verified.")


def test_health_endpoint_provider_toggle():
    print("\n--- Test 4: /health Endpoint Reflects LLM_PROVIDER Toggle ---")
    from main import app, health_check
    import asyncio

    # 1. Test when LLM_PROVIDER = groq
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_test"}):
        app.state.generator = RAGGenerator()
        res_groq = asyncio.run(health_check())
        assert res_groq["llm"]["provider"] == "groq"
        assert res_groq["llm"]["model"] == "openai/gpt-oss-20b"
        print("  --> /health under Groq:", res_groq["llm"])

    # 2. Test when LLM_PROVIDER = ollama
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_MODEL": "educational-llama3",
        "OLLAMA_BASE_URL": "http://localhost:11434"
    }):
        app.state.generator = RAGGenerator()
        res_ollama = asyncio.run(health_check())
        assert res_ollama["llm"]["provider"] == "ollama"
        assert res_ollama["llm"]["model"] == "educational-llama3"
        print("  --> /health under Ollama:", res_ollama["llm"])

    print("[PASS] Dynamic toggling between Groq and Ollama verified on /health.")


if __name__ == "__main__":
    print("=" * 70)
    print("Executing Dual Provider (Groq / Ollama) Test Suite")
    print("=" * 70)

    test_groq_provider_initialization()
    test_ollama_provider_initialization()
    test_ollama_mock_generation()
    test_health_endpoint_provider_toggle()

    print("\n" + "=" * 70)
    print("ALL DUAL PROVIDER SWITCHING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
