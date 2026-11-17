"""
Module: generator.py
====================
Generation Engine for Production-Grade RAG for Education supporting Dual Providers:
1. Groq Cloud API (Default Baseline: 'openai/gpt-oss-20b').
2. Local Ollama Fine-Tuned Model (Phase 6/7: 'educational-llama3' / GGUF).

Features:
1. Dynamic Provider Switching via LLM_PROVIDER ('groq' or 'ollama').
2. Structured JSON Output: Enforces Pydantic model 'RAGResponse' (answer, citations, confidence)
   using LangChain's PydanticOutputParser and native JSON object mode.
3. Two-Tier Refusal Mechanism:
   a. Pre-generation gate: Refuses if all CrossEncoder scores < threshold (e.g., 0.4).
   b. In-generation & Post-generation gate: Refuses if context lacks info or LLM says 'I don't know'.
4. Robust Retry Fallback: Catches invalid JSON or schema violations and automatically
   retries with error feedback up to 2 times before safe fallback.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

# Optional import for Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from prompts import load_prompt_template

# Load environment configuration
PROJECT_ROOT: Path = Path(__file__).resolve().parent
ENV_PATH: Path = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Standardized educational refusal message
REFUSAL_MESSAGE: str = "I am sorry, but the provided context does not contain sufficient information to answer this question."

# Regex pattern for recognizing refusal or uncertainty in LLM answers
REFUSAL_PATTERN = re.compile(
    r"\b(i do not know|i don't know|cannot answer|not enough information|insufficient (information|context)|not provided in the context|not mentioned in the context)\b",
    re.IGNORECASE,
)


# ==============================================================================
# Pydantic Output Model
# ==============================================================================

class RAGResponse(BaseModel):
    """Structured Pydantic response for the RAG generation pipeline."""
    answer: str = Field(
        ...,
        description="Factual educational answer synthesized strictly from the context chunks, or specific refusal message."
    )
    citations: List[int] = Field(
        default_factory=list,
        description="1-based integer indices of context chunks (e.g. [1, 2]) that directly support the answer. Empty on refusal."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 reflecting how well the context supports the answer."
    )


def extract_json_from_text(text: str) -> str:
    """Extracts raw JSON string from potential markdown backticks or commentary."""
    clean = text.strip()
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if clean.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if match:
            clean = match.group(1).strip()

    # If still not starting with {, search for first { and last }
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        clean = clean[start : end + 1]

    return clean


# ==============================================================================
# RAG Generator Class (Dual Provider: Groq & Ollama)
# ==============================================================================

class RAGGenerator:
    """Orchestrates LLM generation using either Groq API or Local Ollama,

    enforcing structured Pydantic JSON outputs and threshold-based refusal gating.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        refusal_threshold: float = 0.4,
        prompt_version: str = "v1",
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
        self.refusal_threshold = float(os.getenv("REFUSAL_THRESHOLD", refusal_threshold))
        self.prompt_version = prompt_version
        self.prompt_template = load_prompt_template(self.prompt_version)

        # LangChain Pydantic Output Parser
        self.parser = PydanticOutputParser(pydantic_object=RAGResponse)
        self.format_instructions = self.parser.get_format_instructions()

        # Provider Configuration: Groq vs Ollama
        if self.provider == "ollama":
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            self.ollama_model = model or os.getenv("OLLAMA_MODEL", "educational-llama3")
            self.model = self.ollama_model
            self.api_key = None
            self.client = None
            print(f"[Generator] Provider configured: 'ollama' (model: '{self.model}', url: '{self.ollama_base_url}')")
        else:
            # Default Groq provider
            self.provider = "groq"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            self.ollama_base_url = None
            self.ollama_model = None

            self.client: Optional[Any] = None
            if GROQ_AVAILABLE and self.api_key:
                try:
                    self.client = Groq(api_key=self.api_key)
                    print(f"[Generator] Groq client initialized with model '{self.model}'.")
                except Exception as exc:
                    print(f"[Generator] Notice: Groq client failed to initialize: {exc}")
                    self.client = None
            else:
                if not self.api_key:
                    print("[Generator] Notice: GROQ_API_KEY is not set. Generator will operate in mock/refusal mode.")

    def is_ollama_ready(self) -> bool:
        """Checks if local Ollama daemon is reachable on OLLAMA_BASE_URL."""
        if not self.ollama_base_url:
            return False
        try:
            r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats the top retrieved/reranked chunks into numbered, clean blocks for prompt context."""
        if not chunks:
            return "No context chunks available."

        formatted_blocks: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "").strip()
            score = chunk.get("score", 0.0)
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page_number", meta.get("page", "N/A"))

            block = (
                f"[{idx}] (Source: {source}, Page: {page}, CrossEncoder Score: {score})\n"
                f"{text}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def check_refusal_threshold(
        self,
        chunks: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Pre-generation gate: Evaluates CrossEncoder scores across all retrieved chunks.

        If all chunk scores are below the threshold, generation is refused immediately.
        """
        active_threshold = threshold if threshold is not None else self.refusal_threshold

        if not chunks:
            return True, "No retrieved context chunks available."

        scores = [float(chunk.get("score", -999.0)) for chunk in chunks]
        max_score = max(scores) if scores else -999.0

        if max_score < active_threshold:
            return True, (
                f"CrossEncoder relevance scores for all chunks (max: {max_score:.4f}) "
                f"fell below the refusal threshold ({active_threshold:.4f})."
            )

        return False, None

    def _call_groq_api(self, prompt: str) -> str:
        """Direct call to Groq Chat Completion with JSON mode."""
        if not self.client:
            raise RuntimeError("Groq client is not initialized. Please set a valid GROQ_API_KEY.")

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an educational assistant that outputs ONLY valid JSON adhering strictly "
                        "to the requested Pydantic schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return chat_completion.choices[0].message.content or ""

    def _call_ollama_api(self, prompt: str) -> str:
        """Direct call to Local Ollama Chat API with JSON schema enforcement."""
        if not self.ollama_base_url:
            raise RuntimeError("Ollama base URL is not configured.")

        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an educational assistant that outputs ONLY valid JSON adhering strictly "
                        "to the requested Pydantic schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "") or ""
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                f"Please ensure Ollama is running ('ollama serve') and model '{self.ollama_model}' exists."
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama API request failed: {exc}")

    def _call_llm(self, prompt: str) -> str:
        """Routes prompt to the active LLM provider (Groq or Ollama)."""
        if self.provider == "ollama":
            return self._call_ollama_api(prompt)
        return self._call_groq_api(prompt)

    def parse_with_retry(
        self,
        raw_text: str,
        question: str,
        context_str: str,
        max_retries: int = 2,
    ) -> Tuple[RAGResponse, bool]:
        """Parses LLM response with PydanticOutputParser and retries on validation errors."""
        cleaned = extract_json_from_text(raw_text)

        # 1. First attempt: parse directly
        try:
            parsed = self.parser.parse(cleaned)
            return parsed, False
        except (OutputParserException, ValidationError, json.JSONDecodeError) as first_err:
            print(f"[Generator] Warning: Initial JSON parse failed ({first_err}). Initiating retry mechanism...")

        current_raw = raw_text
        # 2. Retry loop with error feedback to LLM
        for attempt in range(1, max_retries + 1):
            retry_prompt = (
                f"Your previous JSON response was invalid and failed Pydantic validation.\n"
                f"Original Response:\n{current_raw}\n\n"
                f"Format Instructions:\n{self.format_instructions}\n\n"
                f"Question: {question}\n\n"
                f"Context Chunks:\n{context_str}\n\n"
                f"CRITICAL: Return ONLY a valid JSON object matching the schema with fields 'answer', 'citations', 'confidence'."
            )

            try:
                print(f"[Generator] Retry attempt {attempt}/{max_retries} with error-feedback prompt via {self.provider}...")
                retry_output = self._call_llm(retry_prompt)
                cleaned_retry = extract_json_from_text(retry_output)
                parsed = self.parser.parse(cleaned_retry)
                print(f"[Generator] Retry attempt {attempt} succeeded!")
                return parsed, True
            except Exception as retry_err:
                print(f"[Generator] Retry attempt {attempt} failed: {retry_err}")
                current_raw = retry_output if 'retry_output' in locals() else current_raw

        # 3. Last-resort fallback
        print("[Generator] All retries exhausted. Providing structured fallback response.")
        return (
            RAGResponse(
                answer="Error: Unable to generate a valid structured response after retries.",
                citations=[],
                confidence=0.0,
            ),
            True,
        )

    def generate(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Executes the full generation and refusal pipeline:

        1. Pre-LLM refusal check (CrossEncoder threshold).
        2. Prompt template interpolation (top 5 chunks context + format instructions).
        3. Dynamic LLM invocation (Groq or Ollama, json_object).
        4. Structured JSON parsing with fallback retry mechanism.
        5. Post-generation refusal normalization (check for 'I don't know').
        """
        # Step 1: Pre-generation Refusal Gate (CrossEncoder score threshold)
        refuse_pre, reason_pre = self.check_refusal_threshold(chunks, threshold=threshold)
        if refuse_pre:
            print(f"[Generator] Pre-generation refusal triggered: {reason_pre}")
            return {
                "rag_response": RAGResponse(
                    answer=REFUSAL_MESSAGE,
                    citations=[],
                    confidence=0.0,
                ),
                "refusal_triggered": True,
                "refusal_reason": reason_pre,
                "prompt_used": "",
                "raw_output": "",
            }

        # Format context chunks
        context_str = self.format_context(chunks[:5])

        # Step 2: Assemble Prompt Template
        prompt = self.prompt_template.format(
            context=context_str,
            format_instructions=self.format_instructions,
            question=question,
        )

        # Check Provider Configuration
        if self.provider == "groq" and not self.client:
            reason = "Groq client is not configured (missing GROQ_API_KEY)."
            print(f"[Generator] {reason}")
            return {
                "rag_response": RAGResponse(
                    answer=f"[{reason}] Please configure GROQ_API_KEY in .env or switch to LLM_PROVIDER=ollama.",
                    citations=[],
                    confidence=0.0,
                ),
                "refusal_triggered": False,
                "refusal_reason": reason,
                "prompt_used": prompt,
                "raw_output": "",
            }

        # Step 3: Invoke LLM (Groq or Ollama)
        try:
            raw_output = self._call_llm(prompt)
        except Exception as exc:
            provider_label = self.provider.upper()
            print(f"[Generator Error] {provider_label} API call failed: {exc}")
            return {
                "rag_response": RAGResponse(
                    answer=f"Error connecting to {provider_label} LLM: {str(exc)}",
                    citations=[],
                    confidence=0.0,
                ),
                "refusal_triggered": False,
                "refusal_reason": f"{provider_label} API error: {str(exc)}",
                "prompt_used": prompt,
                "raw_output": "",
            }

        # Step 4: Parse with PydanticOutputParser + Fallback Retry
        rag_response, retried = self.parse_with_retry(
            raw_text=raw_output,
            question=question,
            context_str=context_str,
            max_retries=max_retries,
        )

        # Step 5: Post-generation Refusal Gate
        refusal_triggered = False
        refusal_reason: Optional[str] = None

        if REFUSAL_PATTERN.search(rag_response.answer) or (
            not rag_response.citations and rag_response.confidence < 0.3
        ):
            print("[Generator] Post-generation refusal detected from model output.")
            rag_response = RAGResponse(
                answer=REFUSAL_MESSAGE,
                citations=[],
                confidence=0.0,
            )
            refusal_triggered = True
            refusal_reason = "Model indicated insufficient context or inability to answer."

        return {
            "rag_response": rag_response,
            "refusal_triggered": refusal_triggered,
            "refusal_reason": refusal_reason,
            "prompt_used": prompt,
            "raw_output": raw_output,
        }
