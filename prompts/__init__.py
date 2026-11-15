"""Prompt management package for Production-Grade RAG for Education."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

def load_prompt_template(version: str = "v1") -> str:
    """Loads a versioned prompt template from prompts directory."""
    prompt_file = PROMPTS_DIR / f"rag_{version}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template '{prompt_file.name}' not found in {PROMPTS_DIR}")
    return prompt_file.read_text(encoding="utf-8")
