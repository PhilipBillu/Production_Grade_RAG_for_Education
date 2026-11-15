# Production-Grade RAG for Education

A robust, self-contained Retrieval-Augmented Generation (RAG) ingestion and indexing pipeline built using **LlamaIndex**, **Weaviate**, **PostgreSQL**, and **Hugging Face** dense embeddings.

---

## Architecture

```
                      +-----------------------------+
                      |       Input Documents       |
                      |   (PDFs via pypdf & Web     |
                      |    via BeautifulSoup4)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   LlamaIndex Node Parser    |
                      |      SentenceSplitter       |
                      |  (size: 800, overlap: 100)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Dense Vector Model      |
                      |   HuggingFaceEmbedding      |
                      |  (BAAI/bge-small-en-v1.5)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Weaviate Vector Store   |
                      | (text2vec-huggingface ready)|
                      |    gRPC v4 Client (50051)   |
                      +-----------------------------+
```

---

## Directory Structure

Everything is self-contained directly within this folder:

```
Production_Grade_RAG_for_Education/
├── docker-compose.yml          # Weaviate (text2vec-huggingface enabled) + PostgreSQL
├── .env.example                # Configuration template
├── .env                        # Local environment variables
├── requirements.txt            # Python package dependencies
├── ingest.py                   # Main ingestion script (parsing, chunking, indexing)
├── data/                       # Document storage directory
│   ├── sample.html             # Clean test HTML document for web page parsing
│   └── sample.pdf              # Generated valid multi-page PDF document
├── scripts/                    # Helper scripts
│   └── generate_sample_pdf.py  # Standalone generator for sample PDF documents
├── tests/                      # Unit test suite
│   └── test_parsers_and_chunker.py # Unit tests verifying PDF/HTML parsing & chunking
└── README.md                   # Full step-by-step setup and usage guide
```

---

## Prerequisites

1. **Docker Desktop**: Docker Desktop must be running to host the local Weaviate and PostgreSQL containers.
2. **Python**: Python 3.10+ installed.

---

## Setup & Execution Guide

### 1. Launch Weaviate & PostgreSQL Containers

Start both database engines using Docker Compose:

```bash
docker compose up -d
```

- **Weaviate REST API**: `http://localhost:8080`
- **Weaviate gRPC Port**: `localhost:50051` (high-performance vector ingestion)
- **Module Verification**: Visit `http://localhost:8080/v1/meta` to confirm `text2vec-huggingface` is enabled.
- **PostgreSQL**: `localhost:5432` (`postgres` / `postgres` / `rag_db`)

---

### 2. Set Up the Local Python Virtual Environment

Create an isolated virtual environment directly inside this project folder:

```powershell
# In PowerShell:
python -m venv .venv
.\.venv\Scripts\activate
```

#### Why standard pip can take a while:
Standard PyTorch includes 2.5 GB of CUDA / GPU libraries by default. If your machine is running on CPU, you can install the lightweight CPU-only build (~180 MB) in seconds before installing the remaining packages:

```powershell
# 1. Install lightweight CPU PyTorch wheel:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining project requirements:
pip install -r requirements.txt
```

---

### 3. Run Ingestion

Ingest the sample PDF and HTML files in `data/`:

```bash
python ingest.py
```

What the script performs:
1. **Parses PDFs** using `pypdf` with page-by-page metadata (`source`, `page_number`, `total_pages`).
2. **Parses Web Pages** using `BeautifulSoup4` with `<script>`, `<style>`, `<nav>`, and `<footer>` tag stripping.
3. **Splits text** using `SentenceSplitter(chunk_size=800, chunk_overlap=100)`.
4. **Computes dense embeddings** using `BAAI/bge-small-en-v1.5` (~133 MB local model).
5. **Indexes into Weaviate** via gRPC under collection `DocumentChunk`.
6. **Verifies retrieval** with a sample similarity query against the index.

---

### 4. Ingestion CLI Options

```bash
# Ingest a specific PDF:
python ingest.py --pdf data/sample.pdf

# Ingest and sanitize a live web page:
python ingest.py --url https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# Adjust chunk size and overlap:
python ingest.py --chunk-size 800 --chunk-overlap 100

# Run a custom verification query:
python ingest.py --query-test "What is Weaviate?"
```

---

### 5. Starting the FastAPI Query Server

Launch the production REST API with Uvicorn:

```bash
python main.py
```
*(Or with Uvicorn directly: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`)*

- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health & Readiness Check**: `GET http://localhost:8000/health`

#### Executing a Query via PowerShell / cURL:

```powershell
$body = @{
    question = "What is RAG architecture in educational systems?"
    top_k = 5
    rrf_candidates = 20
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/query" -ContentType "application/json" -Body $body
```

---

### 6. Phase 3: Generation, Structured JSON & Refusal Mechanism

Phase 3 integrates the **Groq API** (`openai/gpt-oss-20b`) for factually grounded educational answer generation:
- **Prompt Template**: Takes the top 5 Cross-Encoder reranked chunks as context (`prompts/rag_v1.txt`).
- **Structured JSON Output**: Enforced via Pydantic model `RAGResponse`:
  ```json
  {
    "answer": "string",
    "citations": [1, 2],
    "confidence": 0.95
  }
  ```
- **Two-Tier Refusal Mechanism**:
  1. *Pre-generation*: If all retrieved chunks score below `refusal_threshold` (default `0.4`), the system returns an immediate refusal without calling Groq.
  2. *In/Post-generation*: If the LLM indicates "I don't know" or insufficient context, it normalizes to the standardized refusal message with `citations: []` and `confidence: 0.0`.
- **Fallback Retry Mechanism**: Automatically catches JSON parse errors and retries with Groq error feedback before falling back gracefully.

---

### 7. Running the Test Suites

Execute all automated unit and integration tests:

```bash
# 1. Parser and chunker offline test suite:
python tests/test_parsers_and_chunker.py

# 2. Hybrid Retriever, RRF, Cross-Encoder & FastAPI test suite:
python tests/test_hybrid_retriever_and_api.py

# 3. Phase 3 Generator, Structured JSON & Refusal test suite:
python tests/test_generator_and_refusal.py
```

---

### 8. Phase 4: Automated Evaluation & CI/CD Regression Gating

Phase 4 introduces offline golden evaluation and pull request quality gating:
- **Golden Dataset**: [dataset/golden_qa.json](file:///C:/Users/ADMIN/OneDrive/Desktop/Production_Grade_RAG_for_Education/dataset/golden_qa.json) containing known in-domain questions and unanswerable out-of-domain questions (expandable to 200+ QA pairs).
- **Evaluation Engine**: [eval_script.py](file:///C:/Users/ADMIN/OneDrive/Desktop/Production_Grade_RAG_for_Education/eval_script.py) runs queries against FastAPI `/query` and computes:
  - **Faithfulness**: Groq LLM Judge evaluating factual grounding against retrieved context.
  - **Refusal Correctness**: Measures refusal on OOD questions and avoids false refusals on valid questions.
  - **Key Fact Correctness**: Verifies recall of essential concepts.
  - **Latency Profiling**: Tracks P50 and P95 response times.
- **CI/CD Quality Gate**: [.github/workflows/rag_eval.yml](file:///C:/Users/ADMIN/OneDrive/Desktop/Production_Grade_RAG_for_Education/.github/workflows/rag_eval.yml) runs on every Pull Request. If `faithfulness < 0.85` or `refusal_correctness < 0.90`, the script exits with code `1`, failing the build and blocking the merge.

#### Run Evaluation Locally:
```bash
python eval_script.py --dataset dataset/golden_qa.json --endpoint http://localhost:8000/query --faithfulness-threshold 0.85 --refusal-threshold 0.90
```

---

## Portability Guarantee

- **No Hardcoded Absolute Paths**: All paths in `ingest.py`, `scripts/`, and `tests/` use `Path(__file__).resolve().parent`.
- **Zip-Ready**: You can zip this entire `Production_Grade_RAG_for_Education` folder, copy it to another laptop, unzip it, and run `docker compose up -d` followed by `python ingest.py`.

<!-- Dummy PR Verification -->
