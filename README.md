# Production-Grade Retrieval-Augmented Generation (RAG) for Education

An enterprise-grade, fully observable, self-contained Retrieval-Augmented Generation (RAG) platform tailored for educational question answering, factual context grounding, hallucination prevention, and offline model alignment.

---

## 🏛 System Architecture

```
                                    +-----------------------------+
                                    |       User Query / API      |
                                    +--------------+--------------+
                                                   |
                        +--------------------------+--------------------------+
                        |                                                     |
                        v                                                     v
         +-----------------------------+                       +-----------------------------+
         |    Dense Semantic Search    |                       |     Lexical Keyword Search  |
         |    Weaviate Vector Store    |                       |      BM25 Okapi Algorithm   |
         |  (BAAI/bge-small-en-v1.5)   |                       |    (Top 20 BM25 Candidates) |
         +--------------+--------------+                       +--------------+--------------+
                        |                                                     |
                        +--------------------------+--------------------------+
                                                   |
                                                   v
                                    +-----------------------------+
                                    |   Reciprocal Rank Fusion    |
                                    |     (RRF, k=60 Fusion)      |
                                    |     -> Top 20 Candidates    |
                                    +--------------+--------------+
                                                   |
                                                   v
                                    +-----------------------------+
                                    |   Neural Cross-Encoder      |
                                    |         Reranker            |
                                    |  (ms-marco-MiniLM-L-6-v2)   |
                                    |      -> Top 5 Chunks        |
                                    +--------------+--------------+
                                                   |
                                                   v
                                    +-----------------------------+
                                    |  Pre-Refusal Gating Check   |
                                    | (All CrossEncoder < 0.40?)  |
                                    +--------------+--------------+
                                          /                \
                             YES (Refuse)/                  \ NO (Proceed)
                                        v                    v
           +---------------------------------+  +---------------------------------+
           |     Standardized Refusal        |  |    Dual LLM Generation Engine   |
           |   "Context does not contain     |  |    (Structured Pydantic JSON)   |
           |     sufficient information"     |  |  • Groq Cloud (openai/gpt-oss-20b)|
           +---------------------------------+  |  • Local Ollama (educational-8b)|
                                                +----------------+----------------+
                                                                 |
                                                                 v
                                                +---------------------------------+
                                                |   Post-Refusal & Uncertainty    |
                                                |      Normalization Check        |
                                                +----------------+----------------+
                                                                 |
                                                                 v
                                                +---------------------------------+
                                                |     Observability Middleware    |
                                                | • PostgreSQL (rag_query_logs)   |
                                                | • Prometheus (/metrics endpoint)|
                                                | • Grafana Dashboards (Port 3000)|
                                                +---------------------------------+
```

---

## 📁 Repository Structure

```
Production_Grade_RAG_for_Education/
├── .github/workflows/
│   └── rag_eval.yml                 # Automated CI/CD evaluation and regression gating
├── data/                            # Document storage directory
│   ├── sample.html                  # Sample web document for HTML parsing
│   └── sample.pdf                   # Sample educational PDF document
├── dataset/
│   └── golden_qa.json               # Benchmark dataset for automated evaluation
├── monitoring/
│   ├── grafana/                     # Auto-provisioned Grafana dashboards & datasources
│   ├── grafana_dashboard.json       # Exportable Grafana dashboard JSON
│   └── prometheus.yml               # Prometheus scrape configuration (host.docker.internal:8000)
├── prompts/
│   └── rag_v1.txt                   # Versioned prompt template for educational grounding
├── scripts/
│   ├── generate_colab_notebooks.py  # Generates Jupyter notebooks for Colab
│   └── generate_sample_pdf.py       # Utility generator for educational PDF documents
├── tests/
│   ├── test_parsers_and_chunker.py  # Unit tests for PDF & HTML ingestion
│   ├── test_hybrid_retriever_and_api.py # Tests for BM25, RRF, CrossEncoder & FastAPI
│   ├── test_generator_and_refusal.py# Tests for Groq, Pydantic schema & refusal gates
│   ├── test_logging_and_metrics.py  # Tests for Postgres logging & Prometheus metrics
│   └── test_provider_switching.py   # Tests for dynamic Groq <-> Ollama switching
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
├── db_logger.py                     # PostgreSQL connection pool & async telemetry logger
├── docker-compose.yml               # Weaviate vector DB + PostgreSQL database
├── docker-compose.monitoring.yml    # Prometheus (port 9090) + Grafana (port 3000)
├── eval_script.py                   # Golden QA evaluation runner & quality gate
├── export_to_ollama.py              # Merges LoRA weights, converts to GGUF, builds Modelfile
├── generator.py                     # Dual-provider LLM generator (Groq / Ollama)
├── ingest.py                        # Document parser, sentence chunker & Weaviate indexer
├── main.py                          # FastAPI server with observability middleware & endpoints
├── metrics.py                       # Prometheus latency histograms, counters, and exporter
├── Modelfile                        # Ollama Modelfile for standalone GGUF serving
├── Modelfile.adapter                # Ollama Modelfile for direct LoRA adapter layering
├── requirements.txt                 # Python dependencies
├── train_sft.ipynb                  # Google Colab notebook: 4-bit QLoRA Supervised Fine-Tuning
├── train_dpo.ipynb                  # Google Colab notebook: Direct Preference Optimization (DPO)
├── verify_10_queries.py             # 10-query validation & Postgres verification script
├── FINAL_PERFORMANCE_REPORT.md      # Comprehensive benchmark report: Base vs SFT vs DPO
└── README.md                        # Master documentation
```

---

## ⚙️ Environment Variables Reference

Create your local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| **`LLM_PROVIDER`** | `groq` | Active LLM backend: `groq` (Cloud API) or `ollama` (Local Model). |
| **`GROQ_API_KEY`** | *Required for Groq* | API key from [Groq Console](https://console.groq.com/). |
| **`GROQ_MODEL`** | `openai/gpt-oss-20b` | Groq model identifier for generation & evaluation. |
| **`OLLAMA_BASE_URL`**| `http://localhost:11434`| HTTP endpoint of local Ollama server. |
| **`OLLAMA_MODEL`** | `educational-llama3` | Name of the fine-tuned model loaded in Ollama. |
| **`REFUSAL_THRESHOLD`**| `0.4` | Cross-Encoder score cutoff. Queries scoring below are refused. |
| **`WEAVIATE_HOST`** | `localhost` | Hostname of Weaviate instance. |
| **`WEAVIATE_PORT`** | `8080` | REST API port for Weaviate. |
| **`WEAVIATE_GRPC_PORT`**| `50051` | High-performance gRPC port for Weaviate. |
| **`POSTGRES_HOST`** | `localhost` | Hostname of PostgreSQL server. |
| **`POSTGRES_PORT`** | `5432` | Port for PostgreSQL. |
| **`POSTGRES_DB`** | `rag_db` | PostgreSQL database name. |
| **`POSTGRES_USER`** | `postgres` | PostgreSQL username. |
| **`POSTGRES_PASSWORD`**| `postgres` | PostgreSQL password. |
| **`INDEX_NAME`** | `DocumentChunk` | Collection name in Weaviate vector store. |
| **`EMBEDDING_MODEL_NAME`**| `BAAI/bge-small-en-v1.5`| Hugging Face dense embedding model name. |
| **`RERANKER_MODEL_NAME`** | `cross-encoder/ms-marco-MiniLM-L-6-v2`| Cross-Encoder neural reranker model name. |

---

## 🚀 Quickstart: Clean Setup from Scratch

Follow these step-by-step instructions to get the entire platform running on a clean machine:

### 1. Clone the Repository
```bash
git clone https://github.com/PhilipBillu/Production_Grade_RAG_for_Education.git
cd Production_Grade_RAG_for_Education
```

### 2. Set Up Python Virtual Environment
```powershell
# In Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\activate

# Install lightweight CPU PyTorch (fast download):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies:
pip install -r requirements.txt
```

### 3. Start Database Containers
Ensure **Docker Desktop** is running, then launch Weaviate and PostgreSQL:
```bash
docker compose up -d
```
Verify readiness:
- Weaviate: [http://localhost:8080/v1/meta](http://localhost:8080/v1/meta)
- PostgreSQL: `localhost:5432` (User: `postgres`, Pass: `postgres`, DB: `rag_db`)

### 4. Run the Ingestion Pipeline
Parse, chunk, embed, and index educational documents into Weaviate:
```bash
python ingest.py
```

### 5. Launch the Observability Stack (Prometheus + Grafana)
```bash
docker compose -f docker-compose.monitoring.yml up -d
```
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (`admin` / `admin`)
- Pre-provisioned Dashboard: [http://localhost:3000/d/rag-education-observability/educational-rag-observability-and-telemetry](http://localhost:3000/d/rag-education-observability/educational-rag-observability-and-telemetry)

### 6. Start the FastAPI API Server
```powershell
.\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus Metrics Feed: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- System Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🔄 Dual Provider Switching: Groq vs. Local Ollama

You can toggle between cloud generation (Groq) and local fine-tuned generation (Ollama) simply by changing `LLM_PROVIDER` in your `.env` file:

### Mode A: Groq Cloud Baseline (Fast & Serverless)
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

### Mode B: Local Ollama Fine-Tuned Model (100% Private & Offline)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=educational-llama3
```
Test the active provider anytime:
```bash
curl http://localhost:8000/health
```

---

## 🎓 Model Fine-Tuning & Alignment Guide

We provide end-to-end notebooks configured for free **Google Colab T4 GPUs**:

### 1. Supervised Fine-Tuning (`train_sft.ipynb`)
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Memory Optimization**: 4-bit NormalFloat (NF4) quantization via `BitsAndBytesConfig`.
- **Method**: QLoRA with rank $r=16$, $\alpha=32$, targeting all linear projections (`q, k, v, o, gate, up, down`).
- **Dataset**: `sft_dataset.jsonl` formatted with official Llama 3 chat template (`apply_chat_template`).
- **Output**: Saves trained LoRA adapter to `./educational_sft_adapter`.

### 2. Direct Preference Optimization (`train_dpo.ipynb`)
- **Purpose**: Teaches the model to prioritize factual context grounding and penalizes hallucinations.
- **Dataset**: `dpo_dataset.jsonl` with paired `prompt`, `chosen` (grounded with citations or polite refusal), and `rejected` (hallucinated).
- **Trainer**: `trl.DPOTrainer` with $\beta=0.1$ and learning rate $5 \times 10^{-6}$.
- **Output**: Saves aligned LoRA adapter to `./educational_dpo_adapter`.

### 3. Export to GGUF & Serve in Ollama (`export_to_ollama.py`)
Merge the LoRA adapter weights and create an Ollama model:
```bash
# 1. Merge weights & generate Modelfile
python export_to_ollama.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_dir ./educational_sft_adapter \
  --output_dir ./merged_educational_llama3 \
  --gguf_output ./educational_llama3_8b.gguf

# 2. Register model in Ollama
ollama create educational-llama3 -f Modelfile

# 3. Test interactively
ollama run educational-llama3
```

---

## 📊 Automated Evaluation & CI/CD Regression Gate

We enforce strict quality gating on every pull request using `eval_script.py`:

```bash
python eval_script.py \
  --dataset dataset/golden_qa.json \
  --endpoint http://localhost:8000/query \
  --faithfulness-threshold 0.85 \
  --refusal-threshold 0.90
```

### Quality Gate Thresholds:
- **Faithfulness Score $\ge 0.85$**: Evaluated via Groq LLM Judge verifying factual context grounding.
- **Refusal Correctness $\ge 0.90$**: Evaluates proper refusal on out-of-domain queries and avoids false refusals on valid queries.
- **Exit Code**: Exits with code `1` (failing the GitHub Action) if either threshold is breached.

See [.github/workflows/rag_eval.yml](.github/workflows/rag_eval.yml) for the automated PR workflow.

---

## 📈 Observability & Monitoring

Every query processed by the platform is captured and exposed across two channels:

### 1. PostgreSQL Telemetry (`request_logs` / `rag_query_logs`)
Connect using your database client (`localhost:5432`, user: `postgres`, pass: `postgres`, db: `rag_db`):
```sql
SELECT 
    id, 
    timestamp, 
    user_query, 
    token_count, 
    latency_ms, 
    refusal_triggered, 
    status_code
FROM request_logs 
ORDER BY id DESC 
LIMIT 10;
```

### 2. Prometheus & Grafana
- **Prometheus** scrapes `http://localhost:8000/metrics` every 5 seconds.
- **Grafana Dashboard** displays:
  - P50, P95, P99 Latency Percentiles (Timeseries)
  - Pipeline Failure Rate (%)
  - Query Request Throughput (req/s)
  - Token Consumption Rate & Total Tokens (tracked via `tiktoken`)
  - Educational Refusal Rate by Stage

---

## 🧪 Comprehensive Test Suites

Run any of the automated test suites:

```bash
# 1. Ingestion Parsers & Chunker:
python tests/test_parsers_and_chunker.py

# 2. Hybrid Retriever, RRF, CrossEncoder & FastAPI:
python tests/test_hybrid_retriever_and_api.py

# 3. Generator, Pydantic Schema & Refusal Gates:
python tests/test_generator_and_refusal.py

# 4. PostgreSQL Logging & Prometheus Metrics:
python tests/test_logging_and_metrics.py

# 5. Dual LLM Provider Switching (Groq <-> Ollama):
python tests/test_provider_switching.py

# 6. End-to-End 10-Query Verification:
python verify_10_queries.py
```

---

## 📄 Performance Benchmark Report

For an in-depth empirical comparison analyzing latency, correctness, and faithfulness across **Base**, **SFT**, and **DPO** models, see the full report:
👉 [FINAL_PERFORMANCE_REPORT.md](FINAL_PERFORMANCE_REPORT.md)
