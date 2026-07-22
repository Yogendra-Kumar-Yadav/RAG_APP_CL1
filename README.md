# Local RAG Application

A fully local Retrieval-Augmented Generation (RAG) application built with Streamlit, Ollama, LlamaIndex, and ChromaDB. No cloud APIs. No data leaves your machine.

---

## Features

- Chat interface to ask questions about your PDF documents
- Local LLM inference via Ollama (`qwen3:8b`)
- Local embeddings via Ollama (`qwen3-embedding:latest`)
- Persistent vector store via ChromaDB
- Cross-encoder reranking via `BAAI/bge-reranker-v2-m3`
- Native PDF text extraction via PyMuPDF with automatic Tesseract OCR fallback
- Inline source citations with file name and page number
- PDF viewer — jump directly to the source page inside the app
- Live pipeline status during answer generation

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | qwen3:8b via Ollama |
| Embeddings | qwen3-embedding:latest via Ollama |
| Vector Store | ChromaDB (persistent) |
| Orchestration | LlamaIndex |
| Reranker | BAAI/bge-reranker-v2-m3 (SentenceTransformers) |
| PDF Processing | PyMuPDF + Tesseract OCR |
| PDF Viewer | streamlit-pdf-viewer |

---

## Folder Structure

```
rag_app/
├── .streamlit/
│   └── config.toml        # Streamlit config (file watcher disabled)
├── data/                  # Place your PDF files here (not tracked by git)
├── storage/               # ChromaDB vector store (auto-created, not tracked by git)
├── app.py                 # Streamlit chat application
├── index_docs.py          # PDF indexing pipeline
├── reranker.py            # BGE cross-encoder reranking
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Prerequisites

### 1. Python 3.11
Download from [python.org](https://www.python.org/downloads/release/python-3110/)

### 2. Ollama
Download and install from [ollama.com](https://ollama.com)

Pull the required models:
```powershell
ollama pull qwen3:8b
ollama pull qwen3-embedding:latest
```

### 3. Tesseract OCR (for scanned/image-based PDFs)
Download the Windows installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

Install to the default path:
```
C:\Program Files\Tesseract-OCR\
```

---

## Installation

```powershell
# Clone the repository
git clone https://github.com/yourusername/rag-app.git
cd rag-app

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Add your PDFs
Create the `data/` folder and copy your PDF files into it:
```powershell
mkdir data
# copy your PDFs into data/
```

### Step 2 — Index your documents
Run this once. Re-run whenever you add new PDFs:
```powershell
python index_docs.py
```

This will:
- Extract text from every PDF page (native extraction + OCR fallback)
- Concatenate each PDF's pages into a single document, recording per-page character
  offsets, so chunks may span page breaks (content split across a page boundary stays together)
- Chunk the text into 500-token segments with 150-token overlap, then tag each chunk
  with the page it starts on for citations
- Embed every chunk using `qwen3-embedding:latest`
- Store all embeddings persistently in `storage/` (a re-run replaces the collection)

### Step 3 — Launch the app
```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## RAG Pipeline

```
User question
  → Embed question (qwen3-embedding:latest)
  → ChromaDB similarity search (top 10 chunks)
  → BGE cross-encoder reranking (top 5 chunks)
  → Context assembly with numbered source labels
  → LLM answer generation (qwen3:8b), citing [Source N]
  → [Source N] remapped to [File: ..., Page: ...] using trusted chunk metadata
  → Answer with inline citations + source panel + PDF viewer
```

---

## Configuration

Key settings are at the top of `app.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | Ollama LLM model |
| `EMBED_MODEL` | `qwen3-embedding:latest` | Ollama embedding model |
| `RETRIEVER_TOP_K` | `10` | Chunks fetched from ChromaDB |
| `RERANKER_TOP_K` | `5` | Chunks kept after reranking |
| `temperature` | `0.1` | LLM temperature, set in `_ollama_generate` (lower = more factual) |
| `think` | `True` | Keep qwen3's reasoning phase on, set in `_ollama_generate` |
| `num_predict` | `6144` | Max tokens per answer (covers thinking + answer), set in `_ollama_generate` |
| `num_ctx` | `10240` | Context window; must fit the ~3k-token RAG prompt (RERANKER_TOP_K chunks at CHUNK_SIZE tokens each, plus system prompt) plus generation |

> Note: `qwen3:8b` is a thinking model and `num_predict` caps thinking + answer combined,
> so a low value can be fully consumed by reasoning and stream a blank answer. Expect a
> short silent pause (~15–35 s) while the model thinks before tokens appear.

Chunking settings are in `index_docs.py`:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Tokens per chunk (kept below the reranker's 1024-token window; sized to usually hold a full paragraph/procedure step) |
| `CHUNK_OVERLAP` | `150` | 30% overlap; also bridges page seams so boundary content isn't orphaned |
| `MIN_WORDS_FOR_OCR` | `30` | Word threshold to trigger OCR |
| `OCR_DPI` | `300` | DPI for OCR image rendering |
| `PAGE_SEPARATOR` | `"\n\n"` | Inserted between pages when concatenating |

---

## Notes

- The `data/` and `storage/` folders are excluded from git
- The `myenv/` virtual environment is excluded from git
- ChromaDB telemetry warnings in the console are harmless
- Reranker model (`BAAI/bge-reranker-v2-m3`) is downloaded from HuggingFace on first run and cached locally
- After the first run all processing is fully offline

---

## Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM | 6 GB | 8 GB+ |
| Storage | 20 GB free | 40 GB free |
| CPU | 8 cores | 12+ cores |

`qwen3:8b` runs fully on GPU with 8 GB VRAM. Below that, some layers spill to CPU and responses will be slower.
