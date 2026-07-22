"""
index_docs.py
-------------
Indexes all PDF files found in the `data/` directory into a persistent
ChromaDB vector store.

Run this script once (or whenever new PDFs are added) before launching app.py.

Pipeline:
  PDF file → PyMuPDF page extraction (native text or Tesseract OCR)
           → concatenate all pages into ONE Document per PDF, recording each
             page's character-offset range (so chunks may span page breaks)
           → SentenceSplitter chunking (500 tokens, 150 overlap)
           → map every chunk back to the page it STARTS on (page_number metadata)
           → Ollama embeddings (qwen3-embedding:latest)
           → ChromaDB persistent storage under storage/

Why concatenate instead of one Document per page:
  Chunking each page in isolation makes every page break a hard chunk boundary,
  so a sentence/paragraph that starts at the bottom of one page and finishes at
  the top of the next gets split across two chunks with no overlap between them.
  Concatenating first lets SentenceSplitter's overlap window bridge the seam;
  we then recover the page number for citations from the recorded offset ranges.
"""

from __future__ import annotations

import bisect
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import chromadb
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL_NAME = "qwen3-embedding:latest"
CHROMA_COLLECTION = "rag_documents"

# Chunk size kept below the reranker's 1024-token window (bge-reranker-v2-m3,
# max_length=1024, see reranker.py) so it scores the whole chunk, not a prefix.
# Sized to usually hold a full paragraph/procedure step end-to-end, and a
# 30% overlap so content near a chunk boundary (e.g. a section's concluding
# lines) is duplicated into the neighboring chunk rather than orphaned.
# Chunks may still span a page break; each is attributed to the page it starts
# on only (see _page_for_offset).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
MIN_WORDS_FOR_OCR = 30          # pages with fewer words trigger OCR
OCR_DPI = 300                   # render resolution for Tesseract
PAGE_SEPARATOR = "\n\n"         # inserted between pages when concatenating

# ---------------------------------------------------------------------------
# Tesseract path — adjust if installed elsewhere on Windows
# ---------------------------------------------------------------------------
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ===========================================================================
# PDF Processing
# ===========================================================================

def _render_page_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image at the requested DPI."""
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _ocr_page(page: fitz.Page) -> str:
    """
    OCR a single PDF page via Tesseract.

    Returns the extracted text, or an empty string if Tesseract fails.
    """
    try:
        img = _render_page_image(page)
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as exc:
        logger.warning("OCR failed on page: %s", exc)
        return ""


def extract_pages(pdf_path: Path) -> List[Tuple[int, str, str]]:
    """
    Extract text from every page of a PDF file.

    Returns a list of (page_number, text, extraction_method) tuples.
    page_number is 1-based.
    extraction_method is "native" or "ocr".
    """
    results: List[Tuple[int, str, str]] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Cannot open PDF '%s': %s", pdf_path.name, exc)
        return results

    logger.info("  Processing '%s' (%d pages)…", pdf_path.name, len(doc))

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        # --- Attempt native text extraction first ---
        native_text = page.get_text().strip()
        word_count = len(native_text.split())

        if word_count >= MIN_WORDS_FOR_OCR:
            results.append((page_number, native_text, "native"))
            logger.debug("    Page %d: native (%d words)", page_number, word_count)
        else:
            # --- Fall back to Tesseract OCR ---
            logger.debug(
                "    Page %d: native text too short (%d words) → OCR",
                page_number,
                word_count,
            )
            ocr_text = _ocr_page(page)
            if not ocr_text:
                logger.warning(
                    "    Page %d of '%s': OCR returned empty text — skipping.",
                    page_number,
                    pdf_path.name,
                )
                continue
            results.append((page_number, ocr_text, "ocr"))

    doc.close()
    return results


def _concatenate_pages(
    pages: List[Tuple[int, str, str]]
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Join a PDF's page texts into one string, recording where each page starts.

    Returns
    -------
    full_text : str
        All page texts concatenated, separated by PAGE_SEPARATOR.
    page_map : list of (start_offset, page_number, extraction_method)
        Sorted ascending by start_offset. start_offset is the character index
        in full_text at which that page's text begins. Used after chunking to
        map a chunk back to the page it starts on.
    """
    text_parts: List[str] = []
    page_map: List[Tuple[int, int, str]] = []
    cursor = 0

    for i, (page_number, text, method) in enumerate(pages):
        page_map.append((cursor, page_number, method))
        text_parts.append(text)
        cursor += len(text)
        if i < len(pages) - 1:
            text_parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)

    return "".join(text_parts), page_map


def _page_for_offset(
    offset: int | None, page_map: List[Tuple[int, int, str]]
) -> Tuple[int, str]:
    """
    Resolve a character offset to (page_number, extraction_method).

    Returns the last page whose start_offset <= offset — i.e. the page on which
    a chunk beginning at `offset` starts. A chunk that spans a page break is
    therefore attributed to its starting page. Falls back to the first page if
    the offset is missing (start_char_idx should always be populated).
    """
    if offset is None:
        return page_map[0][1], page_map[0][2]

    starts = [entry[0] for entry in page_map]
    idx = bisect.bisect_right(starts, offset) - 1
    if idx < 0:
        idx = 0
    return page_map[idx][1], page_map[idx][2]


def build_nodes(data_dir: Path) -> List:
    """
    Walk data_dir and return a flat list of chunked, page-tagged nodes.

    For each PDF: extract pages → concatenate into one Document → chunk with
    SentenceSplitter → stamp every chunk with the page_number and
    extraction_method of the page it starts on.
    """
    pdf_files = sorted(data_dir.glob("**/*.pdf"))

    if not pdf_files:
        logger.error(
            "No PDF files found in '%s'. Add PDFs and re-run.", data_dir
        )
        sys.exit(1)

    logger.info("Found %d PDF file(s).", len(pdf_files))

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_nodes: List = []

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        if not pages:
            logger.warning(
                "No extractable text in '%s' — skipping.", pdf_path.name
            )
            continue

        full_text, page_map = _concatenate_pages(pages)

        # One Document per PDF; only file-level metadata lives here. The
        # per-chunk page_number / extraction_method are assigned after chunking
        # from the offset map, so a chunk that spans a page break still gets a
        # single, unambiguous citation page.
        doc = Document(
            text=full_text,
            metadata={
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
            },
            excluded_embed_metadata_keys=["file_path"],
            excluded_llm_metadata_keys=["file_path"],
        )

        nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
        for node in nodes:
            page_number, method = _page_for_offset(node.start_char_idx, page_map)
            node.metadata["page_number"] = page_number
            node.metadata["extraction_method"] = method

        logger.info(
            "  '%s': %d page(s) → %d chunk(s).",
            pdf_path.name,
            len(pages),
            len(nodes),
        )
        all_nodes.extend(nodes)

    logger.info("Built %d chunk(s) across %d PDF(s).", len(all_nodes), len(pdf_files))
    return all_nodes


# ===========================================================================
# Embedding + Vector Store
# ===========================================================================

def build_index(nodes: list, storage_dir: Path) -> None:
    """
    Embed all nodes with Ollama and persist them in ChromaDB.

    Uses LlamaIndex's StorageContext so the index can be reloaded later
    without re-embedding.
    """
    storage_dir.mkdir(parents=True, exist_ok=True)

    # --- Embedding model (runs via Ollama locally) ---
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        embed_batch_size=1,
        client_kwargs={"trust_env": False},
    )
    Settings.embed_model = embed_model

    # --- ChromaDB persistent client ---
    chroma_client = chromadb.PersistentClient(path=str(storage_dir))
    # Drop any existing collection so re-indexing REPLACES rather than appends
    # (otherwise a second run would add duplicate chunks with fresh IDs).
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
        logger.info("Cleared existing collection '%s'.", CHROMA_COLLECTION)
    except Exception:
        pass  # collection didn't exist yet — first run
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(
        "Embedding %d nodes → ChromaDB collection '%s'…",
        len(nodes),
        CHROMA_COLLECTION,
    )

    # VectorStoreIndex.from_documents() would re-chunk; we pass pre-built nodes.
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("Index persisted to '%s'.", storage_dir)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("=== RAG Indexing Pipeline ===")
    logger.info("Data directory : %s", DATA_DIR)
    logger.info("Storage directory: %s", STORAGE_DIR)

    # 1. Extract + concatenate + chunk + tag pages (one Document per PDF)
    nodes = build_nodes(DATA_DIR)

    # 2. Embed and store
    build_index(nodes, STORAGE_DIR)

    logger.info("=== Indexing complete. You can now run app.py. ===")


if __name__ == "__main__":
    main()
