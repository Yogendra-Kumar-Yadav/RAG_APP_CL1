"""
app.py
------
Streamlit chat interface for the local RAG application.

Run with:
    streamlit run app.py

Prerequisites:
  1. Ollama running locally with qwen3:8b and qwen3-embedding:latest pulled.
  2. index_docs.py has been run at least once to populate storage/.
  3. All packages from requirements.txt installed.

Pipeline per query:
  User question
  → Ollama embedding (qwen3-embedding:latest)
  → ChromaDB similarity search  (top-10)
  → BGE cross-encoder reranking (top-5)
  → Context assembly
  → Ollama LLM answer (qwen3:8b)
  → Streamlit chat message + collapsible source citations + PDF viewer
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import requests
import chromadb
import streamlit as st
from llama_index.core.schema import NodeWithScore, TextNode
from streamlit_pdf_viewer import pdf_viewer

# Bypass any system proxy (Kaspersky etc.) for all requests to Ollama
_NO_PROXY = {"http": None, "https": None}


def _ollama_embed(text: str) -> list[float]:
    """Call Ollama /api/embeddings directly, bypassing system proxies."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        proxies=_NO_PROXY,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _ollama_generate(prompt: str):
    """Call Ollama /api/generate with streaming, bypassing system proxies."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True,
            "think": True,
            # num_predict caps thinking + answer tokens combined; qwen3 can
            # spend 2k+ tokens thinking, so leave headroom for the answer.
            # num_ctx must fit the ~3k-token RAG prompt (RERANKER_TOP_K chunks
            # at CHUNK_SIZE tokens each, plus system prompt and labels) plus
            # all generation — scale this up if CHUNK_SIZE or RERANKER_TOP_K
            # is raised again, to avoid silently truncating context.
            "options": {"temperature": 0.1, "num_predict": 6144, "num_ctx": 10240},
        },
        proxies=_NO_PROXY,
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            import json as _json
            chunk = _json.loads(line)
            if token := chunk.get("response", ""):
                yield token
            if chunk.get("done"):
                break

from reranker import RankedChunk, load_reranker, rerank

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
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "qwen3-embedding:latest"
LLM_MODEL = "qwen3:8b"       
CHROMA_COLLECTION = "rag_documents"

RETRIEVER_TOP_K = 10    # candidates fetched from ChromaDB
RERANKER_TOP_K = 5    # final chunks kept after reranking

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a thorough question-answering assistant working with document excerpts.

Rules you must follow without exception:
0. If the user's message is a greeting, farewell, expression of thanks, or other casual conversational
   remark (e.g. "Hi", "Hello", "Thanks", "Thank you", "Welcome", "Bye"), respond naturally and
   briefly in a friendly tone. Do not search the sources and do not apply rules 1–9 for these messages.
1. For all other messages, answer using ONLY the information present in the numbered source excerpts below.
2. After every sentence or claim, cite the source(s) it came from using ONLY the format: [Source N]
   where N is the number shown in that excerpt's "--- Source N ---" heading. Do NOT write out the
   file name or page number yourself — always use the bare [Source N] tag and nothing else.
3. If the same fact appears in multiple sources, cite all of them, e.g. [Source 1][Source 3].
4. If the answer is not found in any source, say exactly:
   "I could not find an answer to your question in the provided documents."
5. Do NOT hallucinate, infer, or add anything not explicitly stated in the sources.
6. You may Use bullet points or numbered lists for multi-part answers.
7. Where applicable, explain the context, purpose, and any conditions or exceptions mentioned in the sources.
8. Each source excerpt has a "File:" label. Before using any source, judge whether its content is
   genuinely relevant to the question being asked. If a source contains names, roles, committees,
   or context that belong to a different document and are unrelated to the question, ignore that
   source completely — do not quote it, do not cite it, and do not let it influence your answer.
   Only use sources whose content directly addresses the question.
 9. Every single claim, name, role, committee, number, or fact in your answer must be directly
    traceable to an explicit statement in the provided sources. If you cannot find a direct quote
    or explicit statement in the sources to support a claim, do not make that claim.

Citation format example:
  Leadership must establish an environmental policy. [Source 2]
"""


# ===========================================================================
# Cached resource loaders
# (Streamlit re-runs the script on every interaction; @st.cache_resource
#  ensures heavy objects are instantiated only once per server session.)
# ===========================================================================


@st.cache_resource(show_spinner="Connecting to vector database…")
def get_chroma_collection():
    """
    Open the persisted ChromaDB collection and return the raw collection object.
    We query ChromaDB directly to avoid LlamaIndex injecting where:{} filters.
    """
    if not STORAGE_DIR.exists() or not any(STORAGE_DIR.iterdir()):
        raise RuntimeError(
            f"Vector store not found at '{STORAGE_DIR}'. "
            "Run  python index_docs.py  first."
        )

    logger.info("Loading ChromaDB from '%s'…", STORAGE_DIR)
    chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))

    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
    except Exception:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_COLLECTION}' not found. "
            "Run  python index_docs.py  first."
        )

    logger.info("ChromaDB collection ready (%d items).", collection.count())
    return collection


def retrieve_nodes(query: str, top_k: int = RETRIEVER_TOP_K) -> list:
    """
    Embed the query and query ChromaDB's native client directly.

    This completely bypasses LlamaIndex's ChromaVectorStore wrapper, which
    injects an empty `where: {}` filter that ChromaDB 0.5+ rejects.
    Returns a list of NodeWithScore objects for compatibility with reranker.py.
    """
    query_embedding = _ollama_embed(query)

    collection = get_chroma_collection()

    # Call ChromaDB directly — no where filter passed at all
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    nodes_with_scores = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB returns L2 distance; convert to a similarity score
        similarity = 1.0 / (1.0 + dist)
        node = TextNode(text=doc, metadata=meta or {})
        nodes_with_scores.append(NodeWithScore(node=node, score=similarity))

    logger.info("Retrieved %d candidate chunks.", len(nodes_with_scores))
    return nodes_with_scores


@st.cache_resource(show_spinner="Loading reranker model…")
def get_reranker():
    """Load the BGE cross-encoder reranker (cached for the session)."""
    return load_reranker()


# ===========================================================================
# RAG query pipeline
# ===========================================================================

def run_rag_query(query: str, status_callback=None) -> tuple[str, List[RankedChunk]]:
    """
    End-to-end RAG pipeline for a single user query.

    Parameters
    ----------
    query : str
        The user's question.
    status_callback : callable(icon, message) | None
        Optional function called at each pipeline stage to report live progress.

    Returns
    -------
    answer : str
        LLM-generated answer grounded in the retrieved context.
    chunks : List[RankedChunk]
        Reranked source chunks (for citation display).
    """
    def _update(icon: str, msg: str):
        if status_callback:
            status_callback(icon, msg)

    # --- 1. Embed the query ---
    _update("🔍", "Embedding your question…")

    # --- 2. Retrieve top-K candidates from ChromaDB ---
    _update("🗄️", f"Searching vector database for top {RETRIEVER_TOP_K} candidates…")
    raw_nodes = retrieve_nodes(query, top_k=RETRIEVER_TOP_K)

    if not raw_nodes:
        _update("⚠️", "No relevant chunks found in the index.")
        return (
            "I could not find any relevant information in the indexed documents.",
            [],
        )

    _update("✅", f"Retrieved {len(raw_nodes)} candidate chunks from ChromaDB.")

    # --- 3. Rerank with cross-encoder ---
    _update("⚖️", f"Reranking {len(raw_nodes)} chunks with BGE cross-encoder…")
    reranker_model = get_reranker()
    ranked_chunks = rerank(
        query=query,
        nodes=raw_nodes,
        reranker_model=reranker_model,
        top_k=RERANKER_TOP_K,
    )

    top_sources = ", ".join(
        f"{c.file_name} p.{c.page_number}" for c in ranked_chunks
    )
    _update("✅", f"Kept top {len(ranked_chunks)} chunks after reranking → {top_sources}")

    # --- 4. Assemble context string ---
    # Each block is explicitly labelled with file name and page so the LLM can
    # judge relevance and ground its answer — the model is instructed (system
    # prompt rule 2) to cite only the bare [Source N] tag, never the label
    # itself; remap_source_citations() swaps N for the real file/page after
    # generation using our own trusted metadata.
    context_parts = []
    for i, chunk in enumerate(ranked_chunks, start=1):
        context_parts.append(
            f"--- Source {i} ---\n"
            f"File: {chunk.file_name}  |  Page: {chunk.page_number}\n\n"
            f"{chunk.text.strip()}"
        )
    context = "\n\n".join(context_parts)

    # --- 5. Build prompt ---
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== Document Sources ===\n\n"
        f"{context}\n\n"
        f"=== Question ===\n{query}\n\n"
        f"=== Answer (cite every claim with [Source N]) ==="
    )

    # --- 6. Generate answer via Ollama LLM ---
    _update("🤖", f"Generating answer with LLM Model …")       #can replace with {LLM_MODEL}
    answer_generator = _ollama_generate(prompt)

    _update("✅", "Streaming answer…")
    return answer_generator, ranked_chunks


_SOURCE_TAG_RE = re.compile(r"\[Source\s+(\d+)\]")


def remap_source_citations(text: str, chunks: List[RankedChunk]) -> str:
    """
    Replace [Source N] placeholders emitted by the LLM with the real
    [File: ..., Page: ...] citation, using our own trusted chunk metadata.

    We ask the model to cite by source *number* rather than transcribing
    the file name and page itself, because small local models reliably
    lose track of which label belongs to which excerpt when asked to
    copy it verbatim — the content stays correct but the citation drifts
    to the wrong file/page. Remapping by index here removes that failure
    mode entirely, since the mapping always comes from ranked_chunks.
    """

    def _sub(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks):
            chunk = chunks[idx - 1]
            return f"[File: {chunk.file_name}, Page: {chunk.page_number}]"
        return match.group(0)

    return _SOURCE_TAG_RE.sub(_sub, text)


# ===========================================================================
# Source citation UI helpers
# ===========================================================================

def _deduplicate_chunks(chunks: List[RankedChunk]) -> List[RankedChunk]:
    """Remove duplicate (file_name, page_number) pairs, keeping highest score."""
    seen: dict[tuple[str, int], RankedChunk] = {}
    for chunk in chunks:
        key = (chunk.file_name, chunk.page_number)
        if key not in seen or chunk.score > seen[key].score:
            seen[key] = chunk
    return list(seen.values())


def render_sources(chunks: List[RankedChunk], answer_index: int) -> None:
    """
    Render an expandable 'Sources' section beneath an answer.

    Each source shows:
      • File name and page number
      • Reranker score
      • Extraction method badge
      • Snippet of the retrieved text
      • 'Open Page' button that renders the PDF in-page via streamlit-pdf-viewer
    """
    if not chunks:
        return

    deduped = _deduplicate_chunks(chunks)

    with st.expander(f"Sources ({len(deduped)} unique page(s))"):
        for i, chunk in enumerate(deduped):
            col_info, col_btn = st.columns([5, 1])

            with col_info:
                method_badge = (
                    ":blue[native]"
                    if chunk.extraction_method == "native"
                    else ":orange[OCR]"
                )
                st.markdown(
                    f"**{chunk.file_name}** — page {chunk.page_number} "
                    f"| score: `{chunk.score:.4f}` | {method_badge}"
                )
                # Show a trimmed snippet
                snippet = chunk.text[:400].replace("\n", " ").strip()
                if len(chunk.text) > 400:
                    snippet += "…"
                st.caption(snippet)

            with col_btn:
                btn_key = f"open_pdf_{answer_index}_{i}"
                if st.button("Open Page", key=btn_key):
                    # Store the viewer request in session state so it persists
                    # across reruns triggered by button clicks.
                    st.session_state["pdf_viewer"] = {
                        "file_name": chunk.file_name,
                        "page": chunk.page_number,
                    }

            st.divider()


def _render_pdf_viewer(file_name: str, page: int) -> None:
    """
    Open the PDF inside Streamlit and jump to the requested page.

    Uses streamlit-pdf-viewer which renders the PDF natively in the browser
    without exposing file:// URLs.
    """
    pdf_path = DATA_DIR / file_name

    if not pdf_path.exists():
        st.error(
            f"PDF file not found: '{pdf_path}'. "
            "Ensure the file is still in the data/ directory."
        )
        return

    st.subheader(f"Viewing: {file_name} — page {page}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Render only the source page for speed, and scroll directly to it
    pdf_viewer(
        input=pdf_bytes,
        width=900,
        height=700,
        scroll_to_page=page,
        pages_to_render=[page],
        key=f"pdf_viewer_{file_name}_{page}",
    )

    if st.button("Close viewer", key=f"close_pdf_{file_name}_{page}"):
        st.session_state.pop("pdf_viewer", None)
        st.rerun()


# ===========================================================================
# Streamlit page layout
# ===========================================================================

def configure_page() -> None:
    st.set_page_config(
        page_title="Local RAG based Chat-Bot",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> None:
    """Sidebar with status indicators and controls."""
    with st.sidebar:
        st.title("📄 Local RAG")
        st.caption("Powered by Local LLM · ChromaDB")
        st.divider()

        # --- Vector store status ---
        storage_ok = STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir())
        if storage_ok:
            st.success("Vector store: ready", icon="✅")
        else:
            st.error("Vector store: not found", icon="❌")
            st.info("Run  `python index_docs.py`  to build the index.")

        # --- Data directory status ---
        pdfs = list(DATA_DIR.glob("**/*.pdf"))
        if pdfs:
            st.success(f"PDFs indexed: {len(pdfs)} file(s)", icon="📂")
        else:
            st.warning("No PDFs found in data/", icon="⚠️")

        st.divider()

        # --- Model info ---
        st.markdown("**Models**")
        st.code(f"LLM : GenAI LLM\nEmbed: EMBEDDING MODEL", language="text")  # can use {} for llm and embedding model names
        st.markdown("**Reranker**")
        st.code("BGE Reranker", language="text")

        st.divider()

        # --- Clear chat ---
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state.pop("pdf_viewer", None)
            st.rerun()


def render_chat_history() -> None:
    """Replay all messages stored in session state."""
    for i, msg in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("log_lines"):
                with st.expander("Pipeline steps", expanded=False):
                    st.markdown("\n\n".join(msg["log_lines"]))
            if msg["role"] == "assistant" and msg.get("chunks"):
                render_sources(msg["chunks"], answer_index=i)


def handle_user_input() -> None:
    """Accept new user input and run the full RAG pipeline with live status."""
    user_input = st.chat_input("Ask a question about your documents…")
    if not user_input:
        return

    # --- Show user message immediately (render only, no session state yet) ---
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Run RAG pipeline with live progress ---
    with st.chat_message("assistant"):

        # st.status() gives a collapsible live-updating panel
        with st.status("Working on your question…", expanded=True) as status_box:

            # Log lines accumulate inside the status panel
            step_placeholder = st.empty()
            log_lines: list[str] = []

            def update_status(icon: str, message: str) -> None:
                log_lines.append(f"{icon} {message}")
                step_placeholder.markdown("\n\n".join(log_lines))

            try:
                answer_generator, chunks = run_rag_query(user_input, status_callback=update_status)
                status_box.update(
                    label="✅ Done — streaming answer",
                    state="complete",
                )
            except RuntimeError as exc:
                status_box.update(label="❌ Error", state="error", expanded=True)
                st.error(str(exc))
                logger.error("RAG pipeline error: %s", exc)
                return
            except Exception as exc:
                status_box.update(label="❌ Unexpected error", state="error", expanded=True)
                st.error(
                    f"An unexpected error occurred: {exc}\n\n"
                    "Check that Ollama is running and the required models are pulled."
                )
                logger.exception("Unexpected error in RAG pipeline.")
                return

        stream_container = st.empty()
        with stream_container.container():
            raw_answer = st.write_stream(answer_generator)

        full_answer = remap_source_citations(raw_answer, chunks)
        if full_answer != raw_answer:
            # Live-streamed text still shows the raw [Source N] tags;
            # overwrite with the remapped citations once streaming is done.
            stream_container.empty()
            with stream_container.container():
                st.markdown(full_answer)

        answer_index = len(st.session_state.get("messages", [])) + 1
        render_sources(chunks, answer_index=answer_index)

    # --- Persist both messages together, then rerun once to refresh history ---
    st.session_state.setdefault("messages", []).append(
        {"role": "user", "content": user_input, "chunks": []}
    )
    st.session_state["messages"].append(
        {"role": "assistant", "content": full_answer, "chunks": chunks, "log_lines": log_lines}
    )
    st.rerun()


# ===========================================================================
# PDF viewer rendered from session state (persists across button-click reruns)
# ===========================================================================

def render_global_pdf_viewer() -> None:
    """
    Render the PDF viewer panel if the user has clicked 'Open Page'.

    This is rendered separately from the chat so it survives Streamlit reruns
    triggered by other UI interactions.
    """
    viewer_req = st.session_state.get("pdf_viewer")
    if viewer_req:
        st.divider()
        _render_pdf_viewer(viewer_req["file_name"], viewer_req["page"])


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    configure_page()
    render_sidebar()

    st.title("💬 Local RAG Chat")
    st.caption(
        "Ask questions about your Corporate HSE documents."
        " All processing happens locally — no cloud APIs."
    )
    st.divider()

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    render_chat_history()
    handle_user_input()
    render_global_pdf_viewer()


if __name__ == "__main__":
    main()
