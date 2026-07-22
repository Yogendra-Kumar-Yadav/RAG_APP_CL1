"""
reranker.py
-----------
Cross-encoder reranking using BAAI/bge-reranker-v2-m3.

Loads the model once and exposes a single `rerank()` function that takes
a query string and a list of LlamaIndex NodeWithScore objects, scores every
(query, passage) pair, and returns the top-k nodes sorted by reranker score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifier — hosted on HuggingFace, downloaded on first use.
# ---------------------------------------------------------------------------
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


@dataclass
class RankedChunk:
    """A retrieved chunk together with its reranker relevance score."""

    node_id: str
    text: str
    score: float          # cross-encoder score (higher = more relevant)
    file_name: str
    page_number: int
    extraction_method: str  # "native" | "ocr"


# ---------------------------------------------------------------------------
# Model loader (called once; result cached by the caller via @st.cache_resource)
# ---------------------------------------------------------------------------

def load_reranker(model_id: str = RERANKER_MODEL_ID) -> CrossEncoder:
    """
    Download / load the cross-encoder model.

    CrossEncoder runs entirely on the local machine — no cloud API calls.
    The model weights are cached by HuggingFace in ~/.cache/huggingface after
    the first download.
    """
    logger.info("Loading reranker model: %s", model_id)
    model = CrossEncoder(model_id, max_length=1024)
    logger.info("Reranker ready.")
    return model


# ---------------------------------------------------------------------------
# Core reranking function
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    nodes: list,          # list[NodeWithScore] from LlamaIndex
    reranker_model: CrossEncoder,
    top_k: int = 5,
) -> List[RankedChunk]:
    """
    Score every (query, passage) pair with the cross-encoder and return the
    top-k chunks sorted by descending relevance score.

    Parameters
    ----------
    query:
        The user's question.
    nodes:
        LlamaIndex NodeWithScore objects returned by the vector retriever.
    reranker_model:
        A loaded CrossEncoder instance (use `load_reranker()`).
    top_k:
        How many chunks to keep after reranking.

    Returns
    -------
    List[RankedChunk] sorted by score descending, length <= top_k.
    """
    if not nodes:
        logger.warning("rerank() called with empty node list.")
        return []

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, node.node.get_content()) for node in nodes]

    # Score all pairs in a single batched forward pass
    scores: list[float] = reranker_model.predict(pairs).tolist()

    # Zip nodes with their scores and sort
    scored = sorted(
        zip(nodes, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    ranked_chunks: List[RankedChunk] = []
    for node_ws, score in scored[:top_k]:
        meta = node_ws.node.metadata or {}
        ranked_chunks.append(
            RankedChunk(
                node_id=node_ws.node.node_id,
                text=node_ws.node.get_content(),
                score=float(score),
                file_name=meta.get("file_name", "unknown"),
                page_number=int(meta.get("page_number", 0)),
                extraction_method=meta.get("extraction_method", "native"),
            )
        )

    logger.debug(
        "Reranked %d → %d chunks. Top score: %.4f",
        len(nodes),
        len(ranked_chunks),
        ranked_chunks[0].score if ranked_chunks else 0.0,
    )
    return ranked_chunks
