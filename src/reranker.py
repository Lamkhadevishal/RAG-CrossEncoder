"""Cross-Encoder reranking of retrieved documents."""

import logging
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    docs: List[Document],
    model_name: str,
    top_n: int,
) -> List[Tuple[Document, float]]:
    """
    Score each (query, doc) pair with a Cross-Encoder and return
    the top-n documents sorted by descending relevance score.
    """
    logger.info(f"Reranking {len(docs)} documents with Cross-Encoder: {model_name}")
    model = CrossEncoder(model_name)

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)  # shape: (len(docs),)

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top = scored[:top_n]
    logger.info(f"Top-{top_n} reranked documents selected")
    return top  # List of (Document, score)
