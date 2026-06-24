"""Sentence Transformer embeddings wrapper."""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Return a LangChain-compatible HuggingFace embedding model."""
    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
