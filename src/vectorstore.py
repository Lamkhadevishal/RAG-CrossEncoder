"""FAISS vector store: build, persist, and retrieve."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_path: str,
) -> FAISS:
    """Embed chunks and save FAISS index to disk."""
    logger.info("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)
    logger.info(f"Vector store saved to: {persist_path}")
    return vectorstore


def load_vectorstore(
    persist_path: str,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Load an existing FAISS index from disk."""
    logger.info(f"Loading FAISS vector store from: {persist_path}")
    return FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search(
    vectorstore: FAISS,
    query: str,
    top_k: int,
) -> List[Document]:
    """Retrieve top-k documents by vector similarity."""
    logger.info(f"Retrieving top-{top_k} documents for query...")
    return vectorstore.similarity_search(query, k=top_k)
