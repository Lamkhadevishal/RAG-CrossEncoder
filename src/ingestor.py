"""PDF ingestion: save, chunk, embed, and manage FAISS index per document."""

import gc
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import load_and_split
from src.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)

# Permanent storage folder for uploaded PDFs
PDF_STORE = Path("uploaded_pdfs")


def _ensure_store():
    PDF_STORE.mkdir(exist_ok=True)


def ingest_single_pdf(
    uploaded_file,                      # Streamlit UploadedFile
    embeddings: HuggingFaceEmbeddings,
    persist_path: str,
    existing_vs: Optional[FAISS],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[FAISS, int]:
    """
    Persist one uploaded PDF, chunk it, and merge into the existing FAISS index.
    Returns (updated_vectorstore, chunks_added).
    """
    _ensure_store()
    dest = PDF_STORE / uploaded_file.name
    dest.write_bytes(uploaded_file.read())

    chunks = load_and_split(str(dest), chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No text extracted from {uploaded_file.name}")

    logger.info(f"{uploaded_file.name} → {len(chunks)} chunks")

    if existing_vs is None:
        vs = build_vectorstore(chunks, embeddings, persist_path)
    else:
        new_vs = FAISS.from_documents(chunks, embeddings)
        existing_vs.merge_from(new_vs)
        existing_vs.save_local(persist_path)
        vs = existing_vs

    return vs, len(chunks)


def remove_pdf(
    filename: str,
    registry: dict,                     # session_state.doc_registry
    embeddings: HuggingFaceEmbeddings,
    persist_path: str,
) -> Optional[FAISS]:
    """
    Delete a PDF from disk and rebuild the FAISS index without it.
    Returns the new vectorstore (or None if no docs remain).
    """
    # Delete stored PDF
    pdf_path = PDF_STORE / filename
    if pdf_path.exists():
        pdf_path.unlink()
        logger.info(f"Deleted PDF: {filename}")

    # Remove from registry
    registry.pop(filename, None)

    return _rebuild(registry, embeddings, persist_path)


def _force_remove(path: Path) -> None:
    """
    Windows-safe recursive directory removal.
    Clears read-only flags and retries on access errors.
    """
    def _on_error(func, fpath, _exc):
        # Remove read-only flag and retry
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    gc.collect()          # Release any lingering file handles held by Python
    shutil.rmtree(str(path), onerror=_on_error)


def remove_all_pdfs(persist_path: str) -> None:
    """Wipe all stored PDFs and the FAISS index from disk."""
    if PDF_STORE.exists():
        _force_remove(PDF_STORE)
        logger.info("Cleared all uploaded PDFs")

    index_dir = Path(persist_path)
    if index_dir.exists():
        _force_remove(index_dir)
        logger.info("Cleared FAISS index")


def _rebuild(
    registry: dict,
    embeddings: HuggingFaceEmbeddings,
    persist_path: str,
) -> Optional[FAISS]:
    """Re-chunk remaining PDFs and rebuild FAISS from scratch."""
    remaining = [PDF_STORE / name for name in registry if (PDF_STORE / name).exists()]
    if not remaining:
        # Wipe any stale index
        index_dir = Path(persist_path)
        if index_dir.exists():
            _force_remove(index_dir)
        return None

    all_chunks = []
    for pdf_path in remaining:
        chunks = load_and_split(str(pdf_path))
        all_chunks.extend(chunks)

    vs = build_vectorstore(all_chunks, embeddings, persist_path)
    logger.info(f"Rebuilt FAISS with {len(all_chunks)} chunks from {len(remaining)} PDFs")
    return vs
