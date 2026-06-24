"""CLI entry point for the RAG + Cross-Encoder pipeline."""

import logging
import os
import sys

from dotenv import load_dotenv

from src.pipeline import RAGPipeline

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
# Helpers
# ---------------------------------------------------------------------------
def _separator(char: str = "─", width: int = 70) -> str:
    return char * width


def _load_config() -> dict:
    load_dotenv()
    required = ["LLM_API_KEY", "PDF_PATH"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        sys.exit(1)

    return {
        "PDF_PATH": os.getenv("PDF_PATH", ""),
        "FAISS_INDEX_PATH": os.getenv("FAISS_INDEX_PATH", "faiss_index"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "TOP_K": os.getenv("TOP_K", "10"),
        "TOP_N": os.getenv("TOP_N", "3"),
        "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "500"),
        "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "50"),
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "groq"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "llama3-8b-8192"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
    }


def _display_results(result: dict) -> None:
    query = result["query"]
    retrieved = result["retrieved_docs"]
    reranked = result["reranked_docs"]
    answer = result["answer"]

    print(f"\n{_separator('═')}")
    print(f"  QUERY: {query}")
    print(_separator('═'))

    # Retrieved documents (before reranking)
    print(f"\n📄  TOP-{len(retrieved)} RETRIEVED DOCUMENTS (vector similarity)\n{_separator()}")
    for i, doc in enumerate(retrieved, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] Source: {source} | Page: {page}")
        print(f"      {snippet}...\n")

    # Reranked documents with scores
    print(f"🔁  TOP-{len(reranked)} RERANKED DOCUMENTS (Cross-Encoder scores)\n{_separator()}")
    for i, (doc, score) in enumerate(reranked, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] Score: {score:.4f} | Source: {source} | Page: {page}")
        print(f"      {snippet}...\n")

    # Final LLM answer
    print(f"🤖  FINAL ANSWER\n{_separator()}")
    print(f"  {answer.strip()}")
    print(_separator('═') + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = _load_config()

    logger.info("Initialising RAG pipeline...")
    pipeline = RAGPipeline(config)

    print("\n" + _separator('═'))
    print("  RAG System with Cross-Encoder Reranking  |  Type 'exit' to quit")
    print(_separator('═'))

    while True:
        try:
            query = input("\n❓ Enter your query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        try:
            result = pipeline.run(query)
            _display_results(result)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
