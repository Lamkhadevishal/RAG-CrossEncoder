"""RAG pipeline orchestrator with conversational memory."""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embedding_model
from src.vectorstore import load_vectorstore, similarity_search
from src.reranker import rerank
from src.llm import build_llm, generate_answer
from src.memory import rewrite_query, format_chat_history

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end conversational RAG pipeline with Cross-Encoder reranking."""

    def __init__(self, config: dict, vectorstore: Optional[FAISS] = None):
        self.config     = config
        self.embeddings = get_embedding_model(config["EMBEDDING_MODEL"])
        self.llm        = build_llm(
            config["LLM_PROVIDER"],
            config["LLM_MODEL"],
            config["LLM_API_KEY"],
        )

        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            persist_path = config["FAISS_INDEX_PATH"]
            if not (Path(persist_path) / "index.faiss").exists():
                raise ValueError("No FAISS index found. Ingest documents first.")
            self.vectorstore = load_vectorstore(persist_path, self.embeddings)

    def run(self, query: str, chat_history: Optional[List[dict]] = None) -> dict:
        """
        Execute the full conversational pipeline.

        Steps:
          1. Rewrite query using chat history (resolves pronouns/references)
          2. Vector similarity search with standalone query
          3. Cross-Encoder reranking
          4. LLM answer generation with history context
          5. Return all intermediate results for UI display
        """
        chat_history = chat_history or []
        top_k = int(self.config.get("TOP_K", 10))
        top_n = int(self.config.get("TOP_N", 3))

        # 1. Rewrite ambiguous query into standalone question
        standalone_query = rewrite_query(query, chat_history, self.llm)

        # 2. Retrieve using the resolved query
        retrieved = similarity_search(self.vectorstore, standalone_query, top_k)

        # 3. Rerank with Cross-Encoder
        reranked = rerank(
            standalone_query,
            retrieved,
            model_name=self.config["CROSS_ENCODER_MODEL"],
            top_n=top_n,
        )

        # 4. Generate answer with history for conversational continuity
        history_str = format_chat_history(chat_history)
        answer = generate_answer(
            standalone_query,
            reranked,
            provider=self.config["LLM_PROVIDER"],
            model=self.config["LLM_MODEL"],
            api_key=self.config["LLM_API_KEY"],
            chat_history_str=history_str,
        )

        return {
            "original_query":  query,
            "standalone_query": standalone_query,
            "retrieved_docs":  retrieved,
            "reranked_docs":   reranked,
            "answer":          answer,
        }
