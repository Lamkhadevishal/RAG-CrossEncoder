"""Conversational memory: query rewriting and history formatting."""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# System prompt that rewrites a follow-up query into a self-contained question
_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a query rewriter for a conversational RAG system.

Given the chat history and the latest user question, rewrite the question into a
fully self-contained standalone question that resolves ALL pronouns (it, this, that,
they, those, its, their, former, latter, etc.) and references using context from the
chat history.

Rules:
- If the question is already standalone, return it unchanged.
- Replace every ambiguous pronoun or reference with its explicit referent.
- Do NOT answer the question — only rewrite it.
- Return ONLY the rewritten question, nothing else.

Chat History:
{chat_history}

Latest Question: {question}

Standalone Question:"""
)


def format_chat_history(chat_history: List[dict], max_turns: int = 6) -> str:
    """
    Render the last `max_turns` turns as a plain-text dialogue string.
    Each entry is expected to have 'query' and 'answer' keys.
    """
    recent = chat_history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['query']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def rewrite_query(
    query: str,
    chat_history: List[dict],
    llm,
) -> str:
    """
    Use the LLM to resolve pronouns/references and return a standalone query.
    Falls back to the original query on any error.
    """
    if not chat_history:
        return query  # No history → nothing to resolve

    history_str = format_chat_history(chat_history)
    chain = _REWRITE_PROMPT | llm | StrOutputParser()

    try:
        rewritten = chain.invoke({
            "chat_history": history_str,
            "question":     query,
        }).strip()
        logger.info(f"Rewritten query: {rewritten!r}")
        return rewritten or query
    except Exception as e:
        logger.warning(f"Query rewriting failed, using original: {e}")
        return query
