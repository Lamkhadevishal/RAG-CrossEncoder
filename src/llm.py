"""LLM initialisation and answer generation."""

import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Conversational prompt — includes chat history for coherent multi-turn answers
_PROMPT = ChatPromptTemplate.from_template("""
You are an expert AI assistant designed for Retrieval-Augmented Generation (RAG).

Your primary responsibility is to answer the user's question using ONLY the information provided in the retrieved context.

Guidelines:

1. Use the conversation history only to understand references, follow-up questions, and conversational context.

   * Resolve terms such as "it", "this", "that", "they", etc. using the chat history.
   * Do NOT use the chat history as a source of factual information unless that information is also present in the retrieved context.

2. Use ONLY the retrieved context as the source of truth.

3. Do NOT make assumptions, infer missing facts, or use external knowledge.

4. If the context does not contain sufficient information to answer the question:

   * Clearly state that the information is not available in the provided documents.
   * Do not fabricate or guess an answer.

5. Provide detailed and well-structured responses whenever enough information is available.

6. When appropriate, structure the answer using:

   * Overview
   * Key Concepts
   * Detailed Explanation
   * Important Points
   * Conclusion

7. If multiple pieces of context are retrieved:

   * Combine them into a coherent answer.
   * Remove redundancy.
   * Preserve factual accuracy.

8. If the user's question is ambiguous:

   * Use conversation history to resolve ambiguity.
   * If ambiguity still exists, ask for clarification.

Conversation History:
{chat_history}

Retrieved Context:
{context}

User Question:
{question}

Instructions:

* First identify what the user is asking.
* Use the conversation history only for context resolution.
* Extract relevant information from the retrieved context.
* Generate a clear, structured, and accurate response.
* Never include information that is not supported by the retrieved context.

Answer:
""")


def build_llm(provider: str, model: str, api_key: str):
    """Instantiate the appropriate LangChain chat model."""
    provider = provider.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider!r}. Choose groq | openai | google")


def generate_answer(
    query: str,
    reranked_docs: List[Tuple[Document, float]],
    provider: str,
    model: str,
    api_key: str,
    chat_history_str: str = "",
) -> str:
    """Build context from reranked docs and invoke the LLM with conversation history."""
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in reranked_docs)
    llm     = build_llm(provider, model, api_key)
    chain   = _PROMPT | llm | StrOutputParser()

    logger.info(f"Generating answer with {provider}/{model}…")
    return chain.invoke({
        "chat_history": chat_history_str or "No previous conversation.",
        "context":      context,
        "question":     query,
    })
