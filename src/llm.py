"""LLM initialisation and answer generation."""

import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Conversational prompt — includes chat history for coherent multi-turn answers
_PROMPT = ChatPromptTemplate.from_template("""
You are a highly reliable Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question using ONLY the information available in the retrieved context.

## Rules

### 1. Source of Truth

* The retrieved context is the ONLY source of factual information.
* Do NOT use your pre-trained knowledge.
* Do NOT make assumptions or infer facts that are not explicitly supported by the retrieved context.

### 2. Conversation History Usage

Use the conversation history only for:

* Understanding follow-up questions.
* Resolving references such as:

  * "it"
  * "this"
  * "that"
  * "they"
  * "the above"
* Maintaining conversational continuity.

Do NOT use conversation history as a factual source unless the same information is present in the retrieved context.

### 3. Missing Information

If the retrieved context does not contain enough information to answer the question:

* Clearly state:
  "The provided documents do not contain sufficient information to answer this question."
* Mention what information is missing if possible.
* Do NOT guess or fabricate an answer.

### 4. Conflicting Information

If multiple context chunks contain conflicting information:

* Mention the conflict.
* Present the differing statements.
* Do not attempt to determine which is correct unless the context explicitly resolves the conflict.

### 5. Answer Generation

Before answering:

1. Understand the user's question.
2. Resolve references using conversation history if needed.
3. Identify the most relevant information from the retrieved context.
4. Generate an answer strictly grounded in the context.

### 6. Response Style

When sufficient information is available, structure the response as:

#### Overview

Brief summary.

#### Detailed Explanation

Comprehensive explanation based on retrieved information.

#### Key Points

* Important fact 1
* Important fact 2
* Important fact 3

#### Conclusion

Concise final summary.

### 7. Accuracy Requirements

* Every statement in the answer must be supported by the retrieved context.
* Do not introduce new facts.
* Do not speculate.
* Do not provide unsupported recommendations.

### 8. Ambiguous Questions

If the user's question remains ambiguous after considering the conversation history:

* Ask a clarifying question.
* Do not assume the user's intent.

Conversation History:
{chat_history}

Retrieved Context:
{context}

User Question:
{question}

Answer:


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
