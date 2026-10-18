import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from duckduckgo_search import DDGS
from models import get_embeddings_model

# Default knowledge base articles for the Adaptive RAG demo
SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "LangGraph is a library developed by LangChain for building stateful, multi-actor applications with LLMs. "
            "It extends LangChain Expression Language with cyclical workflows, enabling agents to have loops, "
            "conditional branching, human-in-the-loop validation, and persistent memory."
        ),
        metadata={"source": "langgraph_overview.md", "topic": "LangGraph"}
    ),
    Document(
        page_content=(
            "Adaptive RAG is an advanced retrieval-augmented generation strategy that dynamically routes queries, "
            "evaluates retrieval quality, rewrites questions when needed, and falls back to live web search "
            "if internal document stores lack relevant context."
        ),
        metadata={"source": "adaptive_rag_guide.md", "topic": "RAG"}
    ),
    Document(
        page_content=(
            "Open-source AI models hosted on Hugging Face—such as Llama 3.1, Qwen 2.5, and Mistral—provide state-of-the-art "
            "reasoning capabilities. Hugging Face Serverless Inference API allows developers to run inference on these "
            "models without maintaining dedicated GPU infrastructure."
        ),
        metadata={"source": "huggingface_models.md", "topic": "Open Source AI"}
    ),
    Document(
        page_content=(
            "In agentic workflows, self-correction loops use grader components to evaluate model outputs for hallucinations "
            "and factual accuracy against retrieved source documents before returning an answer to the end user."
        ),
        metadata={"source": "agentic_evaluations.md", "topic": "Evaluation"}
    )
]

_vectorstore = None

def get_retriever():
    """
    Returns a retriever backed by ChromaDB initialized with sample knowledge documents.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings_model()
        persist_dir = os.path.join(os.path.dirname(__file__), ".chroma_db")
        _vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCUMENTS,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    return _vectorstore.as_retriever(search_kwargs={"k": 2})

def search_web(query: str, max_results: int = 3) -> List[Document]:
    """
    Executes a web search fallback using DuckDuckGo.
    """
    results = []
    try:
        ddgs = DDGS()
        search_data = ddgs.text(query, max_results=max_results)
        for item in search_data:
            results.append(
                Document(
                    page_content=item.get("body", ""),
                    metadata={"source": item.get("href", "web_search"), "title": item.get("title", "")}
                )
            )
    except Exception as e:
        results.append(
            Document(
                page_content=f"Web search could not retrieve external results for '{query}'. Details: {str(e)}",
                metadata={"source": "web_search_fallback"}
            )
        )
    return results
