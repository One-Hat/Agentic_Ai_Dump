import os
import json
import re
import urllib.request
import urllib.parse
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from models import get_embeddings_model

# Default knowledge base articles for the Adaptive RAG system
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
            "Open-source AI models such as Qwen 2.5/3, Llama 3.1/3.3, and Mistral provide state-of-the-art "
            "reasoning capabilities. Hosted on ultra-fast LPU inference engines like Groq, they achieve "
            "hundreds of tokens per second for real-time agentic decision loops."
        ),
        metadata={"source": "open_source_models.md", "topic": "Open Source AI"}
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

def _search_wikipedia(query: str, max_results: int = 3) -> List[Document]:
    """
    Reliable encyclopedia search fallback using Wikipedia's public REST API.
    """
    results = []
    try:
        clean_q = re.sub(r"[^\w\s]", "", query).strip()
        encoded = urllib.parse.quote(clean_q)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json&utf8=1"
        req = urllib.request.Request(url, headers={"User-Agent": "AdaptiveRAGAgent/1.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
            items = data.get("query", {}).get("search", [])[:max_results]
            for item in items:
                title = item.get("title", "")
                snippet = re.sub(r"<.*?>", "", item.get("snippet", ""))
                results.append(
                    Document(
                        page_content=f"{title}: {snippet}",
                        metadata={"source": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}", "title": title}
                    )
                )
    except Exception as e:
        results.append(
            Document(
                page_content=f"External knowledge retrieval encountered: {str(e)}",
                metadata={"source": "external_search_error"}
            )
        )
    return results

def search_web(query: str, max_results: int = 3) -> List[Document]:
    """
    Searches external web sources using DuckDuckGo with Wikipedia fallback.
    """
    results = []
    # 1. Attempt DuckDuckGo search
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        
        with DDGS(timeout=8) as ddgs:
            search_data = list(ddgs.text(query, max_results=max_results))
            for item in search_data:
                body = item.get("body", "")
                if body:
                    results.append(
                        Document(
                            page_content=body,
                            metadata={"source": item.get("href", "web_search"), "title": item.get("title", "")}
                        )
                    )
    except Exception:
        # Fall back gracefully if DuckDuckGo times out or rate limits
        pass

    # 2. If DDG produced no results, fall back to Wikipedia
    if not results:
        results = _search_wikipedia(query, max_results=max_results)

    return results
