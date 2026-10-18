from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from state import GraphState
from models import get_chat_model
from tools import get_retriever, search_web

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from local vector store based on current question.
    """
    question = state["question"]
    print(f"\n--- [NODE: RETRIEVE] Fetching documents for: '{question}' ---")
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "iterations": state.get("iterations", 0),
        "original_question": state.get("original_question", question)
    }

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant or if no docs are found, triggers web search fallback.
    """
    print("\n--- [NODE: GRADE DOCUMENTS] Evaluating relevance to query ---")
    question = state["question"]
    documents = state["documents"]
    llm = get_chat_model(temperature=0.0)

    filtered_docs = []
    web_search = False

    if not documents:
        print("  -> No documents retrieved. Web search triggered.")
        return {"documents": [], "web_search": True}

    for doc in documents:
        prompt = (
            f"You are a strict relevance grader. Evaluate whether the following document contains information "
            f"relevant to the user question.\n\n"
            f"Document:\n{doc.page_content}\n\n"
            f"Question:\n{question}\n\n"
            f"Respond with only a single word: 'yes' if relevant, or 'no' if irrelevant."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        grade = response.content.strip().lower()

        if "yes" in grade:
            print(f"  -> Document [{doc.metadata.get('source', 'doc')}]: RELEVANT")
            filtered_docs.append(doc)
        else:
            print(f"  -> Document [{doc.metadata.get('source', 'doc')}]: NOT RELEVANT")
            web_search = True

    # If all documents were filtered out, ensure web_search is enabled
    if len(filtered_docs) == 0:
        web_search = True

    return {"documents": filtered_docs, "web_search": web_search}

def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Rewrites the question into an optimized web search query when retrieval is insufficient.
    """
    print("\n--- [NODE: TRANSFORM QUERY] Optimizing question for search ---")
    question = state["question"]
    iterations = state.get("iterations", 0) + 1
    llm = get_chat_model(temperature=0.2)

    prompt = (
        f"You are a query optimizer. The following query failed to retrieve sufficient internal documents.\n"
        f"Rewrite it into a clear, concise keyword-rich search query suitable for web search.\n\n"
        f"Original Query: {question}\n\n"
        f"Optimized Query (reply with only the rewritten query, nothing else):"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    better_question = response.content.strip()
    print(f"  -> Rewritten query: '{better_question}' (Iteration {iterations})")

    return {
        "question": better_question,
        "iterations": iterations
    }

def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes live web search to complement or replace missing local context.
    """
    print("\n--- [NODE: WEB SEARCH] Running live web search ---")
    question = state["question"]
    existing_docs = state.get("documents", [])
    
    web_docs = search_web(question, max_results=3)
    print(f"  -> Retrieved {len(web_docs)} web search results.")
    
    combined_docs = existing_docs + web_docs
    return {"documents": combined_docs, "web_search": False}

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Synthesizes the final answer using retrieved context documents.
    """
    print("\n--- [NODE: GENERATE] Synthesizing final answer ---")
    question = state.get("original_question", state["question"])
    documents = state["documents"]
    llm = get_chat_model(temperature=0.1, max_new_tokens=768)

    context_str = "\n\n".join([f"Source [{doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}" for doc in documents])
    
    system_msg = (
        "You are an expert AI assistant specializing in Agentic AI and open-source models. "
        "Answer the user's question accurately and concisely, basing your response on the provided context. "
        "If the context does not contain sufficient facts to answer completely, acknowledge what is known and clarify limits."
    )
    user_msg = (
        f"Context:\n{context_str}\n\n"
        f"User Question:\n{question}\n\n"
        f"Synthesized Answer:"
    )

    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    return {"generation": response.content.strip()}
