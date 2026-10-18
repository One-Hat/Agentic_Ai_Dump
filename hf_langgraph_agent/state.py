from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the state of the Adaptive RAG LangGraph workflow.
    
    Attributes:
        question: Current query (may be rewritten by the query transformation node).
        original_question: The user's initial question.
        generation: The generated response from the LLM.
        web_search: Flag indicating whether external web search should be executed.
        documents: List of retrieved or fetched documents.
        iterations: Count of query transformation loops to prevent infinite loops.
    """
    question: str
    original_question: str
    generation: str
    web_search: bool
    documents: List[Document]
    iterations: int
