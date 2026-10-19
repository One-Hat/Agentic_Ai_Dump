from langgraph.graph import END, StateGraph
from state import GraphState 
from nodes import retrieve, grade_documents, transform_query, web_search_node, generate
from edges import decide_to_generate, grade_generation_v_documents_and_question

def build_adaptive_rag_graph():
    """
    Constructs and compiles the Adaptive RAG workflow using LangGraph.
    """
    workflow = StateGraph(GraphState)

    # Register Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)

    # Set Workflow Entry Point
    workflow.set_entry_point("retrieve")

    # Connect Nodes & Conditional Edges
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": "transform_query",
            "max_retries": END
        }
    )

    app = workflow.compile()
    return app

# Singleton compiled app
rag_app = build_adaptive_rag_graph()
