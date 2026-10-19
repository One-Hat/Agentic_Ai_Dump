from typing import Literal
from langchain_core.messages import HumanMessage
from state import GraphState
from models import get_chat_model

def decide_to_generate(state: GraphState) -> Literal["transform_query", "generate"]:
    """
    Determines whether to route to query transformation (and then web search)
    or proceed directly to answer generation.
    """
    web_search = state.get("web_search", False)
    iterations = state.get("iterations", 0)

    # Prevent infinite loops: cap transformations at 2
    if web_search and iterations < 2:
        print("--- [DECISION] Retrieved documents insufficient or irrelevant -> Routing to TRANSFORM QUERY ---")
        return "transform_query"
    else:
        print("--- [DECISION] Documents sufficient (or iteration limit reached) -> Routing to GENERATE ---")
        return "generate"

def grade_generation_v_documents_and_question(state: GraphState) -> Literal["useful", "not useful", "max_retries"]:
    """
    Self-correction edge: Evaluates if the generated response is grounded in facts
    and addresses the original question.
    """
    print("\n--- [EVALUATION] Checking response accuracy and usefulness ---")
    generation = state["generation"]
    question = state.get("original_question", state["question"])
    documents = state["documents"]
    iterations = state.get("iterations", 0)

    if iterations >= 2:
        print("  -> Max evaluation retries reached. Finalizing output.")
        return "max_retries"

    llm = get_chat_model(temperature=0.0)
    context_str = "\n\n".join([doc.page_content for doc in documents])

    # 1. Hallucination check only if substantial retrieved context exists
    if len(context_str.strip()) > 50:
        hallucination_prompt = (
            f"You are an evaluator grading whether an answer contradicts the provided facts.\n"
            f"Facts:\n{context_str}\n\n"
            f"Answer:\n{generation}\n\n"
            f"Does the answer directly contradict the facts? Reply 'yes' if it contradicts, or 'no' if it is faithful."
        )
        h_res = llm.invoke([HumanMessage(content=hallucination_prompt)])
        contradicts = "yes" in h_res.content.strip().lower()

        if contradicts:
            print("  -> Self-Correction Alert: Output contradicts provided context.")
            return "not useful"

    # 2. Answer usefulness check
    usefulness_prompt = (
        f"You are an evaluator grading whether an answer helpfully addresses the user question.\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{generation}\n\n"
        f"Does the answer directly address and answer the user question? Reply with only 'yes' or 'no'."
    )
    u_res = llm.invoke([HumanMessage(content=usefulness_prompt)])
    is_useful = "yes" in u_res.content.strip().lower()

    if is_useful:
        print("  -> Quality Verified: Generation is helpful and directly answers the question.")
        return "useful"
    else:
        print("  -> Output did not completely address the question. Re-routing for query rewrite.")
        return "not useful"
