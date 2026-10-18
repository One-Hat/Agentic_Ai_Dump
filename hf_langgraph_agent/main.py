import sys
import argparse
from config import IS_TOKEN_CONFIGURED, HF_MODEL_ID
from graph import rag_app

def print_banner():
    print("=" * 70)
    print("   🤖 OPEN-SOURCE HUGGING FACE + LANGRAPH ADAPTIVE RAG AGENT")
    print(f"   Model: {HF_MODEL_ID}")
    print("=" * 70)

def run_agent(question: str):
    """
    Executes the Adaptive RAG workflow for a given question.
    """
    if not IS_TOKEN_CONFIGURED:
        print("\n❌ Error: Hugging Face API token is missing!")
        print("Please copy .env.example to .env and set your HUGGINGFACEHUB_API_TOKEN:")
        print("  1. Visit: https://huggingface.co/settings/tokens")
        print("  2. Create a free read token")
        print("  3. Set HUGGINGFACEHUB_API_TOKEN=hf_... in .env\n")
        return

    print(f"\n[QUERY RECEIVED]: '{question}'")
    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "web_search": False,
        "generation": "",
        "iterations": 0
    }

    print("\n🚀 Executing LangGraph workflow...\n")
    final_state = None
    
    # Stream events to trace node execution
    for output in rag_app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"✓ Step completed by [{node_name}]")
            final_state = state_update

    print("\n" + "=" * 70)
    print("🎯 FINAL SYNTHESIZED RESPONSE:")
    print("=" * 70)
    if final_state and "generation" in final_state:
        print(final_state["generation"])
    else:
        print("Could not obtain a final answer.")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Adaptive RAG Agent using Hugging Face & LangGraph")
    parser.add_argument("--query", "-q", type=str, help="Single query to ask the agent")
    parser.add_argument("--mermaid", action="store_true", help="Print the Mermaid diagram of the graph")
    args = parser.parse_args()

    print_banner()

    if args.mermaid:
        print("\n--- LangGraph Workflow Architecture (Mermaid) ---")
        try:
            print(rag_app.get_graph().draw_mermaid())
        except Exception as e:
            print(f"Could not render mermaid graph: {e}")
        return

    if args.query:
        run_agent(args.query)
    else:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_input = input("Ask a question: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                run_agent(user_input)
                print("\n")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

if __name__ == "__main__":
    main()
