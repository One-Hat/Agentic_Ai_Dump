import os
import streamlit as st
from config import MODEL_NAME, EMBEDDING_MODEL_ID, IS_CONFIGURED
from graph import rag_app

# Page configuration
st.set_page_config(
    page_title="Adaptive RAG Agent | LangGraph + Groq",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 999px;
        background-color: #f0f2f6;
        color: #31333f;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Agent Settings")
    st.markdown(f"**LLM Model:** `{MODEL_NAME}`")
    st.markdown("**Inference Engine:** `Groq (LPU)`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL_ID}`")
    st.markdown("**Vector Store:** `ChromaDB (Local)`")
    st.divider()

    st.subheader("🏛️ Workflow Architecture")
    st.markdown("""
    This agent runs an **Adaptive RAG** loop in LangGraph:
    1. **Retrieve**: Searches local knowledge base.
    2. **Grade Docs**: Evaluates relevance.
    3. **Transform Query**: Rewrites query if local docs are insufficient.
    4. **Web Search**: Live search fallback (DuckDuckGo + Wikipedia).
    5. **Generate**: Synthesizes response.
    6. **Self-Correction**: Verifies grounding and completeness.
    """)

    with st.expander("View LangGraph Workflow Flowchart"):
        st.code("""
flowchart TD
    Start([User Query]) --> Retrieve[ChromaDB Vector Store]
    Retrieve --> GradeDocs{Relevance Grader}
    GradeDocs -- Relevant --> Generate[Synthesize Answer]
    GradeDocs -- Irrelevant --> Transform[Query Optimizer]
    Transform --> Search[Web & Wiki Fallback]
    Search --> Generate
    Generate --> Eval{Self-Correction Check}
    Eval -- Verified --> End([Final Response])
    Eval -- Incomplete --> Transform
        """, language="mermaid")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.markdown('<div class="main-title">🤖 Adaptive RAG Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stateful multi-actor agent built with <b>LangGraph</b>, '
    '<b>Groq LPU inference</b>, and <b>ChromaDB</b>.</div>',
    unsafe_allow_html=True
)

if not IS_CONFIGURED:
    st.error("⚠️ GROQ_API_KEY is not configured! Please set it in your `.env` file.")
    st.stop()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Adaptive RAG Agent. You can ask me questions about **LangGraph**, **Agentic AI**, or general topics—I'll retrieve internal documents, verify them, or fetch live web information when needed!",
            "steps": [],
            "sources": []
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("🔍 View Execution Steps", expanded=False):
                for step in msg["steps"]:
                    st.markdown(f"- {step}")
        if msg.get("sources"):
            with st.expander("📚 View Retrieved Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- **{src.get('title', 'Document')}** ({src.get('source', 'unknown')}):\n  > *{src.get('content', '')[:200]}...*")

# Chat input
if user_query := st.chat_input("Ask a question (e.g. 'What is LangGraph?' or 'What is the theory of relativity?')..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process assistant response with LangGraph
    with st.chat_message("assistant"):
        execution_steps = []
        retrieved_sources = []
        final_answer = ""

        # Real-time execution status container
        with st.status("🚀 Executing Adaptive RAG workflow...", expanded=True) as status:
            initial_state = {
                "question": user_query,
                "original_question": user_query,
                "documents": [],
                "web_search": False,
                "generation": "",
                "iterations": 0
            }

            final_state = {}
            for output in rag_app.stream(initial_state):
                for node_name, state_update in output.items():
                    final_state.update(state_update)

                    if node_name == "retrieve":
                        docs = state_update.get("documents", [])
                        msg = f"**Retrieve**: Fetched `{len(docs)}` document(s) from local ChromaDB."
                        st.write(f"🔍 {msg}")
                        execution_steps.append(msg)

                    elif node_name == "grade_documents":
                        web_search = state_update.get("web_search", False)
                        docs = state_update.get("documents", [])
                        if web_search:
                            msg = f"**Grade Documents**: Internal documents insufficient/irrelevant. Routing to external search."
                            st.write(f"⚖️ {msg}")
                        else:
                            msg = f"**Grade Documents**: Internal documents verified relevant (`{len(docs)}` kept)."
                            st.write(f"⚖️ {msg}")
                        execution_steps.append(msg)

                    elif node_name == "transform_query":
                        new_q = state_update.get("question", "")
                        iteration = state_update.get("iterations", 1)
                        msg = f"**Transform Query** (Pass {iteration}): Rewrote query to: *\"{new_q}\"*"
                        st.write(f"🔄 {msg}")
                        execution_steps.append(msg)

                    elif node_name == "web_search":
                        docs = state_update.get("documents", [])
                        msg = f"**Web Search**: Retrieved live external context (`{len(docs)}` total sources)."
                        st.write(f"🌐 {msg}")
                        execution_steps.append(msg)

                    elif node_name == "generate":
                        msg = "**Generate**: Synthesized response and verified through self-correction evaluation."
                        st.write(f"✍️ {msg}")
                        execution_steps.append(msg)

            status.update(label="✅ Workflow Execution Complete!", state="complete", expanded=False)

        final_answer = final_state.get("generation", "Could not generate an answer.")
        st.markdown(final_answer)

        # Collect source documents for reference
        for doc in final_state.get("documents", []):
            retrieved_sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", doc.metadata.get("topic", "Source")),
                "content": doc.page_content
            })

        if retrieved_sources:
            with st.expander("📚 View Retrieved Sources", expanded=False):
                for src in retrieved_sources:
                    st.markdown(f"- **{src.get('title', 'Document')}** (`{src.get('source', '')}`):\n  > *{src.get('content', '')[:200]}...*")

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "steps": execution_steps,
            "sources": retrieved_sources
        })
