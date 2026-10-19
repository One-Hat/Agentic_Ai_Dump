from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL_ID, IS_CONFIGURED

def get_chat_model(temperature: float = 0.1, max_new_tokens: int = 512) -> ChatGroq:
    """
    Initializes the open-source Chat LLM via Groq.
    """
    if not IS_CONFIGURED:
        raise ValueError(
            "GROQ_API_KEY is not configured! Please add your free key to the .env file."
        )

    return ChatGroq(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_new_tokens
    )

def get_embeddings_model():
    """
    Initializes a local sentence-transformers embedding model for document retrieval.
    Runs locally on CPU, 100% offline and free.
    """
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_ID)
