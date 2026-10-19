from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    HF_TOKEN,
    HF_MODEL_ID,
    EMBEDDING_MODEL_ID,
    IS_CONFIGURED,
    USE_GROQ
)

def get_chat_model(temperature: float = 0.1, max_new_tokens: int = 512):
    """
    Initializes an open-source Chat LLM via Groq (free, high-speed)
    or falls back to Hugging Face Inference API.
    """
    if not IS_CONFIGURED:
        raise ValueError(
            "No LLM provider configured! Please set GROQ_API_KEY in your .env file."
        )

    if USE_GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_new_tokens
        )
    else:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id=HF_MODEL_ID,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            task="text-generation"
        )
        return ChatHuggingFace(llm=endpoint)

def get_embeddings_model():
    """
    Initializes a Hugging Face sentence transformer embedding model (runs locally).
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
