from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from config import HF_TOKEN, HF_MODEL_ID, EMBEDDING_MODEL_ID, IS_TOKEN_CONFIGURED

def get_chat_model(temperature: float = 0.7, max_new_tokens: int = 512):
    """
    Initializes an open-source Chat LLM via Hugging Face Serverless Inference API.
    """
    if not IS_TOKEN_CONFIGURED:
        raise ValueError(
            "Hugging Face API token is not configured! Please set HUGGINGFACEHUB_API_TOKEN in your .env file."
        )

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
    Initializes a Hugging Face sentence transformer embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
