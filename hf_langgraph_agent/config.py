import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve Groq key and model
raw_groq_key = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY = raw_groq_key.strip().strip('"').strip("'")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant").strip().strip('"').strip("'")

# Retrieve Hugging Face token (fallback if Groq is not used)
raw_hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "") or os.getenv("HF_TOKEN", "")
HF_TOKEN = raw_hf_token.strip().strip('"').strip("'")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct").strip().strip('"').strip("'")

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2").strip().strip('"').strip("'")

# Set in environment for underlying libraries
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Validation check
IS_CONFIGURED = bool(
    (GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_your_free_key")) or
    (HF_TOKEN and not HF_TOKEN.startswith("hf_your_token"))
)
USE_GROQ = bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_your_free_key"))
