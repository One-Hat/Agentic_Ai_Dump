import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve token and sanitize any stray spaces or quotes
raw_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "") or os.getenv("HF_TOKEN", "")
HF_TOKEN = raw_token.strip().strip('"').strip("'")

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct").strip()
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2").strip()

# Check if token is valid placeholder
IS_TOKEN_CONFIGURED = bool(HF_TOKEN and not HF_TOKEN.startswith("hf_your_token"))
