import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Groq API configuration
raw_groq_key = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY = raw_groq_key.strip().strip('"').strip("'")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.8-27b").strip().strip('"').strip("'")

# Local embedding model configuration
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2").strip().strip('"').strip("'")

# Export to environment for underlying clients
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

IS_CONFIGURED = bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_your_free_key"))
