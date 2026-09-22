# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))
os.environ.setdefault("LANGSMITH_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "pharmacy_RAG"))
os.environ.setdefault("LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CHUNKS_DIR = DATA_DIR / "chunks"
IMAGES_DIR = DATA_DIR / "images"
TABLES_DIR = DATA_DIR / "tables"
FORMULAS_DIR = DATA_DIR / "formulas"
CHUNK_VECTORS_DIR = DATA_DIR / "chunk_vectors_cache"

SUBJECTS = [
    "biopharmaceutics", "clinical_pharmacy", "industrial_pharmacy", "hospital_pharmacy", "pqm",
    "anatomy", "biochemistry", "medical_physiology", "organic_chemistry",
]

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = "SarimAli26/pharmacy-rag-checkpoints"

# Postgres
POSTGRES_URI = os.environ.get("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/pharmacy_rag")

# Memory settings
MAX_TOKENS_STM = 4000        # max tokens to keep in short term memory
MAX_MESSAGES_STM = 20        # max messages to keep before trimming