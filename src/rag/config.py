"""Load settings from the project .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / os.environ["DATA_DIR"]
CHROMA_DIR = PROJECT_ROOT / os.environ["CHROMA_DIR"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_REVISION = os.environ["EMBEDDING_REVISION"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]
OLLAMA_URL = os.environ["OLLAMA_URL"]
CHUNK_SIZE = int(os.environ["CHUNK_SIZE"])
CHUNK_OVERLAP = int(os.environ["CHUNK_OVERLAP"])
RETRIEVAL_K = int(os.environ["RETRIEVAL_K"])
