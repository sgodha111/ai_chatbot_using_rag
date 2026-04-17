from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_store"
CHROMA_COLLECTION = "rag_documents"
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K_RESULTS = 4
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
