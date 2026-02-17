import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Embeddings Configuration
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDINGS_DIMENSION = int(os.getenv("EMBEDDINGS_DIMENSION", "384"))

# Vector Database Configuration
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector_db/chromadb")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./vector_db/faiss_indices")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
