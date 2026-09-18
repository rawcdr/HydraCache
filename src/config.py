import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant Settings
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "apple_10k")
    
    # Embedding Model (FastEmbed)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Data paths
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/apple_10k_2025.pdf")
    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.pkl")
    
    # Phase 5 API Settings
    API_KEY = os.getenv("API_KEY", "default-dev-key")
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
