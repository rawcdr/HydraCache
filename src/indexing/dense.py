import logging
from typing import List
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from src.config import Config

logger = logging.getLogger(__name__)

class DenseIndexer:
    def __init__(self, host: str = Config.QDRANT_HOST, port: int = Config.QDRANT_PORT):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = Config.COLLECTION_NAME
        self.model_name = Config.EMBEDDING_MODEL
        
        # Configure FastEmbed native support in QdrantClient
        logger.info(f"Setting FastEmbed model: {self.model_name}")
        self.client.set_model(self.model_name)

    def index(self, chunks: List[Document]):
        """
        Embeds and indexes documents into Qdrant natively.
        """
        logger.info(f"Indexing {len(chunks)} chunks into Qdrant collection '{self.collection_name}'...")
        
        texts = [chunk.page_content for chunk in chunks]
        metadata = [{"page": chunk.metadata.get("page", 0), "chunk_id": chunk.metadata.get("chunk_id", "")} for chunk in chunks]
        
        # client.add generates embeddings natively using FastEmbed and handles collection creation
        self.client.add(
            collection_name=self.collection_name,
            documents=texts,
            metadata=metadata,
            ids=list(range(1, len(texts) + 1))  # Simple int IDs for qdrant
        )
        logger.info(f"Successfully indexed {len(texts)} chunks into Qdrant.")
