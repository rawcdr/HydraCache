import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import QueryResponse
from src.config import Config
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, collection_name: str = Config.COLLECTION_NAME, host: str = Config.QDRANT_HOST, port: int = Config.QDRANT_PORT):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        
        # We reuse the exact embedding model used in Phase 1 indexing
        logger.info(f"Initializing DenseRetriever with embedding model {Config.EMBEDDING_MODEL}")
        self.client.set_model(Config.EMBEDDING_MODEL)

    def to_retrieval_result(self, point) -> RetrievalResult:
        """
        Converts a Qdrant QueryResponse point (or ScoredPoint depending on API) into our strict RetrievalResult.
        """
        metadata = point.metadata if hasattr(point, "metadata") else point.payload
        
        return RetrievalResult(
            chunk_id=metadata.get("chunk_id", ""),
            text=metadata.get("document", ""),  # fastembed injects the text into "document" payload
            score=float(point.score),
            source_file=metadata.get("source_file", ""),
            page_number=metadata.get("page_number", 0),
            chunk_type=metadata.get("chunk_type", "text"),
            parent_id=metadata.get("parent_id"),
            retrieval_source="dense"
        )

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        logger.info(f"Executing Dense retrieval for query: '{query}' (top_k={top_k})")
        
        if not query.strip():
            logger.warning("Empty query received; returning empty list.")
            return []
            
        try:
            # We use QdrantClient.query() which automatically embeds query_text using the configured model
            results = self.client.query(
                collection_name=self.collection_name,
                query_text=query,
                limit=top_k
            )
            
            retrieval_results = [self.to_retrieval_result(pt) for pt in results]
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Failed dense retrieval from Qdrant: {str(e)}")
            raise
