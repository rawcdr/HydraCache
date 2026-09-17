import logging
import pickle
from typing import List
from src.config import Config
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self, index_path: str = Config.BM25_INDEX_PATH):
        self.index_path = index_path
        self.bm25 = None
        self.chunks = None
        self._load_index()

    def _load_index(self):
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunks = data['chunks']
            logger.info(f"Successfully loaded BM25 index from {self.index_path}")
        except FileNotFoundError:
            logger.error(f"BM25 index not found at {self.index_path}. Please run Phase 1 indexing first.")
            raise

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        logger.info(f"Executing BM25 retrieval for query: '{query}' (top_k={top_k})")
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue # Skip zero-match documents
                
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.metadata.get("chunk_id", ""),
                text=chunk.page_content,
                score=float(score),
                source_file=chunk.metadata.get("source_file", ""),
                page_number=chunk.metadata.get("page_number", 0),
                chunk_type=chunk.metadata.get("chunk_type", "text"),
                parent_id=chunk.metadata.get("parent_id"),
                retrieval_source="bm25"
            ))
            
        return results
