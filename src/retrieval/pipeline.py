import logging
from typing import List
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

class PipelineRetriever:
    def __init__(self, rrf_k: int = 60, pool_size: int = 20, top_k: int = 5):
        self.hybrid_retriever = HybridRetriever(rrf_k=rrf_k, pool_size=pool_size)
        
        try:
            self.reranker = Reranker()
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.reranker = None
            
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievalResult]:
        logger.info(f"Starting end-to-end retrieval pipeline for query: '{query}'")
        
        # 1-4. BM25, Dense, RRF, Top-20 candidate pool
        candidates = self.hybrid_retriever.retrieve_hybrid(query)
        
        # 5-6. Cross-encoder reranking, Top-5 final results
        if self.reranker:
            final_results = self.reranker.rerank(query, candidates, top_k=self.top_k)
            logger.info("Reranking applied successfully.")
            return final_results
        else:
            logger.error("Reranker unavailable. Cannot fulfill Phase 2 strict requirements.")
            raise RuntimeError("Reranking failed: Reranker is unavailable.")
