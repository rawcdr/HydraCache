import logging
import concurrent.futures
from typing import List
from src.retrieval.sparse import SparseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.models import RetrievalResult
from src.retrieval.fusion import rrf_fuse

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, rrf_k: int = 60, pool_size: int = 20, bm25_top_k: int = 20, dense_top_k: int = 20):
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever()
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k

    def retrieve_hybrid(self, query: str) -> List[RetrievalResult]:
        logger.info(f"Executing Hybrid retrieval for query: '{query}'")
        
        # Run BM25 and Dense retrieval concurrently using a ThreadPool
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_bm25 = executor.submit(self.sparse_retriever.retrieve, query, self.bm25_top_k)
            future_dense = executor.submit(self.dense_retriever.retrieve, query, self.dense_top_k)
            
            # Wait for both to complete
            bm25_results = future_bm25.result()
            dense_results = future_dense.result()
            
        logger.info(f"Retrieved {len(bm25_results)} from BM25, {len(dense_results)} from Dense")
        
        # Perform RRF fusion
        fused_results = rrf_fuse(
            results_lists=[bm25_results, dense_results],
            k=self.rrf_k,
            top_k=self.pool_size
        )
        
        logger.info(f"Fusion complete. Candidate pool size: {len(fused_results)}")
        return fused_results
