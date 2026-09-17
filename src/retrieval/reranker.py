import logging
from typing import List
from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Initializing CrossEncoder reranker with model {model_name}")
        self.model = TextCrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        if not candidates:
            return []
            
        logger.info(f"Reranking {len(candidates)} candidates for query: '{query}'")
        
        # Prepare documents for reranking
        documents = [c.text for c in candidates]
        
        try:
            # Generate rerank scores using fastembed TextCrossEncoder
            # It yields an Iterable[float] corresponding to the documents
            scores = list(self.model.rerank(query, documents))
        except Exception as e:
            logger.error(f"Failed to rerank candidates: {e}")
            raise

        # Assign rerank scores to copies of candidate objects
        reranked_results = []
        for candidate, score in zip(candidates, scores):
            # Create a shallow-ish copy manually to assign rerank score
            reranked_result = RetrievalResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=candidate.score, # original retrieval score
                source_file=candidate.source_file,
                page_number=candidate.page_number,
                chunk_type=candidate.chunk_type,
                parent_id=candidate.parent_id,
                retrieval_source=candidate.retrieval_source,
                provenance=candidate.provenance,
                rrf_score=candidate.rrf_score,
                rerank_score=float(score)
            )
            reranked_results.append(reranked_result)
            
        # Sort by rerank score descending
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return reranked_results[:top_k]
