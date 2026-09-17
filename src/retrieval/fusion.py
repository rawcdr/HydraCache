from typing import List, Dict, Tuple
from src.retrieval.models import RetrievalResult

def rrf_fuse(results_lists: List[List[RetrievalResult]], k: int = 60, top_k: int = 20) -> List[RetrievalResult]:
    """
    Fuses multiple ranked lists of RetrievalResult using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF_score = sum(1 / (k + rank_i)) for each list i where the document appears.
    Rank is 1-indexed.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievalResult] = {}
    provenance_map: Dict[str, set] = {}

    for results in results_lists:
        for rank_zero_indexed, result in enumerate(results):
            rank = rank_zero_indexed + 1
            chunk_id = result.chunk_id
            
            # Initialize if not seen
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                chunk_map[chunk_id] = result
                provenance_map[chunk_id] = set()
            
            # Add to RRF score
            rrf_scores[chunk_id] += 1.0 / (k + rank)
            
            # Track provenance
            provenance_map[chunk_id].add(result.retrieval_source)

    # Sort chunks by RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    # Take top_k
    top_chunk_ids = sorted_chunk_ids[:top_k]
    
    # Build final result list
    fused_results = []
    for chunk_id in top_chunk_ids:
        original = chunk_map[chunk_id]
        # Create a new instance to avoid mutating the original
        fused_result = RetrievalResult(
            chunk_id=original.chunk_id,
            text=original.text,
            score=original.score, # Keeping original raw score from whatever retriever hit it first
            source_file=original.source_file,
            page_number=original.page_number,
            chunk_type=original.chunk_type,
            parent_id=original.parent_id,
            retrieval_source="hybrid",
            provenance=sorted(list(provenance_map[chunk_id])),
            rrf_score=rrf_scores[chunk_id]
        )
        fused_results.append(fused_result)
        
    return fused_results
