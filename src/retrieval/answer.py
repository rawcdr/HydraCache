import time
import logging
from typing import List, Optional
from dataclasses import dataclass
from src.retrieval.models import RetrievalResult
from src.retrieval.pipeline import PipelineRetriever
from src.cache.semantic_cache import SemanticCache
from src.generation.synthesize import generate_answer

from src.generation.decompose import analyze_query

logger = logging.getLogger(__name__)

@dataclass
class AnswerResult:
    answer: str
    cache_hit: bool
    cache_distance: Optional[float] = None
    cache_latency: float = 0.0
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context: Optional[List[RetrievalResult]] = None
    query_mode: str = "single_hop"
    sub_questions: List[str] = None


class AnswerPipeline:
    """
    The central orchestration layer for HydraCache query processing.
    
    This pipeline manages the complete lifecycle of a query:
    1. Semantic Cache Lookup
    2. Query Classification & Decomposition (Multi-Hop)
    3. Parallel Sparse/Dense Retrieval
    4. Evidence Merging & Deduplication (for Multi-Hop)
    5. Cross-Encoder Reranking
    6. LLM Synthesis
    7. Cache Storage
    """
    def __init__(self, pipeline: Optional[PipelineRetriever] = None, cache: Optional[SemanticCache] = None):
        self.pipeline = pipeline or PipelineRetriever()
        self.cache = cache or SemanticCache()
        
    def answer(self, query: str) -> AnswerResult:
        """
        Processes a query through the full HydraCache pipeline.
        
        Args:
            query (str): The natural language query from the user.
            
        Returns:
            AnswerResult: A comprehensive object containing the synthesized answer, 
                          latency telemetry, token usage, query mode, and citations.
        """
        logger.info(f"Answering query: '{query}'")
        start_total = time.time()
        
        # 1. Semantic Cache Lookup
        cache_res = self.cache.lookup(query)
        cache_latency = cache_res.latency
        
        if cache_res.hit and cache_res.answer:
            logger.info("Cache hit successfully bypassed retrieval and generation.")
            return AnswerResult(
                answer=cache_res.answer,
                cache_hit=True,
                cache_distance=cache_res.distance,
                cache_latency=cache_latency,
                retrieval_latency=0.0,
                generation_latency=0.0,
                total_latency=(time.time() - start_total) * 1000,
                context=[], # Context was already baked into the answer
                query_mode="single_hop",
                sub_questions=[]
            )
            
        # 2. Phase 6 Multi-hop Detection
        plan = analyze_query(query)
        
        # 3. Retrieval
        start_retrieval = time.time()
        
        if plan.mode == "single_hop" or not plan.sub_questions:
            context = self.pipeline.retrieve(query)
        else:
            logger.info(f"Executing multi-hop retrieval for {len(plan.sub_questions)} sub-questions.")
            combined_context = []
            seen_chunk_ids = set()
            
            for sub_q in plan.sub_questions:
                sub_ctx = self.pipeline.retrieve(sub_q)
                for chunk in sub_ctx:
                    if chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk_id)
                        combined_context.append(chunk)
                        
            # Final Reranking of combined evidence pool against the original query
            # We take Top-7 to provide more context for multi-hop synthesis
            if self.pipeline.reranker and combined_context:
                context = self.pipeline.reranker.rerank(query, combined_context, top_k=7)
            else:
                context = combined_context[:7]
                
        retrieval_latency = (time.time() - start_retrieval) * 1000
        
        # 4. Answer Generation
        start_gen = time.time()
        try:
            gen_result = generate_answer(query, context)
            generation_latency = (time.time() - start_gen) * 1000
            
            # 5. Cache the successful result with original query as key
            self.cache.store(query, gen_result["answer"])
            
            return AnswerResult(
                answer=gen_result["answer"],
                cache_hit=False,
                cache_distance=cache_res.distance,
                cache_latency=cache_latency,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=(time.time() - start_total) * 1000,
                prompt_tokens=gen_result["prompt_tokens"],
                completion_tokens=gen_result["completion_tokens"],
                total_tokens=gen_result["total_tokens"],
                context=context,
                query_mode=plan.mode,
                sub_questions=plan.sub_questions
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            generation_latency = (time.time() - start_gen) * 1000
            # Do NOT cache failed requests
            return AnswerResult(
                answer=f"Generation failed: {e}",
                cache_hit=False,
                cache_distance=cache_res.distance,
                cache_latency=cache_latency,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=(time.time() - start_total) * 1000,
                context=context,
                query_mode=plan.mode,
                sub_questions=plan.sub_questions
            )
